#/usr/bin/env python3.11.2-deb12

import config

import numpy as np
from scipy import signal

import os, gc, sys, glob
from numba import jit
from tqdm import tqdm

import time
from datetime import datetime

from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy import units as u
from astropy.coordinates import SkyCoord

#=========================================================================================================================

@jit(nopython=True, cache=True, fastmath=False)
def fct_DIoU(box1, box2):
	"""
	Compute the Distance Intersection over Union (DIoU) between two 3D bounding boxes.

	DIoU extends classical IoU by penalizing the spatial distance between the center points
	of the two boxes, encouraging not only overlap but also closeness.

	-> typical results in 2D explained in Cornu+2024, Fig A.3

	Parameters:
		box1: array-like of shape (n>=6,) with format
			[x_min, y_min, z_min, x_max, y_max, z_max, ...]

		box2: array-like in the same format

	Returns:
		DIoU: float

	DIoU score, higher is a better match.
	1.0 => perfect match
	DIoU<0 => miss matched (miss centered or away from target)
	"""

	#Compute the width and height (RA, DEC), and depth (Freq) of the intersection box
	inter_w = max(0.0, min(box1[3], box2[3]) - max(box1[0], box2[0]))	#along RA
	inter_h = max(0.0, min(box1[4], box2[4]) - max(box1[1], box2[1]))	#along DEC
	inter_d = max(0.0, min(box1[5], box2[5]) - max(box1[2], box2[2]))	#along Freq
	#Compute the intersection volume (rectangular cuboid)
	inter_3d = float(inter_w * inter_h * inter_d)

	#Compute the union volume (total volume discounting the intersection)
	vol_box1 = abs(box1[3]-box1[0])*abs(box1[4]-box1[1])*abs(box1[5]-box1[2])
	vol_box2 = abs(box2[3]-box2[0])*abs(box2[4]-box2[1])*abs(box2[5]-box2[2])
	union_3d = float(vol_box1 + vol_box2 - inter_3d)

	#Classical Intersection over Union
	IoU = inter_3d/union_3d

	#Compute the dimensions of the smallest enclosing square box
	enclose_w = max(box1[3], box2[3]) - min(box1[0], box2[0])
	enclose_h = max(box1[4], box2[4]) - min(box1[1], box2[1])
	enclose_d = max(box1[5], box2[5]) - min(box1[2], box2[2])
	#Euclidean distance between the two opposite corner of the enclosing box (i.e. diagonal)
	diag_enclose = float(np.sqrt(enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d))

	#Center coordinates of each box
	cx_1 = (box1[3] + box1[0])*0.5 ; cx_2 = (box2[3] + box2[0])*0.5
	cy_1 = (box1[4] + box1[1])*0.5 ; cy_2 = (box2[4] + box2[1])*0.5
	cz_1 = (box1[5] + box1[2])*0.5 ; cz_2 = (box2[5] + box2[2])*0.5
	#Euclidean distance between box centers
	dist_cent = float(np.sqrt((cx_1 - cx_2)*(cx_1 - cx_2) + (cy_1 - cy_2)*(cy_1 - cy_2) + (cz_1 - cz_2)*(cz_1 - cz_2)))

	#DIoU = IoU - (distance between centers / diagonal of enclosing box)
	return IoU - (dist_cent/diag_enclose)

#=========================================================================================================================

@jit(nopython=True, cache=True, fastmath=False)
def tile_filter(c_pred, c_box, c_tile, nb_box_tot):
	"""
	Extracts predicted bounding boxes from a YOLO-like 3D grid output and filters them based on objectness score.

	Parameters:
		c_pred: ndarray of shape (nb_box_tot*(8+nb_param), f, dec, ra)
			Raw network output containing bounding box parameters.
		c_box: temporary 1D array (length = 8 + nb_param + 1) used to build boxes.
		c_tile: output buffer to store kept boxes (modified in-place).
		nb_box_tot: number of boxes predicted per cell.

	Returns:
		c_nb_box: int

	Number of valid boxes stored in c_tile.
	"""

	c_nb_box = 0  #Counter for valid boxes

	for p_f in range(config.yolo_nb_freq_reg):			#Loop over frequency regions
		for p_dec in range(config.yolo_nb_sky_reg):		#Loop over declination regions
			for p_ra in range(config.yolo_nb_sky_reg):	#Loop over right ascension regions

				for k in range(nb_box_tot):		#Loop over all detected boxes

					offset = int(k*(8+config.nb_param))

					#Extract probability and objectness
					c_box[6] = c_pred[offset+6,p_f,p_dec,p_ra] #probability
					c_box[7] = c_pred[offset+7,p_f,p_dec,p_ra] #objectness

					#Manual objectness penality on the edges of the images (help for both obj selection and NMS)
					if(p_ra == 0 or p_ra == config.yolo_nb_sky_reg-1 or \
                                           p_dec == 0 or p_dec == config.yolo_nb_sky_reg-1 or\
                                           p_f == 0 or p_f == config.yolo_nb_freq_reg-1):

						c_box[6] = max(0.03,c_box[6]-0.05)
						c_box[7] = max(0.03,c_box[7]-0.05)

					#Filter by objectness threshold
					if (c_box[7] >= 0.2):
						#Compute box center (bx/by/bz) and size (bw/bh/bd)
						bx = (c_pred[offset+0,p_f,p_dec,p_ra] + c_pred[offset+3,p_f,p_dec,p_ra])*0.5
						by = (c_pred[offset+1,p_f,p_dec,p_ra] + c_pred[offset+4,p_f,p_dec,p_ra])*0.5
						bz = (c_pred[offset+2,p_f,p_dec,p_ra] + c_pred[offset+5,p_f,p_dec,p_ra])*0.5

						bw = max(5.0, c_pred[offset+3,p_f,p_dec,p_ra] - c_pred[offset+0,p_f,p_dec,p_ra])
						bh = max(5.0, c_pred[offset+4,p_f,p_dec,p_ra] - c_pred[offset+1,p_f,p_dec,p_ra])
						bd = max(5.0, c_pred[offset+5,p_f,p_dec,p_ra] - c_pred[offset+2,p_f,p_dec,p_ra])

						#Convert to corner format: [x_min, y_min, z_min, x_max, y_max, z_max]
						c_box[0] = bx - bw*0.5; c_box[3] = bx + bw*0.5
						c_box[1] = by - bh*0.5; c_box[4] = by + bh*0.5
						c_box[2] = bz - bd*0.5; c_box[5] = bz + bd*0.5

						#Save box
						c_box[8] = k
						c_box[9:9+config.nb_param] = c_pred[offset+8:offset+8+config.nb_param,p_f,p_dec,p_ra]
						c_box[-1] = p_f*config.yolo_nb_freq_reg*config.yolo_nb_sky_reg + p_dec*config.yolo_nb_sky_reg + p_ra

						#Store the final box
						c_tile[c_nb_box,:] = c_box[:]
						c_nb_box += 1

	return c_nb_box

#=========================================================================================================================

@jit(nopython=True, cache=True, fastmath=False)
def first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, nms_threshold):
	"""
	Perform a Non-Maximum Suppression (NMS) within a YOLO sub-cube patches.

	Parameters:
		c_tile: output buffer to store kept boxes (modified in-place).
		c_tile_kept: output array of same shape to hold filtered boxes after NMS.
		c_box: temporary array for copying and comparing boxes.
		c_nb_box: int, number of boxes initially present in c_tile.
		nms_threshold: float, DIoU threshold above which boxes are considered overlapping and suppressed.

	Returns:
		c_nb_box_final: int

	Number of valid boxes stored in c_tile.
	"""

	c_box_size_prev = c_nb_box	#Number of boxes to scan
	c_nb_box_final = 0		#Number of boxes kept

	#Loop over all the boxes until they are either kept or rejected
	while c_nb_box > 0:
		#Find the box with the maximum objectness score
		max_objct = np.argmax(c_tile[:c_box_size_prev,7])

		#Copy the best-scoring box into c_box
		c_box = np.copy(c_tile[max_objct])

		#Set its objectness to 0.0 to mark it as treated
		c_tile[max_objct,7] = 0.0

		#Keep it in the final output
		c_tile_kept[c_nb_box_final] = c_box
		c_nb_box_final += 1; c_nb_box -= 1; i = 0

		#Compare this box with all others in the tile
		for i in range(c_box_size_prev):
			if c_tile[i, 7] < 0.0000000001:
				continue

			#Compute Distance-IoU
			IoU = fct_DIoU(c_box[:6], c_tile[i,:6])

			#Suppress if IoU exceeds threshold
			if(IoU >  nms_threshold):
				c_tile[i,7] = 0.0  #Set its objectness to 0.0 to mark it as treated
				c_nb_box -= 1       #Reduce count

	return c_nb_box_final

#=========================================================================================================================

@jit(nopython=True, cache=True, fastmath=False)
def inter_patch_NMS(boxes, comp_boxes, c_tile, direction, overlap, patch_shift, patch_size, nms_threshold):
	"""
	Apply Non-Maximum Suppression across neighboring YOLO sub-cube patches	to eliminate overlapping predictions along shared borders.

	Parameters:
		boxes: ndarray (N, M), bounding boxes in the current patch.
		comp_boxes: ndarray (N, M), boxes from the neighboring patch to compare with.
			This array is modified in-place (shifted by direction * patch_shift).
		c_tile: output buffer to store kept boxes (modified in-place).
		direction: array-like of shape (3,), indicating neighbor direction in x/y/z (values in [-1, 0, 1]).
		overlap: array-like, size of the overlapping region to preserve.
		patch_shift: array-like, size of the step between adjacent patches (to realign neighbors).
		patch_size: array-like, full size of a patch along each axis (RA, DEC, FREQ).
		nms_threshold: float, DIoU threshold to suppress overlapping boxes.

	Returns:
		nb_box_kept: int

	Number of valid boxes stored in c_tile.
	"""

	#Reset output
	c_tile[:,:] = 0.0
	nb_box_kept = 0

	#Determine which boxes are fully inside non-overlapping region
	mask_keep = np.where((boxes[:,0] > overlap[0]) & (boxes[:,3] < patch_shift[0]) &\
                                                 (boxes[:,1] > overlap[1]) & (boxes[:,4] < patch_shift[1]) &\
                                                 (boxes[:,2] > overlap[2]) & (boxes[:,5] < patch_shift[2]))[0]

	#Determine boxes that lie on the shared border and need NMS
	mask_remain = np.where((boxes[:,0] <= overlap[0]) | (boxes[:,3] >= patch_shift[0]) |\
                                                   (boxes[:,1] <= overlap[1]) | (boxes[:,4] >= patch_shift[1]) |\
                                                   (boxes[:,2] <= overlap[2]) | (boxes[:,5] >= patch_shift[2]))[0]

	#Copy safe boxes directly to output
	nb_box_kept = np.shape(mask_keep)[0]
	c_tile[0:nb_box_kept,:] = boxes[mask_keep,:]

	#Shift comparison boxes into current patch's frame of reference
	comp_boxes[:,0:3] += direction[:]*patch_shift[:]
	comp_boxes[:,3:6] += direction[:]*patch_shift[:]

	#Keep only comparison boxes that are within patch bounds
	comp_mask_keep = np.where((comp_boxes[:,0] < patch_size[0]) & (comp_boxes[:,3] > 0) &\
                                                          (comp_boxes[:,1] < patch_size[1]) & (comp_boxes[:,4] > 0) &\
                                                          (comp_boxes[:,2] < patch_size[2]) & (comp_boxes[:,5] > 0))[0]

	#For each border box, check if it overlaps too much with a better prediction in the neighbor
	for b_ref in mask_remain:
                found = 0
                for b_comp in comp_mask_keep:
                        IoU = fct_DIoU(boxes[b_ref,:6], comp_boxes[b_comp,:6])

                        if(IoU > nms_threshold and boxes[b_ref,7] < comp_boxes[b_comp,7]):
                                found = 1
                                break
                if(found == 0):
                        c_tile[nb_box_kept,:] = boxes[b_ref,:]
                        nb_box_kept += 1

	return nb_box_kept

#=========================================================================================================================

#Expect fits files without the extension
def cube_norm(cube_path, prefix):
	"""
	Normalize and optionally noise-augment a 3D radio data cube for inference.

	This function loads a FITS cube, optionally adds correlated noise to simulate higher
	frequency resolution (via freq_exp_factor), computes per-channel normalization factors,
	and writes normalized binary sub-cubes to disk for model input.

	Parameters:
		cube_path: Path to the FITS cube to be normalized.

		prefix: Prefix used for output files written to disk.
	"""

	global cube_data, continuum_data, c_norm
	global nb_sky_patch_per_subcube, sub_cube_pixel_size, nb_sub_cube_per_dim
	global name_prefix

	name_prefix = prefix

	print("Loading cube to normalize ...")
	hdul = fits.open(cube_path, memmap=True)
	wcs_cube = WCS(hdul[0].header)

	cube_data = hdul[0].data[0]


	#Note: Be careful to compute std **BEFORE** converting NaN values to 0.
	#Otherwise, the empty areas in the data will have a huge impact on the normalization.
	c_norm = np.zeros(np.shape(cube_data)[0])
	for i in tqdm(range(0, np.shape(cube_data)[0])):
                cube_slice = np.asarray(cube_data[i], dtype="float32")
                c_norm[i] = np.nanstd(cube_slice, axis=(0,1))

	np.nan_to_num(cube_data[:,:,:], copy=False, nan=0.0)

	#Decide how to split the cube into sub-cubes for disk-saving and memory handling
	if(config.cube_spliting):
		encoding_bytes = 2
		nb_sky_patch_per_subcube = int((np.sqrt(config.size_file_split_limit*1e9/(encoding_bytes*(config.map_pixel_freq_size+2*config.orig_offset_freq))) - config.overlap_sky)/config.patch_shift_sky)
		sub_cube_pixel_size = nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky
		nb_sub_cube_per_dim = int(np.ceil((config.map_pixel_size + config.orig_offset_sky*2)/sub_cube_pixel_size))
		print(sub_cube_pixel_size, nb_sub_cube_per_dim)
	else:
		nb_sky_patch_per_subcube = config.nb_area_sky
		sub_cube_pixel_size = nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky
		nb_sub_cube_per_dim = 1


	#Tried parallelization on chunck loading but reduced performances, strongly I/O limited
	for n_h in tqdm(range(0,nb_sub_cube_per_dim)):
		for n_w in range(0,nb_sub_cube_per_dim):

			#Allocate buffer for this subcube, with border padding
			norm_data = np.zeros((config.map_pixel_freq_size+2*config.orig_offset_freq,
                                min(sub_cube_pixel_size, (config.map_pixel_size + config.orig_offset_sky*2 - n_h*(sub_cube_pixel_size-config.overlap_sky))),
                                min(sub_cube_pixel_size, (config.map_pixel_size + config.orig_offset_sky*2 - n_w*(sub_cube_pixel_size-config.overlap_sky)))), dtype="uint16")

			#Compute spatial coordinates of current patch
			y_min = n_h*nb_sky_patch_per_subcube*config.patch_shift_sky - config.orig_offset_sky
			y_max = (n_h+1)*nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky - config.orig_offset_sky
			x_min = n_w*nb_sky_patch_per_subcube*config.patch_shift_sky - config.orig_offset_sky
			x_max = (n_w+1)*nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky - config.orig_offset_sky

			#Clamp edges to cube dimensions
			orig_y_min = max(0, y_min)
			orig_y_max = min(config.map_pixel_size, y_max)
			orig_x_min = max(0, x_min)
			orig_x_max = min(config.map_pixel_size, x_max)

			#Extract corresponding data block
			data = np.asarray(cube_data[:,orig_y_min:orig_y_max,orig_x_min:orig_x_max], dtype="float32")

			#If required, smooth the frequency axis
			if(config.freq_smoothing):
				kernel = np.zeros((config.kernel_size)) + 1.0
				conv = np.zeros((config.map_pixel_freq_size))
				for k in tqdm(range(0, config.map_pixel_size)):
					for l in range(0, config.map_pixel_size):
						conv[:] = signal.convolve(data[:,k,l], kernel, mode="same")
						data[:,k,l] = conv[:]

			#Renormalize by each channel noise, scale, and tanh transform, them remap in [0, 1]
			for i in range(0, np.shape(data)[0]):
				data[i,:,:] = (np.tanh(config.prenorm_scaling*data[i,:,:] / c_norm[i]) + 1.0)*0.5

			end_y_min = max(0, -y_min)
			end_y_max = sub_cube_pixel_size - max(0, y_max - config.map_pixel_size)
			end_x_min = max(0, -x_min)
			end_x_max = sub_cube_pixel_size - max(0, x_max - config.map_pixel_size)

                        #Scale to fit in uint16 format and save binary
			norm_data[:,:,:] = 0.5*65535.0
			norm_data[config.orig_offset_freq:-config.orig_offset_freq,end_y_min:end_y_max,end_x_min:end_x_max] = np.asarray(data[:,:,:] * 65535.0, dtype="uint16")
			norm_data.tofile(config.work_path+name_prefix+"_%d_%d.bin"%(n_h, n_w))

			del (norm_data)
			gc.collect()
