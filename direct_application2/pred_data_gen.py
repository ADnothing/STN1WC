#!/usr/bin/env python3.11.2-deb12

import config

from aux_fct import *

def init_data_gen():
	"""
	Initialize the prediction data generation process.

	This function handles:
		Optional normalization of the data cube (via cube_norm if do_norm is 1 (True)).
		Opening the cube and reading its WCS header.
		Calculating the total padded cube size.
		Setting up sub-cube partitioning logic depending on whether cube splitting is enabled.
		Loading the already normalized cube binary if not using dynamic splitting.

	Sets the following global variables:
		norm_data: The full or partial normalized cube data (if preloaded).
		l_map_pixel_size, l_map_pixel_freq_size: Padded dimensions of the data cube.
		sub_cube_pixel_size, nb_sky_patch_per_subcube, nb_sub_cube_per_dim: Parameters for patch tiling.
	"""

	global norm_data, input_test, targets_test
	global l_map_pixel_size, l_map_pixel_freq_size
	global sub_cube_pixel_size, nb_sky_patch_per_subcube, nb_sub_cube_per_dim

	if(config.do_norm):
		#Full cube norm
		cube_norm(config.cube_file_path, "pred")

	hdul = fits.open(config.cube_file_path, memmap=True)
	wcs_cube = WCS(hdul[0].header)

	#Compute padded size of the data cube (adding margin offsets for tiling)
	l_map_pixel_size = config.map_pixel_size + config.orig_offset_sky*2
	l_map_pixel_freq_size = config.map_pixel_freq_size+2*config.orig_offset_freq

	if(config.cube_spliting):
		#Determine number of patches per sub-cube based on max allowed size
		encoding_bytes = 2
		nb_sky_patch_per_subcube = int((np.sqrt(config.size_file_split_limit*1e9/(encoding_bytes*(config.map_pixel_freq_size+2*config.orig_offset_freq))) - config.overlap_sky)/config.patch_shift_sky)
		sub_cube_pixel_size = nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky
		nb_sub_cube_per_dim = int(np.ceil((config.map_pixel_size + config.orig_offset_sky*2)/sub_cube_pixel_size))

	else:
		#Using cube splitting is expected here, but it would be possible to distribute individual sub-cube predictions by adjusting the coordinate bellow
		nb_sky_patch_per_subcube = config.nb_area_sky
		sub_cube_pixel_size = nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky
		nb_sub_cube_per_dim = 1
		norm_data = np.fromfile(config.work_path+"pred_0_0.bin", dtype="uint16")
		norm_data = np.reshape(norm_data, (l_map_pixel_freq_size, l_map_pixel_size, l_map_pixel_size))

#=========================================================================================================================

def create_test_batch(subcube_dec, subcube_ra):
	"""
	Generate test data batches for a given sub-cube position.

	This function reads normalized binary sub-cube data from disk, extracts overlappingpatches, flattens them,
	and fills input tensors formatted for YOLO-CIANNA.

	Parameters:
		subcube_dec: Index of the sub-cube in the DEC (vertical) direction.
		subcube_ra: Index of the sub-cube in the RA (horizontal) direction.

	Returns:
		input_test: ndarray
		targets_test: ndarray
	"""

	if(config.cube_spliting):
		subcube_w = min(sub_cube_pixel_size, (config.map_pixel_size + config.orig_offset_sky*2) - subcube_ra*nb_sky_patch_per_subcube*config.patch_shift_sky)
		subcube_h = min(sub_cube_pixel_size, (config.map_pixel_size + config.orig_offset_sky*2) - subcube_dec*nb_sky_patch_per_subcube*config.patch_shift_sky)

		l_nb_area_ra  = int(subcube_w/config.patch_shift_sky)
		l_nb_area_dec = int(subcube_h/config.patch_shift_sky)

		norm_data = np.fromfile(config.work_path+"pred_%d_%d.bin"%(subcube_dec, subcube_ra), dtype="uint16")
		norm_data = np.reshape(norm_data, (l_map_pixel_freq_size, subcube_h, subcube_w))
	else:
		l_nb_area_ra  = config.nb_area_sky
		l_nb_area_dec = config.nb_area_sky

	nb_images = l_nb_area_ra*l_nb_area_dec*config.nb_area_freq

	input_test = np.zeros((nb_images,1*config.sky_size*config.sky_size*config.freq_size), dtype="float32")
	targets_test = np.zeros((nb_images,1+config.max_nb_obj_per_image*(7+config.nb_param)), dtype="float32")

	#Diagnostic
	print ("STD subcube", np.std((norm_data.flatten()/65535)*2.0 - 1.0))

	for patch_freq in range(0,config.nb_area_freq):		#Loop over frequency regions
		for patch_dec in range(0,l_nb_area_dec):	#Loop over declination regions
			for patch_ra in range(0,l_nb_area_ra):	#Loop over right ascension regions

				#Linear index in the input_test array
				i = patch_freq*l_nb_area_dec*l_nb_area_ra + patch_dec*l_nb_area_ra + patch_ra

				#Compute coordinate offset for the patch
				p_ra   = patch_ra*config.patch_shift_sky
				p_dec  = patch_dec*config.patch_shift_sky
				p_freq = patch_freq*config.patch_shift_freq

				patch = np.copy(norm_data[p_freq:p_freq+config.freq_size, p_dec:p_dec+config.sky_size, p_ra:p_ra+config.sky_size])

				input_test[i,0:config.freq_size*config.sky_size*config.sky_size] = (patch.flatten("C")/65535.0)*2.0 - 1.0

				targets_test[i,:] = 0.0

	if(config.cube_spliting):
		del(norm_data)
		gc.collect()

	return input_test, targets_test

#=========================================================================================================================

def process_pred(process_file, save_file, subcube_dec, subcube_ra):
	"""
	Post-process the raw model predictions for a given sub-cube:
		* Parses model outputs from file.
		* Converts predicted boxes to usable format.
		* Applies two-stage Non-Maximum Suppression (tile-local and inter-tile).
		* Converts coordinates from patch-relative to cube-relative.
		* Saves filtered predictions to disk.

	Parameters:
		process_file: Path to the raw binary output from the model forward pass (flattened predictions).
		save_file: Path where the final filtered predictions will be saved (as a .txt).
		subcube_dec: Index of the sub-cube in the DEC (vertical) direction.
		subcube_ra: Index of the sub-cube in the RA (horizontal) direction.

	Returns:
		Returns 0 if successful, or early exit if no objects are found.
	"""

	if(config.cube_spliting):
		#Determine sub-cube size in padded space
		subcube_w = min(sub_cube_pixel_size, (config.map_pixel_size + config.orig_offset_sky*2) - subcube_ra*nb_sky_patch_per_subcube*config.patch_shift_sky)
		subcube_h = min(sub_cube_pixel_size, (config.map_pixel_size + config.orig_offset_sky*2) - subcube_dec*nb_sky_patch_per_subcube*config.patch_shift_sky)

		#Number of local patches in this sub-cube
		l_nb_area_ra  = int(subcube_w/config.patch_shift_sky)
		l_nb_area_dec = int(subcube_h/config.patch_shift_sky)

	else:
		l_nb_area_ra  = int(config.nb_area_sky)
		l_nb_area_dec = int(config.nb_area_sky)

	#Load raw prediction array
	pred_data = np.fromfile(process_file, dtype="float32")
	pred_data = np.reshape(pred_data, (config.nb_area_freq, l_nb_area_dec, l_nb_area_ra, config.nb_box*(8+config.nb_param), config.yolo_nb_freq_reg, config.yolo_nb_sky_reg, config.yolo_nb_sky_reg))

	final_boxes = []

	#First NMS working buffers
	c_tile = np.zeros((config.yolo_nb_sky_reg*config.yolo_nb_sky_reg*config.yolo_nb_freq_reg*config.nb_box,(8+1+config.nb_param+1)),dtype="float32")
	c_tile_kept = np.zeros((config.yolo_nb_sky_reg*config.yolo_nb_sky_reg*config.yolo_nb_freq_reg*config.nb_box,(8+1+config.nb_param+1)),dtype="float32")
	c_box = np.zeros((8+1+config.nb_param+1),dtype="float32")

	total_nb_box = 0

	#First NMS
	for p_freq in range(0,config.nb_area_freq):
		for p_dec in range(0,l_nb_area_dec):
			for p_ra in range(0,l_nb_area_ra):

				c_tile[:,:] = 0.0
				c_tile_kept[:,:] = 0.0

				c_pred = pred_data[p_freq,p_dec,p_ra,:,:,:]
				c_nb_box = tile_filter(c_pred, c_box, c_tile, config.nb_box)

				c_nb_box_final = c_nb_box
				c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, 0.1)

				total_nb_box += c_nb_box_final
				final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

	print(total_nb_box)
	if(total_nb_box < 1):
		print ("No prediction found, closing fwd.")
		if(os.path.isfile("pred_subfiles/filtered_pred_%d_%d.txt"%(subcube_dec,subcube_ra))):
			os.remove("pred_subfiles/filtered_pred_%d_%d.txt"%(subcube_dec,subcube_ra))
		return 0
	else:
		final_boxes = np.reshape(np.array(final_boxes, dtype="object"), (config.nb_area_freq, l_nb_area_dec, l_nb_area_ra))


	#Second NMS working buffer
	c_tile = np.zeros((config.yolo_nb_sky_reg*config.yolo_nb_sky_reg*config.yolo_nb_freq_reg*config.nb_box,(8+1+config.nb_param+1)),dtype="float32")

	A = np.array([-1, 0, 1])
	B = np.array([-1, 0, 1])
	C = np.array([-1, 0, 1])

	x, y, z = np.meshgrid(A, B, C)
	dir_array = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
	l_overlap = np.array((config.overlap_sky, config.overlap_sky, config.overlap_freq))
	l_patch_shift = np.array((config.patch_shift_sky, config.patch_shift_sky, config.patch_shift_freq))
	l_patch_size = np.array((config.sky_size, config.sky_size, config.freq_size))

	#Second NMS over all the overlapping patches
	for p_freq in range(0,config.nb_area_freq):
		for p_dec in range(0,l_nb_area_dec):
			for p_ra in range(0,l_nb_area_ra):
				boxes = np.copy(final_boxes[p_freq,p_dec,p_ra])
				for l in range(0,np.shape(dir_array)[0]):
					if(p_freq+dir_array[l,2] >= 0 and p_freq+dir_array[l,2] <= config.nb_area_freq-1 and\
					   p_dec +dir_array[l,1] >= 0 and p_dec +dir_array[l,1] <= l_nb_area_dec-1  and\
					   p_ra  +dir_array[l,0] >= 0 and p_ra  +dir_array[l,0] <= l_nb_area_ra-1 ):
						comp_boxes = np.copy(final_boxes[p_freq+dir_array[l,2],p_dec+dir_array[l,1],p_ra+dir_array[l,0]])
						c_nb_box = inter_patch_NMS(boxes, comp_boxes, c_tile, dir_array[l], l_overlap, l_patch_shift, l_patch_size, -0.3)
						boxes = np.copy(c_tile[0:c_nb_box,:])

				final_boxes[p_freq,p_dec,p_ra] = np.copy(boxes)

	#Convert to subcube pixel coordinates (and pixel centered coordinate system, expected for fits skycoord conversions)
	final_boxes_scaled = np.copy(final_boxes)
	for p_freq in range(0,config.nb_area_freq):
		box_freq_offset = p_freq*config.patch_shift_freq
		for p_dec in range(0,l_nb_area_dec):
			box_dec_offset = p_dec*config.patch_shift_sky
			for p_ra in range(0,l_nb_area_ra):
				box_ra_offset = p_ra*config.patch_shift_sky

				final_boxes_scaled[p_freq,p_dec,p_ra][:,0] = box_ra_offset   + final_boxes_scaled[p_freq,p_dec,p_ra][:,0] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,1] = box_dec_offset  + final_boxes_scaled[p_freq,p_dec,p_ra][:,1] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,2] = box_freq_offset + final_boxes_scaled[p_freq,p_dec,p_ra][:,2] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,3] = box_ra_offset   + final_boxes_scaled[p_freq,p_dec,p_ra][:,3] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,4] = box_dec_offset  + final_boxes_scaled[p_freq,p_dec,p_ra][:,4] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,5] = box_freq_offset + final_boxes_scaled[p_freq,p_dec,p_ra][:,5] - 0.5

	#Flatten and sort boxes by objectness score
	flat_kept_scaled = np.vstack(final_boxes_scaled.flatten())
	flat_kept_scaled = flat_kept_scaled[flat_kept_scaled[:,7].argsort(),:][::-1]

	np.savetxt(save_file, flat_kept_scaled)

	return 0

#=========================================================================================================================

def assemble_and_build_catalog(cat_path):
	"""
	Assemble and merge all subcube predictions into a final source catalog.

	This function:
		* Loads all saved prediction results per subcube.
		* Applies a final round of inter-subcube 3D NMS to remove duplicates across borders of subcubes.
		* Reconstructs the full-cube pixel coordinates from subcube-relative ones.
		* Converts pixel coordinates to sky coordinates using WCS.
		* Applies parameter rescaling using normalization statistics from training.
		* Outputs two files:
			net_pred_filtered_repos_rescaled.dat: raw cube-space box data
        		final_pred_catalog.txt: human-readable, calibrated catalog
        		
	Parameters:
		cat_path: path and name of the catalog. 
	"""

	subcube_grid = np.empty((nb_sub_cube_per_dim,nb_sub_cube_per_dim), dtype="object")
	grid_density = np.zeros((nb_sub_cube_per_dim,nb_sub_cube_per_dim), dtype="int")

	for k in range(0,nb_sub_cube_per_dim):
		for l in range(0,nb_sub_cube_per_dim):
			if(os.path.isfile("pred_subfiles/filtered_pred_%d_%d.txt"%(k,l))):
				l_cat = np.loadtxt("pred_subfiles/filtered_pred_%d_%d.txt"%(k,l))
			else:
				l_cat = np.array([])

			subcube_grid[k,l] = l_cat
			grid_density[k,l] = np.shape(l_cat)[0]

	#Third NMS working buffers
	A = np.array([-1, 0, 1])
	B = np.array([-1, 0, 1])
	C = np.array([0])

	x, y, z = np.meshgrid(A, B, C)
	dir_array = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
	l_overlap = np.array((config.overlap_sky, config.overlap_sky, 0))
	l_patch_shift = np.array((nb_sky_patch_per_subcube*config.patch_shift_sky, nb_sky_patch_per_subcube*config.patch_shift_sky, 0))
	l_patch_size = np.array((sub_cube_pixel_size, sub_cube_pixel_size, config.map_pixel_freq_size+2*config.orig_offset_freq))

	c_tile = np.zeros((np.max(grid_density),(8+1+config.nb_param+1)),dtype="float32")

	#Third NMS over all the overlapping patches
	for p_h in tqdm(range(0,nb_sub_cube_per_dim)):
		for p_w in range(0,nb_sub_cube_per_dim):
			if(grid_density[p_h,p_w] == 0):
				continue
			boxes = np.copy(subcube_grid[p_h,p_w]).reshape((-1,(8+1+config.nb_param+1)))
			for l in range(0,np.shape(dir_array)[0]):
				if(p_h+dir_array[l,1] >= 0 and p_h+dir_array[l,1] <= nb_sub_cube_per_dim-1 and\
				   p_w+dir_array[l,0] >= 0 and p_w+dir_array[l,0] <= nb_sub_cube_per_dim-1 ):
					if(grid_density[p_h+dir_array[l,1],p_w+dir_array[l,0]] == 0):
						continue
					comp_boxes = np.copy(subcube_grid[p_h+dir_array[l,1],p_w+dir_array[l,0]]).reshape((-1,(8+1+config.nb_param+1)))
					c_nb_box = inter_patch_NMS(boxes, comp_boxes, c_tile, dir_array[l], l_overlap, l_patch_shift, l_patch_size, -0.3)
					boxes = np.copy(c_tile[0:c_nb_box,:])

			subcube_grid[p_h,p_w] = np.copy(boxes)
			grid_density[p_h,p_w] = np.shape(boxes)[0]

	print (grid_density)

	#Convert to fullcube pixel coordinates
	if(config.cube_spliting):

		#Convert to fullcube pixel coordinates
		for p_h in tqdm(range(0,nb_sub_cube_per_dim)):
			box_h_offset = p_h*nb_sky_patch_per_subcube*config.patch_shift_sky
			for p_w in range(0,nb_sub_cube_per_dim):
				box_w_offset = p_w*nb_sky_patch_per_subcube*config.patch_shift_sky
				if(np.shape(subcube_grid[p_h,p_w])[0] > 0):
					subcube_grid[p_h,p_w][:,0] += box_w_offset
					subcube_grid[p_h,p_w][:,3] += box_w_offset
					subcube_grid[p_h,p_w][:,1] += box_h_offset
					subcube_grid[p_h,p_w][:,4] += box_h_offset

	mask = np.where(grid_density > 0)

	box_cat = np.vstack(subcube_grid[mask].flatten())
	box_cat = box_cat[box_cat[:,7].argsort(),:][::-1]

	#Remove orig offset
	box_cat[:,0] -= config.orig_offset_sky
	box_cat[:,1] -= config.orig_offset_sky
	box_cat[:,2] -= config.orig_offset_freq
	box_cat[:,3] -= config.orig_offset_sky
	box_cat[:,4] -= config.orig_offset_sky
	box_cat[:,5] -= config.orig_offset_freq

	print (np.shape(box_cat))

	#Optional filter near borders (disabled by default)
	if(0):
		index = np.where((box_cat[:,0] < 64) | (box_cat[:,1] < 64) | (box_cat[:,2] < 256) |\
						 (box_cat[:,3] > config.map_pixel_size-64) | (box_cat[:,4] > config.map_pixel_size-64) | (box_cat[:,5] > config.map_pixel_freq_size-256))[0]
		box_cat = np.delete(box_cat, index, axis=0)

	np.savetxt("net_pred_filtered_repos_rescaled.dat", box_cat)

	hdul = fits.open(config.cube_file_path, memmap=True)
	wcs_cube = WCS(hdul[0].header)

	cls = utils.pixel_to_skycoord((box_cat[:,3]+box_cat[:,0])*0.5, (box_cat[:,4]+box_cat[:,1])*0.5, wcs_cube)
	ra_dec_coords = np.array([cls.ra.deg, cls.dec.deg])

	cat_size = int(np.shape(box_cat)[0])

	cat_header = "id ra dec hi_size line_flux_integral central_freq pa i w20 obj prob"
	final_box_cat = np.zeros((cat_size,11), dtype="float32")

	#Load normalization constants used during training
	if(config.bootstrap):
		lims = np.loadtxt("train_cat_lims_bootstrap.txt")
	else:
		lims = np.loadtxt("train_cat_lims.txt")

	final_box_cat[:,0] = np.arange(0,cat_size)
	final_box_cat[:,[1,2]] = ra_dec_coords.T
	final_box_cat[:,3] = box_cat[:,10]*lims[1,0] + lims[1,1]
	final_box_cat[:,4] = np.exp(box_cat[:,9]*lims[0,0] + lims[0,1])
	final_box_cat[:,5] = (box_cat[:,5]+box_cat[:,2])*0.5*config.pixel_size_freq + config.min_cube_freq
	final_box_cat[:,6] = np.mod(np.arctan2(np.clip(box_cat[:,12],0.0,1.0)*2.0-1.0, np.clip(box_cat[:,13],0.0,1.0)*2.0-1.0)*180.0/np.pi,360.0)
	final_box_cat[:,7] = np.arccos(np.clip(box_cat[:,14],0.0,1.0))*180.0/np.pi
	final_box_cat[:,8] = ((np.exp(box_cat[:,11]*lims[2,0] + lims[2,1])*config.pixel_size_freq)/final_box_cat[:,5])*299792.458
	final_box_cat[:,9] = box_cat[:,7]
	final_box_cat[:,10]= box_cat[:,6]

	np.savetxt(cat_path, final_box_cat, header=cat_header, comments="", fmt="%d %3.13f %2.13f %1.13f %1.13f %10.1f %3.13f %2.13f %3.13f %1.6f %1.6f")
