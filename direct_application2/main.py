#/usr/bin/env python3.11.2-deb12

import config

from aux_fct import *
from pred_data_gen import *
from pred import *

CIANNA_path = "/minerva/dcornu/CIANNA/src/build/lib.*/"
sys.path.insert(0,glob.glob(CIANNA_path)[-1])
import CIANNA as cnn

if __name__ == "__main__":

	config.net_path = sys.argv[1]
	data_list_file = sys.argv[2]

	data_path_list = []
	if not os.path.exists(data_list_file):
		raise FileNotFoundError(f"File not found: {data_list_file}")
	with open(data_list_file, 'r') as file:
		for line in file:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split()
			if len(parts) != 2:
				raise ValueError(f"Invalid line in data list file: {line}")
			path, scaling = parts
			if not os.path.exists(path):
				raise FileNotFoundError(f"File not found: {path}")
			data_path_list.append((path, float(scaling)))


	#Initialize CIANNA model
	cnn.init(in_dim=i_ar([config.sky_size, config.sky_size,config.freq_size]), in_nb_ch=1, 
                        out_dim=1+config.max_nb_obj_per_image*(7+config.nb_param),
                        bias=0.1, b_size=16, comp_meth='C_CUDA', dynamic_load=1, 
                        mixed_precision="FP16C_FP32A", inference_only=1, adv_size=30)

	nb_yolo_filters = cnn.set_yolo_params(no_override = 0, raw_output = 0)

	cnn.load(config.net_path,config.load_epoch, bin=1)

	for c_path, c_scaling in data_path_list:
	
		config.cube_file_path = c_path
		config.prenorm_scaling = c_scaling

		print("Data:", config.cube_file_path)

		hdul = fits.open(config.cube_file_path, memmap=True)

		config.map_pixel_size = hdul[0].data[0].shape[1]
		config.map_pixel_freq_size = hdul[0].data[0].shape[0]
		config.pixel_size_freq = hdul[0].header["CDELT3"]
		config.min_cube_freq = hdul[0].header["CRVAL3"]
		config.rest_freq = hdul[0].header["RESTFRQ"]
		hdul.close()

		#####   INFERENCE RELATED GLOBAL VARIABLES    #####
		config.yolo_nb_sky_reg = int(config.sky_size/config.c_size_sky)
		config.yolo_nb_freq_reg = int(config.freq_size/config.c_size_freq)

		config.overlap_sky = config.c_size_sky
		config.overlap_freq = 2*config.c_size_freq
		config.patch_shift_sky = config.sky_size - config.overlap_sky
		config.patch_shift_freq = config.freq_size - config.overlap_freq

		config.orig_offset_sky = config.patch_shift_sky - ((int(config.map_pixel_size/2) - int(config.sky_size/2) + config.patch_shift_sky)%config.patch_shift_sky)
		config.orig_offset_freq = config.patch_shift_freq - ((int(config.map_pixel_freq_size/2) - int(config.freq_size/2) + config.patch_shift_freq)%config.patch_shift_freq)

		config.nb_area_sky = int((config.map_pixel_size+2*config.orig_offset_sky)/config.patch_shift_sky)
		config.nb_area_freq = int((config.map_pixel_freq_size+2*config.orig_offset_freq)/config.patch_shift_freq)

		date_str = datetime.now().strftime("%d_%m_%y")
		z_max = (config.rest_freq - config.min_cube_freq)/config.min_cube_freq
		z_max_str = str(np.round(z_max, 3)).replace(".", "-")
		config.cat_name = f"./catalogs/LADUMA_pred_z{z_max_str}_{date_str}.txt"
		n = 1
		while os.path.exists(config.cat_name):
			config.cat_name = f"./catalogs/LADUMA_pred_z{z_max_str}_{date_str}_{n}.txt"
			n += 1


		init_data_gen()
		fwd_process()
