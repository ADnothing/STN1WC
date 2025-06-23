#/usr/bin/env python3.11.2-deb12

import config

from aux_fct import *
from pred_data_gen import *
from pred import *

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
		config.c_size_sky = 8
		config.c_size_freq = 16
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
		z_max_str = str(np.round(z_max)).replace(".", "-")
		config.cat_name = f"./catalogs/LADUMA_pred_z{z_max_str}_{date_str}.txt"
		n = 1
		while os.path.exists(config.cat_name):
			config.cat_name = f"./catalogs/LADUMA_pred_z{z_max_str}_{date_str}_{n}.txt"
			n += 1


		init_data_gen()
		fwd_process()
