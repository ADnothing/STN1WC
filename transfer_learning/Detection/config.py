#/usr/bin/env python3.11.2-deb12


######    HARD CODED GLOBAL VARIABLES AND DATA    #####

global work_path

work_path = "/scratch/aanthore/LADUMA_data/WORK2/" #Temp directory

#Normalization parameters

global do_norm, freq_smoothing
global kernel_size, cube_spliting, size_file_split_limit

do_norm = 1
freq_smoothing = 0
kernel_size = 20
cube_spliting = 1
size_file_split_limit = 4.0 #In Gigabytes

global prenorm_scaling

prenorm_scaling = None #Normalization factor

#####    NETWORK RELATED GLOBAL VARIABLES     #####

global sky_size, freq_size, nb_param, nb_box

sky_size = 64
freq_size = 256
nb_param = 6
nb_box = 1

#####    TRAINING RELATED GLOBAL VARIABLES    #####

global bootstrap, max_nb_obj_per_image

bootstrap = 0
max_nb_obj_per_image = 10

global load_epoch, net_path, cube_file_path, cat_name

load_epoch = 0
net_path = ""
cube_file_path = ""
cat_name = ""

global map_pixel_size, map_pixel_freq_size
global pixel_size_freq, min_cube_freq, rest_freq

map_pixel_size = None
map_pixel_freq_size = None
pixel_size_freq = None
min_cube_freq = None
rest_freq = None

#####   INFERENCE RELATED GLOBAL VARIABLES    #####
global c_size_sky, c_size_freq, yolo_nb_sky_reg, yolo_nb_freq_reg
global overlap_sky, overlap_freq, patch_shift_sky, patch_shift_freq
global orig_offset_sky, orig_offset_freq
global nb_area_sky, nb_area_freq

c_size_sky = 8
c_size_freq = 16
yolo_nb_sky_reg = None
yolo_nb_freq_reg = None

overlap_sky = None
overlap_freq = None
patch_shift_sky = None
patch_shift_freq = None

orig_offset_sky = None
orig_offset_freq = None

nb_area_sky = None
nb_area_freq = None
