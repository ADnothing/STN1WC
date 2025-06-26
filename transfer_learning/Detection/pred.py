#/usr/bin/env python3.11.2-deb12

import config

from threading import Thread
import subprocess

from aux_fct import *
from pred_data_gen import *


CIANNA_path = "/minerva/dcornu/CIANNA/src/build/lib.*/"
sys.path.insert(0,glob.glob(CIANNA_path)[-1])
import CIANNA as cnn


def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def load_next_cube(k,l):
	input_test, targets_test = create_test_batch(k,l)
	cnn.delete_dataset("TEST_buf", silent=1)
	cnn.create_dataset("TEST_buf", int(np.shape(input_test)[0]), input_test[:,:], targets_test[:,:], silent=1)
	return

def post_process_cube(epoch, k,l):
	process_pred("fwd_res/net0_%04d_%d_%d.dat"%(epoch,k,l), "pred_subfiles/filtered_pred_%d_%d.txt"%(k,l), k, l)
	return


# === INFERENCE EXECUTION BLOCK ===
def fwd_process():
	

	#Calculate how the cube is split into subcubes
	if(config.cube_spliting):
		encoding_bytes = 2
		nb_sky_patch_per_subcube = int((np.sqrt(config.size_file_split_limit*1e9/(encoding_bytes*(config.map_pixel_freq_size+2*config.orig_offset_freq))) - config.overlap_sky)/config.patch_shift_sky)
		sub_cube_pixel_size = nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky
		nb_sub_cube_per_dim = int(np.ceil((config.map_pixel_size + config.orig_offset_sky*2)/sub_cube_pixel_size))
	else:
		nb_sky_patch_per_subcube = config.nb_area_sky
		sub_cube_pixel_size = nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky
		nb_sub_cube_per_dim = 1

	#Load the first subcube for inference
	input_test, targets_test = create_test_batch(0,0)
	cnn.create_dataset("TEST", int(np.shape(input_test)[0]), input_test[:,:], targets_test[:,:], silent=1)

	#Loop over all subcubes to forward pass and post-process
	for i in range(0,nb_sub_cube_per_dim*nb_sub_cube_per_dim):
		k = int(i / nb_sub_cube_per_dim)
		l = int(i % nb_sub_cube_per_dim)
		print ("Forwarding subcube:", k, l)

		#Preload next subcube in background
		if(i+1 < nb_sub_cube_per_dim*nb_sub_cube_per_dim):
			k_l = int((i+1) / nb_sub_cube_per_dim)
			l_l = int((i+1) % nb_sub_cube_per_dim)
			t = Thread(target=load_next_cube, args=(k_l,l_l,))
			t.start()

		#Forward pass current subcube
		cnn.forward(saving=2, no_error=1)
		os.system("mv fwd_res/net0_%04d.dat fwd_res/net0_%04d_%d_%d.dat"%(config.load_epoch, config.load_epoch, k, l))

		if(i > 0):
			t2.join()

		t2 = Thread(target=post_process_cube, args=(config.load_epoch,k,l,))
		t2.start()

		#Swap data buffers when next subcube is ready
		if(i+1 < nb_sub_cube_per_dim*nb_sub_cube_per_dim):
			t.join()
			cnn.swap_data_buffers("TEST")

	# === POST-PROCESS ===
	if(config.cube_spliting):
		encoding_bytes = 2
		nb_sky_patch_per_subcube = int((np.sqrt(config.size_file_split_limit*1e9/(encoding_bytes*(config.map_pixel_freq_size+2*config.orig_offset_freq))) - config.overlap_sky)/config.patch_shift_sky)
		sub_cube_pixel_size = nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky
		nb_sub_cube_per_dim = int(np.ceil((config.map_pixel_size + config.orig_offset_sky*2)/sub_cube_pixel_size))
	else:
		nb_sky_patch_per_subcube = config.nb_area_sky
		sub_cube_pixel_size = nb_sky_patch_per_subcube*config.patch_shift_sky + config.overlap_sky
		nb_sub_cube_per_dim = 1

	for i in range(0,nb_sub_cube_per_dim*nb_sub_cube_per_dim):
		k = int(i / nb_sub_cube_per_dim)
		l = int(i % nb_sub_cube_per_dim)
		print ("Forwarding subcube:", k, l)
		post_process_cube(config.load_epoch,k,l)


	#Wait for the last post-processing thread to finish
	if("t2" in locals()):
		t2.join()
		
	# === FINAL CATALOG ASSEMBLY ===
	if(1):
		assemble_and_build_catalog(config.cat_name)
		cnn.delete_dataset("TEST")

