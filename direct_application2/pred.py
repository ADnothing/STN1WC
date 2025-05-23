#/usr/bin/env python

from threading import Thread
import subprocess

from aux_fct import *
from pred_data_gen import *


sys.path.insert(0,glob.glob(CIANNA_path)[-1])
import CIANNA as cnn

start_time = time.time()

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def load_next_cube(k,l):
	input_test, targets_test = create_test_batch(k,l)
	cnn.delete_dataset("TEST_buf", silent=1)
	cnn.create_dataset("TEST_buf", int(np.shape(input_test)[0]), input_test[:,:], targets_test[:,:], silent=1)
	return

def post_process_cube(load_epoch, k,l):
	process_pred("fwd_res/net0_%04d_%d_%d.dat"%(load_epoch,k,l), "pred_subfiles/filtered_pred_%d_%d.txt"%(k,l), k, l)
	return

load_epoch = 0
init_data_gen()
net_path = sys.argv[1]

# === INFERENCE EXECUTION BLOCK ===
if(1):
	#Initialize CIANNA model
	cnn.init(in_dim=i_ar([sky_size,sky_size,freq_size]), in_nb_ch=1, 
                        out_dim=1+max_nb_obj_per_image*(7+nb_param),
                        bias=0.1, b_size=16, comp_meth='C_CUDA', dynamic_load=1, 
                        mixed_precision="FP16C_FP32A", inference_only=1, adv_size=30)

	nb_yolo_filters = cnn.set_yolo_params(no_override = 0, raw_output = 0)

	cnn.load(net_path,load_epoch, bin=1)

	#Calculate how the cube is split into subcubes
	if(cube_spliting):
		encoding_bytes = 2
		nb_sky_patch_per_subcube = int((np.sqrt(size_file_split_limit*1e9/(encoding_bytes*(map_pixel_freq_size+2*orig_offset_freq))) - overlap_sky)/patch_shift_sky)
		sub_cube_pixel_size = nb_sky_patch_per_subcube*patch_shift_sky + overlap_sky
		nb_sub_cube_per_dim = int(np.ceil((map_pixel_size + orig_offset_sky*2)/sub_cube_pixel_size))
	else:
		nb_sky_patch_per_subcube = nb_area_sky
		sub_cube_pixel_size = nb_sky_patch_per_subcube*patch_shift_sky + overlap_sky
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
		os.system("mv fwd_res/net0_%04d.dat fwd_res/net0_%04d_%d_%d.dat"%(load_epoch, load_epoch, k, l))

		if(i > 0):
			t2.join()

		t2 = Thread(target=post_process_cube, args=(load_epoch,k,l,))
		t2.start()

		#Swap data buffers when next subcube is ready
		if(i+1 < nb_sub_cube_per_dim*nb_sub_cube_per_dim):
			t.join()
			cnn.swap_data_buffers("TEST")

# === POST-PROCESS ONLY MODE ===
elif(1):
	if(cube_spliting):
		encoding_bytes = 2
		nb_sky_patch_per_subcube = int((np.sqrt(size_file_split_limit*1e9/(encoding_bytes*(map_pixel_freq_size+2*orig_offset_freq))) - overlap_sky)/patch_shift_sky)
		sub_cube_pixel_size = nb_sky_patch_per_subcube*patch_shift_sky + overlap_sky
		nb_sub_cube_per_dim = int(np.ceil((map_pixel_size + orig_offset_sky*2)/sub_cube_pixel_size))
	else:
		nb_sky_patch_per_subcube = nb_area_sky
		sub_cube_pixel_size = nb_sky_patch_per_subcube*patch_shift_sky + overlap_sky
		nb_sub_cube_per_dim = 1

	for i in range(0,nb_sub_cube_per_dim*nb_sub_cube_per_dim):
		k = int(i / nb_sub_cube_per_dim)
		l = int(i % nb_sub_cube_per_dim)
		print ("Forwarding subcube:", k, l)
		post_process_cube(load_epoch,k,l)


#Wait for the last post-processing thread to finish
if("t2" in locals()):
	t2.join()

# === FINAL CATALOG ASSEMBLY ===
if(1):
	assemble_and_build_catalog()

end_time = time.time()
print("Inference duration: %f s"%(end_time-start_time))
