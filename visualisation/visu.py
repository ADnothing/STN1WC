#/usr/bin/env python3.11.2-deb12

import sys

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

cube_file_path = "/minerva/dcornu/LADUMA_data/laduma_dr1.2_image.1304~1420MHz_clean.fits"
cat_path = sys.argv[1]

def full_visu(hdul, cat_path):
	"""
	"""
	
	backgound_data = np.squeeze(hdul[0].data)[hdul[0].header["CRPIX3"]]
	
	cat = np.loadtxt(cat_path, skiprows=1, usecols=(1, 2, 5, 10))
	
	
		
	
def sources_visu(cat_path):
	"""
	"""
	
	cat = np.loadtxt(cat_path, skiprows=1, usecols=(1, 2, 3, 5, 8, 10))
	
if __name__ == "__main__":

	
	
	hdul = fits.open(cube_file_path, memmap=True)
	wcs_cube = WCS(hdul[0].header)

	full_visu(hdul, cat_path)
	sources_visu(cat_path)
