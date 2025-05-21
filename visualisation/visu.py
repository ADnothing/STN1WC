#/usr/bin/env python3.11.2-deb12

import sys, os

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke


cube_file_path = "/minerva/dcornu/LADUMA_data/laduma_dr1.2_image.1304~1420MHz_clean.fits"
cat_path = sys.argv[1]

def full_visu(hdul, cat_path):
	"""
	"""
	
	background_data = np.squeeze(hdul[0].data)[hdul[0].header["CRPIX3"] - 1]
	wcs = WCS(hdul[0].header, naxis=2)
    
	cat = np.loadtxt(cat_path, skiprows=1, usecols=(1, 2, 5, 9, 10))
	ra, dec, freq, objectness, prob = cat.T

	pix_x, pix_y = wcs.world_to_pixel_values(ra, dec)

	norm = simple_norm(background_data, 'sqrt', percent=90)

	fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': wcs})
	ax.imshow(background_data, origin='lower', cmap='gray', norm=norm)

	cmap = plt.cm.viridis
	norm_freq = plt.Normalize(freq.min(), freq.max())
	colors = cmap(norm_freq(freq))
	size = 100*objectness
	
	ax.scatter(pix_x, pix_y,
               c=colors,
               s=size,
               marker='o',
               alpha=0.7,
               zorder=3)
               
	for x, y, p in zip(pix_x, pix_y, prob):
		ax.text(x, y, f'{p:.2f}', color='white', fontsize=8, ha='center', va='center', path_effects=[withStroke(linewidth=2, foreground='black')])

	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_freq)
	sm.set_array([])
	plt.colorbar(sm, ax=ax, label='Central Frequency')

	ax.set_xlabel('RA')
	ax.set_ylabel('DEC')
	plt.tight_layout()
	plt.savefig("./images/full.pdf", dpi=200)
	plt.close(fig)
	
	
		
	
def sources_visu(hdul, cat_path):
	"""
	"""
	
	cat = np.loadtxt(cat_path, skiprows=1, usecols=(0, 1, 2, 3, 5, 8, 9, 10))
	sou_id, ra, dec, freq_size, central_freq, w20_arcsec, objness, proba = cat.T
		
	
	
	
if __name__ == "__main__":

	if not os.path.exists("./images/"):
		os.makedirs("./images/")
	
	
	hdul = fits.open(cube_file_path, memmap=True)
	wcs_cube = WCS(hdul[0].header)

	full_visu(hdul, cat_path)
	sources_visu(hdul, cat_path)
