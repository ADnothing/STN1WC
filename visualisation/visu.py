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

def full_visu(hdul, cat_path, disp_prob=False):
	"""
	Visualizes all predicted sources from a catalog over a representative slice of the data cube.

	Parameters:
		hdul: The FITS file handle representing the 3D radio cube.
		cat_path: Path to the source catalog file.
		disp_prob (optionnal): Display the probabilities for each source in the image.

	Saves an image "images/full.pdf" with a 2D view of RA/DEC and markers indicating source locations,
	colored by frequency and size scaled by objectness.
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

	if disp_prob:
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
	Visualizes sources from a catalog on a spectral data cube.

	Parameters:
	- hdul: FITS HDUList containing the data cube.
	- cat_path: Path to the text catalog containing source information.
	"""

	cat = np.loadtxt(cat_path, skiprows=1, usecols=(0, 1, 2, 3, 5, 8, 9, 10))
	sou_id, ra, dec, freq_size, central_freq, w20_arcsec, objness, proba = cat.T

	header = hdul[0].header
	cube_data = np.squeeze(hdul[0].data)
	wcs_full = WCS(header)
	wcs_sky = WCS(header, naxis=2)

	for idx in range(len(sou_id)):

		source_id = int(sou_id[idx])
		center_coord = SkyCoord(ra[idx]*u.deg, dec[idx]*u.deg, frame='icrs')

		#Convert sky coordinates to pixel positions
		x_pix, y_pix = utils.skycoord_to_pixel(center_coord, wcs_full.celestial, origin=0)
		z_pix = int(np.round((central_freq[idx] - header["CRVAL3"]) / header["CDELT3"] + header["CRPIX3"] - 1))

		#Round and clip pixel positions to valid cube indices
		x_pix = int(np.round(x_pix))
		y_pix = int(np.round(y_pix))
		z_pix = int(np.clip(z_pix, 0, cube_data.shape[0] - 1))

		#Extract a small cube around the source and average along the frequency axis (20 channels)
		cutout_2d = cube_data[
					max(0, z_pix - 10):min(cube_data.shape[0], z_pix + 10),
					max(0, y_pix - 50):min(cube_data.shape[1], y_pix + 50),
					max(0, x_pix - 50):min(cube_data.shape[2], x_pix + 50)
				     ].mean(axis=0)

		cutout_wcs_2d = wcs_full.celestial.slice((slice(None), slice(None)))
		cutout_wcs_2d.wcs.crpix -= [max(0, x_pix - 50), max(0, y_pix - 50)]

		fig = plt.figure(figsize=(8, 10))
		gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.6], hspace=0.3)

		#ALYCS:
		#Anthore LADUMA YOLO-CIANA Sources
		title_str = f"ALYCS{source_id:04d}  O = {int(round(objness[idx]*100))}%  P = {int(round(proba[idx]*100))}%"
		fig.suptitle(title_str, fontsize=14, fontweight='bold')


		ax1 = fig.add_subplot(gs[0], projection=cutout_wcs_2d)
		norm = simple_norm(cutout_2d, 'asinh', percent=90)
		ax1.imshow(cutout_2d, origin='lower', cmap='hot', norm=norm)

		w20_pix = w20_arcsec[idx]/(abs(header['CDELT1'])*3600)
		rect1 = Rectangle((50 - w20_pix/2, 50 - w20_pix/2), w20_pix, w20_pix,
					edgecolor='green', facecolor='none', lw=2)
		ax1.add_patch(rect1)

		ax1.coords[0].set_axislabel("RA")
		ax1.coords[1].set_axislabel("DEC")

		ra_center = int(np.round(x_pix))
		dec_center = int(np.round(y_pix))
		freq_start = max(0, z_pix - 80)
		freq_end = min(cube_data.shape[0], z_pix + 80)
		dec_start = max(0, dec_center - 50)
		dec_end = min(cube_data.shape[2], dec_center + 50)
		ra_slice = slice(max(0, ra_center - 10), min(cube_data.shape[1], ra_center + 10))

		slice_freq_ra = cube_data[freq_start:freq_end, ra_slice, dec_start:dec_end].mean(axis=1)

		freq_axis = header['CRVAL3'] + header['CDELT3'] * (np.arange(freq_start, freq_end) - (header["CRPIX3"] - 1))
		freq_axis_GHz = freq_axis / 1e9

		dec_pix = np.arange(dec_start, dec_end)
		N = len(dec_pix)
		pix_coords = np.zeros((N, 4))
		pix_coords[:, 0] = ra_center
		pix_coords[:, 1] = dec_pix
		pix_coords[:, 2] = z_pix
		pix_coords[:, 3] = 0

		world_coords = wcs_full.wcs_pix2world(pix_coords, 0)
		dec_deg = world_coords[:, 1]

		ax2 = fig.add_subplot(gs[1])
		ax2.imshow(slice_freq_ra.T, aspect='auto', origin='lower',
				extent=[freq_axis_GHz[0], freq_axis_GHz[-1], dec_deg[0], dec_deg[-1]],
				cmap='hot')
		ax2.set_xlabel("Frequency (GHz)")
		ax2.set_ylabel("DEC (deg)")

		box_freq_width = freq_size[idx]*header["CDELT3"]/1e9  #GHz
		box_dec_height = w20_arcsec[idx]/3600  #degrees

		rect2 = Rectangle((central_freq[idx]/1e9 - box_freq_width/2, dec[idx] - box_dec_height/2),
					box_freq_width, box_dec_height,
					edgecolor='green', facecolor='none', lw=2)
		ax2.add_patch(rect2)

		plt.tight_layout()
		plt.savefig(f"./images/source_ALYCS{source_id:04d}.pdf", dpi=200)
		plt.close(fig)


if __name__ == "__main__":

	if not os.path.exists("./images/"):
		os.makedirs("./images/")


	hdul = fits.open(cube_file_path, memmap=True)
	wcs_cube = WCS(hdul[0].header)

	full_visu(hdul, cat_path)
	sources_visu(hdul, cat_path)
