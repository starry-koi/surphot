#!/usr/bin/env python
import numpy as np
import reproject as rp
import matplotlib.pyplot as plt
import photutils.segmentation as seg
import scipy.stats as scstats
import warnings
import decimal
from py_scripts import robust_stats as rstats #robust statistics functions based on IDL routines
from py_scripts import mask_enlarge as m
import ds9norm
import subprocess as sp
import commentjson as cjson
import os
import sys
from os import listdir
from os.path import isfile,join
from astropy.io import fits
from astropy import stats
from astropy import wcs
from astropy.utils.exceptions import AstropyWarning
from scipy import ndimage
from distutils.util import strtobool

#Author: Lilikoi Latimer
#Created May 2018
#Updated October 2022

'''
=========================================================================================


### Quirks and possible issues ###

- Have to have mask_enlarge and bresenham_march in the same directory, as they're both 
  user-made functions

- Only works for .fits files

- Sources at the edge of an image can git a bit weird with the differing background
  apertures, so if you have a source near an edge double-check that that isn't happening
  
- Code will not progress until the reference image matplotlib popup is closed

- Uses ONLY ONE REFERENCE IMAGE (in the /reference folder) and DOESN'T FIND FLUX FOR THE 
  REFERENCE IMAGE, so it is suggested that the user puts a copy of the reference image in
  the /total folder to find the flux. For example, the /reference folder could have im1.fits,
  and the /total folder could have im1.fits, im2.fits, im3.fits.

- Have to MANUALLY set the reference and total images (in the respective folders).

- Have to MANUALLY set (check VARIABLES section, though should set in params.config):
	- What sigma level to use for sigma-clipping 			(sigma_level)
	- What sigma level to use for identifying sources 		(sigma_source_id)
	- What sigma level to use for measuring sources			(sigma_measure)

- Some other variables which may need changing depending on your needs are:
  -focus_var and the associated center and box_side variables, which are for focusing in
   specific regions of the image instead of analyzing the whole thing in one go
  -deblent_var, which controls whether to deblend overlapping sources
  -plot_var, which controls how the main plot for the reference image looks. If analyzing
   an HST image, for instance, you may want this on 'log' so it looks better, while 
   radio data will likely look better with it on 'linear'

- The code creates a /results folder to store all the output plots and fluxes in. If you
  rerun the code, it will overwrite the files in the /results folder, so it's recommended
  to either delete the /results folder or to rename it to e.g. /results_old before 
  rerunning the code.

- For source-specific background apertures, the code makes 3 related background apertures
  at different distances from the source (small, original, large) as an attempt to get a
  more accurate estimate of the background for each source. This process excludes any
  nearby sources, so the background won't be contaminated by nearby sources.

- The aperture can sometimes get a bit messy if it kind of crosses back over on itself
  with respect to a line drawn from the center of mass of the source to the line used to
  find the aperture - basically, if the contour line is multiple-valued in polar coords
  with respect to the center of mass of the source being the point (0,0). 
  Essentially, if the contours look weird the aperture can look weird too.
  
- The code pops up a matplotlib plot so you can check that it's detecting sources the
  way you want it to. If correct, just x out of the plot and let the code continue 
  (the plot will get saved and outputted in the results folder), and if you want to
  change something, then after x-ing out of the popup, stop the code from running,
  change the sigma-related variables, and run the code again.
  


### Broad Overview ###

This program detects sources and finds fluxes of those sources in an image using sigma-clipping (like contour levels of the image noise). Specifically, this program reads in a single reference image (which the user has to put in the /reference folder) and uses that image to determine how many sources there are in the image, the extent of the sources, and what to make the source and background apertures (so we can find the background, and subtract the background from the flux). It also deblends the sources (if the deblent_var flag is set to True) e.g. if there are two sources that are not separated - see [https://photutils.readthedocs.io/en/stable/segmentation.html#source-deblending] for more.

We then take the arrays consisting of the sources and background apertures for the reference image, and reproject them into the coordinates of the images we want to find the flux of, which are stored in the /total folder. We then find the fluxes, backgrounds, and final background-subtracted fluxes for these sources in all the images in the /total folder.

Note that, in an attempt to estimate error on our fluxes, we calculate the final (background subtracted) flux using three different background apertures at different radii from the source. The final flux is the mean of these three calculated fluxes, and the final error on the flux is the standard deviation of these three calculated fluxes.

Finally, we plot all the sources with their respective background apertures, and write out information to the output.txt file in the /results folder, saving the plots as well.


=========================================================================================
'''
#################################################
################### VARIABLES ###################
#################################################
'''
Defaults of:

sigma_level 	= 3 		
sigma_source_id = 7 
sigma_measure	= 7
		
focused_var	= False
center		= np.array([0,0])
box_side	= 20

deblent_var	= True
plot_var	= 'linear'
show_plots	= False
bg_method 	= 'median'
print_option_1 	= 'everything' 	
savefig 	= 'savepdf'
print_type	= 'source' 
			
pix_thresh 	= 5
pref_bg 	= np.array([1.75,3.5]) 	
bg_fac_small	= 0.8
bg_fac_large	= 1.5
in_out_factor 	= 0.7 	
small_in_out	= 0.7
large_in_out	= 0.6
min_in_factor 	= 1 		
ap_step 	= 0.01 			
round_dec 	= 2 
ds9_contrast	= 1.14583
ds9_bias	= 0.46748			

'''


#Full paths to the folder containing the reference image and folder containing the images
#we want to find the flux of. Note that the reference folder can only contain 1 image.
#We also make the results folder and save the path to it for later use.
ref_path 	= os.getcwd() + "/reference/"
total_path 	= os.getcwd() + "/total/"
results_path 	= os.getcwd() + "/results/"
sp.run(["mkdir", "-p",  results_path])


#We used to declare all the variables here in the script. The variables/parameters have since been
#moved to the params.config file, and we just read the values in from that. The following
#commented-out section was kept so that definitions of all the variables could still be found 
#in the actual code (which is useful when editing the code, for example).

'''
############ Sigma-level variables #############

sigma_level 	= 3 			#What sigma level to use for sigma-clipping

sigma_source_id = 7			#What sigma level to use to identify sources

sigma_measure   = 5			#What sigma level to use to measure sources


########### Focused variables ###############
#Focusing is when, instead of analyzing the entire image, we focus in on a user-defined
#box and analyze the part of the image inside the box.

focused_var	= False			#Whether to focus in on a specific part of the
					#image. If True, will need to make sure the
					#center and box_side variables are also set
					#appropriately.

center		= np.array([0,0])	#Describes the center of the box, in sky
					#coordinates (in degrees).
					#Form of np.array([RA,Dec])

box_side	= 20			#Side length of the box, in arcseconds


############ Other flag-type variables ################

deblent_var 	= True			#Flag on whether to deblend the detected sources
					#Occasionally, will have one big source that we
					#don't want to break apart into connected smaller
					#ones

plot_var	= 'linear'			#What style to plot the main image for the 
					#reference plot. 'log' gets you ds9 settings,
					#the contrast and bias of which are specified
					#below in the Other variables/parameters 
					#section. 'linear' gets you just a normal plot.
					#OPTIONS are 'log' or 'linear'

show_plots	= False			#Whether to pop up the individual source plots
					#at the end of a code run (plots will still be
					#saved according to the savefig variable).
					#OPTIONS are True or False.

bg_method 	= 'median' 		#Method to find the background from the aperture
					#OPTIONS are 'mode','mean','mad_std','rmode',
					#'rmean', 'median'
					
print_option_1 	= 'everything' 		#What to print
					#OPTIONS are 'everything' and 'fluxes'
				
savefig 	= 'savepdf' 		#Whether to save the plots, and how to save them
					#OPTIONS are 'savepdf','savepng','saveps',
					#'savesvg', anything else and it won't save

print_type	= 'source'		#How to group the printed information - what the
					#highest level of information is - for 'source'
					#prints out a source, then iterates through all
					#the different images to find the information
					#for that source in every image.
					#OPTIONS are 'source' or 'image'


########### Other variables/parameters - these shouldn't need much changing ###########

pix_thresh 	= 5 			#Minimum number of pixels a source can have

pref_bg 	= np.array([1.75,2.5]) 	#Preferred factors for aperture - can think of
					#this as factors multiplying the contour
					#outlining the source

bg_fac_small    = 0.8			#What factor to decrease the background aperture
					#factor by, to measure error (while keeping the 
					#small_in_out_factor)

bg_fac_large    = 1.5			#What factor to enlarge the background aperture
					#factor by, to measure error (while keeping the 
					#large_in_out_factor)
				
in_out_factor 	= 0.7 			#If the preferred factors overlap a source, the 
					#code will find the correct outer factor and
					#this tells us the preferred ratio of 
					#inner/outer factors

small_in_out    = 0.7			#Same, but for the small aperture

large_in_out    = 0.6			#Same, but for the large aperture
				
min_in_factor 	= 1 			#Minimum factor for inner part of aperture

ap_step 	= 0.01 			#How much to step the bg factor down by for fine 
					#tuning - e.g. if ap_step=0.01 and pref_bg[1] =
					#3.5, it will try 3.49,3.48,3.47... until it
					#finds one that works.

round_dec 	= 2 			#How many decimal places to round to for
					#display, scientific notation-wise

ds9_contrast 	= 1.14583		#For plotting the reference image - since we're
					#using the ds9 cmapping tools, we need to specify
					#these. These shouldn't need changing, but if they
					#do, go into ds9, switch scaling to logarithm,
					#adjust the image to your liking, then go to 
					#colormap parameters and copy the contrast and
					#bias to here.

ds9_bias 	= 0.46748		#Same as above.
'''

#Import all the variables from the params.config file. There are several variables that
#are boolean True/False that need to be properly converted from a string to a bool value,
#and there are two variables that need to be converted from a list to an array.
with open('params.config', 'r') as paramfile:
	
	json_obj	= cjson.load(paramfile)
	
	sigma_level 	= json_obj['sigma_level']
	sigma_source_id = json_obj['sigma_source_id']
	sigma_measure	= json_obj['sigma_measure']
			
	focused_var	= bool(strtobool(json_obj['focused_var']))
	center		= np.array(json_obj['center'])
	box_side	= json_obj['box_side']

	deblent_var	= bool(strtobool(json_obj['deblent_var']))
	plot_var	= json_obj['plot_var']
	show_plots	= bool(strtobool(json_obj['show_plots']))
	bg_method 	= json_obj['bg_method']
	print_option_1 	= json_obj['print_option_1']
	savefig 	= json_obj['savefig']
	print_type	= json_obj['print_type']
				
	pix_thresh 	= json_obj['pix_thresh']
	pref_bg 	= np.array(json_obj['pref_bg'])
	bg_fac_small	= json_obj['bg_fac_small']
	bg_fac_large	= json_obj['bg_fac_large']
	in_out_factor 	= json_obj['in_out_factor']
	small_in_out	= json_obj['small_in_out']
	large_in_out	= json_obj['large_in_out']
	min_in_factor 	= json_obj['min_in_factor']
	ap_step 	= json_obj['ap_step']
	round_dec 	= json_obj['round_dec']
	ds9_contrast	= json_obj['ds9_contrast']
	ds9_bias	= json_obj['ds9_bias']


#========================================================================================
#========================================================================================

#################################################
############ DECLARING FOR LATER USE ############
#################################################


#Suppresses annoying warnings when using reproject - as far as I can tell the warnings
#don't actually do anything.
warnings.simplefilter('ignore',AstropyWarning)
warnings.simplefilter('ignore',FutureWarning)

#Getting a variable to round things to our specified number of decimal points
sci_rnd = '%.'+str(round_dec)+'E'

#Getting the reference file name
ref_file_temp = [f for f in listdir(ref_path) if isfile(join(ref_path,f))]
ref_file = ref_path + ref_file_temp[0]

#Getting the list of all files from the specified folder
total_files = [f for f in listdir(total_path) if isfile(join(total_path,f))]
total_files.sort()

#Print the names of the reference file and analysis files to the screen
print("Reference file:")
print(ref_file_temp[0])
print("Files to be analyzed:")
for n in range(0,len(total_files)):
	print(total_files[n])

#Declaring a whole bunch of lists for information that we'll eventually be printing to
#the screen
total_image_list = list([])
total_label_list = list([]) 
aperture_size_list = list([])
aperture_small_size_list = list([])
aperture_large_size_list = list([])
total_raw_flux_list = list([])
total_source_npix_list = list([])
total_bg_npix_list = list([])
total_bg_npix_small_list = list([])
total_bg_npix_large_list = list([])
total_bg_list = list([])
total_bg_small_list = list([])
total_bg_large_list = list([])
total_ps_list = list([])
total_flux_list = list([])
total_flux_orig_list = list([])
total_flux_small_list = list([])
total_flux_large_list = list([])
total_flux_er_list = list([])

#========================================================================================
#========================================================================================

#################################################
############ SIGMA,SOURCES AND PLOTS ############
#################################################

#Getting the data and header from the reference image. Additionally, getting rid of any
#extraneous dimensions in the data, e.g. if it had dimensions (1,1,512,512) the
#np.squeeze part turns that into (512,512). We also save the shape, as we'll need to use
#that later on to create blank arrays.
data_ref_full,header_ref = fits.getdata(ref_file,header=True)
data_ref = np.squeeze(data_ref_full)
data_ref_dim = np.shape(data_ref)

#Now checking on whether we're analyzing the whole image or just a specific part of it.
if focused_var == False:
	print("\nDetecting sources in entire image...")
	#Here we get the sigma value that we'll use - sigma clipping works by taking the image,
	#finding the std of it, then zeroing out anything with standard deviation above that. It
	#then iterates over that process (default is 5 iterations). I also chose to use the
	#mad_std function for standard deviation, as that seems to return better results.
	sigma_array = stats.sigma_clip(data_ref,sigma=sigma_level,stdfunc=stats.mad_std)
	sigma_clipped = stats.mad_std(sigma_array)

	#We set the sigma threshold for source detection and the sigma threshold we'll use to
	#measure the flux
	sigma_source_detect = sigma_clipped*sigma_source_id
	sigma_source_measure = sigma_clipped*sigma_measure

	#Detecting the sources using photutils detect_sources function for both the 'source
	#detection' threshold and the 'flux measuring' threshold
	rough_sources_detect = seg.detect_sources(data_ref,sigma_source_detect,pix_thresh)
	rough_sources_measure = seg.detect_sources(data_ref,sigma_source_measure,pix_thresh)
else:
	#Drawing a box around the galaxy we care about so we can exclude all the extraneous
	#sources in the image. The galaxy coordinates we'll have to input ourselves.
	#galaxy coordinates will be gal_ra, gal_dec --in degrees
	#also define (in arcsec) the box side length (for the full side, not just half the side)
	print("\nDetecting sources in focused region...")
	print("Center (Ra,Dec) in degrees:",center)
	print("Box sides (arcsec):",box_side)

	box_side = box_side/3600 #getting into degrees

	w_img = wcs.WCS(header_ref,ref_file)

	#defining the corners of the box (in sky coords)
	wcs_corner1 = center + np.array([-box_side/2,-box_side/2])
	wcs_corner2 = center + np.array([-box_side/2,box_side/2])
	wcs_corner3 = center + np.array([box_side/2,-box_side/2])
	wcs_corner4 = center + np.array([box_side/2,box_side/2])

	#getting the corners to pixel coords
	corner1 = w_img.all_world2pix(wcs_corner1[0],wcs_corner1[1],0)
	corner2 = w_img.all_world2pix(wcs_corner2[0],wcs_corner2[1],0)
	corner3 = w_img.all_world2pix(wcs_corner3[0],wcs_corner3[1],0)
	corner4 = w_img.all_world2pix(wcs_corner4[0],wcs_corner4[1],0)

	#gathering the row,column values of our box
	corner_row = np.array([corner1[0],corner2[0],corner3[0],corner4[0]])
	corner_col = np.array([corner1[1],corner2[1],corner3[1],corner4[1]])

	#setting any negative values(indicating a corner of the box lies outside the
	#array) to zero.
	corner_row[corner_row<0] = 0
	corner_col[corner_col<0] = 0

	#finding the minimum and maximum row and column values, which will make up the 
	#corners of our box. The +1 is so the beginnings and endings are nice and 
	#symmetric (as x[a:b] goes from x[a] to x[b-1]). We also round everything to
	#the nearest integer as arrays only take ints as arguments.
	row_start = int(round(np.amin(corner_row)))
	col_start = int(round(np.amin(corner_col)))
	row_end = int(round(np.amax(corner_row))) + 1
	col_end = int(round(np.amax(corner_col))) + 1


	#We first just take the data inside the box and run sigmaclip on it to get our
	#sigma levels.
	data_foc_temp = data_ref[row_start:row_end, col_start:col_end]
	sigma_array = stats.sigma_clip(data_ref,sigma=sigma_level,stdfunc=stats.mad_std)
	sigma_clipped = stats.mad_std(sigma_array)

	#We set the sigma threshold for source detection
	sigma_source_detect = sigma_clipped*sigma_source_id
	sigma_source_measure = sigma_clipped*sigma_measure

	#making a new 'focused' array that has the data from the original array inside
	#the box and zeros outside. We then run the source detection function on this.
	data_foc = np.zeros(data_ref_dim)
	data_foc[row_start:row_end,col_start:col_end] = data_foc_temp

	rough_sources_detect = seg.detect_sources(data_foc,sigma_source_detect,pix_thresh)
	rough_sources_measure = seg.detect_sources(data_foc,sigma_source_measure,pix_thresh)


#So rough_sources_measure might have additional sources in there that we don't actually
#want to measure - we only want to measure the ones that we detect (at the higher sigma
#value). So we go through and keep only the measure_sources that match up with the 
#detected_sources.
kill_labels = np.array([])
for n in range(0,rough_sources_measure.nlabels):
	
	src_measure = rough_sources_measure.copy()
	src_measure.keep_labels(labels = n+1)
	overlap_check = np.count_nonzero(src_measure.data*rough_sources_detect.data)
	if overlap_check == 0:
		kill_labels = np.append(kill_labels,n+1)

kill_labels = [int(n) for n in kill_labels] #getting labels as integers
rough_sources = rough_sources_measure.copy()
rough_sources.remove_labels(labels=kill_labels,relabel=True)


#################
#################
#Plotting things: We plot the original image, the image after sigma-clipping, what
#we found for our deblent sources using the detection threshold, and what we found 
#for our deblent sources using the measuring threshold.

#Getting the deblended ('deblent') sources
if deblent_var == True:
	total_deblent_srcs_detect = seg.deblend_sources(data_ref,rough_sources_detect,pix_thresh)
	total_deblent_sources = seg.deblend_sources(data_ref,rough_sources,pix_thresh)
	print("\nUsing deblended sources...")
elif deblent_var == False:
	total_deblent_srcs_detect = rough_sources_detect.copy()
	total_deblent_sources = rough_sources.copy()
	print("\nUsing rough (not deblended) sources...")
else:
	sys.exit('Specify whether to deblend sources or not - deblent_var')



plt.figure(1,figsize=(15,15))

#Origin as 'lower' for all of these because normally python plots things upside-down
if plot_var == 'linear':
	plt.subplot(2,2,1)
	plt.imshow(data_ref,origin='lower')
	plt.title("Reference Image: " + ref_file_temp[0],fontsize=10)
elif plot_var == 'log': 
	plt.subplot(2,2,1)
	plt.imshow(data_ref,origin='lower', norm=ds9norm.DS9Normalize(stretch='log',contrast=ds9_contrast,bias=ds9_bias))
	plt.title("Reference Image: " + ref_file_temp[0],fontsize=10)
else:
	print("Specify plot_var variable!")
	sys.exit()

plt.subplot(2,2,2)
plt.imshow(sigma_array,vmin=np.amin(data_ref),vmax=np.amax(data_ref),origin='lower')
plt.title("Sigma-clipped Image at " + str(sigma_level) + "σ",fontsize=10)

plt.subplot(2,2,3)
plt.imshow(total_deblent_srcs_detect.data,origin='lower',cmap=total_deblent_srcs_detect.cmap())
plt.title("Deblended Sources (detect) at "+ str(sigma_source_id) + "σ",fontsize=10)

plt.subplot(2,2,4)
plt.imshow(total_deblent_sources.data,origin='lower',cmap=total_deblent_sources.cmap())
plt.title("Deblended Sources (measure) at "+ str(sigma_measure) + "σ",fontsize=10)

plt.tight_layout() #Makes the plots look nicer

#Saving the figure, if that was indicated in variables
cur_dir = os.getcwd()
os.chdir(results_path)
if savefig == 'savepdf':
	plt.savefig("Reference.pdf",bbox_inches='tight')
elif savefig == 'savepng':
	plt.savefig("Reference.png",bbox_inches='tight')
elif savefig == 'saveps':
	plt.savefig("Reference.ps",dpi=300,bbox_inches='tight')
elif savefig == 'savesvg':
	plt.savefig("Reference.svg",bbox_inches='tight')
os.chdir(cur_dir)

print("\n--Check the reference image and see if it's detecting the sources you want.")
print("--If not, then close the popup and hit ctrl+c to stop the run and go change the sigma-related variables in the code.")
print("--If it is fine, then just close the popup and let the rest of the code run.\n")
plt.show()
#========================================================================================
#========================================================================================

#################################################
############### FINDING APERTURES ###############
#################################################
'''
First, we pick a specific source. Then we get the mask for that source (essentially an
array of ones and zeros, with ones where the source is and zeros everwhere else). The 
mask is basically just a shape made out of ones. We find the outline of that shape, 
and expand it by the specified factors. 

We then check to make sure that the aperture isn't overlapping any other sources, and if
it is, we fix it, whittling the aperture down until nothing overlaps. We then store the
aperture in an array for later use.
'''

print("Finding apertures...")

#We'll declare this array for later use - essentially we'll just store the individual
#array masks in here.
aperture_masks = np.zeros((rough_sources.nlabels,data_ref_dim[0],data_ref_dim[1]))
aperture_masks_small = np.zeros((rough_sources.nlabels,data_ref_dim[0],data_ref_dim[1]))
aperture_masks_large = np.zeros((rough_sources.nlabels,data_ref_dim[0],data_ref_dim[1]))


#Iterate through all our sources (rough sources, not the deblent ones)
for n in range(0,rough_sources.nlabels):
	
	src = rough_sources.copy() #Because of the way python deals with variables
	src.keep_labels(labels = n+1) #Focus in on one source
	src.relabel_consecutive() #Change labels to 1, essentially
	outline = src.copy() #Copying just in case
	srcmask = src.copy()
	
	#Find center of mass of the source mask, and find the coordinates of the outline
	ycent,xcent = np.array(ndimage.measurements.center_of_mass(srcmask.data))
	yline,xline = np.nonzero(outline.outline_segments())
	
	#Basically just getting the outline in a new array
	temparr = np.zeros(data_ref_dim)
	temparr[yline.astype(int),xline.astype(int)]=1
	
	#Finding the outer mask - note that if the curve crosses back over on itself i.e.
	#is multiple-valued in polar coordinates where the center of mass is (0,0), then
	#the mask can look a little messed up. It shouldn't be crazy, though, and it
	#doesn't matter too much as this is just to find the background.
	out_fac = pref_bg[1]
	out_fac_large = pref_bg[1]*bg_fac_large
	
	maskout = m.mask_enlarge(xline,yline,xcent,ycent,out_fac,data_ref_dim)
	maskout_large = m.mask_enlarge(xline,yline,xcent,ycent,out_fac_large,data_ref_dim)
	
	#Check if the mask overlaps with any sources.
	aperture_check = (maskout - srcmask.data)*rough_sources.data
	aperture_large_check = (maskout_large - srcmask.data)*rough_sources.data
	
	sumcheck = np.count_nonzero(aperture_check)
	sumcheck_large = np.count_nonzero(aperture_large_check)
	
	#Whittling down - start with the mask at the original factor, then try the
	#original factor - ap_step, see if there's overlap, then step down again, see
	#if there's overlap, etc.
	while sumcheck != 0:
		out_fac = out_fac - ap_step
		maskout = m.mask_enlarge(xline,yline,xcent,ycent,out_fac,data_ref_dim)
		aperture_check = (maskout - srcmask.data)*rough_sources.data
		sumcheck = np.count_nonzero(aperture_check)
	
	while sumcheck_large != 0:
		out_fac_large = out_fac_large - ap_step
		maskout_large =m.mask_enlarge(xline,yline,xcent,ycent,out_fac_large, data_ref_dim)
		aperture_check_large = (maskout_large - srcmask.data)*rough_sources.data
		sumcheck_large = np.count_nonzero(aperture_check_large)
	
	#Bit of error check - clearly the final outer factor shouldn't be bigger than 
	#what we originally specified as preferred, and the final inner factor shouldn't
	#be bigger than what we originally specified. We additionally make sure that the
	#inner factor is at least one, as otherwise our aperture would cross our source.
	out_final = min(out_fac,pref_bg[1])
	in_final = max(1,min(out_final*in_out_factor,pref_bg[0]))
	
	out_final_small = out_final*bg_fac_small
	in_final_small = max(1,min(out_final_small*small_in_out,pref_bg[0]*bg_fac_small))
	
	out_final_large = min(out_fac_large,pref_bg[1]*bg_fac_large)
	in_final_large = max(1,min(out_final_large*large_in_out,pref_bg[0]*bg_fac_large))
	
	#Storing the aperture factors
	temp_aperture = ( round(in_final,round_dec), round(out_final,round_dec) )
	temp_aperture_small = (round(in_final_small,round_dec),round(out_final_small,round_dec))
	temp_aperture_large = (round(in_final_large,round_dec),round(out_final_large,round_dec))
	
	aperture_size_list.append(temp_aperture)
	aperture_small_size_list.append(temp_aperture_small)
	aperture_large_size_list.append(temp_aperture_large)
	
	#Making the final masks
	maskin = m.mask_enlarge(xline,yline,xcent,ycent,in_final,data_ref_dim)
	maskout = m.mask_enlarge(xline,yline,xcent,ycent,out_final,data_ref_dim)
	
	maskin_small = m.mask_enlarge(xline,yline,xcent,ycent,in_final_small,data_ref_dim)
	maskout_small =m.mask_enlarge(xline,yline,xcent,ycent,out_final_small,data_ref_dim)
	
	maskin_large = m.mask_enlarge(xline,yline,xcent,ycent,in_final_large,data_ref_dim)
	maskout_large =m.mask_enlarge(xline,yline,xcent,ycent,out_final_large,data_ref_dim)
	
	#Making the aperture and making sure it isn't negative
	aperturemask = maskout - maskin
	aperturemask_small = maskout_small - maskin_small
	aperturemask_large = maskout_large - maskout_small
	
	aperturemask[aperturemask<0]=0
	aperturemask_small[aperturemask_small<0]=0
	aperturemask_large[aperturemask_large<0]=0
	
	#Storing the final aperture
	aperture_masks[n,:,:] = aperturemask
	aperture_masks_small[n,:,:] = aperturemask_small
	aperture_masks_large[n,:,:] = aperturemask_large
	
#========================================================================================
#========================================================================================

#################################################
################ FINDING FLUXES #################
#################################################
'''
Here we take a specific file (not the reference file) and find the flux for it. We do
this by using reproject to get the source masks and the aperture masks into that image's
pixel frame (array coordinates), then simply multiply the source and aperture by the 
image and sum up the remaining values. We then iterate over all the images.

There's a bit of mess with dealing with deblended sources, but that gets covered in a 
for loop, explained in more detail below.
'''

for k in range(0,len(total_files)):
	#Import the new file, eliminate extraneous dimensions, the usual
	total_file = total_path + total_files[k]
	data_file_full,header_file = fits.getdata(total_file,header=True)
	data_file = np.squeeze(data_file_full)
	data_file[np.isnan(data_file)] = 0 #get rid of any nan's in the data
	
	#Make some temporary lists - these will encompass all our knowledge about our
	#sources in a single image - they will change from image to image, though.
	nslabels = list([])
	nsraw_fluxes = list([])
	nssrc_npixes = list([])
	nsbg_npixes = list([])
	nsbg_npixes_small = list([])
	nsbg_npixes_large = list([])
	nsbgs = list([])
	nsbgs_small = list([])
	nsbgs_large = list([])
	nspoint_srcs = list([])
	nsfluxes = list([])
	nsfluxes_orig = list([])
	nsfluxes_small = list([])
	nsfluxes_large = list([])
	nsfluxes_er = list([])
	
	#Printing roadmarks to screen e.g. "Analyzing file (1/3) -- filename.fits"
	print("Analyzing file ("+str(k+1)+"/"+str(len(total_files))+") -- "+str(total_files[k]))
	
	for n in range(0,rough_sources.nlabels):
		#Iterate through all our sources, doing the same as previously - focus
		#on a single source at a time, keep only that source's mask
		src = rough_sources.copy()
		src.keep_labels(labels = n+1)
		src.relabel_consecutive()
		
		#Get the aperture mask for our source
		aperturemask = aperture_masks[n,:,:]
		aperturemask_small = aperture_masks_small[n,:,:]
		aperturemask_large = aperture_masks_large[n,:,:]
		
		#Deblend sources - this tells us whether there are actually multiple
		#sources in this rough source that aren't separated at the sigma level
		#we searched at.
		if deblent_var == True:
			fine_source_mask = seg.deblend_sources(data_ref,src,pix_thresh,relabel=True)
		elif deblent_var == False:
			fine_source_mask = src.copy()
		else:
			sys.exit('Specify whether to deblend sources or not - deblent_var')
		#Declaring more temporary lists - these will vary from source to source.
		labels = list([])
		raw_fluxes = list([])
		src_npixes = list([])
		bg_npixes = list([])
		bg_npixes_small = list([])
		bg_npixes_large = list([])
		bgs = list([])
		bgs_small = list([])
		bgs_large = list([])
		point_srcs = list([])
		fluxes = list([])
		fluxes_orig = list([])
		fluxes_small = list([])
		fluxes_large = list([])
		fluxes_er = list([])
		label_multiple = list([])
		
		for m in range(0,fine_source_mask.nlabels):
			
			print("Source "+str((n+1)+(m+1)/100)+"...")
			
			#Iterating through all the deblent sources, now. Same form.
			deblent_source = fine_source_mask.copy()
			deblent_source.keep_labels(labels = m+1)
			deblent_source.relabel_consecutive()
			
			#Making these new arrays because we have to reshape them in order
			#to put them in the reproject function.
			proj_source = deblent_source.data
			proj_aperture = aperturemask
			proj_aperture_small = aperturemask_small
			proj_aperture_large = aperturemask_large
			
			#Since our radio images have four axes, e.g. (1,1,512,512),
			#we have to reshape our source and aperture masks to put them
			#in this shape. Even though the other two axes don't really carry
			#any information pertinent to the fluxes
			#Note that this will still work for images with only the usual
			#two axes (e.g. not radio images) just fine
			proj_source = np.reshape(proj_source,np.shape(data_ref_full))
			proj_aperture = np.reshape(proj_aperture,np.shape(data_ref_full))
			proj_aperture_small = np.reshape(proj_aperture_small,np.shape(data_ref_full))
			proj_aperture_large = np.reshape(proj_aperture_large,np.shape(data_ref_full))
			
			#As reproject takes arguments in the form of (data, header)
			src = (proj_source,header_ref)
			apt = (proj_aperture,header_ref)
			apt_small = (proj_aperture_small,header_ref)
			apt_large = (proj_aperture_large,header_ref)
			
			#Having that independent_celestial_slice = True is VERY IMPORTANT
			# - since we have more than just celestial axes - we have
			#frequency and stokes in addition to just RA and Dec. This just
			#tells reproject that the celestial coordinates (RA,Dec) should
			#be considered independently of what the other two axes values
			#are - otherwise two images of the same region of the sky, but
			#at different frequencies, would return zero.
			src_ref,src_foot = rp.reproject_interp(src,header_file,independent_celestial_slices = True)
			apt_ref,apt_foot = rp.reproject_interp(apt,header_file,independent_celestial_slices = True)
			apt_small_ref,apt_small_foot = rp.reproject_interp(apt_small,header_file,independent_celestial_slices = True)
			apt_large_ref,apt_large_foot = rp.reproject_interp(apt_large,header_file,independent_celestial_slices = True)
			
			#Removing extra dimensions again
			src_ref = np.squeeze(src_ref)
			apt_ref = np.squeeze(apt_ref)
			apt_small_ref = np.squeeze(apt_small_ref)
			apt_large_ref = np.squeeze(apt_large_ref)
			
			#As sometimes, if the reprojected image doesn't cover part of the
			#image we're projecting to, the array will just be filled with 
			#NaN's in the part without any overlap.
			src_ref[np.isnan(src_ref)]=0
			apt_ref[np.isnan(apt_ref)]=0
			apt_small_ref[np.isnan(apt_small_ref)]=0
			apt_large_ref[np.isnan(apt_large_ref)]=0
			
			#We just find all the information we want now and add it to the
			#appropriate list
			label = (n+1) + (m+1)/100
			labels.append(label)
			
			if fine_source_mask.nlabels > 1:
				label_multiple.append(label) #for plot titles, later
			
			raw_flux = np.sum(src_ref*data_file)
			raw_fluxes.append(raw_flux)
			
			src_npix = np.count_nonzero(src_ref)
			src_npixes.append(src_npix)
			
			bg_npix = np.count_nonzero(apt_ref*data_file)
			bg_npixes.append(bg_npix)
			
			bg_npix_small = np.count_nonzero(apt_small_ref*data_file)
			bg_npixes_small.append(bg_npix_small)
			
			bg_npix_large = np.count_nonzero(apt_large_ref*data_file)
			bg_npixes_large.append(bg_npix_large)
			
			
			#We want to consider only the values in the background, as we
			#want to find some sort of mean or mode and if we include the
			#zeros in the rest of the array it will mess with our 
			#statistics
			bg_indices = np.nonzero(apt_ref)
			bg_values = np.ndarray.flatten(data_file[bg_indices])
			
			bg_indices_small = np.nonzero(apt_small_ref)
			bg_values_small = np.ndarray.flatten(data_file[bg_indices_small])
			
			bg_indices_large = np.nonzero(apt_large_ref)
			bg_values_large = np.ndarray.flatten(data_file[bg_indices_large])
			
			if bg_method == 'mode':
				#This gives an array of the form (mode,counts)
				bgmodearr = scstats.mode(bg_values,axis=None)
				bgmodearr = np.squeeze(bgmodearr)
				bg = bgmodearr[0]
				
				bgmodearr_small = scstats.mode(bg_values_small,axis=None)
				bgmodearr_small = np.squeeze(bgmodearr_small)
				bg_small = bgmodearr_small[0]
				
				bgmodearr_large = scstats.mode(bg_values_large,axis=None)
				bgmodearr_large = np.squeeze(bgmodearr_large)
				bg_large = bgmodearr_large[0]
			elif bg_method == 'mean':
				bg = np.mean(bg_values)
				bg_small = np.mean(bg_values_small)
				bg_large = np.mean(bg_values_large)
			elif bg_method == 'mad_std':
				bg = stats.mad_std(bg_values)
				bg_small = stats.mad_std(bg_values_small)
				bg_large = stats.mad_std(bg_values_large)
			elif bg_method == 'rmean':
				bg = rstats.mean(bg_values)
				bg_small = rstats.mean(bg_values_small)
				bg_large = rstats.mean(bg_values_large)
			elif bg_method == 'rmode':
				bg = rstats.mode(bg_values)
				bg_small = rstats.mode(bg_values_small)
				bg_large = rstats.mode(bg_values_large)
			elif bg_method == 'median':
				bg = np.median(bg_values)
				bg_small = np.median(bg_values_small)
				bg_large = np.median(bg_values_large)
			else:
				sys.exit("Specify background preference!")
			
			bgs.append(bg)
			bgs_small.append(bg_small)
			bgs_large.append(bg_large)
			
			ps_val = np.amax(src_ref*data_file)
			point_srcs.append(ps_val)
			
			flux_orig = raw_flux - src_npix*bg
			fluxes_orig.append(flux_orig)
			
			flux_small = raw_flux - src_npix*bg_small
			fluxes_small.append(flux_small)
			
			flux_large = raw_flux - src_npix*bg_large
			fluxes_large.append(flux_large)
			
			flux = np.mean(np.array([flux_orig,flux_small,flux_large]))
			fluxes.append(flux)
			
			flux_er = np.std(np.array([flux_orig,flux_small,flux_large]))
			fluxes_er.append(flux_er)
			
			#Now we make plots of the specific aperture we're working with,
			#the specific source we're working with, and all the other
			#sources, just for visualization purposes. We only do this once
			#as we're using the same aperture and source masks for every
			#image, so no need to do it multiple times.
			if k==0:
				#Make a copy of the total source masks, remove the source
				#that we're currently working on,
				srcs_plot = rough_sources.copy()
				srcs_plot.remove_labels(labels=n+1)
				
				#List of the form [1,2,3,...], and again removing the
				#label of the source we're currently working on
				to_relabel = list(range(1,rough_sources.nlabels+1))
				to_relabel.remove(n+1)
				
				#Relabel all the sources as 3 (previously they consisted
				#of little islands of values of 1, or 2, etc.). We want
				#to rename them all to 3 to make the plots consistent.
				srcs_plot.relabel(labels=to_relabel,new_label=3)
				
				#Plot the deblent source we're currently working with,
				#along with all the rest of the sources, along with the
				#aperture
				'''
				plt.figure(n+2,figsize = (9,6))
				plt.subplot(1,fine_source_mask.nlabels,m+1)
				#plt.imshow(deblent_source.data+2*aperturemask+srcs_plot.data + 3*aperturemask_small + 4*aperturemask_large, origin='lower', vmin=0, vmax=6)
				plt.imshow(deblent_source.data+srcs_plot.data+ 2*aperturemask, origin='lower')
				plt.title("Source: "+ str(label) + " \n Aperture factors: " + str(aperture_small_size_list[n]) + ", " + str(aperture_size_list[n]) + ", " + str(aperture_large_size_list[n]))
				plt.tight_layout()
				'''
				plt.figure(n+2,figsize = (15,11))
				
				if fine_source_mask.nlabels > 1:
					plt.suptitle("Sources: " + str(label_multiple))
				else:
					plt.suptitle("Source: "+ str(label))
				
				#plt.subplot(fine_source_mask.nlabels,3,m+1)
				plt.subplot(fine_source_mask.nlabels,3,3*m+1)
				plt.imshow(deblent_source.data + 2*aperturemask_small + srcs_plot.data, origin='lower',vmin=0,vmax=5)
				plt.title("Aperture factor: " + str(aperture_small_size_list[n]))
				
				#plt.subplot(fine_source_mask.nlabels,3, fine_source_mask.nlabels + (m+1))
				plt.subplot(fine_source_mask.nlabels,3,3*m+2)
				plt.imshow(deblent_source.data + 2*aperturemask + srcs_plot.data, origin='lower',vmin=0,vmax=5)
				plt.title("Aperture factor: " + str(aperture_size_list[n]))
				
				#plt.subplot(fine_source_mask.nlabels,3, 2*fine_source_mask.nlabels + (m+1))
				plt.subplot(fine_source_mask.nlabels,3,3*m+3)
				plt.imshow(deblent_source.data + 2*aperturemask_large + srcs_plot.data, origin='lower',vmin=0,vmax=5)
				plt.title("Aperture factor: " + str(aperture_large_size_list[n]))
				
				plt.tight_layout()
				
		#This is just saving those plots that we just made, if we decided that
		#we want to by specifying it in the variables section. We switch over
		#to the results folder, save the image, then switch back to the 
		#current directory
		if k==0:		
			lbltemp = str(n+1)
			cur_dir = os.getcwd()
			os.chdir(results_path)
			if savefig == 'savepdf':
				plt.savefig("Source_"+ lbltemp + ".pdf",bbox_inches='tight')
			elif savefig == 'savepng':
				plt.savefig("Source_"+ lbltemp + ".png",bbox_inches='tight')
			elif savefig == 'saveps':
				plt.savefig("Source_"+ lbltemp + ".ps",dpi=300,bbox_inches='tight')
			elif savefig == 'savesvg':
				plt.savefig("Source_"+ lbltemp + ".svg",bbox_inches='tight')
			if show_plots == False:
				plt.close(n+2)
			os.chdir(cur_dir)	
				
		
		#Appending our information to the appropriate lists
		nslabels.append(labels)
		nsraw_fluxes.append(raw_fluxes)
		nssrc_npixes.append(src_npixes)
		nsbg_npixes.append(bg_npixes)
		nsbg_npixes_small.append(bg_npixes_small)
		nsbg_npixes_large.append(bg_npixes_large)
		nsbgs.append(bgs)
		nsbgs_small.append(bgs_small)
		nsbgs_large.append(bgs_large)
		nspoint_srcs.append(point_srcs)
		nsfluxes.append(fluxes)
		nsfluxes_orig.append(fluxes_orig)
		nsfluxes_small.append(fluxes_small)
		nsfluxes_large.append(fluxes_large)
		nsfluxes_er.append(fluxes_er)
	
	#Storing the information	
	total_image_list.append(total_files[k])	
	total_label_list.append(nslabels)
	total_raw_flux_list.append(nsraw_fluxes)
	total_source_npix_list.append(nssrc_npixes)
	total_bg_npix_list.append(nsbg_npixes)
	total_bg_npix_small_list.append(nsbg_npixes_small)
	total_bg_npix_large_list.append(nsbg_npixes_large)
	total_bg_list.append(nsbgs)
	total_bg_small_list.append(nsbgs_small)
	total_bg_large_list.append(nsbgs_large)
	total_ps_list.append(nspoint_srcs)
	total_flux_list.append(nsfluxes)
	total_flux_orig_list.append(nsfluxes_orig)
	total_flux_small_list.append(nsfluxes_small)
	total_flux_large_list.append(nsfluxes_large)
	total_flux_er_list.append(nsfluxes_er)

if show_plots == True:
	plt.show()

#========================================================================================
#========================================================================================

#################################################
############# DISPLAYING THE RESULTS ############
#################################################
'''
We print out some relevant variables (like the sigma level used to identify sources, and
the sigma level used for sigma-clipping, etc.) and then print out information regarding
each image.

This prints out all the information for each source in each image. For 'image':

-Image 1
	-Source 1
		-flux
		-npix
		-etc.
	-Source 2
		-flux
		-npix
		-etc.
	-etc.
-Image 2
	-Source 1
		-... etc.
		
So the results are clustered by image, then by source.

-------------------------

For 'source':

-Source 1
	-Image 1
		-flux
		-npix
		-etc.
	-Image 2
		-flux
		-...
	-...
-Source 2
	-Image 1
		-... etc.
		


So print_type tells the code what the highest level of the grouping is - image or
source.	

The code looks messy because I had to get everything in the right format for printing -
the right amount of decimals, nicely formatted, etc.
'''	

#Now to redirect all this output to a file for later perusal instead of just printing
#it all to the screen.
sys.stdout = open(results_path + 'output.txt', 'wt')

print('--------------------------------------')
print('--------------------------------------')
print('--------------------------------------')
print('--------------------------------------')
print("Reference file:")
print(ref_file_temp[0])
print("Files to be analyzed:")
for n in range(0,len(total_files)):
	print(total_files[n])

print("\n")
print("Variables Used:")
plc = "   "
print(plc,"Sigma level of reference image:            " + str(sigma_clipped))
print(plc,"Sigma level for sigma-clipping:            " + str(sigma_level) +"σ")
print(plc,"Sigma level for source identification:     " + str(sigma_source_id)+"σ --", sigma_source_detect)
print(plc,"Sigma level for measuring fluxes:          " + str(sigma_measure) + "σ --", sigma_source_measure)
print(plc,"Were sources deblended:                    " + str(deblent_var))
print(plc,"Was focusing used:			       " + str(focused_var))
if focused_var == True:
	print(plc,"Box center coords [Ra, Dec] (degrees):    ",center)
	print(plc,"Side length of focusing box (arcsec):     ",box_side*3600)
print(plc,"Plotting variable for reference image:     " + str(plot_var))
print(plc,"Pixel threshold for source identification:",pix_thresh,"pixels")
print(plc,"Preferred aperture factors:               ",pref_bg)
print(plc,"Preferred aperture radii ratio:           ",in_out_factor)
print(plc,"Preferred small aperture multiplier:      ",bg_fac_small)
print(plc,"Preferred small aperture radii ratio:     ",small_in_out)
print(plc,"Preferred large aperture multiplier:      ",bg_fac_large)
print(plc,"Preferred large aperture radii ratio:     ",large_in_out)
print(plc,"Minimum factor for inner aperture:        ",min_in_factor)
print(plc,"Aperture factor step size:                ",ap_step)
print(plc,"Background method:                         " + bg_method)
print(plc,"Decimals to round to:                     ",round_dec)
print(plc,"DS9 contrast for plotting:                ",ds9_contrast)
print(plc,"DS9 bias for plotting:                    ",ds9_bias)
print('--------------------------------------')
print('--------------------------------------')

#Grouping by image
if print_type == 'image':
		
	for n in range(0,len(total_image_list)):
		print('--------------------------------------')
		print('--------------------------------------')
		print(total_image_list[n])
		print(' ')
	
		for m in range(0,rough_sources.nlabels):
		
			for k in range(0,len(total_label_list[n][m])):
			
				print("Source:",total_label_list[n][m][k])
				tab = "            "
				raw_flux_print = sci_rnd % decimal.Decimal(str(total_raw_flux_list[n][m][k]))
				bg_print = sci_rnd % decimal.Decimal(str(total_bg_list[n][m][k]))
				bg_print_small = sci_rnd % decimal.Decimal(str(total_bg_small_list[n][m][k]))
				bg_print_large = sci_rnd % decimal.Decimal(str(total_bg_large_list[n][m][k]))
				
				flux_print = sci_rnd % decimal.Decimal(str(total_flux_list[n][m][k]))
				flux_orig_print = sci_rnd % decimal.Decimal(str(total_flux_orig_list[n][m][k]))
				flux_small_print = sci_rnd % decimal.Decimal(str(total_flux_small_list[n][m][k]))
				flux_large_print = sci_rnd % decimal.Decimal(str(total_flux_large_list[n][m][k]))
				
				flux_er_print = sci_rnd % decimal.Decimal(str(total_flux_er_list[n][m][k]))
				ps_print = sci_rnd % decimal.Decimal(str(total_ps_list[n][m][k]))
			
				if print_option_1 == 'everything':
					print(tab,"Aperture factors:", aperture_small_size_list[m], aperture_size_list[m], aperture_large_size_list[m])
					print(tab,"Raw Flux:                 " +raw_flux_print)
					print(tab,"Point Source:             "+ps_print)
					print(tab,"Source npix:             ", total_source_npix_list[n][m][k])
					print(tab)
					print(tab,"Background npix: small - ", total_bg_npix_small_list[n][m][k])
					print(tab,"              original - ", total_bg_npix_list[n][m][k])
					print(tab,"                 large - ", total_bg_npix_large_list[n][m][k])
					print(tab)
					print(tab,"Background:      small -  " +bg_print_small)
					print(tab,"              original -  " +bg_print)
					print(tab,"                 large -  " +bg_print_large)
					print(tab)
					print(tab,"Flux:            small -  " +flux_small_print)
					print(tab,"              original -  " +flux_orig_print)
					print(tab,"                 large -  " +flux_large_print)
					print(tab)
					print(tab,"Flux (final):             " +flux_print + " +/- " + flux_er_print)
				elif print_option_1 == 'fluxes':
					print(tab,"Aperture factors:", aperture_small_size_list[m], aperture_size_list[m], aperture_large_size_list[m])
					print(tab,"Flux (final):             " +flux_print + " +/- " + flux_er_print)
				else:
					sys.exit("Specify printing preference!")


#Grouping by source
elif print_type == 'source':

	for m in range(0,rough_sources.nlabels):
	
		for k in range(0,len(total_label_list[0][m])):
		
			print('--------------------------------------')
			print('--------------------------------------')
			print("Source:",total_label_list[0][m][k])
			print("Aperture factors:", aperture_small_size_list[m], aperture_size_list[m], aperture_large_size_list[m])
			print(' ')
		
			for n in range(0,len(total_image_list)):
				
				tab = "            "
				print(tab,"-------------------------")
				print(total_image_list[n])
				raw_flux_print = sci_rnd % decimal.Decimal(str(total_raw_flux_list[n][m][k]))
				bg_print = sci_rnd % decimal.Decimal(str(total_bg_list[n][m][k]))
				bg_print_small = sci_rnd % decimal.Decimal(str(total_bg_small_list[n][m][k]))
				bg_print_large = sci_rnd % decimal.Decimal(str(total_bg_large_list[n][m][k]))
				
				flux_print = sci_rnd % decimal.Decimal(str(total_flux_list[n][m][k]))
				flux_orig_print = sci_rnd % decimal.Decimal(str(total_flux_orig_list[n][m][k]))
				flux_small_print = sci_rnd % decimal.Decimal(str(total_flux_small_list[n][m][k]))
				flux_large_print = sci_rnd % decimal.Decimal(str(total_flux_large_list[n][m][k]))
				
				flux_er_print = sci_rnd % decimal.Decimal(str(total_flux_er_list[n][m][k]))
				ps_print = sci_rnd % decimal.Decimal(str(total_ps_list[n][m][k]))
			
				if print_option_1 == 'everything':
					print(tab,"Raw Flux:                 " +raw_flux_print)
					print(tab,"Point Source:             "+ps_print)
					print(tab,"Source npix:             ", total_source_npix_list[n][m][k])
					print(tab)
					print(tab,"Background npix: small - ", total_bg_npix_small_list[n][m][k])
					print(tab,"              original - ", total_bg_npix_list[n][m][k])
					print(tab,"                 large - ", total_bg_npix_large_list[n][m][k])
					print(tab)
					print(tab,"Background:      small -  " +bg_print_small)
					print(tab,"              original -  " +bg_print)
					print(tab,"                 large -  " +bg_print_large)
					print(tab)
					print(tab,"Flux:            small -  " +flux_small_print)
					print(tab,"              original -  " +flux_orig_print)
					print(tab,"                 large -  " +flux_large_print)
					print(tab)
					print(tab,"Flux (final):             " +flux_print + " +/- " + flux_er_print)
			
				elif print_option_1 == 'fluxes':
					print(tab,"Flux (final):             " +flux_print + " +/- " + flux_er_print)
				
				else:
					sys.exit("Specify printing preference!")

else:
	print("No print_type specified, so no information printed.")

print('--------------------------------------')
print('--------------------------------------')
print('--------------------------------------')
print('--------------------------------------')

#========================================================================================
#========================================================================================

#################################################
##################### OTHER #####################
#################################################

#Alternate way to do the aperture stuff - not as accurate, but might not take as long if
#the aperture is way off. Basically tries to figure out the factor roughly by dividing
#the distance to the overlap by the distance to the normal outline.

'''
if sumcheck != 0:
	ybad,xbad = np.nonzero(aperture_check)
	
	dist_bad = np.sqrt( (xbad-xcent)**2 + (ybad-ycent)**2 )
	theta_bad = np.arctan2(ybad - ycent,xbad - xcent)
	
	#original distances and thetas to outline coords
	rarr = np.sqrt( (xcent-xline)**2 + (yline-ycent)**2 )
	thetarr = np.arctan2(yline - ycent,xline - xcent)
	
	min_bad_dist = np.amin(dist_bad)
	min_bad_dist_element = np.where(dist_bad == min_bad_dist)
	min_bad_theta = theta_bad[min_bad_dist_element]
	min_factor_array = np.array([])
	
	for m in range(0,len(min_bad_theta)):
		bad_theta_temp = min_bad_theta[m]
		line_theta_index = (np.abs(thetarr-bad_theta_temp)).argmin()
		line_r = rarr[line_theta_index]
		temp_factor = min_bad_dist/line_r
		min_factor_array = np.append(min_factor_array,temp_factor)
		#the necessity of a branch cut in theta can mess this up a little
		#bit as well
		
	new_out = np.amin(min_factor_array)
	maskout = m.mask_enlarge(xline,yline,xcent,ycent,new_out,data_ref_dim)
	
	
	bg_aperture = (round(new_in,round_dec),round(new_out,round_dec))

'''

#========================================================================================
#========================================================================================
#########################################################################################
########################################## END ##########################################
#########################################################################################
