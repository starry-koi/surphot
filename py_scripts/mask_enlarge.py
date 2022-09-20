#!/usr/bin/env python
import numpy as np
from scipy import ndimage
#from bresenham_march import bresenham_march #user file so don't move the file
from py_scripts import bresenham_march as b
'''
Specifically written for the surphotv2.py program. Essentially, you have a closed curve
or closed line (of pixels) in a 2d array. This program takes that curve, along with 
the center of mass of the object you would have if you filled in the curve with ones,
and makes the curve bigger. Then it fills the curve in with ones, and returns that array.

DOESN'T WORK WELL WHEN: the curve is double-valued in polar coordinates (e.g. two 
different r values for a single theta value, or the curve crosses back and forth
on itself). This is because the bresenham_march algorithm needs **ordered points** to
work, and the way I order them is by their theta value.

Inputs:
xline,yline - array coordinates of the line (need first and last element to be same)
xcent,ycent - coordinates of the center of mass of filled in object
factor - factor by which to increase the size (e.g. 2, 2.5, 3.27, whatever)
output_dim - desired dimensions of the output 2d array as a tuplet (dim1,dim2) 
'''


def mask_enlarge(xline,yline,xcent,ycent,factor,output_dim):
	
	
	#Essentially treating xcent,ycent as our zero and getting the coordinates of the
	#line in polar form.
	rarr = np.sqrt( (xcent-xline)**2 + (ycent-yline)**2 )
	thetarr = np.arctan2(yline - ycent,xline - xcent)
	
	#However, our xline,yline might not have come in sorted. So we have to sort these
	#points into sequentially going around the curve, otherwise the algorithm won't
	#work. We'll sort theta, then apply that to theta and r - as theta tells us the
	#angle, which is how we want to sort by. Note that this doesn't always work,
	#like if the curve is messed up and double valued, polar wise.
	sort_indices = np.argsort(thetarr)
	thetarr = thetarr[sort_indices]
	rarr = rarr[sort_indices]
	
	#Factor increases the distance from the center of mass
	outrarr = factor*rarr
	
	#Getting back in normal array coordinates
	xout = outrarr*np.cos(thetarr) + xcent
	yout = outrarr*np.sin(thetarr) + ycent
	
	#We also want to make sure that we catch that last line - between (xend,yend) and
	#(x0,y0) - this will complete our closed curve. To do this we append xout,yout
	#with their first value. 
	xout = np.append(xout,xout[0])
	yout = np.append(yout,yout[0])
		
	#As in general xout,yout won't be integers
	xoutint = xout.astype(int)
	youtint = yout.astype(int)
	
	#Our new line has breaks in it - it's not completely closed. This is because
	#we've increased the perimeter by some factor but we haven't added any new
	#pixels. To fix this, we consider each sequential pair of points individually,
	#then use bresenham_march to draw a line between them and get the coordinates
	#of the pixels that fall on that line (e.g. between (x0,y0) and (x1,y1), then
	#between (x1,y1) and (x2,y2) ). Note that we assume that xline,yline have the 
	#same first and last point - yline[end] = yline[0], and same for xline.
	xoutf = np.array([])
	youtf = np.array([])
	tempimage = np.zeros(output_dim)
	
	for m in range(0,(len(xoutint)-1)):
		x0 = xoutint[m]
		y0 = youtint[m]
		x1 = xoutint[m+1]
		y1 = youtint[m+1]
		
		values,coords = b.bresenham_march(tempimage,(x0,y0),(x1,y1))
		
		xnew = [x[0] for x in coords]
		ynew = [y[1] for y in coords]
		
		xoutf = np.append(xoutf,xnew)
		youtf = np.append(youtf,ynew)
	
	#Now we create an array, fill it with zeros, and set all the coordinates of our
	#newly found line to 1. We set both the bresenham_march found coordinates, as 
	#well as the original coordinates for the enlarged line, equal to 1, as 
	#sometimes using just the breseham_march coords can leave you with holes.
	#We also check that the enlarged x and y coords aren't out of bounds of the
	#original image, and only fill them in on the mask if they're in bounds.
	maskout = np.zeros(output_dim)
	y_max = output_dim[0]
	x_max = output_dim[1]
	for m in range(0,len(xoutf)):
		
		#Getting specific values for xoutf and youtf coords and putting them
		#into the mask array if they aren't out of bounds of the original
		#image.
		y_sp = youtf[m].astype(int)
		x_sp = xoutf[m].astype(int)
		
		if y_sp < y_max and x_sp < x_max:
			maskout[y_sp, x_sp] = 1
	
	#Now that but with the xoutint and youtint coords
	for m in range(0,len(xoutint)):
		y_sp = youtint[m]
		x_sp = xoutint[m]
		
		if y_sp < y_max and x_sp < x_max:
			maskout[y_sp, x_sp] = 1
		
	#Filling in the curve and getting it from boolean to numbers.
	maskout = ndimage.morphology.binary_fill_holes(maskout)
	maskout = np.multiply(maskout,1)
	
	return maskout
