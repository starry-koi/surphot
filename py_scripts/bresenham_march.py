#!/usr/bin/env python
import numpy as np
import math
#import cv2 as cv2
import matplotlib.pyplot as plt
'''
p1, p2 are tuples (x0,y0) and (x1,y1) - the points between which we make the line. img 
I think just has to essentially be an array of the same shape as the one you care about.

This function returns values,coordinates where values is a list of 3-tuples:
[(val01,val02,val03),(val11,val12,val13)] etc. - values at the points of the line

Coordinates is a list of tuples [(x0,y0),(x1,y1),(x2,y2)...] the array coordinates of
the line as mapped onto our array.

Copy-pasted this off the internet, with some minor changes.
'''

def bresenham_march(img, p1, p2):
	x1 = p1[0]
	y1 = p1[1]
	x2 = p2[0]
	y2 = p2[1]
	steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
	if steep:
		t = x1
		x1 = y1
		y1 = t
		
		t = x2
		x2 = y2
		y2 = t
	also_steep = x1 > x2
	if also_steep:
	
		t = x1
		x1 = x2
		x2 = t
		
		t = y1
		y1 = y2
		y2 = t
		
	dx = x2 - x1
	dy = math.fabs(y2 - y1)
	error = 0.0
	delta_error = 0.0; # Default if dx is zero
	if dx != 0:
		delta_error = math.fabs(dy/dx)
		
	if y1 < y2:
		y_step = 1 
	else:
		y_step = -1
	
	y = y1
	ret = list([])
	coords = list([])
	for x in range(x1, x2):
		if steep:
			p = (y, x)
		else:
			p = (x, y)
		
		(b,r,g,a) = (-1,)*4
		if p[0] < np.shape(img)[1] and p[1] < np.shape(img)[0]:
			#(b,r,g, a) = cv2.cv.Get2D(img, p[1], p[0])
			(b,r,g, a) = (img[int(p[1]),int(p[0])],)*4
		
		ret.append((p,(b,r,g)))
		coords.append(p)
		
		error += delta_error
		if error >= 0.5:
			y += y_step
			error -= 1
		
	if also_steep:
		ret.reverse()
	
	return ret,coords
	

'''
mat = np.zeros((512,512))
#mat[150,150]=1
#mat[437,482]=1

stuff,newcoords = bresenham_march(mat,(150,150),(437,482))
#(xn,yn),(val1,val2,val3) = bresenham_march(mat,(150,150),(437,482))

#print(newcoords[0])
#print(newcoords[1][0])
#print(newcoords[0:2][0])

xnew = [x[0] for x in newcoords]
ynew = [y[1] for y in newcoords]


matnew = mat
matnew[xnew,ynew]=1

plt.figure(1)
plt.imshow(matnew,origin='lower')
plt.show()

'''






















