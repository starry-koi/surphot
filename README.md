Surphot
===

Source detection and analysis using image noise levels. Specifically, uses a single 'reference image' (which the user has to put in the /reference folder) to find sources and source backgrounds, then uses those source and background masks to both calculate and estimate error on the source fluxes for each of the images in the /total folder (which the user also has to manually supply).


Setup
---
You will need to delete the `placeholder.fits` files in the /reference and /total folders and replace them with the images you want to analyze. There should only be one image in the /reference folder, which the code will use to determine source and background masks. All the images you want to use these masks on (all the images you want to find source fluxes for) will need to be placed in the /total folder. Note that this may include a copy of the image in the /reference folder, assuming you also want to analyze that image.


Brief Rundown of the Code
---

A more in-depth walkthrough is available in the `surphot.py` code file, with comments explaining each step in detail. The overview is as follows:

* We start by analyzing the reference image (the sole image in the /reference folder). We first check whether to analyze the entire image or focus in on part of it (via the `focused_var` variable). We find the noise in the image using the `astropy.stats.sigma_clip` function ([doc](https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clip.html)) using the `sigma_level` variable as the number of standard deviations for the clipping limit.
* We detect sources at a multiple of the noise as defined by the `sigma_source_id` variable and [deblend](https://photutils.readthedocs.io/en/stable/segmentation.html#source-deblending) them or not according to the `deblent_var` variable.
* We make a plot of the original reference image, the sigma-clipped image, the source masks at the detection level (using `sigma_source_id`) and the source masks at the measuring level (using `sigma_measure`). This plot pops up and the code will not proceed until the plot is closed.
* We then make the background apertures. We make three for each source, at varying distances from the source.
* We find the flux of each source, and correct for background contamination using the background apertures. The final reported flux is the mean of the three calculated fluxes (based on the three different background apertures) and the final error on the flux is the standard deviation of the three calculated fluxes.
* Finally, we save all the results to a created /results folder. This includes the fluxes (to `output.txt`), the Reference pop-up plot, and plots of all the sources along with their background apertures.


Variables the User Will Have to Care About
---

All variables are assigned and explained in more detail in the `params.config` file; you will likely have to go into this file to change some of these variables before running the code. What follows is an overview:

* `sigma_source_id`
  * What sigma level to use to identify sources. e.g., if `sigma_source_id` is set to 5, then the code will identify sources at 5 times the image's noise level.
* `sigma_measure`
  * What sigma level to use to measure the identified sources. Setting this to a different, lower value than `sigma_source_id` can be useful if you want to use a higher threshold for source detection to limit spurious detections, then relax to a lower threshold to ensure you're measuring the full extent of the source.
* `deblent_var`
  * Whether or not to deblend sources.
* `plot_var`
  * Adjusts the scaling when plotting the image in the /reference folder to 'linear' or 'log'. In my experience 'linear' works better for radio images, and 'log' works better for optical images.

Additionally, there is an option to focus in on a smaller region of the image rather than analyzing the entire image. This draws a box around a point and only analyzes the part of the image that's in the box.

* `focused_var`
  * Whether or not to focus in. 
* `center`
  * The center of the box, in sky coordinates (specifically degrees).
* `box_side`
  * The desired side length of the box, in arcseconds.


A Sample Run
---

Let's say you have 3 HST images in different optical-type bands that you'd like to analyze. You first delete the `placeholder.fits` files, copy the image you want to use for source detection into the /reference folder, then copy all three of the images into the /total folder. You run the code, and the Reference plot pops up. After inspecting it, you see that you're getting a spurious source detection, so you close the Reference plot and stop the code from running further. You go into the `params.config` file and change the value of the `sigma_source_id` variable to a higher value, then rerun the code. Again, the Reference plot pops up and while you're detecting only the sources you want to be, you're not measuring as much of the extent as you'd like to. So you close the Reference plot, stop the code from running, and adjust the `sigma_measure` variable down a bit. Rerunning the code and viewing the Reference plot, you see that everything looks the way you want it to look, so you click out of the plot and let the code run its course. 


Understanding the Output
---

All results are saved to the /results folder. It will contain:
* `output.txt`
  * This contains all the fluxes for each source in each image. These are listed in three flavors 'small', 'medium' and 'large', which correspond to which kind of background aperture was used. The final reported flux is the mean of the three fluxes using the three different background apertures, and the final reported flux error is the standard deviation of the three different fluxes.
* `Reference.pdf`
  *  This is the Reference plot that pops up in each code run, saved for perusal.
*  `Source_*.pdf`
   *  These are the plots of each source mask used with the accompanying three background aperture masks.


Quirks
---

* If the source is too small (has only a few pixels) the background apertures might be slightly wonky. Easiest way to fix this is to decrease the `sigma_measure` variable to capture more of the source extent.
* This code will only work on .fits images.
* Sources at the edge of an image can make the background apertures not function correctly, so check that this isn't happening when the code pops up the Reference plot.
* The Reference plot that pops up halts the code until closed. This is so the user can check that the sources are being detected as desired before running through the rest of the code. If it looks correct, click out of the plot and let the code run its course - if not, you can click out of the plot and stop the code (ctrl+c), adjust the sigma-related variables, and run it again.




