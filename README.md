Surphot
===

Source detection and analysis using image noise levels. Specifically, uses a single 'reference image' (which the user has to put in the /reference folder) to find sources and source backgrounds, then uses those source and background masks to both calculate and estimate error on the source fluxes for each of the images in the \total folder (which the user also has to manually supply).


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


A Sample Run
---



Understanding the Output
---


Quirks
---

* If the source is too small (has only a few pixels) the background apertures might be slightly wonky. Easiest way to fix this is to decrease the `sigma_measure` variable to capture more of the source extent.
* This code will only work on .fits images.
* Sources at the edge of an image can make the background apertures not function correctly, so check that this isn't happening when the code pops up the Reference plot.
* The Reference plot that pops up halts the code until closed. This is so the user can check that the sources are being detected as desired before running through the rest of the code. If it looks correct, click out of the plot and let the code run its course - if not, you can click out of the plot and stop the code (ctrl+c), adjust the sigma-related variables, and run it again.




