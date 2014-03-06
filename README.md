PostPTV
=======

Post processing routines for analysing PTV data.

The OpenPTV software (www.openptv.net) produces few types of the ASCII files, the results of the identification, localization
(triangulation), correspondence (stereo-matching) and tracking procedures. 

The output of the first step of the image processing routines is the location of the particles in the images. These files are called _targets
and located in the `/img` folder


The stereo-matching and localization results are stored in the `rt_is.###` files where `###` is the number of the frame. 


The final result is the positions of the particles in the 3D space, linked together, called `ptv_is.###` files. The file format is
the single row header and 5 columns:

the header is the number of particles in the frame, N (i.e. the number of rows in the file)

The five columns are: 
`previous, next, x, y, z`

previous is the row number of the particle in the previous frame, counts from zero (i.e. first row -> previous = 0)
next is the row number in the next frame
x,y,z are in millimeters

if the particle has no link in the previous frame, then previous = -1
if the particle has no link in the next frame, then next = -2


See example of the `ptv_is` file in https://github.com/alexlib/postptv/blob/master/data/tracers/ptv_is.10001


There is a post-processing software, created by Beat Luethi that is available from https://github.com/3dptv/3d-ptv-post-process where 
the latest version is https://github.com/3dptv/3d-ptv-post-process/tree/master/version_January_2013 This software will get into OpenPTV 
but at the moment we can simply add Python codes that will visualize the results of the post-processing. The software creates files
that are named `xuap.###` (x for position, u for velocity, a for acceleration of the particles) and also `trajPoint.###`. The files are
explained on https://github.com/3dptv/3d-ptv-post-process/tree/master/version_January_2013 

The Python codes in this repository shall work equally well with any kind of Lagrangian trajectories (and maybe also other types), showing
the same types of figures and plots for `rt_is.###`, `ptv_is.###`, `xuap.###` or `trajPoint.###` files. 


Example
---------

http://nbviewer.ipython.org/gist/openptv-user/9067177




