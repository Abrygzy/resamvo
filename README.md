# resamvo
A naive package to resample data from Voronoi binning method.

The basic idea is shown below:
1. For the original data, create Voronoi bins with boundary following this [link](https://stackoverflow.com/a/33602171);
2. For each Voronoi cell, use Convex Hull define them;
3. Do other things:
   1. Test whether other data is in the Voronoi cell;
   2. Calculate the volume of the Voronoi cell;
   3. ...
