# Research-Spring2021

  
#### To get/construct GFED datasets:
1. Get original data:
a. Download original GFED data from https://www.geo.vu.nl/~gwerf/GFED/GFED4/, place in folder "GFED4s"
b. Construct GFED data with appropriate datasets multiplied by cell area:
2. Create folder "GFED4s_timesArea"
a. Copy all .hdf5 files from the original GFED data (from folder "GFED4s") into folder "GFED4s_timesArea"
b. Run "code/multiply_by_area_gfed4s.py"