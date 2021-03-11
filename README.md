# Research-Spring2021

  
#### Earth Observations for Health (EO4HEALTH) : http://www.geohealthcop.org/eo4health

#### To get/construct GFED datasets:
- Get original data:
  - Download original GFED data from https://www.geo.vu.nl/~gwerf/GFED/GFED4/, place in folder "GFED4s"
- Construct GFED data with appropriate datasets multiplied by cell area:
  - Copy all .hdf5 files from the original GFED data (from folder "GFED4s") into new folder "GFED4s_timesArea"
  - Run "code/multiply_by_area_gfed4s.py"
- Get PM2.5 grid data
  - Download files for each year: https://sedac.ciesin.columbia.edu/data/set/sdei-global-annual-gwr-pm2-5-modis-misr-seawifs-aod/data-download#close
  - Place files into new folder "Global Annual PM2.5 Grids"
  - Run "code/unzip_pm2-5.py 

##### If encounter error message where a directory does not exist, create it in the path described
  
#### Links to potentially useful datasets/tools
- ARSET: https://appliedsciences.nasa.gov/what-we-do/capacity-building/arset
- Gridded Population of the World (GPW): https://sedac.ciesin.columbia.edu/data/collection/gpw-v4
- CDC Asthma: https://www.cdc.gov/asthma/data-visualizations/default.htm
- Our World in Data - Wildfires: https://ourworldindata.org/natural-disasters#wildfires

  
 
