# Research-Spring2021
### [In-Progress Undergraduate Research Project]
##

#### To get/construct GFED datasets:
- Get original data:
  - Download original GFED data from <https://www.geo.vu.nl/~gwerf/GFED/GFED4/>, place in folder "GFED4s"
- Construct GFED data with appropriate datasets multiplied by cell area:
  - Copy all .hdf5 files from the original GFED data (from folder "GFED4s") into new folder "GFED4s_timesArea"
  - Run "code/multiply_by_area_gfed4s.py"
- Get PM2.5 grid data
  - Download files for each year: <https://sedac.ciesin.columbia.edu/data/set/sdei-global-annual-gwr-pm2-5-modis-misr-seawifs-aod/data-download#close>
  - Place files into new folder "Global Annual PM2.5 Grids"
  - Run "code/unzip_pm2-5.py 
##
#### 2000-2018 Surface PM2.5 datasets (monthly, 0.01 deg resolution)
- Corresponding article: <https://pubs.acs.org/doi/full/10.1021/acs.est.0c01764>
- More info + instructions: <http://fizz.phys.dal.ca/~atmos/martin/?page_id=140http://fizz.phys.dal.ca/~atmos/martin/?page_id=140page_id=140http://fizz.phys.dal.ca/~atmos/martin/?page_id=140>
  - files downloaded via FTP, as per instructions. I used FileZilla to do this.
  - put PM2.5 files into path "Atmospheric Composition Analysis Group/V4NA03/[filetype]/NA/PM25"
    - where [filetype] = NetCDF or ArcGIS
##
##### If in any case, you encounter error message where a directory does not exist, create it in the path described
##
#### Other datasets/tools being used
- Gridded Population of the World (GPW): <https://sedac.ciesin.columbia.edu/data/collection/gpw-v4>
- CDC Asthma: <https://www.cdc.gov/asthma/data-visualizations/default.htm>
- CDC WONDER - Underlying Cause of Death - saved query: <https://wonder.cdc.gov/controller/saved/D76/D133F078>

#### Potentially useful
- Our World in Data - Wildfires: <https://ourworldindata.org/natural-disasters#wildfires>