# Research-Spring2021
### [Undergraduate Research Project]
##

#### To get/construct GFED datasets:
- Get original data:
  - Download original GFED data from <https://www.geo.vu.nl/~gwerf/GFED/GFED4/>, place in folder "GFED4s"
- Construct GFED data with appropriate datasets multiplied by cell area:
  - Copy all .hdf5 files from the original GFED data (from folder "GFED4s") into new folder "GFED4s_timesArea"
  - Run "code/multiply_by_area_gfed4s.py"
- Get PM2.5 grid data (not used for study)
  - Download files for each year: <https://sedac.ciesin.columbia.edu/data/set/sdei-global-annual-gwr-pm2-5-modis-misr-seawifs-aod/data-download#close>
  - Place files into new folder "Global Annual PM2.5 Grids"
  - Run "code/unzip_pm2-5.py 
#### To get 2000-2018 Surface PM2.5 datasets (monthly, 0.01 deg resolution)
- Corresponding article: <https://pubs.acs.org/doi/full/10.1021/acs.est.0c01764>
- More info + instructions: <https://sites.wustl.edu/acag/datasets/surface-pm2-5/>



#### To get mortality data (death counts by county, month):
- CDC WONDER - Underlying Cause of Death - Chronic lower respiratory diseases - (saved query): <https://wonder.cdc.gov/controller/saved/D76/D133F078>


#### To write all data files (after downloading original data files from source):
1. Run "adjust_sup_deaths_data_by_pop.py"
2. Run "read_acag_pm2-5.py", "read_gfed4s.py"
3. Run "write_county_month_pm2-5.py"
4. Run, in any order:
    - Run "write_county_month_gfed.py"
    - Run "write_county_month_clim.py"
    - Run "write_county_month_gfed.py"
    - Run "write_county_month_median-income.py"
5. To write AQI data (optional)
    - Run "impute_county_month_AQI.py"
    - Run "write_county_month_AQI_main.py"


##
##### If in any case, you encounter error message where a directory does not exist, create it in the path described
##


#### To run Random Forest and RFECV (recursive feature elimination and cross-validated selection)
- To tune/test hyperparameters and static combinations of features, run "random_forest.py"
  - Set hyperparameters by editing "param_grid" variable
  - Set combinations of features by editing "columns_list" variable
- To perform feature selection, run "random_forest_RFECV.py"
  - Adjust starting features by editing "columns" variable
  - Note: not for tuning hyperparameters due to runtime; hyperparameters ("param_grid" variable) can be set for a single iteration of RFECV

