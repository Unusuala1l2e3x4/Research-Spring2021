import pathlib, os
import geopandas as gpd
from numpy.lib.utils import info
import pandas as pd

# pd.set_option('display.max_columns', None)

dir = str(pathlib.Path(__file__).parent.absolute())

# https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.2000.html

shp = 'co99_d00'
if 'co99_d00_shp' in os.listdir(dir):
  os.rename(os.path.join(dir,'co99_d00_shp'), os.path.join(dir,shp))
inFile = gpd.read_file(os.path.join(dir, shp, shp + '.shp'))
inFile = inFile.sort_values(by = ['STATE', 'COUNTY'])
# inFile.to_file(  os.path.join(dir, shp + '.geojson'), driver='GeoJSON')
print(pd.DataFrame(inFile))
# print(inFile.keys())


# exit(0)


# https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html

shp = 'cb_2019_us_state_500k'
inFile = gpd.read_file(os.path.join(dir, shp, shp + '.shp'))
inFile = inFile.sort_values(by = 'STATEFP')
# inFile.to_file(  os.path.join(dir, shp + '.geojson'), driver='GeoJSON')
print(pd.DataFrame(inFile))
# print(inFile.keys())

shp = 'cb_2019_us_county_500k'
inFile = gpd.read_file(os.path.join(dir, shp, shp + '.shp'))
inFile = inFile.sort_values(by = ['STATEFP', 'COUNTYFP'])
# inFile.to_file(  os.path.join(dir, shp + '.geojson'), driver='GeoJSON')
print(pd.DataFrame(inFile))
# print(inFile.keys())

