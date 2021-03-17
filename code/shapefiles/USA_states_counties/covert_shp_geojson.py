import pathlib, os
import geopandas
from numpy.lib.utils import info



dir = str(pathlib.Path(__file__).parent.absolute())

shp = 'cb_2019_us_state_500k'
inFile = geopandas.read_file(os.path.join(dir, shp, shp + '.shp'))
inFile = inFile.sort_values(by = 'STATEFP')
inFile.to_file(  os.path.join(dir, shp + '.geojson'), driver='GeoJSON')


shp = 'cb_2019_us_county_500k'
inFile = geopandas.read_file(os.path.join(dir, shp, shp + '.shp'))
inFile = inFile.sort_values(by = ['STATEFP', 'COUNTYFP'])
inFile.to_file(  os.path.join(dir, shp + '.geojson'), driver='GeoJSON')
