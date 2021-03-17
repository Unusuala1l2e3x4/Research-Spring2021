import numpy as np
from numpy import cos, sin, arctan2, arccos
import pathlib, os
import json

# https://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python
# https://stackoverflow.com/a/19398136 <-- source of code

WGS84_RADIUS = 6378137
WGS84_RADIUS_SQUARED = WGS84_RADIUS**2
d2r = np.pi/180

def greatCircleBearing(lon1, lat1, lon2, lat2):
    dLong = lon1 - lon2
    s = cos(d2r*lat2)*sin(d2r*dLong)
    c = cos(d2r*lat1)*sin(d2r*lat2) - sin(lat1*d2r)*cos(d2r*lat2)*cos(d2r*dLong)
    return arctan2(s, c)

def quad_area(lon, lat, deg):
  deg = deg / 2
  lons = [lon+deg,lon+deg,lon-deg,lon-deg]
  lats = [lat+deg,lat-deg,lat-deg,lat+deg]
  N = 4 # len(lons)
  angles = np.empty(N)
  for i in range(N):
      phiB1, phiA, phiB2 = np.roll(lats, i)[:3]
      lB1, lA, lB2 = np.roll(lons, i)[:3]
      # calculate angle with north (eastward)
      beta1 = greatCircleBearing(lA, phiA, lB1, phiB1)
      beta2 = greatCircleBearing(lA, phiA, lB2, phiB2)
      # calculate angle between the polygons and add to angle array
      angles[i] = arccos(cos(-beta1)*cos(-beta2) + sin(-beta1)*sin(-beta2))
  return (np.sum(angles) - (N-2)*np.pi)*WGS84_RADIUS_SQUARED



# pt = {'lon':-102.055,'lat':40.995}
# area = quad_area(pt['lat'], pt['lon'], 0.1)

pt = [-102.055, 40.995]
area = quad_area(pt[0], pt[1], 0.1)

print(area / 1000000, 'sq km')





dir = str(pathlib.Path(__file__).parent.absolute())
statesDir = os.path.join( dir, 'shapefiles\\USA_states_counties\\us_states')

filename = ''

with open(os.path.join(dir, filename + '.geojson'), 'r') as f:
  stateItems = json.load(f)['features']










# # area of the earth
# pt = {'lon':0,'lat':0}
# area = quad_area(pt['lat'], pt['lon'], 179.6)
# print(area*2 / 1000000, 'sq km')