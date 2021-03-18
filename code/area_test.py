import numpy as np
from numpy import cos, sin, arctan2, arccos
import pathlib, os
import json

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon

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


def polygon_area(coords):
  N = len(coords)

  coords = np.rot90(coords, k=3)
  lons = coords[0]
  lats = coords[1]
  
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
pt = [-102.055, 40.995]
area = quad_area(pt[0], pt[1], 0.1)
print(area / 1000000, 'sq km')


dir = str(pathlib.Path(__file__).parent.absolute())
statesDir = os.path.join( dir, 'shapefiles\\USA_states_counties\\us_states')
filename = '08-CO-Colorado'
with open(os.path.join(statesDir, filename + '.geojson'), 'r') as f:
  stateItems = json.load(f)
print(polygon_area(stateItems['features'][0]['geometry']['coordinates'][0]), 'sq m (coords from \'08-CO-Colorado.geojson\', using polygon_area)')
# doesnt match

# polygon = Polygon(stateItems['features'][0]['geometry'])
# polygon = Polygon(stateItems['features'][0]['geometry'])
# print(polygon.area, 'sq m (coords from \'08-CO-Colorado.geojson\', using shapely area function)')


corners = [[-109.050076,41.000659],[-109.050076,36.999084],[-102.042091,36.999084],[-102.042091,41.000659]]
print(polygon_area(corners), 'sq m (corners from google maps)')


# "ALAND": 268419875371, "AWATER": 1184637800
print(268419875371 + 1184637800, 'sq m (ALAND + AWATER from \'08-CO-Colorado.geojson\')')

