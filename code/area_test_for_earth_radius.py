import numpy as np
import pandas as pd
from numpy import cos, sin, arctan2, arccos
import pathlib, os
import json, h5py

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

def quad_area(lon, lat, deg, radius):
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
  return (np.sum(angles) - (N-2)*np.pi)*radius**2



def gfed_filenames(gfed_fnames):
  ret = []
  for filename in gfed_fnames:
    if '.hdf5' in filename:
      ret.append(filename)
  return ret

if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  gfedDir = os.path.join(ppPath, 'GFED4s')
  gfedDir_timesArea = os.path.join(ppPath, 'GFED4s_timesArea')
  outputDir = os.path.join(pPath, 'read_gfed4s-outfiles')

  deg = 0.25
  
  
  gfed_fnames = os.listdir(gfedDir)
  gfed_fnames = sorted(gfed_filenames(gfed_fnames)) # same as in gfedDir_timesArea

  fd_timesArea = h5py.File(os.path.join(gfedDir_timesArea, gfed_fnames[0]), 'r')

  latsArea = pd.DataFrame()
  latsArea['lat'] = [i for i in np.unique(fd_timesArea['lat']) if i > 0]
  latsArea['area'] = sorted(np.unique(fd_timesArea['ancill/grid_cell_area']), reverse=True)

  
  radius = WGS84_RADIUS

  # 6361753.50975 avg of Spherical Earth Approx. of Radius (RE), nominal "zero tide" polar
  # 6366707.0195 Spherical Earth Approx. of Radius (RE)

  radiusErrors = pd.DataFrame()
  radiusErrors['radius'] = np.arange(6366113.579189921, 6366113.579189923, step=0.0000000005)
  print(len(radiusErrors))
  temp = [np.mean(np.abs([quad_area(0, lat, deg, row.radius) for lat in latsArea.lat] - latsArea.area) / latsArea.area) for row in radiusErrors.itertuples()]

  radiusErrors['percenterror'] = temp


  print(radiusErrors)
  print(radiusErrors.loc[radiusErrors.percenterror == np.min(radiusErrors.percenterror),:])

  errorFinal = float(radiusErrors.loc[radiusErrors.percenterror == np.min(radiusErrors.percenterror),'percenterror'])
  radiusFinal = float(radiusErrors.loc[radiusErrors.percenterror == np.min(radiusErrors.percenterror),'radius'])
  print(radiusFinal, errorFinal)
  print(list(radiusErrors.loc[:,'percenterror'])) # 1.5168794457058445e-06
  print(list(radiusErrors.loc[:,'radius']))       # 6366113.579189922

  # cannot split further, because float precision



  




