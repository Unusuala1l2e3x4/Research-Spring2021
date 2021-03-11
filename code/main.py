
import os
import pathlib
import time
import numpy as np

def run(dir, file, args=None):
  # a = "python " + dir + '/' + file
  a = "C:/Users/Alex/anaconda3/python.exe " + dir + '/' + file
  if type(args) is list:
    for arg in args:
      a += ' ' + str(arg)
  elif type(args) is str:
    a += ' ' + args
  print(a)
  os.system(a)


def timer_start():
  return time.time()

def timer_elapsed(t0):
  return time.time() - t0

def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()



if __name__ == "__main__":

  dir = str(pathlib.Path(__file__).parent.absolute())

  # cmap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html
  # startYear startMonth endYear endMonth group dataset cmap

  # run(dir, 'read_gfed4s.py', '2016 12 2016 12 burned_area burned_fraction cool')

  # run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'burned_area', 'burned_fraction', 'cool', 'TENA'])

  t0 = timer_start()
  t1 = t0

  # run(dir, 'read_gfed4s.py', [2020, 10, 2020, 10, 'emissions', 'C', 'YlOrRd', 'TENA'])
  
  # run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'burned_area', 'burned_fraction', 'YlOrRd', 'TENA'])
  # t0 = timer_restart(t0, '')
  # run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'burned_area', 'burned_fraction', 'YlOrRd'])
  # t0 = timer_restart(t0, '')
  # run(dir, 'read_gfed4s.py', [1997, 1, 2020, 10, 'emissions', 'C', 'YlOrRd', 'TENA'])
  # t0 = timer_restart(t0, '')
  # run(dir, 'read_gfed4s.py', [1997, 1, 2020, 10, 'emissions', 'C', 'YlOrRd'])
  # t0 = timer_restart(t0, '')
  # run(dir, 'read_gfed4s.py', [1997, 1, 2020, 10, 'emissions', 'DM', 'YlOrRd', 'TENA'])
  # t0 = timer_restart(t0, '')
  # run(dir, 'read_gfed4s.py', [1997, 1, 2020, 10, 'emissions', 'DM', 'YlOrRd'])
  # t0 = timer_restart(t0, '')



  states = os.path.join('USA_states_counties', 'cb_2018_us_state_500k', 'cb_2018_us_state_500k.shp')
  counties = os.path.join('USA_states_counties', 'cb_2017_us_county_500k', 'cb_2017_us_county_500k.shp')

  # 1998, 2016
  years = np.arange(2016, 2016 + 1)

  for y in years:
    run(dir, 'read_pm2-5.py', [y, y, 'YlOrRd', 'basisregions', 'TENA.geo.json', states])


  # run(dir, 'read_pm2-5.py', [2016, 2016, 'YlOrRd', os.path.join('USA_states_counties', 'us_states'), 'CA-California.geojson', counties])
  # t0 = timer_restart(t0, '')


  t1 = timer_restart(t1, 'total time')
  