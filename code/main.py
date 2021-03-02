
import os
import pathlib
import time

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

  run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'burned_area', 'burned_fraction', 'YlOrRd', 'TENA'])
  t0 = timer_restart(t0, '')
  run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'burned_area', 'burned_fraction', 'YlOrRd'])
  t0 = timer_restart(t0, '')
  run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'emissions', 'C', 'YlOrRd', 'TENA'])
  t0 = timer_restart(t0, '')
  run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'emissions', 'C', 'YlOrRd'])
  t0 = timer_restart(t0, '')
  run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'emissions', 'DM', 'YlOrRd', 'TENA'])
  t0 = timer_restart(t0, '')
  run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'emissions', 'DM', 'YlOrRd'])
  t0 = timer_restart(t0, '')
  