
import os
import pathlib


def run(dir, file, args=None):
  a = "python " + dir + '/' + file
  if type(args) is list:
    for arg in args:
      a += ' ' + str(arg)
  elif type(args) is str:
    a += ' ' + args
  print(a)
  os.system(a)


if __name__ == "__main__":

  dir = str(pathlib.Path(__file__).parent.absolute())

  # cmap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html

  # startYear startMonth endYear endMonth group dataset cmap

  # run(dir, 'read_gfed4s.py', '2016 12 2016 12 burned_area burned_fraction cool')
  # run(dir, 'read_gfed4s.py', [2016, 8, 2016, 8, 'burned_area', 'burned_fraction', 'cool'])
  # run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12,'burned_area', 'burned_fraction', 'cool'])
  # run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'burned_area', 'burned_fraction', 'cool', 'TENA'])
  # run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'emissions', 'C', 'YlOrRd', 'TENA'])
  run(dir, 'read_gfed4s.py', [1997, 1, 2016, 12, 'burned_area', 'burned_fraction', 'YlOrRd', 'TENA'])