
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

  t0 = timer_start()
  t1 = t0

  run(dir, 'write_county_month_pop.py', ['r','hdf5'])
  run(dir, 'write_county_month_pop.py', ['r','csv'])

  run(dir, 'write_county_month_deaths.py', ['w','hdf5'])
  run(dir, 'write_county_month_deaths.py', ['w','csv'])
  


  t1 = timer_restart(t1, 'adjust_sup_deaths_data_by_pop total time')
  