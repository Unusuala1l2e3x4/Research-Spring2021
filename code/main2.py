
import os
import pathlib
from multiprocessing import Process

import importlib
fc = importlib.import_module('functions')



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






def main():
  dir = str(pathlib.Path(__file__).parent.absolute())

  t0 = fc.timer_start()
  t1 = t0


  limit_tests = [16, 20, 24]
  methods = ['linear', 'nearest', 'cubic']


  
  for method in methods:
    proc = []
    for limit in limit_tests:
      p = Process(target=run, args=(dir, 'write_county_month_AQI.py', [limit, method] ))
      proc.append(p)
    for p in proc:
      p.start()
    for p in proc:
      p.join()


  

  # run(dir, 'write_county_month_AQI.py', [])

  t1 = fc.timer_restart(t1, 'main total time')
  







if __name__ == "__main__":
  main()
