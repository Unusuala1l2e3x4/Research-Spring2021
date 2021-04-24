
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


  # limit_tests_all = [16, 20, 24]

  # limit_tests_all = list(range(0,25,4))

  limit_tests_all = []
  limit_tests_all.append([0, 3, 6])
  limit_tests_all.append([9, 12, 15])
  limit_tests_all.append([18, 21, 24])
  # limit_tests_all.append([12])
  
  methods = ['linear', 'nearest', 'cubic'] # 45 min + 7.4 hrs + 93 min = 9.76 hrs
  # methods = ['cubic']

  # print(limit_tests_all)
  # print(methods)

  # exit()



  for method in methods:
    proc = []
    for limit_tests in limit_tests_all:
      p = Process(target=run, args=(dir, 'write_county_month_AQI.py', [method] + limit_tests ))
      proc.append(p)
      # print('\t\t\t',[method] + limit_tests)
    # print('\t\t\t'+str(proc))
    for p in proc:
      p.start()
    for p in proc:
      p.join()
    t0 = fc.timer_restart(t0, 'time for multiprocess')

  t1 = fc.timer_restart(t1, 'main total time')
  







if __name__ == "__main__":
  main()
