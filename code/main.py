
import os
import pathlib
import time
from typing import Optional
import numpy as np
from multiprocessing import Process, process

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


dates = ['199901', '199902', '199903', '199904', '199905', '199906', '199907', '199908', '199909', '199910', '199911', '199912', '200001', '200002', '200003', '200004', '200005', '200006', '200007', '200008', '200009', '200010', '200011', '200012', '200101', '200102', '200103', '200104', '200105', '200106', '200107', '200108', '200109', '200110', '200111', '200112', '200201', '200202', '200203', '200204', '200205', '200206', '200207', '200208', '200209', '200210', '200211', '200212', '200301', '200302', '200303', '200304', '200305', '200306', '200307', '200308', '200309', '200310', '200311', '200312', '200401', '200402', '200403', '200404', '200405', '200406', '200407', '200408', '200409', '200410', '200411', '200412', '200501', '200502', '200503', '200504', '200505', '200506', '200507', '200508', '200509', '200510', '200511', '200512', '200601', '200602', '200603', '200604', '200605', '200606', '200607', '200608', '200609', '200610', '200611', '200612', '200701', '200702', '200703', '200704', '200705', '200706', '200707', '200708', '200709', '200710', '200711', '200712', '200801', '200802', '200803', '200804', '200805', '200806', '200807', '200808', '200809', '200810', '200811', '200812', '200901', '200902', '200903', '200904', '200905', '200906', '200907', '200908', '200909', '200910', '200911', '200912', '201001', '201002', '201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012', '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110', '201111', '201112', '201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308', '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907']


# world = os.path.join('UIA_World_Countries_Boundaries-shp', 'World_Countries__Generalized_.shp')
USAstates = os.path.join('USA_states_counties', 'cb_2019_us_state_500k', 'cb_2019_us_state_500k.shp')
USAcounties = os.path.join('USA_states_counties', 'cb_2019_us_county_500k', 'cb_2019_us_county_500k.shp')

def runMulti(dir, file, args=None):
  startDates = args[0]
  endDates = args[1]
  startDates = [startDates] if type(startDates) is not list else startDates
  endDates = [endDates] if type(endDates) is not list else endDates
  proc = []
  for i in range(min(len(startDates),len(endDates))):
    p = Process(target=run, args=(dir, file, [startDates[i], endDates[i]] + args[2:] ))
    proc.append(p)
  for p in proc:
    p.start()
  for p in proc:
    p.join()




def main():
  dir = str(pathlib.Path(__file__).parent.absolute())

  # cmap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html
  # startYear startMonth endYear endMonth group dataset cmap

  t0 = timer_start()
  t1 = t0

  startDates = ['200001', '200501', '201001', '201501']
  endDates = ['200412', '200912', '201412', '201812']

  # startDate (200001), endDate (201812), cmap, regionDir, regionFile, mapFile, isYearly, maxMappedValue (optional)
  # run(dir, 'read_acag_pm2-5.py', ['200001', '201812', 'gist_stern', 'geo-countries', 'geo-countries-union.json', USAstates, True, 115])
  # run(dir, 'read_acag_pm2-5.py', ['200001', '201812', 'gist_stern', 'basisregions', 'TENA.geo.json', USAcounties, True])
  # run(dir, 'read_acag_pm2-5.py', ['200001', '201812', 'gist_stern', 'basisregions', 'TENA.geo.json', USAstates, True, 50])
  # run(dir, 'read_acag_pm2-5.py', ['200101', '200112', 'YlOrRd', 'basisregions', 'TENA.geo.json', USAcounties, True])
  # run(dir, 'read_acag_pm2-5.py', ['200001', '200012', 'YlOrRd', os.path.join('USA_states_counties', 'us_states'), '06-CA-California.geojson', USAcounties, True])

  # runMulti(dir, 'read_acag_pm2-5.py', [startDates, endDates, 'gist_stern', 'basisregions', 'TENA.geo.json', USAstates, False, 215]) # ~ 5.8 min (dont set save_maxVals = True)
  # runMulti(dir, 'read_acag_pm2-5.py', [startDates, endDates, 'gist_stern', 'basisregions', 'TENA.geo.json', USAstates, True, 50]) # 36.56 sec
  run(dir, 'read_acag_pm2-5.py', ['200001', '200012', 'gist_stern', 'basisregions', 'TENA.geo.json', USAstates, True, 50]) # 36.56 sec

  # run(dir, 'read_acag_pm2-5.py', ['200001', '201812', 'gist_stern', 'basisregions', 'TENA.geo.json', USAstates, False, 215]) # ~ 17.6 min

  # for filename in ['06-CA-California','08-CO-Colorado','13-GA-Georgia','22-LA-Louisiana','51-VA-Virginia']: # 
  #   run(dir, 'read_acag_pm2-5.py', ['200101', '200112', 'YlOrRd', os.path.join('USA_states_counties', 'us_states'), \
  #     filename + '.geojson', os.path.join('USA_states_counties', 'us_states_counties', filename + '.geojson'), True, 26.5])


  # best: gist_stern
  # bad: terrain
  # for filename in ['16-ID-Idaho']: # ,'30-MT-Montana'
  #   for cmap in ['gist_stern']: # ,'turbo','rainbow','gist_ncar','terrain','nipy_spectral'
  #     run(dir, 'read_acag_pm2-5.py', ['200006', '200010', cmap, os.path.join('USA_states_counties', 'us_states'), \
  #       filename + '.geojson', os.path.join('USA_states_counties', 'us_states_counties', filename + '.geojson'), False])


  # for filename in ['16-ID-Idaho','30-MT-Montana']: # ,'22-LA-Louisiana'
  #   for maxval in [25,50,100,200,300,400,500]:
  #     run(dir, 'read_acag_pm2-5.py', ['200008', '200008', 'YlOrRd', os.path.join('USA_states_counties', 'us_states'), \
  #       filename + '.geojson', os.path.join('USA_states_counties', 'us_states_counties', filename + '.geojson'), False, maxval])




  # # plot_usa.py - requires changing parameters in the file
  # run(dir, 'plot_usa.py', [dates[0], dates[-1], 'CountyDeaths'])
  # # run(dir, 'plot_usa.py', [dates[0], dates[-1], 'StateDeaths']) 


  # # startYYYYMM, endYYYYMM, pltTitle
  # dates01 = [i for i in dates if i.endswith('01')]
  # dates12 = [i for i in dates if i.endswith('12')]
  # l = min(len(dates01), len(dates12))
  # dates01 = dates01[:l]
  # dates12 = dates12[:l]
  # for i in range(0,l,4):
  #   runMulti(dir, 'plot_usa.py', [dates01[i:i+4], dates12[i:i+4], 'CountyDeaths', 3150])




  t1 = timer_restart(t1, 'main total time')
  


if __name__ == "__main__":
  main()
