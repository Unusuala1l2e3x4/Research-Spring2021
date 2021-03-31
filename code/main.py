
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


dates = ['199901', '199902', '199903', '199904', '199905', '199906', '199907', '199908', '199909', '199910', '199911', '199912', '200001', '200002', '200003', '200004', '200005', '200006', '200007', '200008', '200009', '200010', '200011', '200012', '200101', '200102', '200103', '200104', '200105', '200106', '200107', '200108', '200109', '200110', '200111', '200112', '200201', '200202', '200203', '200204', '200205', '200206', '200207', '200208', '200209', '200210', '200211', '200212', '200301', '200302', '200303', '200304', '200305', '200306', '200307', '200308', '200309', '200310', '200311', '200312', '200401', '200402', '200403', '200404', '200405', '200406', '200407', '200408', '200409', '200410', '200411', '200412', '200501', '200502', '200503', '200504', '200505', '200506', '200507', '200508', '200509', '200510', '200511', '200512', '200601', '200602', '200603', '200604', '200605', '200606', '200607', '200608', '200609', '200610', '200611', '200612', '200701', '200702', '200703', '200704', '200705', '200706', '200707', '200708', '200709', '200710', '200711', '200712', '200801', '200802', '200803', '200804', '200805', '200806', '200807', '200808', '200809', '200810', '200811', '200812', '200901', '200902', '200903', '200904', '200905', '200906', '200907', '200908', '200909', '200910', '200911', '200912', '201001', '201002', '201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012', '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110', '201111', '201112', '201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308', '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907']


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


  # world = os.path.join('UIA_World_Countries_Boundaries-shp', 'World_Countries__Generalized_.shp')
  states = os.path.join('USA_states_counties', 'cb_2019_us_state_500k', 'cb_2019_us_state_500k.shp')
  counties = os.path.join('USA_states_counties', 'cb_2019_us_county_500k', 'cb_2019_us_county_500k.shp')

  # # 1998, 2016
  # years = np.arange(1998, 2016 + 1)

  # for y in years:
  #   # run(dir, 'read_sedac_pm2-5.py', [y, y, 'YlOrRd', 'basisregions', 'TENA.geo.json', states])
  #   run(dir, 'read_sedac_pm2-5.py', [y, y, 'YlOrRd', 'geo-countries', 'geo-countries-union.json', world])

  # run(dir, 'read_sedac_pm2-5.py', [2000, 2000, 'YlOrRd', 'basisregions', 'TENA.geo.json', states])
  # run(dir, 'read_sedac_pm2-5.py', [2000, 2000, 'YlOrRd', os.path.join('USA_states_counties', 'us_states'), '06-CA-California.geojson', counties])
  # t0 = timer_restart(t0, '')

  # exit()

  # max range = '200001', '201812'
  run(dir, 'read_acag_pm2-5.py', ['200001', '201812', 'YlOrRd', 'geo-countries', 'geo-countries-union.json', states, True])
  # run(dir, 'read_acag_pm2-5.py', ['200001', '201812', 'YlOrRd', 'basisregions', 'TENA.geo.json', counties, True])
  # run(dir, 'read_acag_pm2-5.py', ['200001', '200001', 'YlOrRd', os.path.join('USA_states_counties', 'us_states'), '06-CA-California.geojson', states, True])
  
  # exit()


  # # startYYYYMM, endYYYYMM, pltTitle
  # dates01 = [i for i in dates if i.endswith('01')]
  # dates12 = [i for i in dates if i.endswith('12')]
  # print(dates01)
  # print(dates12)

  # for d in range(min(len(dates01), len(dates12))):
  #   run(dir, 'plot_usa.py', [dates01[d], dates12[d], 'CountyDeaths']) 




  # for d in dates:
  #   run(dir, 'plot_usa.py', [d, d, 'CountyDeaths']) 
  
  # run(dir, 'plot_usa.py', [dates[0], dates[-1], 'CountyDeaths']) 
  # run(dir, 'plot_usa.py', [dates[30], dates[30], 'CountyDeaths']) 



  t1 = timer_restart(t1, 'main total time')
  