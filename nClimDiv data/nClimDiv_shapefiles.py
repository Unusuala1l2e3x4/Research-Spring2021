
import numpy as np
import geopandas as gpd
import pandas as pd

import os, pathlib, io, sys, copy, time

# import importlib
# fc = importlib.import_module('functions')


def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()



climStateCodes = {'01':'Alabama','02':'Arizona','03':'Arkansas','04':'California','05':'Colorado','06':'Connecticut','07':'Delaware','08':'Florida','09':'Georgia', 
'10':'Idaho','11':'Illinois','12':'Indiana','13':'Iowa','14':'Kansas','15':'Kentucky','16':'Louisiana','17':'Maine','18':'Maryland','19':'Massachusetts','20':'Michigan', 
'21':'Minnesota','22':'Mississippi','23':'Missouri','24':'Montana','25':'Nebraska','26':'Nevada','27':'New Hampshire','28':'New Jersey','29':'New Mexico','30':'New York', 
'31':'North Carolina','32':'North Dakota','33':'Ohio','34':'Oklahoma','35':'Oregon','36':'Pennsylvania','37':'Rhode Island','38':'South Carolina','39':'South Dakota', 
'40':'Tennessee','41':'Texas','42':'Utah','43':'Vermont','44':'Virginia','45':'Washington','46':'West Virginia','47':'Wisconsin','48':'Wyoming'} # ,'50':'Alaska'

def tempstring(temp):
  temp = str(temp)
  while len(temp) != 11:
    temp = '0' + temp
  return temp

dates = ['199901', '199902', '199903', '199904', '199905', '199906', '199907', '199908', '199909', '199910', '199911', '199912', '200001', '200002', '200003', '200004', '200005', '200006', '200007', '200008', '200009', '200010', '200011', '200012', '200101', '200102', '200103', '200104', '200105', '200106', '200107', '200108', '200109', '200110', '200111', '200112', '200201', '200202', '200203', '200204', '200205', '200206', '200207', '200208', '200209', '200210', '200211', '200212', '200301', '200302', '200303', '200304', '200305', '200306', '200307', '200308', '200309', '200310', '200311', '200312', '200401', '200402', '200403', '200404', '200405', '200406', '200407', '200408', '200409', '200410', '200411', '200412', '200501', '200502', '200503', '200504', '200505', '200506', '200507', '200508', '200509', '200510', '200511', '200512', '200601', '200602', '200603', '200604', '200605', '200606', '200607', '200608', '200609', '200610', '200611', '200612', '200701', '200702', '200703', '200704', '200705', '200706', '200707', '200708', '200709', '200710', '200711', '200712', '200801', '200802', '200803', '200804', '200805', '200806', '200807', '200808', '200809', '200810', '200811', '200812', '200901', '200902', '200903', '200904', '200905', '200906', '200907', '200908', '200909', '200910', '200911', '200912', '201001', '201002', '201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012', '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110', '201111', '201112', '201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308', '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907']


def main():
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'River Basins - smaller files')
  


  dirname = 'CONUS_CLIMATE_DIVISIONS'
  filename = 'GIS.OFFICIAL_CLIM_DIVISIONS'
  


  t0 = timer_start()
  t1 = t0


  shapeData = gpd.read_file(os.path.join(pPath, dirname, filename + '.shp')).sort_values(by=['CLIMDIV']).reset_index(drop=True)
  t0 = timer_restart(t0, 'read shp')
  shapeData = shapeData.replace('Massachusettes' , 'Massachusetts') # fix typo
  print(shapeData)


  climStateCodes2 = dict()

  prev = ''
  for i in range(len(shapeData)):
    statecode = '{:02d}'.format(int(shapeData.STATE_CODE[i]))
    if statecode > '48':
      break
    if shapeData.STATE[i] != prev:
      climStateCodes2[statecode] = shapeData.STATE[i]
      prev = shapeData.STATE[i]
      

  print(climStateCodes)
  print(climStateCodes2)
  assert climStateCodes == climStateCodes2

  exit()

  shapeData.to_file(os.path.join(pPath, dirname, filename + '.geojson'), driver='GeoJSON')
  t0 = timer_restart(t0, 'write geojson')

  # divs = 20

  # n=0
  # for i in range(0,len(shapeData),int(len(shapeData) / divs)):
  #   j = int(len(shapeData) / divs)
  #   print(i, i + j)

  #   # os.mkdir(os.path.join(outputDir, 'RiverBasins'+str(n)))
  #   shapeData.iloc[i:j,:].to_file(os.path.join(outputDir,'RiverBasins'+str(n)+'.geojson'), driver='GeoJSON')

  #   n += 1






  t1 = timer_restart(t1, 'total time')














if __name__ == "__main__":
  main()

