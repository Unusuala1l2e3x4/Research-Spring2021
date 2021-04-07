
import numpy as np

import geopandas as gpd
import pandas as pd

import os, pathlib, io, sys, copy

import importlib
fc = importlib.import_module('functions')

climStateCodes = {'01':'Alabama','02':'Arizona','03':'Arkansas','04':'California','05':'Colorado','06':'Connecticut','07':'Delaware','08':'Florida','09':'Georgia', 
'10':'Idaho','11':'Illinois','12':'Indiana','13':'Iowa','14':'Kansas','15':'Kentucky','16':'Louisiana','17':'Maine','18':'Maryland','19':'Massachusetts','20':'Michigan', 
'21':'Minnesota','22':'Mississippi','23':'Missouri','24':'Montana','25':'Nebraska','26':'Nevada','27':'New Hampshire','28':'New Jersey','29':'New Mexico','30':'New York', 
'31':'North Carolina','32':'North Dakota','33':'Ohio','34':'Oklahoma','35':'Oregon','36':'Pennsylvania','37':'Rhode Island','38':'South Carolina','39':'South Dakota', 
'40':'Tennessee','41':'Texas','42':'Utah','43':'Vermont','44':'Virginia','45':'Washington','46':'West Virginia','47':'Wisconsin','48':'Wyoming','50':'Alaska'}



def tempstring(temp):
  temp = str(temp)
  while len(temp) != 11:
    temp = '0' + temp
  return temp


dates = ['199901', '199902', '199903', '199904', '199905', '199906', '199907', '199908', '199909', '199910', '199911', '199912', '200001', '200002', '200003', '200004', '200005', '200006', '200007', '200008', '200009', '200010', '200011', '200012', '200101', '200102', '200103', '200104', '200105', '200106', '200107', '200108', '200109', '200110', '200111', '200112', '200201', '200202', '200203', '200204', '200205', '200206', '200207', '200208', '200209', '200210', '200211', '200212', '200301', '200302', '200303', '200304', '200305', '200306', '200307', '200308', '200309', '200310', '200311', '200312', '200401', '200402', '200403', '200404', '200405', '200406', '200407', '200408', '200409', '200410', '200411', '200412', '200501', '200502', '200503', '200504', '200505', '200506', '200507', '200508', '200509', '200510', '200511', '200512', '200601', '200602', '200603', '200604', '200605', '200606', '200607', '200608', '200609', '200610', '200611', '200612', '200701', '200702', '200703', '200704', '200705', '200706', '200707', '200708', '200709', '200710', '200711', '200712', '200801', '200802', '200803', '200804', '200805', '200806', '200807', '200808', '200809', '200810', '200811', '200812', '200901', '200902', '200903', '200904', '200905', '200906', '200907', '200908', '200909', '200910', '200911', '200912', '201001', '201002', '201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012', '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110', '201111', '201112', '201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308', '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907']

if __name__ == "__main__":
  # mode = sys.argv[1]
  # ext = sys.argv[2]

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'plot_usa-outfiles')

  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  nClimDivDir = os.path.join(ppPath, 'nClimDiv data')

  # PARAMS
  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title

  testing = False
  doChanges = True
  origDataMonth = '07'
  suppValString = '-1'

  mode = 'w'

  # END PARAMS


  t0 = fc.timer_start()
  t1 = t0


  startYYYYMM = '200001'
  endYYYYMM = '201812'
  months = ['01','02','03','04','05','06','07','08','09','10','11','12']
  dates = [i for i in dates if i >= startYYYYMM and i <= endYYYYMM]

  stateMapFile = 'cb_2019_us_state_500k'
  stateMapData = gpd.read_file(os.path.join(usaDir, stateMapFile, stateMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  countyMapFile = 'cb_2019_us_county_500k'
  countyMapData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  countyMapData = fc.clean_states_reset_index(countyMapData)
  countyMapData = fc.county_changes_deaths_reset_index(countyMapData)

  t0 = fc.timer_restart(t0, 'read map data')

  climToStateCodes = dict()

  # print(stateMapData)
  # print(climStateCodes)

  # print(fc.clean_states_reset_index(stateMapData))

  for code in climStateCodes.keys():
    climToStateCodes[code] = fc.stateGEOIDstring(int(stateMapData.loc[stateMapData.NAME==climStateCodes[code],'GEOID']))

  t0 = fc.timer_restart(t0, 'climToStateCodes')
  
  # print(climToStateCodes)

  # exit()


  # for testing
  data_GEOIDs = []
  all_filenames = []
  
  # temp/precip files
  filenames = [name for name in os.listdir(nClimDivDir) if name.startswith('climdiv') and name.endswith('.0-20210304')]
  print(filenames)
  # exit()
  for filename in filenames:
    print(filename)
    outFileName = '.'.join(filename.split('.')[:-1])


    df = open(os.path.join(nClimDivDir, filename), 'r').read()
    df = pd.read_csv(io.StringIO(df), names=['temp']+months, sep='\s+')

    df['year'] = [str(i)[-4:] for i in df.temp]

    df = df[df.year >= startYYYYMM[:4]]
    df = df[df.year <= endYYYYMM[:4]]

    df.temp = [tempstring(i) for i in df.temp]

    df['STATEFP'] = [climToStateCodes[i[:2]] for i in df.temp]

    df['GEOID'] = [s+t[2:5] for s,t in zip(df.STATEFP, df.temp)]
    df = df.drop(columns=['temp'])

    ## CHANGES
    if doChanges:
      # remove Alaska
      df = df[df.STATEFP != '02']

      # replace 24511 with 11001 (Washington DC)
      df.loc[:,'GEOID'] = df.loc[:,'GEOID'].replace('24511','11001')
      df.STATEFP = [i[:2] for i in df.GEOID]
      
      # 51678 not in clim data -> set values to bounding county (51163), append to df
      df51163 = copy.deepcopy(df[df.GEOID == '51163'])
      df51163.loc[:,'GEOID'] = df51163.loc[:,'GEOID'].replace('51163','51678')
      df = df.append(df51163)

    dfOut = pd.DataFrame()
    dfOut['GEOID'] = sorted(set(df.GEOID))
    dfOut['STATEFP'] = [item[0:2] for item in dfOut.GEOID]
    dfOut[dates] = None

    
    ## CHECK IS CHANGES EXECUTED
    # print(set(dfOut['GEOID']) - set(countyMapData.GEOID)) # {'24511' - fixed} Baltimore, MD?     MD = 18 in climdiv
    # print(set(countyMapData.GEOID) - set(dfOut['GEOID'])) # {'11001' - fixed, '51678 - fixed'} - Washinton, DC; Lexington City, VA

    # print(dfOut[dfOut.GEOID == '51678']) # 51678 not in clim data -> set values to bounding county (51163)
    # print(countyMapData[countyMapData.GEOID == '51678'])

    # print(sorted(countyMapData.GEOID) == sorted(dfOut['GEOID']))
    # print(sorted(countyMapData.GEOID) == sorted(set(df['GEOID'])))
    
    for geoid in dfOut['GEOID']:
      temp = df[df['GEOID'] == geoid]
      dfOut.loc[dfOut.GEOID == geoid, dates] = np.ravel(temp.loc[:,months])
    print(dfOut)

    dfOut = dfOut.drop(columns=['STATEFP'])

    # dfOut = fc.clean_states_reset_index(dfOut) # does nothing
    # dfOut = fc.county_changes_deaths_reset_index(dfOut) # does nothing

    t0 = fc.timer_restart(t0, 'convert '+outFileName)

    fc.save_df(dfOut, nClimDivDir, outFileName, 'hdf5')
    t0 = fc.timer_restart(t0, 'save hdf5 '+outFileName)
    fc.save_df(dfOut, nClimDivDir, outFileName, 'csv')
    t0 = fc.timer_restart(t0, 'save csv '+outFileName)



  t1 = fc.timer_restart(t1, 'total time')


