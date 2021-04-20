
import numpy as np
from numpy import cos, sin, arctan2, arccos
import geopandas as gpd
from numpy.core.numeric import NaN
import pandas as pd

import os, pathlib, io, copy, re, json
from shapely.geometry import shape, GeometryCollection
from shapely.ops import unary_union
import rasterio, rasterio.features, rasterio.warp


import importlib
fc = importlib.import_module('functions')


def combineBorders(*geoms):
  polys = [GeometryCollection(g) for g in geoms]
  bufs = [p.buffer(0) for p in polys]
  newShape = unary_union([b for b in bufs])
  return newShape


def tempstring11(temp):
  temp = str(temp)
  while len(temp) != 11:
    temp = '0' + temp
  return temp


def tempstring10(temp):
  temp = str(temp)
  while len(temp) != 10:
    temp = '0' + temp
  return temp


dates = ['199901', '199902', '199903', '199904', '199905', '199906', '199907', '199908', '199909', '199910', '199911', '199912', '200001', '200002', '200003', '200004', '200005', '200006', '200007', '200008', '200009', '200010', '200011', '200012', '200101', '200102', '200103', '200104', '200105', '200106', '200107', '200108', '200109', '200110', '200111', '200112', '200201', '200202', '200203', '200204', '200205', '200206', '200207', '200208', '200209', '200210', '200211', '200212', '200301', '200302', '200303', '200304', '200305', '200306', '200307', '200308', '200309', '200310', '200311', '200312', '200401', '200402', '200403', '200404', '200405', '200406', '200407', '200408', '200409', '200410', '200411', '200412', '200501', '200502', '200503', '200504', '200505', '200506', '200507', '200508', '200509', '200510', '200511', '200512', '200601', '200602', '200603', '200604', '200605', '200606', '200607', '200608', '200609', '200610', '200611', '200612', '200701', '200702', '200703', '200704', '200705', '200706', '200707', '200708', '200709', '200710', '200711', '200712', '200801', '200802', '200803', '200804', '200805', '200806', '200807', '200808', '200809', '200810', '200811', '200812', '200901', '200902', '200903', '200904', '200905', '200906', '200907', '200908', '200909', '200910', '200911', '200912', '201001', '201002', '201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012', '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110', '201111', '201112', '201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308', '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907']

if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'plot_usa-outfiles')

  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  nClimDivDir = os.path.join(ppPath, 'nClimDiv data')

  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  pm_fnames = [re.split('.nc',l)[0] for l in sorted(os.listdir(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25')))]

  # PARAMS
  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title

  doChanges = True
  origDataMonth = '07'
  suppValString = '-1'
  # END PARAMS



  t0 = fc.timer_start()
  t1 = t0


  startDate, endDate = '200001', '201812'
  months = ['01','02','03','04','05','06','07','08','09','10','11','12']
  dates = [i for i in dates if i >= startDate and i <= endDate]

  stateMapFile = 'cb_2019_us_state_500k'
  stateMapData = gpd.read_file(os.path.join(usaDir, stateMapFile, stateMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  countyMapFile = 'cb_2019_us_county_500k'
  countyMapData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  countyMapData = fc.clean_states_reset_index(countyMapData)
  countyMapData = fc.county_changes_deaths_reset_index(countyMapData)


  climdivMapData = gpd.read_file(os.path.join(nClimDivDir, 'CONUS_CLIMATE_DIVISIONS', 'GIS.OFFICIAL_CLIM_DIVISIONS' + '.shp')).sort_values(by=['CLIMDIV']).reset_index(drop=True)
  climdivMapData = climdivMapData.replace('Massachusettes' , 'Massachusetts') # fix typo
  # t0 = fc.timer_restart(t0, 'read shp')
  # print(climdivMapData)

  t0 = fc.timer_restart(t0, 'read map data')

  climToStateCodes = dict()
  prev = ''
  for i in range(len(climdivMapData)):
    statecode = '{:02d}'.format(int(climdivMapData.STATE_CODE[i]))
    name = climdivMapData.STATE_FIPS[i]
    if statecode > '48':
      break
    if name != prev:
      climToStateCodes[statecode] = name
      prev = name
  t0 = fc.timer_restart(t0, 'climToStateCodes')


  latlonGEOID, pmMat, tf, geoidMat, areaMat, latAreas = None, None, None, None, None, None
  deg, templon = 0.01, 0
  regionDir, regionFile = 'basisregions', 'TENA.geo.json'
  with open(os.path.join(shapefilesDir, regionDir, regionFile), 'r') as f:
    contents = json.load(f)
    basisregion = shape(contents['features'][0]['geometry'])
  t0 = fc.timer_restart(t0, 'read basisregion')
  if regionFile.split('.')[0] + '_rounded.hdf5' not in os.listdir(os.path.join(pmDir, 'points_in_region_rounded')):
    latlonGEOID = fc.read_df(os.path.join(pmDir, 'points_in_region'), regionFile.split('.')[0], 'hdf5')
    latlonGEOID.lat = [round(i, 3) for i in latlonGEOID.lat]
    latlonGEOID.lon = [round(i, 3) for i in latlonGEOID.lon]
    fc.save_df(latlonGEOID, os.path.join(pmDir, 'points_in_region_rounded'), regionFile.split('.')[0] + '_rounded', 'hdf5')
  else:
    latlonGEOID = pd.DataFrame(fc.read_df(os.path.join(pmDir, 'points_in_region_rounded'), regionFile.split('.')[0] + '_rounded', 'hdf5'))
  t0 = fc.timer_restart(t0, 'get latlonGEOID')
  fd = fc.read_df(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25'), pm_fnames[0], 'nc') # for checking
  pmMat = fd.variables['PM25'][:] # for checking
  tf = rasterio.transform.from_origin(np.round(np.min(fd.variables['LON'][:]), 2), np.round(np.max(fd.variables['LAT'][:]), 2), deg,deg)
  minLat, maxLat, minLon, maxLon = fc.get_bound_indices(basisregion.bounds, tf)
  xy = rasterio.transform.xy(tf, range(fd.dimensions['LAT'].size), range(fd.dimensions['LAT'].size))
  lats_1d = np.array(xy[1])
  xy = rasterio.transform.xy(tf, range(fd.dimensions['LON'].size), range(fd.dimensions['LON'].size))
  lons_1d = np.array(xy[0])
  bounded_mat = pmMat[minLat:maxLat,minLon:maxLon]
  latlonGEOID = latlonGEOID.reindex(pd.Index(np.arange(0,bounded_mat.shape[0]*bounded_mat.shape[1])))
  latlonGEOID['lat'], latlonGEOID['lon'] = fc.bound_ravel(lats_1d, lons_1d, basisregion.bounds, tf)
  latlonGEOID['GEOID'] = latlonGEOID['GEOID'].replace(NaN,'')
  temp = np.reshape(list(latlonGEOID['GEOID']), bounded_mat.shape)
  geoidMat = np.empty(pmMat.shape, dtype='<U5')
  geoidMat[minLat:maxLat,minLon:maxLon] = temp
  latAreas = pd.DataFrame(np.reshape(list(latlonGEOID['lat']), bounded_mat.shape)[:,0], columns=['lat'])
  latAreas['area'] = [fc.quad_area(templon, lat, deg) for lat in latAreas.lat]
  temp = np.matrix([np.repeat(a, bounded_mat.shape[1]) for a in latAreas.area])
  areaMat = np.empty(pmMat.shape)
  areaMat[minLat:maxLat,minLon:maxLon] = temp
  t0 = fc.timer_restart(t0, 'initialize (areaMat, geoidMat; pm25, lats/lons for checking)')







  dfClimDiv = None

  # rst_fn = 'temp_raster.tif'
  # out_fn

  # by clim div, need to aggregate by county
  # drought PDSI file
  filenames = [name for name in os.listdir(nClimDivDir) if name.startswith('climdiv') and name.endswith('.0-20210406')]
  # print(filenames)
  for filename in filenames: # ~14 min each
    # continue
    print(filename)
    outFileName = '.'.join(filename.split('.')[:-1])

    if 'by_climdiv_'+outFileName + '.hdf5' in os.listdir(nClimDivDir):
      dfClimDiv = fc.read_df(nClimDivDir, 'by_climdiv_'+outFileName, 'hdf5' )
      t0 = fc.timer_restart(t0, 'load by_climdiv_'+outFileName)
    else:
      df = open(os.path.join(nClimDivDir, filename), 'r').read()
      df = pd.read_csv(io.StringIO(df), names=['temp']+months, sep='\s+')
      df['year'] = [str(i)[-4:] for i in df.temp]
      df = df[df.year >= startDate[:4]]
      df = df[df.year <= endDate[:4]]
      df.temp = [tempstring10(i) for i in df.temp]
      df = df[df.temp.str[:2] <= '48']
      df['STATEFP'] = [climToStateCodes[i[:2]] for i in df.temp]
      df['CD'] = [i[2:4] for i in df.temp]
      df['FIPS_CD'] = [s+t for s,t in zip(df.STATEFP, df.CD)]
      df = df.drop(columns=['temp']).reset_index(drop=True)
      dfClimDiv = pd.DataFrame()
      dfClimDiv['FIPS_CD'] = sorted(set(df.FIPS_CD))
      dfClimDiv['STATEFP'] = [item[:2] for item in dfClimDiv.FIPS_CD]
      dfClimDiv[dates] = None
      for geoid in dfClimDiv['FIPS_CD']:
        temp = df[df['FIPS_CD'] == geoid]
        dfClimDiv.loc[dfClimDiv.FIPS_CD == geoid, dates] = np.ravel(temp.loc[:,months])
      fc.save_df(dfClimDiv, nClimDivDir, 'by_climdiv_'+outFileName, 'hdf5')
      fc.save_df(dfClimDiv, nClimDivDir, 'by_climdiv_'+outFileName, 'csv')
      t0 = fc.timer_restart(t0, 'write by_climdiv_'+outFileName)

    # map values to 0.01 deg grid (matrix)             # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
    assert list(dfClimDiv.FIPS_CD) == list(climdivMapData.FIPS_CD)


    dfOut = pd.DataFrame()
    dfOut['GEOID'] = countyMapData.GEOID
    # dfOut = dfOut.drop(columns=['STATEFP'])

    for date in dates:
      print(date)
      shapes = ((geom,value) for geom, value in zip(climdivMapData.geometry, dfClimDiv[date]))
      mat = rasterio.features.rasterize(shapes=shapes, fill=NaN, out_shape=pmMat.shape, transform=tf)
      dfOut[date] = fc.aggregate_by_geoid(areaMat, mat, geoidMat, tf, countyMapData)

    t0 = fc.timer_restart(t0, 'write '+outFileName)
    print(dfOut)

    # print(df)
    # break

    # # dfOut = fc.clean_states_reset_index(dfOut) # does nothing
    # # dfOut = fc.county_changes_deaths_reset_index(dfOut) # does nothing

    # t0 = fc.timer_restart(t0, 'convert '+outFileName)

    fc.save_df(dfOut, nClimDivDir, outFileName, 'hdf5')
    t0 = fc.timer_restart(t0, 'save hdf5 '+outFileName)
    fc.save_df(dfOut, nClimDivDir, outFileName, 'csv')
    t0 = fc.timer_restart(t0, 'save csv '+outFileName)





  # already by county
  # temp/precip files
  filenames = [name for name in os.listdir(nClimDivDir) if name.startswith('climdiv') and name.endswith('.0-20210304')]
  for filename in filenames:
    # continue
    print(filename)
    outFileName = '.'.join(filename.split('.')[:-1])

    df = open(os.path.join(nClimDivDir, filename), 'r').read()
    df = pd.read_csv(io.StringIO(df), names=['temp']+months, sep='\s+')

    df['year'] = [str(i)[-4:] for i in df.temp]

    df = df[df.year >= startDate[:4]]
    df = df[df.year <= endDate[:4]]

    df.temp = [tempstring11(i) for i in df.temp]

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

    dfOut = dfOut.drop(columns=['STATEFP'])

    # dfOut = fc.clean_states_reset_index(dfOut) # does nothing
    # dfOut = fc.county_changes_deaths_reset_index(dfOut) # does nothing

    t0 = fc.timer_restart(t0, 'convert '+outFileName)

    fc.save_df(dfOut, nClimDivDir, outFileName, 'hdf5')
    t0 = fc.timer_restart(t0, 'save hdf5 '+outFileName)
    fc.save_df(dfOut, nClimDivDir, outFileName, 'csv')
    t0 = fc.timer_restart(t0, 'save csv '+outFileName)




  t1 = fc.timer_restart(t1, 'total time')


