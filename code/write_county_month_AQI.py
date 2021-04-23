
from datetime import timedelta
import numpy as np

import geopandas as gpd
from numpy.core.numeric import NaN
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as cl
import matplotlib.cm as cm
import matplotlib.font_manager as fm

import os, pathlib, re, copy, json, sys

from scipy.interpolate import interp1d, griddata
import zipfile

from shapely.geometry import shape
import rasterio, rasterio.features, rasterio.warp

import importlib
fc = importlib.import_module('functions')


dates = ['199901', '199902', '199903', '199904', '199905', '199906', '199907', '199908', '199909', '199910', '199911', '199912', '200001', '200002', '200003', '200004', '200005', '200006', '200007', '200008', '200009', '200010', '200011', '200012', '200101', '200102', '200103', '200104', '200105', '200106', '200107', '200108', '200109', '200110', '200111', '200112', '200201', '200202', '200203', '200204', '200205', '200206', '200207', '200208', '200209', '200210', '200211', '200212', '200301', '200302', '200303', '200304', '200305', '200306', '200307', '200308', '200309', '200310', '200311', '200312', '200401', '200402', '200403', '200404', '200405', '200406', '200407', '200408', '200409', '200410', '200411', '200412', '200501', '200502', '200503', '200504', '200505', '200506', '200507', '200508', '200509', '200510', '200511', '200512', '200601', '200602', '200603', '200604', '200605', '200606', '200607', '200608', '200609', '200610', '200611', '200612', '200701', '200702', '200703', '200704', '200705', '200706', '200707', '200708', '200709', '200710', '200711', '200712', '200801', '200802', '200803', '200804', '200805', '200806', '200807', '200808', '200809', '200810', '200811', '200812', '200901', '200902', '200903', '200904', '200905', '200906', '200907', '200908', '200909', '200910', '200911', '200912', '201001', '201002', '201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012', '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110', '201111', '201112', '201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308', '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907']
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

def endOfMonth(yyyymm):
  if yyyymm.month == 12:
    return yyyymm.replace(year=yyyymm.year+1, month=1)-timedelta(days=1) # last day of year
  else:
    return yyyymm.replace(month=yyyymm.month+1)-timedelta(days=1) # last day of year


# def daysInMonth(date):
#   return len(pd.date_range(pd.to_datetime(date, format='%Y%m', errors='coerce'), endOfMonth(pd.to_datetime(date, format='%Y%m', errors='coerce')), freq='d'))


def aggregate_fully_interpolated_months(df):
  # row.date: number of rows aggregated for that month -> equal to days in month
  # row.Index: date as yyyymm
  grouped = df.groupby(df.date.dt.strftime('%Y%m'))
  t = grouped.count()
  # print(t)
  monthdays = pd.DataFrame(np.rot90([list(t.date)]*len(df.columns), k=-1))
  monthdays.index, monthdays.columns = t.index, t.columns
  # print(monthdays)
  mask = monthdays == t
  ret = grouped.mean()*mask.replace(False, NaN)
  return ret.loc[:, ret.columns != 'date']



def plot_percent_nan(df, title):
  print(title)
  sites = sorted(set(df.columns) - {'date'})

  df.date = [np.datetime64(i) for i in df.date]
  df['percent nans'] = df[sites].isnull().sum(axis=1) * 100 / df[sites].shape[1]
  # count missing values for each date
  fig, ax = plt.subplots()
  ax.plot('date', 'percent nans', data=df)

  plt.ylabel('% nan')
  plt.xlabel('date')
  plt.title(title)

  ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

  datemin = np.datetime64(np.min(df['date']), 'Y')
  datemax = np.datetime64(np.max(df['date']), 'Y') + np.timedelta64(1, 'Y')
  ax.set_xlim(datemin, datemax)
  ax.format_xdata = mdates.DateFormatter('%Y-%m')
  ax.grid(True)
  fig.autofmt_xdate()

  plt.show()


tickSpacings = [0.1,0.2,0.5,1,2,5,10]

maxMappedValue = 400
def plot_interpolation(m, plttitle):
  res = 3
  maxUnit = np.nanmax(m)
  minUnit = np.nanmin(m)

  cmap = copy.copy(cm.get_cmap('gist_stern'))
  norm = cl.Normalize(vmin=0, vmax=maxMappedValue, clip=False) # clip=False is default; see https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html#matplotlib.colors.Normalize
  mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range

  tickSpacing = fc.closest(tickSpacings, maxMappedValue/15)
  with plt.style.context(("seaborn", "ggplot")):
    shapeData.boundary.plot(figsize=(18*res,10*res), edgecolor='black', color="white", linewidth=0.3*res)

    plt.xlabel("Longitude", fontsize=7*res)
    plt.ylabel("Latitude", fontsize=7*res)
    plt.xticks(fontsize=7*res)
    plt.yticks(fontsize=7*res)
    plt.title(plttitle, fontsize=7*res)

    plt.xlim((lons_1d[minLon] - deg, lons_1d[maxLon] + deg))
    plt.ylim((lats_1d[maxLat] - deg, lats_1d[minLat] + deg))

    ## contour lines
    levelsDiscrete = np.arange(0,maxMappedValue + tickSpacing,tickSpacing)
    img = plt.imshow(m, vmin=minUnit, vmax=maxMappedValue, cmap=cmap, origin='lower', extent=[lons_1d[minLon], lons_1d[maxLon], lats_1d[minLat], lats_1d[maxLat]])
    img.cmap.set_over('black')
    img.cmap.set_under('white')
    img.changed()

    ticks = sorted(list(set([maxMappedValue] + list(levelsDiscrete) ))) # [minUnit, maxUnit] + 
    cb = plt.colorbar(mapper, ticks=ticks, drawedges=True, label='AQI', pad=0.001)
    cb.ax.yaxis.label.set_font_properties(fm.FontProperties(size=7*res))
    cb.ax.tick_params(labelsize=5*res)
    plt.grid(False)

    fc.save_plt(plt, os.path.join(epaAqDir, 'images'), plttitle+'_' + "{:.3f}".format(maxUnit) + '_' + "{:.3f}".format(maxMappedValue) + '_' + fc.utc_time_filename(), 'png')

    # plt.show()
    plt.close()



if __name__ == "__main__":
  limit_tests, methods = None, None

  numArgs = len(sys.argv)
  if numArgs > 1:
    methods = [sys.argv[1]]
    limit_tests = [int(i) for i in sys.argv[2:]]
    

  print(limit_tests, methods)
  # exit()



  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())

  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')


  title = 'daily_aqi_by_county'
  epaAqDir = os.path.join(ppPath, 'EPA AQ data')
  aqiDir = os.path.join(epaAqDir, title)
  
  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  pm_fnames = [re.split('.nc',l)[0] for l in sorted(os.listdir(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25')))]
  

  latlonGEOID, pmMat, tf, geoidMat, areaMat, latAreas = None, None, None, None, None, None
  deg, templon = 0.01, 0
  regionDir, regionFile = 'basisregions', 'TENA.geo.json'

  # PARAMS
  testing = False
  origDataMonth = '07'
  suppValString = '-1'
  ext = 'hdf5'

  startDate, endDate = '200001', '201812'
  startDateTemp, endDateTemp = '199901', '202001'  # startDate - endDate should be strict subset
  # END PARAMS

  if limit_tests is None:
    limit_tests = list(range(0,25,4))
  if methods is None:
    methods = ['linear', 'nearest', 'cubic']
  


  dates = [i for i in dates if i >= startDate and i <= endDate]
  dailyDates = list(pd.date_range(pd.to_datetime(startDate, format='%Y%m', errors='coerce'), endOfMonth(pd.to_datetime(endDate, format='%Y%m', errors='coerce')), freq='d'))
  dailyDatesExtra = list(pd.date_range(pd.to_datetime(startDateTemp, format='%Y%m', errors='coerce'), pd.to_datetime(endDateTemp, format='%Y%m', errors='coerce'), freq='d'))

  # print(dailyDates)
  # print(dailyDatesExtra)

  # exit()

  t0 = fc.timer_start()
  t1 = t0


  USAstates = os.path.join('USA_states_counties', 'cb_2019_us_state_500k', 'cb_2019_us_state_500k.shp')
  shapeData = gpd.read_file(os.path.join(shapefilesDir, USAstates)).sort_values(by=['GEOID']).reset_index(drop=True)
  shapeData = fc.clean_states_reset_index(shapeData)
  shapeData = fc.county_changes_deaths_reset_index(shapeData)

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

  # print(pmMat)
  meshes = tuple(np.meshgrid(lons_1d, lats_1d))
  # print(meshes[0])
  # print(meshes[1])

  # exit()








  countyMapFile = 'cb_2019_us_county_500k'
  countyMapData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  countyMapData = fc.clean_states_reset_index(countyMapData)
  countyMapData = fc.county_changes_deaths_reset_index(countyMapData)
  allgeoids = set(countyMapData.GEOID)

  sites = pd.read_csv(zipfile.ZipFile(os.path.join(epaAqDir, 'aqs_sites'+'.zip')).open('aqs_sites' + '.csv', 'r'))[['State Code','County Code','Site Number','Latitude','Longitude']]
  sites = sites[sites['State Code'] <= '56']
  sites.index = [ '{:02d}-{:03d}-{:04d}'.format(int(x),y,z) for x,y,z in zip(sites['State Code'], sites['County Code'], sites['Site Number'])]
  sites = sites[['Latitude','Longitude']]
  sites.columns = ['lat','lon']
  allsites = set(sites.index)

  sitesLonLat = sites.to_dict('index')
  for i in sitesLonLat:
    sitesLonLat[i] = (sitesLonLat[i]['lon'], sitesLonLat[i]['lat']) # siteid : (lon, lat)

  # print(countyMapData)
  # print(sites)
  # print(sitesLonLat)
  # exit()


  outFileName = 'TENA_county_AQI_'+startDate+'_'+endDate


  fd, dailydf, dailyDefiningSite = None, None, None
  prevyear = ''

  
  dailydfName = title + '_' + startDateTemp + '_' + endDateTemp
  dailyDefiningSiteName = title + '_defining_site_' + startDateTemp + '_' + endDateTemp


  # if dailydfName + '.zip' in os.listdir(epaAqDir):
  #   dailydf = fc.read_df(epaAqDir, dailydfName, 'zip')
  #   t0 = fc.timer_restart(t0, 'load dailydf')
  # else:
  #   print('writing ' + dailydfName)
  #   dailydf = pd.DataFrame(columns=['date'] + sorted(allgeoids))
  #   dailydf.date = dailyDatesExtra
  #   # read all data
  #   for fname in [i for i in os.listdir(aqiDir) if i.endswith('.zip')]: # ~ 11.8 min
  #     print(fname)
  #     fd = pd.read_csv(zipfile.ZipFile(os.path.join(aqiDir, fname)).open(fname.split('.')[-2] + '.csv', 'r'))
  #     fd.Date = pd.to_datetime(fd.Date, errors='coerce')
  #     fd = fd[fd['State Code'].astype(str) <= '56']
  #     fd['GEOID'] = [ '{:02d}{:03d}'.format(int(x),y) for x,y in zip(fd['State Code'], fd['County Code'])]
  #     fd = fc.clean_states_reset_index(fd)

  #     for geoid in allgeoids.intersection(set(fd.GEOID)): # only missed GEOID is 23901, which has no data after 2004, and is inside another county with much more sites
  #       dailydf.loc[dailydf.date.isin(fd.loc[fd.GEOID == geoid, 'Date']), geoid ] = fd.loc[(fd.GEOID == geoid) & (fd.Date.isin(dailydf.date)), 'AQI' ].tolist()

  #   t0 = fc.timer_restart(t0, 'write dailydf') # ~12 min
  #   fc.save_df(dailydf, epaAqDir, dailydfName, 'zip')
  #   t0 = fc.timer_restart(t0, 'save dailydf zip')



  if dailyDefiningSiteName + '.zip' in os.listdir(epaAqDir):
    dailyDefiningSite = fc.read_df(epaAqDir, dailyDefiningSiteName, 'zip')
    t0 = fc.timer_restart(t0, 'load dailyDefiningSite')
  else:
    fd = None
    print('writing ' + dailyDefiningSiteName)
    dailyDefiningSite = pd.DataFrame(columns=['date'] + sorted(allsites))
    dailyDefiningSite.date = dailyDatesExtra
    # read all data
    for fname in [i for i in os.listdir(aqiDir) if i.endswith('.zip')]: # ~ 24.3 min
      print(fname)
      fd = pd.read_csv(zipfile.ZipFile(os.path.join(aqiDir, fname)).open(fname.split('.')[-2] + '.csv', 'r'))
      fd.Date = pd.to_datetime(fd.Date, errors='coerce')
      fd = fd[fd['State Code'].astype(str) <= '56']
      fd['GEOID'] = [ '{:02d}{:03d}'.format(int(x),y) for x,y in zip(fd['State Code'], fd['County Code'])]
      fd = fc.clean_states_reset_index(fd)

      for siteid in allsites.intersection(set(fd['Defining Site'])): # only missed GEOID is 23901, which has no data after 2004, and is inside another county with much more sites
        dailyDefiningSite.loc[dailyDefiningSite.date.isin(fd.loc[fd['Defining Site'] == siteid, 'Date']), siteid ] = fd.loc[(fd['Defining Site'] == siteid) & (fd.Date.isin(dailyDefiningSite.date)), 'AQI' ].tolist()

    dailyDefiningSite = dailyDefiningSite.dropna(axis=1, how='all')

    t0 = fc.timer_restart(t0, 'write dailyDefiningSite') # ~12 min
    fc.save_df(dailyDefiningSite, epaAqDir, dailyDefiningSiteName, 'zip')
    t0 = fc.timer_restart(t0, 'save dailyDefiningSite zip')
  allsites = set(dailyDefiningSite.columns) - {'date'}
  allsitesList = sorted(allsites)

  # print(dailyDefiningSite)

  # plot_percent_nan(dailydf, dailydfName + ' - % NaN')
  # plot_percent_nan(dailyDefiningSite, dailyDefiningSiteName + ' - % NaN')
  # plot_percent_nan(temp, 'interpolated ' + dailyDefiningSiteName + ' - % NaN')



  # interpolate each date's value for each county individually by temporal

  temp, prevlimit, counts = None, None, []


  for limit in limit_tests:
    print(limit)
    if temp is None:
      temp = copy.deepcopy(dailyDefiningSite)
      temp['date'] = temp['date'].astype(np.datetime64)
      if limit != 0:
        temp = temp.interpolate(method='linear',limit_area="inside", axis=0, limit=limit, limit_direction="both")
    else:
      temp = temp.interpolate(method='linear',limit_area="inside", axis=0, limit=limit-prevlimit, limit_direction="both")
    prevlimit = limit
    monthMean = aggregate_fully_interpolated_months(temp)
    counts.append(monthMean.notnull().sum().sum())

    # continue



    for method in methods:
      outName = outFileName + '_limit-' + str(limit) + '_' + method
                  # if outName+'.'+ext in os.listdir(epaAqDir):  # test (comment out)
                  #   continue
      print('\t',outName)

      outData = pd.DataFrame(countyMapData.GEOID, columns=['GEOID'])

      assert allsitesList == list(monthMean.columns)
      for date in dates: #  [7:8]   # test
        # print(date)
        pointval = np.array([ [sitesLonLat[site], i] for site,i in zip(allsitesList, monthMean.loc[date, :]) if not np.isnan(i)] , dtype=object)
        mat = griddata(list(pointval[:,0]), list(pointval[:,1]), meshes, method=method)
        
        if date.endswith('08') and limit in [6,12,18]:   # test
          plttitle = 'limit-' + str(limit) + '_' + method + '_' + date
          plot_interpolation(mat[minLat:maxLat,minLon:maxLon], plttitle)
          
        outData[date] = fc.aggregate_by_geoid_remove_incomplete(areaMat, mat, geoidMat, tf, countyMapData)

      # continue 
      fc.save_df(outData, epaAqDir, outName, ext) # test
      fc.save_df(outData, os.path.join(epaAqDir, 'csv'), outName, 'csv') # test



  t0 = fc.timer_restart(t0, 'save limit files')

  # exit()


  print(list(limit_tests))
  print(counts)
  limit_counts = pd.DataFrame(np.rot90([counts,limit_tests], k=-1), columns=['full month count','limit'])
  fc.save_df(limit_counts, os.path.join(epaAqDir, 'limit arg test for interpolate function'), 'limit arg test for interpolate function '+fc.utc_time_filename(), 'csv')


  t1 = fc.timer_restart(t1, 'write_county_month_AQI total time')



