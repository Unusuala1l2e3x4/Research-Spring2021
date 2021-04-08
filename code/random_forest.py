from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


import os, pathlib, re, json, sys, copy

import importlib
fc = importlib.import_module('functions')



if __name__ == '__main__':
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())

  outputDir = os.path.join(pPath, 'random_forest-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets')
  usCensusDir = os.path.join(ppPath, 'US Census Bureau', 'population')
  nClimDivDir = os.path.join(ppPath, 'nClimDiv data')


  countyMapFile = 'cb_2019_us_county_500k'

  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title
  countySupEstTitle = countyTitle + ', suppressed estimates'

  ext = 'hdf5'

  # startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]
  startYYYYMM, endYYYYMM = '200001', '201812'

  deathsData = fc.makeCountyFileGEOIDs(fc.read_df(cdcWonderDir, countySupEstTitle, ext))
  dates = sorted(i for i in deathsData if i != 'GEOID' and i >= startYYYYMM and i <= endYYYYMM)
  # print(dates)

  shapeData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  shapeData = fc.clean_states_reset_index(shapeData)
  shapeData = fc.county_changes_deaths_reset_index(shapeData)
  print(list(shapeData.GEOID) == list(deathsData.GEOID)) # True

  popData = fc.read_df(usCensusDir, 'TENA_county_pop_1999_2019', ext)
  print(list(shapeData.GEOID) == list(popData.GEOID)) # True

  precipData = fc.read_df(nClimDivDir, 'climdiv-pcpncy-v1.0', ext)
  print(list(shapeData.GEOID) == list(precipData.GEOID))

  tempData = fc.read_df(nClimDivDir, 'climdiv-tmpccy-v1.0', ext)
  print(list(shapeData.GEOID) == list(tempData.GEOID))

  deathsSum = np.sum(deathsData.loc[:, dates], axis=1)
  popSum = np.sum(popData.loc[:, dates], axis=1)
  unit = 'monthly_death_rate'
  shapeData[unit] = deathsSum / popSum




  # print(shapeData)
  # print(deathsData)
  # print(popData)
  # print(precipData)
  # print(tempData)




  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor

  
  
  # print("Hello!")
  # df = pd.read_csv(inpath)
  # count_columns = [column for column in df.columns if '_count' in column]

  # X = df[count_columns]
  # y = df['mean_property_value']
  # print(X)
  # print(y)




  # clf = RandomForestRegressor(random_state=0)
  # clf.fit(X,y)
  # print("World!")


  # importances = pd.DataFrame(dict(name=count_columns, importance=clf.feature_importances_)).sort_values('importance', ascending=False).head(10)
  # print(importances)
  # plt.bar(importances['name'], importances['importance'])
  # plt.xticks(rotation=10)
  # plt.xlabel("Species")
  # plt.ylabel("Gini Importance")
  # plt.show()





