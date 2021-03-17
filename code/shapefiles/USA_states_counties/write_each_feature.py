import json
import copy
import pathlib
import os


def create_dir_if_exist(dir):
  if not os.path.isdir(dir):
    os.mkdir(dir)

# def create_dirs_if_exist(dirs):
#   if isinstance(dirs, list):
#     for dir in dirs:
#       create_dir_if_exist(dir)
#   else:
#     create_dir_if_exist(dirs)


temp = {"type":"FeatureCollection","features":[]}

dir = str(pathlib.Path(__file__).parent.absolute())

dirStates = os.path.join(dir, 'us_states')
dirCounties = os.path.join(dir, 'us_states_counties')
dirCountiesFolders = os.path.join(dir, 'us_counties_by_state')

with open(os.path.join(dir, 'cb_2019_us_state_500k.geojson'), 'r') as f:
  stateItems = json.load(f)['features']
with open(os.path.join(dir, 'cb_2019_us_county_500k.geojson'), 'r') as f:
  countyItems = json.load(f)['features']



stateCount = 0
countyCount = 0


for state in stateItems:
  stateCount += 1
  stateName = state['properties']['GEOID'] + '-' + state['properties']['STUSPS'] + '-' + state['properties']['NAME']

  stateData = copy.deepcopy(temp)
  countyData = copy.deepcopy(temp)
  
  stateData['features'].append(state)

  dirCountiesFolder = os.path.join(dirCountiesFolders, stateName)

  create_dir_if_exist(dirCountiesFolder)

  while countyCount < len(countyItems) and countyItems[countyCount]['properties']['STATEFP'] == state['properties']['STATEFP']:
    countyName = countyItems[countyCount]['properties']['GEOID'] + '-' + state['properties']['STUSPS'] + '-' + countyItems[countyCount]['properties']['NAME']
    countySingleData = copy.deepcopy(temp)

    countyData['features'].append(countyItems[countyCount])
    countySingleData['features'].append(countyItems[countyCount])

    with open(os.path.join(dirCountiesFolder, countyName + '.geojson'), 'w') as outfile:
      json.dump(countySingleData, outfile)

    countyCount += 1

  with open(os.path.join(dirStates, stateName + '.geojson'), 'w') as outfile:
    json.dump(stateData, outfile)

  with open(os.path.join(dirCounties, stateName + '.geojson'), 'w') as outfile:
    json.dump(countyData, outfile)


print(stateCount)
print(countyCount)
# 56
# 3233