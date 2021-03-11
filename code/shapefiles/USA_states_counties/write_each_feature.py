import json
import copy
import pathlib
import os

temp = {"type":"FeatureCollection","features":[]}

dir = str(pathlib.Path(__file__).parent.absolute())

dirStates = os.path.join(dir, 'us_states')

# with open(os.path.join(dir, 'cb_2018_us_state_500k.geojson'), 'r') as f:
with open(os.path.join(dir, 'cb_2018_us_state_500k.geojson'), 'r') as f:
  mainfile = json.load(f)

i = 0

# 5 to 259
# 255 total
# 255 - 17 = 238 with ISO_A3

name = None

for feature in mainfile['features']:

  name = feature['properties']['STUSPS'] + '-' + feature['properties']['NAME']
  
  data = copy.deepcopy(temp)
  data['features'].append(feature)

  with open(os.path.join(dirStates, name + '.geojson'), 'w') as outfile:
    json.dump(data, outfile)
  i += 1


print(i)

    