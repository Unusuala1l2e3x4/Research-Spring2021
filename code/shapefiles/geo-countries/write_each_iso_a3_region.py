import json
import copy
import pathlib

temp = {"type":"FeatureCollection","features":[]}

dir = str(pathlib.Path(__file__).parent.absolute()) + '/'

dirCountries = str(pathlib.Path(__file__).parent.absolute()) + '/geo-countries-countries/'

with open(dir + 'geo-countries.json', 'r') as f:
  mainfile = json.load(f)

i = 0

# 5 to 259
# 255 total
# 255 - 17 = 238 with ISO_A3

name = None

for feature in mainfile['features']:
  if feature['properties']['ISO_A3'] == '-99' or feature['properties']['ISO_A2'] == '-':
    name = feature['properties']['ADMIN'].replace(' ', '_')
  else:
    name = feature['properties']['ISO_A3']
  
  data = copy.deepcopy(temp)
  data['features'].append(feature)

  with open(dirCountries + name + '.geo.json', 'w') as outfile:
    json.dump(data, outfile)
  i += 1


print(i)

    