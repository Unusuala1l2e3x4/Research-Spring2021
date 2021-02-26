
import json
import copy
from shapely.geometry import shape, GeometryCollection, Point
import pathlib




temp = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "ADMIN": "-",
        "id": "-"
      },
      "geometry": {
        "type": "MultiPolygon",
        "coordinates": [
          # [
          #   [
          #     [
          #       -171.737,
          #       25.7944
          #     ],
          #     [
          #       -171.7206,
          #       25.7616
          #     ],
          #     [
          #       -171.748,
          #       25.7506
          #     ]
          #   ]
          # ],
        ]
      }
    }
  ]
}


# [[['lon', 'lat'],['lon', 'lat']]] # each enclosed polygon in the area


areas = {'TENA':copy.deepcopy(temp), 'AK':copy.deepcopy(temp), 'HI':copy.deepcopy(temp)}

areas['TENA']['features'][0]['properties']['ADMIN'] = 'Temperate North America'
areas['AK']['features'][0]['properties']['ADMIN'] = 'Alaska'
areas['HI']['features'][0]['properties']['ADMIN'] = 'Hawaii'
areas['TENA']['features'][0]['properties']['id'] = 'TENA'
areas['AK']['features'][0]['properties']['id'] = 'AK'
areas['HI']['features'][0]['properties']['id'] = 'HI'




# areas['HI']['features'][0]['geometry']['coordinates'].append()

# HI: lat < 40, lon < -140
# AK: lat > 50


dir = str(pathlib.Path(__file__).parent.absolute()) + '/'


with open(dir + 'USA.geo.json', 'r') as f:
  mainfile = json.load(f)


loc = None

for polygon in mainfile['features'][0]['geometry']['coordinates']:
  firstpoint = polygon[0][0]
  lon = firstpoint[0]
  lat = firstpoint[1]
  if lat < 40 and lon < -140:
    loc = 'HI'
  elif lat > 50:
    loc = 'AK'
  else:
    loc = 'TENA'
  areas[loc]['features'][0]['geometry']['coordinates'].append(polygon)


# print(areas)

for area in areas:
  # print(areas[area])
  name = areas[area]['features'][0]['properties']['id']
  with open(dir + name + '.geo.json', 'w') as outfile:
    json.dump(areas[area], outfile)


