
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
        ]
      }
    }
  ]
}





areas = {'FRA-EURO':copy.deepcopy(temp), 'NLD-EURO':copy.deepcopy(temp), 'NOR-EURO':copy.deepcopy(temp), 'PRT-EURO':copy.deepcopy(temp)}

areas['FRA-EURO']['features'][0]['properties']['ADMIN'] = 'France mainland'
areas['NLD-EURO']['features'][0]['properties']['ADMIN'] = 'Netherlands mainland'
areas['NOR-EURO']['features'][0]['properties']['ADMIN'] = 'Norway mainland'
areas['PRT-EURO']['features'][0]['properties']['ADMIN'] = 'Portugal mainland'


areas['FRA-EURO']['features'][0]['properties']['id'] = 'FRA-EURO'
areas['NLD-EURO']['features'][0]['properties']['id'] = 'NLD-EURO'
areas['NOR-EURO']['features'][0]['properties']['id'] = 'NOR-EURO'
areas['PRT-EURO']['features'][0]['properties']['id'] = 'PRT-EURO'




dir = str(pathlib.Path(__file__).parent.absolute()) + '/'
polygons = []


with open(dir + 'FRA.geo.json', 'r') as f:
  mainfile = json.load(f)
for polygon in mainfile['features'][0]['geometry']['coordinates']:
  polygons.append(polygon)

with open(dir + 'NLD.geo.json', 'r') as f:
  mainfile = json.load(f)
for polygon in mainfile['features'][0]['geometry']['coordinates']:
  polygons.append(polygon)

with open(dir + 'NOR.geo.json', 'r') as f:
  mainfile = json.load(f)
for polygon in mainfile['features'][0]['geometry']['coordinates']:
  polygons.append(polygon)

with open(dir + 'PRT.geo.json', 'r') as f:
  mainfile = json.load(f)
for polygon in mainfile['features'][0]['geometry']['coordinates']:
  polygons.append(polygon)


loc = None


for polygon in polygons:
  firstpoint = polygon[0][0]
  lon = firstpoint[0]
  lat = firstpoint[1]

  if lat < 42.4 and lat > 36 and lon > -11 and lon < -5:
    loc = 'PRT'
  elif lat > 56 and lon > 3:
    loc = 'NOR'
  elif lat > 50.73 and lon > 3.3 and lat < 54 and lon < 8:
    loc = 'NLD'
  elif lat < 51.2 and lat > 41.28 and lon > -6:
    loc = 'FRA'
  else:
    continue
  areas[loc + '-EURO']['features'][0]['geometry']['coordinates'].append(polygon)


for area in areas:
  name = areas[area]['features'][0]['properties']['id']
  with open(dir + name + '.geo.json', 'w') as outfile:
    json.dump(areas[area], outfile)


