
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





areas = {'ESP-EURO':copy.deepcopy(temp)}

areas['ESP-EURO']['features'][0]['properties']['ADMIN'] = 'Spain mainland'

areas['ESP-EURO']['features'][0]['properties']['id'] = 'ESP-EURO'




dir = str(pathlib.Path(__file__).parent.absolute()) + '/'
polygons = []


with open(dir + 'ESP.geo.json', 'r') as f:
  mainfile = json.load(f)
for polygon in mainfile['features'][0]['geometry']['coordinates']:
  polygons.append(polygon)


loc = None


for polygon in polygons:
  firstpoint = polygon[0][0]
  lon = firstpoint[0]
  lat = firstpoint[1]

  if lat > 32:
    loc = 'ESP'
  else:
    continue
  areas[loc + '-EURO']['features'][0]['geometry']['coordinates'].append(polygon)



for area in areas:
  name = areas[area]['features'][0]['properties']['id']
  with open(dir + name + '.geo.json', 'w') as outfile:
    json.dump(areas[area], outfile)


