
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





areas = {'FRA-SHAF':copy.deepcopy(temp)}

areas['FRA-SHAF']['features'][0]['properties']['ADMIN'] = 'France SHAF'


areas['FRA-SHAF']['features'][0]['properties']['id'] = 'FRA-SHAF'




dir = str(pathlib.Path(__file__).parent.absolute()) + '/'
polygons = []


with open(dir + 'FRA.geo.json', 'r') as f:
  mainfile = json.load(f)
for polygon in mainfile['features'][0]['geometry']['coordinates']:
  polygons.append(polygon)

loc = None


for polygon in polygons:
  firstpoint = polygon[0][0]
  lon = firstpoint[0]
  lat = firstpoint[1]

  if lat < -11:
    loc = 'FRA-SHAF'
  else:
    continue
  areas[loc]['features'][0]['geometry']['coordinates'].append(polygon)


for area in areas:
  name = areas[area]['features'][0]['properties']['id']
  with open(dir + name + '.geo.json', 'w') as outfile:
    json.dump(areas[area], outfile)


