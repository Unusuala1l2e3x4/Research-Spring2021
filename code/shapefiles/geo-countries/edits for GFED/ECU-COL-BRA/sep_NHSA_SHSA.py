
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


# BOAS - Boreal Asia - Russia, excluding area south of 55 N between the Ukraine and Kazakhstan


areas = {'ECU-NHSA':copy.deepcopy(temp), 'ECU-SHSA':copy.deepcopy(temp), 'COL-NHSA':copy.deepcopy(temp), 'COL-SHSA':copy.deepcopy(temp),  'BRA-NHSA':copy.deepcopy(temp), 'BRA-SHSA':copy.deepcopy(temp)}




areas['ECU-NHSA']['features'][0]['properties']['ADMIN'] = 'Ecuador-NHSA'
areas['COL-NHSA']['features'][0]['properties']['ADMIN'] = 'Colombia-NHSA'
areas['BRA-NHSA']['features'][0]['properties']['ADMIN'] = 'Brazil-NHSA'

areas['ECU-SHSA']['features'][0]['properties']['ADMIN'] = 'Ecuador-SHSA'
areas['COL-SHSA']['features'][0]['properties']['ADMIN'] = 'Colombia-SHSA'
areas['BRA-SHSA']['features'][0]['properties']['ADMIN'] = 'Brazil-SHSA'


areas['ECU-NHSA']['features'][0]['properties']['id'] = 'ECU-NHSA'
areas['COL-NHSA']['features'][0]['properties']['id'] = 'COL-NHSA'
areas['BRA-NHSA']['features'][0]['properties']['id'] = 'BRA-NHSA'

areas['ECU-SHSA']['features'][0]['properties']['id'] = 'ECU-SHSA'
areas['COL-SHSA']['features'][0]['properties']['id'] = 'COL-SHSA'
areas['BRA-SHSA']['features'][0]['properties']['id'] = 'BRA-SHSA'



dir = str(pathlib.Path(__file__).parent.absolute()) + '/'



def split_north_south(loc, latitude):
  polygons = []

  with open(dir + loc + '.geo.json', 'r') as f:
    mainfile = json.load(f)
  for polygon in mainfile['features'][0]['geometry']['coordinates']:
    polygons.append(polygon)

  for polygon in polygons:
    north = [[]]
    south = [[]]
    for item in polygon[0]:
      lon = item[0]
      lat = item[1]
      if lat < latitude:
        south[0].append(item)
      else:
        north[0].append(item)
    if north[0]:
      areas[loc + '-NHSA']['features'][0]['geometry']['coordinates'].append(north)
    if south[0]:
      areas[loc + '-SHSA']['features'][0]['geometry']['coordinates'].append(south)






if __name__ == "__main__":
  split_north_south('ECU', 0)
  split_north_south('COL', 0)
  split_north_south('BRA', 0)












  for area in areas:
    name = areas[area]['features'][0]['properties']['id']
    with open(dir + name + '.geo.json', 'w') as outfile:
      json.dump(areas[area], outfile)


