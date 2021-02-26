
import json
import copy
from shapely.geometry import shape, GeometryCollection, Point
import pathlib
import os

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

areas = dict()
regions = ['MIDE','NHAF']


def geojson_filenames(filenames):
  ret = []
  for filename in filenames:
    if '.geo' in filename:
      ret.append(filename)
  return ret


# BOAS - Boreal Asia - Russia, excluding area south of 55 N between the Ukraine and Kazakhstan



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
      areas[loc + '-' + regions[0]]['features'][0]['geometry']['coordinates'].append(north)
    if south[0]:
      areas[loc + '-' + regions[1]]['features'][0]['geometry']['coordinates'].append(south)




if __name__ == "__main__":
  dir = str(pathlib.Path(__file__).parent.absolute()) + '/'

  filenames = os.listdir(dir)
  filenames = geojson_filenames(filenames)
  codes = [f.split('.')[0] for f in filenames]

  


  for c in codes:
    with open(dir + c + '.geo.json', 'r') as f:
      contents = json.load(f)
      admin = contents['features'][0]['properties']['ADMIN']
      for r in regions:
        a = c + '-' + r
        areas[a] = copy.deepcopy(temp)
        areas[a]['features'][0]['properties']['ADMIN'] = admin + '-' + r
        areas[a]['features'][0]['properties']['id'] = a


  for c in codes:
    split_north_south(c, 23.5)


  for area in areas:
    name = areas[area]['features'][0]['properties']['id']
    with open(dir + 'result/' + name + '.geo.json', 'w') as outfile:
      json.dump(areas[area], outfile)


