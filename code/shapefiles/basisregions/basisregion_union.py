
import json
import copy
from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import pathlib
import os
import geojson

temp = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        # "ADMIN": "-",
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

def folder_filenames(filenames):
  ret = []
  for filename in filenames:
    if ' - ' in filename:
      ret.append(filename)
  return ret


def combineBorders(*geoms):
  polys = [GeometryCollection(g) for g in geoms]
  bufs = [p.buffer(0) for p in polys]
  newShape = unary_union([b for b in bufs])
  return newShape


# BOAS - Boreal Asia - Russia, excluding area south of 55 N between the Ukraine and Kazakhstan



if __name__ == "__main__":
  regionsDir = str(pathlib.Path(__file__).parent.absolute()) + '/'

  regions = folder_filenames( os.listdir(regionsDir) )
  # codes = [f.split(' ')[0] for f in regions]
  # print(codes)


  for r in regions:
    
    rDir = regionsDir + r + '/'
    countries = geojson_filenames( os.listdir(rDir) )
    if len(countries) == 1:
      continue

    print(r)

    cDict = dict()

    for c in countries:
      with open(rDir + c, 'r') as f:
        contents = json.load(f)
        cDict[c] = shape(contents['features'][0]['geometry'])
        # print(len(cDict[c]))
        # print(c)

    
    temp2 = copy.deepcopy(temp)
    name = r.split(' ')[0]
    temp2['features'][0]['geometry'] = combineBorders([cDict[item] for item in cDict.keys()])
    temp2['features'][0]['properties']['id'] = name

    # print('---' + str(len(union)))

    with open(regionsDir + name + '.geo.json', 'w') as o:
      geojson.dump(temp2, o)



