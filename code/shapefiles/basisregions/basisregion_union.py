
import json
import copy
from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import pathlib
import os
import geojson


feature = {
  "type": "Feature",
  "properties": {
    "id": "-"
  },
  "geometry": {
    "type": "MultiPolygon",
    "coordinates": []
  }
}

featurecollection = {
  "type": "FeatureCollection",
  "features": []
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
  combined = copy.deepcopy(featurecollection)


  regionsDir = str(pathlib.Path(__file__).parent.absolute()) + '/'

  regions = folder_filenames( os.listdir(regionsDir) )
  # codes = [f.split(' ')[0] for f in regions]
  # print(codes)


  for r in regions:
    name = r.split(' ')[0]

    rDir = regionsDir + r + '/'
    countries = geojson_filenames( os.listdir(rDir) )
    # if len(countries) == 1:
    #   continue

    print(r)

    cDict = dict()

    for c in countries:
      with open(rDir + c, 'r') as f:
        contents = json.load(f)
        cDict[c] = shape(contents['features'][0]['geometry'])
        # print(len(cDict[c]))
        # print(c)

    
    temp = copy.deepcopy(feature)
    single = copy.deepcopy(featurecollection)
    
    temp['geometry'] = combineBorders([cDict[item] for item in cDict.keys()])
    temp['properties']['id'] = name

    combined['features'].append(temp)
    single['features'].append(temp)

    # print('---' + str(len(union)))


    with open(regionsDir + 'results/' +  name + '.geo.json', 'w') as o:
      geojson.dump(single, o)

  with open(regionsDir + 'GFED_basis_regions' + '.geo.json', 'w') as o:
    geojson.dump(combined, o)



