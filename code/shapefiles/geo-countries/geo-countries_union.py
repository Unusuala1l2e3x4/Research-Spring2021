
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
  combined['features'].append(copy.deepcopy(feature))

  regionsDir = str(pathlib.Path(__file__).parent.absolute())

  with open(os.path.join(regionsDir, 'geo-countries.json'), 'r') as f:
    contents = json.load(f)

    combined['features'][0]['geometry'] = shape( combineBorders([shape(ft['geometry']) for ft in contents['features']]) )
  
    with open(os.path.join(regionsDir, 'geo-countries-union.json'), 'w') as o:
      geojson.dump(combined, o)



