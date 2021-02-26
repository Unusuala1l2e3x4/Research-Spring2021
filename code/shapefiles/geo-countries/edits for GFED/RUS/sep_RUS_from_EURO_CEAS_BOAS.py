
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


areas = {'RUS-EURO':copy.deepcopy(temp), 'RUS-CEAS':copy.deepcopy(temp), 'RUS-BOAS':copy.deepcopy(temp)}

areas['RUS-EURO']['features'][0]['properties']['ADMIN'] = 'Russia EURO'
areas['RUS-CEAS']['features'][0]['properties']['ADMIN'] = 'Russia CEAS'
areas['RUS-BOAS']['features'][0]['properties']['ADMIN'] = 'Russia BOAS'

areas['RUS-EURO']['features'][0]['properties']['id'] = 'RUS-EURO'
areas['RUS-CEAS']['features'][0]['properties']['id'] = 'RUS-CEAS'
areas['RUS-BOAS']['features'][0]['properties']['id'] = 'RUS-BOAS'



dir = str(pathlib.Path(__file__).parent.absolute()) + '/'
polygons = []


with open(dir + 'RUS.geo.json', 'r') as f:
  mainfile = json.load(f)
for polygon in mainfile['features'][0]['geometry']['coordinates']:
  polygons.append(polygon)


loc = None
largest_polygon = None

maxsize = 0

# find EURO portion first
# save largest polygon for mainland RUS, for CEAS/BOAS seperation
for polygon in polygons:
  size = len(polygon[0])
  if size > maxsize:
    maxsize = size
    largest_polygon = polygon

  firstpoint = polygon[0][0]
  lon = firstpoint[0]
  lat = firstpoint[1]

  if lon > 18 and lon < 24:
    loc = 'EURO'
    print(size)
  # elif :
  #   loc = 'CEAS'
  # elif :
  #   loc = 'BOAS'
  else:
    continue
  areas['RUS-' + loc]['features'][0]['geometry']['coordinates'].append(polygon)


# 213 total polygons
print(len(largest_polygon[0]))

rus_north_of_55 = [[]]
rus_south_of_55 = [[]]

for item in largest_polygon[0]:
  lon = item[0]
  lat = item[1]
  if lat < 54.9 and lon < 68.33:
    rus_south_of_55[0].append(item)
  else:
    rus_north_of_55[0].append(item)

print(len(rus_north_of_55[0]))
print(len(rus_south_of_55[0]))

print(len(rus_south_of_55[0]) + len(rus_north_of_55[0]))

# do CEAS
loc = 'CEAS'
areas['RUS-' + loc]['features'][0]['geometry']['coordinates'].append(rus_south_of_55)

# do BOAS
loc = 'BOAS'
areas['RUS-' + loc]['features'][0]['geometry']['coordinates'].append(rus_north_of_55)


for polygon in polygons:
  size = len(polygon[0])
  if size == maxsize: # skip mainland portion
    continue

  firstpoint = polygon[0][0]
  lon = firstpoint[0]
  lat = firstpoint[1]

  if lon > 18 and lon < 24:
    continue
  areas['RUS-' + loc]['features'][0]['geometry']['coordinates'].append(polygon)





for area in areas:
  name = areas[area]['features'][0]['properties']['id']
  with open(dir + name + '.geo.json', 'w') as outfile:
    json.dump(areas[area], outfile)


