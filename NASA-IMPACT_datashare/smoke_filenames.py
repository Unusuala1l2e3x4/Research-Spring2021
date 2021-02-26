# smoke_labeled/
#      |> time-<timestamp>-loc-<west_south_east_north>.bmp
#      |> time-<timestamp>-loc-<west_south_east_north>.tif


# coordinates (N, W) = top left of image

# thomas/rye/creek fire: time-20172822115414-loc--120.0_32.7_-117.0_34.3

# Scatter Plots on Maps 
  # https://coderzcolumn.com/tutorials/data-science/plotting-static-maps-with-geopandas-working-with-geospatial-data#4
  # https://geopandas.org/



import geopandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm

import os
import pathlib



# import warnings

# warnings.filterwarnings('ignore')

# %matplotlib inline

# geopandas.datasets.available
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
# print("Geometry Column Name : ", world.geometry.name)
# print("Dataset Size : ", world.shape)
# world.head()

# print(list(world.head()))






if __name__ == "__main__":

  current_dir = str(pathlib.Path(__file__).parent.absolute())

  files = os.listdir(current_dir + '\smoke-labeled')
  # https://careerkarma.com/blog/python-list-files-in-directory/

  files = set(files)
  files2 = set()

  for f in files:
    files2.add(f[:-4])

  df = pd.DataFrame(None, columns = ['N', 'W', 'timestamp'])

  # time-20172822115414-loc--120.0_32.7_-117.0_34.3
  # time-<timestamp>-loc-<west_south_east_north>
  for f in sorted(files2):
    parsed = f.split('c-')
    coords = parsed[1].split('_')
    # print(coords)
    ts = parsed[0].split('-')
    df = df.append({'N':float(coords[3]),'W':float(coords[0]),'timestamp':int(ts[1])}, ignore_index=True)

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html

  print(df)

  # plot image onto map?
  # https://matplotlib.org/tutorials/introductory/images.html
  # https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html
  # https://matplotlib.org/3.3.3/tutorials/intermediate/imshow_extent.html


  color_buf = 1e8
  color_min = min(df.timestamp) - color_buf
  color_max = max(df.timestamp) + color_buf
  # print(color_min)
  # print(color_max)
  norm = cl.Normalize(vmin=color_min, vmax=color_max, clip=True)

  mapper = cm.ScalarMappable(norm=norm, cmap='brg')  # cmap color range
  # https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html


  df['color'] = [mapper.to_rgba(v) for v in df.timestamp]

  # for c in range(len(df.color)):
  #   print(df.timestamp[c],' ',df.color[c])

  with plt.style.context(("seaborn", "ggplot")):
    world.plot(figsize=(18,10),
                color="white",
                edgecolor = "grey")

    plt.scatter(df.W, df.N, s=2, c=df.color, alpha=0.3, marker='s')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # plt.xlim((-135,-5))
    # plt.ylim((15,70))
    plt.title("smoke-labeled")
    plt.show()
  

  # df.to_csv(current_dir + '\smoke_N_W_timestamp.csv', index=False)