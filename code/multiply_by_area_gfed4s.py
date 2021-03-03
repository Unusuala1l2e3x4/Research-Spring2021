import h5py
import numpy as np

import pandas as pd

import os
import pathlib
import sys

import time





def month_str(month):
  if month < 10:
    return '0' + str(month)
  else:
    return str(month)

def gfed_filenames(filenames):
  ret = []
  for filename in filenames:
    if '.hdf5' in filename:
      ret.append(filename)
  return ret

def flatten_list(regular_list):
  return [item for sublist in regular_list for item in sublist]


def timer_start():
  return time.time()

def timer_elapsed(t0):
  return time.time() - t0

def timer_restart(t0, msg):
  # print(timer_elapsed(t0), msg)
  return timer_start()


def save_df(df, cldf, stats, dest, name):
  fd = pd.HDFStore(os.path.join(dest, name + '.hdf5'))
  # del df['color']
  del df['region']
  fd.put('data', df, format='table', data_columns=True)
  fd.put('colormap', cldf, format='table', data_columns=True)
  fd.put('stats', stats, format='table', data_columns=True)
  fd.close()




# 'DM_AGRI', 'DM_BORF', 'DM_DEFO', 'DM_PEAT', 'DM_SAVA', 'DM_TEMF'
# 'C_AGRI', 'C_BORF', 'C_DEFO', 'C_PEAT', 'C_SAVA', 'C_TEMF'


def fslash(a, b):
  return a + '/' + b



  


# dname = name.split('/')[-1]
# if dname in ['small_fire_fraction', 'lat', 'lon', 'basis_regions', 'grid_cell_area', 'source'] or 'day_' in dname or 'UTC' in dname or 'C_' in dname or 'DM_' in dname:
#   return




def visitor_func(name, node):
  
  if isinstance(node, h5py.Dataset):
    # node is a dataset
    return
  else:
    namesplitted = name.split('/')
    if len(namesplitted) in [1, 3]:
      return
    # group = namesplitted[-1]
    groupParent = namesplitted[-2]
    for key in node.keys():
      if groupParent == 'emissions' and key not in ['C', 'DM'] or groupParent == 'burned_area' and key == 'source':
        continue
      # print(key)

      path = fslash(name, key)
      # print(path)

      # newdata = np.matrix(fd[path])
      # print('\t',np.max(newdata))
      newdata = np.multiply(np.matrix(fd[path]), grid_cell_area)
      # print('\t',np.max(newdata))

      del fd[path]
      fd.create_dataset(path, data=newdata)

      # print(name)




if __name__ == "__main__":

# https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
# https://stackoverflow.com/questions/31146036/how-do-i-traverse-a-hdf5-file-using-h5py
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  dir = os.path.join(ppPath, 'GFED4s_timesArea')

  filenames = os.listdir(dir)
  filenames = sorted(gfed_filenames(filenames))


  grid_cell_area = h5py.File(os.path.join(dir, filenames[10]), 'r')['ancill/grid_cell_area']
  grid_cell_area = np.matrix(grid_cell_area)
  
  for filename in filenames[10:11]:


    print(filename)

    fd = h5py.File(os.path.join(dir, filename), 'r+')

    # print(fd)
    fd.visititems(visitor_func)
    

    # print(l)

      
    fd.close()





