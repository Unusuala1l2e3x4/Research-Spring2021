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




if __name__ == "__main__":

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  dir_in = os.path.join(ppPath, 'GFED4s')
  dir_out = os.path.join(ppPath, 'GFED4s_timesArea')


  filenames = os.listdir(dir_in)
  filenames = sorted(gfed_filenames(filenames))

  # print(filenames)

  # https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html

  for filename in filenames[10:11]:
    filenameNew = filename.split('.hdf5')[0] + '_timesArea' + '.hdf5'

    print(filenameNew)


    in_fd = h5py.File(os.path.join(dir_in, filename), 'r')
    out_fd = h5py.File(os.path.join(dir_out, filenameNew), 'w')


    for g1 in in_fd.keys():
      print(g1)
      if g1 in ['lon', 'lat']:
        out_fd.create_dataset(g1, data = in_fd[g1])
        continue
      group = out_fd.create_group(g1)
      for g2 in in_fd[g1].keys():
        print(g2)
        if g1 == 'ancill':
          group.create_dataset(g2, data = in_fd[g1 + '/' + g2])
          continue
        # for g3 in in_fd[g1 + '/' + g2].keys():
        #   print(g3)
  

    # ancill = out_fd.create_group('ancill')
    # ancill.create_dataset('basis_regions', data = in_fd['ancill/basis_regions'])
    # ancill.create_dataset('grid_cell_area', data = in_fd['ancill/grid_cell_area'])



    # biosphere = out_fd.create_group('ancill')






     
    in_fd.close()
    out_fd.close()





