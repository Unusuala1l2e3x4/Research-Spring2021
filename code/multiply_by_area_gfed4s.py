import h5py
import numpy as np
import os
import pathlib

import importlib
fc = importlib.import_module('functions')


def fslash(a, b):
  return a + '/' + b

# https://docs.h5py.org/en/latest/high/group.html#h5py.Group.visititems
def visit_timesArea(name, node):
  if isinstance(node, h5py.Dataset):
    return
  else:
    namesplitted = name.split('/')
    if len(namesplitted) in [1, 3]:
      return
    layer = namesplitted[-2]
    for key in node.keys():
      if layer == 'emissions' and key not in ['C', 'DM'] or layer == 'burned_area' and key == 'source':
        continue
      path = fslash(name, key)
      newdata = np.multiply(np.matrix(fd[path]), grid_cell_area)
      del fd[path]
      fd.create_dataset(path, data=newdata)



if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  dir = os.path.join(ppPath, 'GFED4s_timesArea')

  filenames = os.listdir(dir)
  filenames = sorted(fc.gfed_filenames(filenames))


  grid_cell_area = h5py.File(os.path.join(dir, filenames[10]), 'r')['ancill/grid_cell_area']
  grid_cell_area = np.matrix(grid_cell_area)
  
  for filename in filenames:
    print(filename)
    fd = h5py.File(os.path.join(dir, filename), 'r+')
    fd.visititems(visit_timesArea)
    fd.close()





