

import pandas as pd
import numpy as np

# from time import time

import os
import pathlib
import ast

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import importlib
fc = importlib.import_module('functions')


def plotxyz(df, x_, y_, z_):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.set_title(''.join(['x:',x_,'  y:',y_]))


  x, y, z = df[x_], df[y_], df[z_]
  # ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='black', linewidth=0.5, alpha=0.7)
  ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5, alpha=0.2)
  
  # max point
  dfmax = df.loc[df[z_] == max(df[z_])]
  x, y, z = dfmax[x_], dfmax[y_], dfmax[z_]
  ax.scatter(x, y, z, color='red', linewidth=0.5, alpha=1)
  

  plt.show()


def main():

  t0 = fc.timer_start()
  t1 = t0

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())

  outputDir = os.path.join(pPath, 'random_forest-outfiles')

  df = fc.read_df(os.path.join(outputDir, 'final param tuning'),
                  'maxsamples minleaf minsplit', 'csv')
  # print(df)

  df = df[['param_max_samples', 'param_min_samples_leaf',
            'param_min_samples_split', 'mean_test_r2', 'test_set_r2']]
  # print(df)
  
  print(df.loc[df['mean_test_r2'] == max(df['mean_test_r2'])])
  print(df.loc[df['test_set_r2'] == max(df['test_set_r2'])])


  plotxyz(df, 'param_max_samples', 'param_min_samples_leaf', 'mean_test_r2')
  plotxyz(df, 'param_max_samples', 'param_min_samples_split', 'mean_test_r2')
  plotxyz(df, 'param_min_samples_leaf', 'param_min_samples_split', 'mean_test_r2')

  plotxyz(df, 'param_max_samples', 'param_min_samples_leaf', 'test_set_r2')
  plotxyz(df, 'param_max_samples', 'param_min_samples_split', 'test_set_r2')
  plotxyz(df, 'param_min_samples_leaf', 'param_min_samples_split', 'test_set_r2')




if __name__ == '__main__':
  main()
