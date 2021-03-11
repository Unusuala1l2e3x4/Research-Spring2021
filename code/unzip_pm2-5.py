import os
import pathlib
import zipfile

def zip_filenames(filenames):
  ret = []
  for filename in filenames:
    if '.zip' in filename:
      ret.append(filename)
  return ret

if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')

  filenames = os.listdir(pmDir)
  filenames = sorted(zip_filenames(filenames)) # same as in gfedDir_timesArea

  # print(len(filenames))

  for filename in filenames:
    z = zipfile.ZipFile(os.path.join(pmDir, 'data_0.01', filename), 'r')
    tiffilename = sorted(z.namelist())[1]
    z.extract(tiffilename, path=os.path.join(pmDir, 'data_0.01'))
    





