import os
import glob
import functools
import operator

if __name__ == "__main__":
  extensions = ['vert', 'frag', 'geom', 'tesc', 'tese', 'comp']
  filenames = functools.reduce(operator.add, [glob.glob(f'*.{extension}') for extension in extensions])

  for filename in filenames:
    print(f'compiling {filename}:')
    if os.system(f'glslc.exe {filename} -o {filename}.spv') != 0:
      # delete previously compiled spv file
      if os.path.exists(f'{filename}.spv'):
        os.remove(f'{filename}.spv')
