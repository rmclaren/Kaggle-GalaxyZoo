
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension('source', ['cython_functions.pyx'],
                        include_dirs=[np.get_include()])]

setup(
    name='GalaxyZoo',
    ext_modules=cythonize(extensions, gdb_debug=True))
