#!/data/reddylab/software/miniconda2/envs/YoungSook/bin/python2.7

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'CalculateOnebp Part',
  ext_modules = cythonize("calculateOnebp.pyx"),
)


