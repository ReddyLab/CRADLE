import sys
import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [
	Extension(
		'CRADLE.CorrectBias.trainCovariates',
		['CRADLE/CorrectBias/trainCovariates.pyx'],
		extra_compile_args=["-fno-strict-aliasing"]),
	Extension(
		'CRADLE.CorrectBias.taskCovariates',
		['CRADLE/CorrectBias/taskCovariates.pyx'],
		extra_compile_args=["-fno-strict-aliasing"]),
	Extension(
		'CRADLE.CorrectBias.regression',
		['CRADLE/CorrectBias/regression.pyx'],
		extra_compile_args=["-fno-strict-aliasing"]),
	Extension(
		'CRADLE.CorrectBias.readCounts',
		['CRADLE/CorrectBias/readCounts.pyx'],
		extra_compile_args=["-fno-strict-aliasing"]),
	Extension(
		'CRADLE.CorrectBias.covariateUtils',
		['CRADLE/CorrectBias/covariateUtils.pyx'],
		extra_compile_args=["-fno-strict-aliasing"]),
	Extension(
		'CRADLE.CalculateCovariates.taskCovariates',
		['CRADLE/CalculateCovariates/taskCovariates.pyx'],
		extra_compile_args=["-fno-strict-aliasing"]),
	Extension(
		'CRADLE.CallPeak.calculateRC',
		['CRADLE/CallPeak/calculateRC.pyx'],
		extra_compile_args=["-fno-strict-aliasing"]),
	Extension(
		'CRADLE.correctbiasutils.cython',
		['CRADLE/correctbiasutils/cython.pyx'],
		extra_compile_args=["-fno-strict-aliasing"])
]

with open("requirements.txt", "r") as requirements_file:
	install_requirements = requirements_file.readlines()

setup(
	install_requires = install_requirements,
	ext_modules = cythonize(
		ext_modules,
		include_path = [numpy.get_include()],
		language_level = int(sys.version[:1]),
	),
	cmdclass = { 'build_ext': build_ext },
)
