

from setuptools import setup, Extension
import subprocess 
import sys

def main():

	#with open("README.md", "r") as fh:
	#	long_description = fh.read()

	subprocess.call([sys.executable, '-m', 'pip', 'install', '{}>={}'.format('numpy', '1.14.3')])
	subprocess.call([sys.executable, '-m', 'pip', 'install', '{}>={}'.format('argparse', '1.1')])
	subprocess.call([sys.executable, '-m', 'pip', 'install', '{}>={}'.format('py2bit', '0.3.0')]) 
	subprocess.call([sys.executable, '-m', 'pip', 'install', '{}>={}'.format('pyBigWig', '0.3.11')])
	subprocess.call([sys.executable, '-m', 'pip', 'install', '{}>={}'.format('statsmodels', '0.8.0')])
	subprocess.call([sys.executable, '-m', 'pip', 'install', '{}>={}'.format('scipy', '1.0.1')])	
	subprocess.call([sys.executable, '-m', 'pip', 'install', '{}>={}'.format('matplotlib', '1.5.3')]) 
	subprocess.call([sys.executable, '-m', 'pip', 'install', '{}>={}'.format('h5py', '2.6.0')])

	import numpy

	ext_modules = [
		Extension('CRADLE.CorrectBias.calculateOnebp', ['CRADLE/CorrectBias/calculateOnebp.c'], include_dirs=[numpy.get_include()], extra_compile_args=["-fno-strict-aliasing"]),
		Extension('CRADLE.CorrectBiasStored.calculateOnebp', ['CRADLE/CorrectBiasStored/calculateOnebp.c'], include_dirs=[numpy.get_include()], extra_compile_args=["-fno-strict-aliasing"]),
		Extension('CRADLE.CallPeak.calculateRC', ['CRADLE/CallPeak/calculateRC.c'], include_dirs=[numpy.get_include()], extra_compile_args=["-fno-strict-aliasing"])
	]

	setup(name = "CRADLE",
	      version = "0.1.4",
	      description = "Correct Read Counts and Analysis of Differently Expressed Regions",
	      #long_description = long_description,
	      author = "Young-Sook Kim",
	      author_email = "kys91240@gmail.com",
	      url = "https://github.com/Young-Sook/CRADLE",
	      packages = ['CRADLE', 'CRADLE.CorrectBias', 'CRADLE.CorrectBiasStored', 'CRADLE.CallPeak'], # package names
	      package_dir = {'CRADLE': 'CRADLE'}, # It calls ./CRADLE/CorrectBias/__init__.py
	      scripts = ["bin/cradle"], # python scource code, intended to be started from the command line.
	      ext_modules = ext_modules,
	      classifiers = [
			"Programming Language :: Python :: 2.7",
			"Programming Language :: Python :: 3.6",
			"License :: OSI Approved :: MIT License"
	      ]

	      ### If meta data is needed -> add classifier = [ 'Development status:', 'Operating system'] 
	)



if __name__ == '__main__':
	main()



