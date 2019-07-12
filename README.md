# CRADLE


## Installation
You can install CRADLE either with using pip or git repository.
1) Using pip
```
pip install cradle
```
recommend to install the newest version.

2) Using git repository
```
git clone https://github.com/Young-Sook/CRADLE.git
python setup.py install
```

## Dependencies
CRADLE requires
```
- numpy (>= 1.14.3)
- argparse (>= 1.1)
- py2bit (>= 0.3.0)
- pyBigWig (>= 0.3.11)
- statsmodels (>= 0.8.0)
- scipy (>= 1.0.1)
- matplotlib (>= 1.5.3)
- h5py (>= 2.6.0)
- Cython (>= 0.24.1)
```
Those dependencies will be automatically installed when you install CRADLE either with pip or git repository.


## Commands
```
cradle {correctBias, correctBias_stored, callPeak}
```

### 1) correctBias
This command corrects technical biases (shear, PCR, mappability, G-quadruplex) from read counts. Not recommended to run this command when you have large regions (see 'correctBias_stored' below). 
* Required Arguments
  -  -ctrlbw <br /> 
      Ctrl bigwig files. Un-noramlized files are recommended. Each file name should be spaced
  -  -expbw <br />
      Experimental bigwig files. Un-noramlized files are recommended. Each file name should be spaced.
  -  -l <br />
      Fragment length
  -  -r <br /> 
      Text file that shows regions of analysis. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. ex) chr1\t100\t3000
  -  -biasType <br /> 
      Type of biases you want to correct among 'shear', 'pcr', 'map', 'gquad'. If you want to correct 'shear' and 'pcr' bias, you should type -biasType shear pcr. If you type map, -mapFile and -kmer are required. If you type gquad, -gquadFile is required
  -  -faFile <br /> 
       .2bit files. You can download .2bit files in UCSC genome browser. <br/> <br/> 
* Optional Arguments <br />
   !! Warning !! Some optional arguments are required depending on parameters in required arguments. <br />
  -  -binSize <br /> 
      The size of bin (bp) for correction. If you put '1', it means you want to correct read counts in single-bp resolution. (default=1) 
  -  -mi <br /> 
      The minimum number of fragments. Positions that have less fragments than this value are filtered out. (default=10)
  -  -mapFile <br /> 
      Mappability file in bigwig format. Required when 'map' is in '-biasType'. See 'Reference' if you want to download human mappability files (36mer, 50mer, 100mer for hg19 and hg38). 
  -  -kmer <br /> 
      The length of sequencing reads. If you have paired-end sequencing with 50mer from each end, type '50'. Required when 'map' is in '-biasType'
  -  -gquadFile <br /> 
      Gqaudruplex files in bigwig format. Multiple files are allowed. Required when 'gquad' is in '-biasType'. 
      See 'Reference' if you want to download human Gqaudruplex files (hg19 and hg38). 
  -  -gquadMax <br /> 
      The maximum gquad score. This is used to normalize Gquad score. default=78.6
  -  -o <br /> 
      Output directory. All corrected bigwig files will be stored here. If the directory doesn't exist, cradle will make the directory. 
  -  -p <br /> 
      The number of cpus. default=(available cpus)/2
  -  -bl <br /> 
      Text file that shows regions you want to filter out. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. ex) chr1\t1\t100

Example of running correctBias: 
```
cradle correctBias -ctrlbw ctrl1.bw ctrl2.bw ctrl3.bw
                   -expbw exp1.bw exp2.bw exp3.bw
                   -l 500 
                   -r /data/YoungSook/target_region.bed
                   -biasType shear pcr map gquad
                   -faFile /data/YoungSook/hg38.2bit
                   -kmer 50
                   -o /data/YoungSook/CRADLE_result
                   -bl /data/YoungSook/blacklist_regions.bed
```

### 2) correctBias_stored
This command corrects technical biases (shear, PCR, mappability, G-quadruplex) from read counts when there are precomputed covariate files(.hdf). This command is much faster than running correctBias. Using 'correctBias_stored' is highly recommended when you have large regions, especially if you have whole genome data. You can download covariate files in synpase ().

* Required Arguments

* Optional Arguments

### 3) callPeak
* Required Arguments

* Optional Arguments


## Tips on running CRADLE


## References
* DNAShape 
* Mappability
* G-quadruplex sturcture
