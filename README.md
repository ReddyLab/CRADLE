# CRADLE
CRADLE (Correcting Read counts and Analysis of DifferentiaLly Expressed regions) is a package that was developed to analyze STARR-seq data. CRADLE removes technical biases from sonication, PCR, mappability and G-quadruplex sturcture, and generates bigwig files with corrected read counts. CRADLE then uses those corrected read counts and detects both activated and repressed enhancers. CRADLE will help find enhancers with better accuracy and credibility.

## DISCLAIMER
CRADLE callPeak subcommand is designed to call peaks using the read counts 'CORRECTED' by either correctBias or correctBias_stored subcommand in CRADLE. CRADLE callPeak subcommand assumes read counts follow a gaussian distribution, so it might not be ideal to use for uncorrected read counts.


## Installation
You can install CRADLE either with using pip or git repository.
1) Using pip
```
pip install cradle
```
Recommend to install the newest version.

2) Using git repository
```
git clone https://github.com/ReddyLab/CRADLE.git
make install
```

or, alternatively

```
git clone https://github.com/ReddyLab/CRADLE.git
pip install build # If the 'build' package isn't already installed
python -m build # Build cradle
pip install dist/*.whl # Install cradle
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
These dependencies will be automatically installed when you install CRADLE either with pip or git repository.


## Commands
```
cradle <correctBias | correctBias_stored | callPeak | normalize> [options]
```

### 1) correctBias
This command corrects technical biases (shear, PCR, mappability, G-quadruplex) from read counts. This command takes bigwig files as input and outputs bigwig files with corrected read counts. Not recommended to run this command when you have large regions (see 'correctBias_stored' below).  <br/> <br/>

Example of running correctBias:
```
cradle correctBias -ctrlbw ctrl1.bw ctrl2.bw ctrl3.bw
                   -expbw exp1.bw exp2.bw exp3.bw
                   -l 500
                   -r /data/YoungSook/target_region.bed
                   -biasType shear pcr map gquad
                   -genome /data/YoungSook/hg38.2bit
                   -kmer 50
                   -o /data/YoungSook/CRADLE_result
                   -bl /data/YoungSook/blacklist_regions.bed
```

* Required Arguments
  -  -ctrlbw <br />
      Ctrl bigwig files. Un-normalized files are recommended. Each file name should be spaced
  -  -expbw <br />
      Experimental bigwig files. Un-normalized files are recommended. Each file name should be spaced.
  -  -l <br />
      Fragment length
  -  -r <br />
      Text file that shows regions of analysis. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. ex) chr1\t100\t3000
  -  -biasType <br />
      Type of biases you want to correct among 'shear', 'pcr', 'map', 'gquad'. If you want to correct 'shear' and 'pcr' bias, you should type -biasType shear pcr. If you type map, -mapFile and -kmer are required. If you type gquad, -gquadFile is required
  -  -genome <br />
       The human genome sequence, in .2bit format. For information on downloading the genome, see [https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/) §§ "Files" and "How to Download"<br/> <br/>
  -  -faFile <br/>
      The same as `-genome`. This argument is deprecated and may be removed in a future release. Please use `-genome` instead.<br/> <br/>
* Optional Arguments <br />
   !! Warning !! Some optional arguments are required depending on what you put in required arguments. <br />
  -  -binSize <br />
      The size of bin (bp) for correction. If you put '1', it means you want to correct read counts in single-bp resolution. (default=1)
  -  -mi <br />
      The minimum number of fragments. Positions that have less fragments than this value are filtered out. default=the number of samples
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
      Output directory. All corrected bigwig files will be stored here. If the directory doesn't exist, cradle will make the directory. default=CRADLE_correctionResult.
  -  -p <br />
      The number of cpus. default=(available cpus)/2
  -  -bl <br />
      Text file that shows regions you want to filter out. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. ex) chr1\t1\t100
  -  -norm <br/>
     Whether normalization is needed for input bigwig files. Choose either 'True' or 'False'. default=True
  -  -generateNormBW <br />
     If you want to generate normalized observed bigwig files, type 'True' (only works when '-norm True'). If you don't want, type 'False'. default=False
  -  -rngSeed <br />
     Set the seed value for the RNG. This enables repeatable runs. default=None


### 2) correctBias_stored
This command corrects technical biases (shear, PCR, mappability, G-quadruplex) from read counts when there are precomputed covariate files(.hdf). This command takes bigwig files as input and outputs bigwig files with corrected read counts. This command is much faster than running correctBias. Using 'correctBias_stored' is highly recommended when you have large regions, especially if you have whole genome data. You can download covariate files from [synpase](http://www.synapse.org).

Example of running correctBias_stored:
```
cradle correctBias_stored -ctrlbw ctrl1.bw ctrl2.bw ctrl3.bw
                          -expbw exp1.bw exp2.bw exp3.bw
                          -r /data/YoungSook/target_region.bed
                          -biasType shear pcr map gquad
                          -covariDir /data/YoungSook/hg39_fragLen500_kmer50
                          -genome /data/YoungSook/hg38.2bit
                          -kmer 50
                          -o /data/YoungSook/CRADLE_result
                          -bl /data/YoungSook/blacklist_regions.bed
```

* Required Arguments
  -  -ctrlbw <br />
      Ctrl bigwig files. Un-normalized files are recommended. Each file name should be spaced
  -  -expbw <br />
      Experimental bigwig files. Un-normalized files are recommended. Each file name should be spaced.
  -  -r <br />
      Text file that shows regions of analysis. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. ex) chr1\t100\t3000
  -  -biasType <br />
      Type of biases you want to correct among 'shear', 'pcr', 'map', 'gquad'. If you want to correct 'shear' and 'pcr' bias, you should type -biasType shear pcr.
  -  -covariDir <br />
      The directory of hdf files that have covariate values. The directory name of covariate files should be 'refGenome_fragLen(fragment length)_kmer(the length of sequenced reads)' ex) hg38_fragLen300_kmer36
  -  -genome <br/>
      The human genome sequence, in .2bit format. For information on downloading the genome, see [https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/) §§ "Files" and "How to Download"<br/> <br/>
  -  -faFile <br/>
      The same as `-genome`. This argument is deprecated and may be removed in a future release. Please use `-genome` instead.<br/> <br/>

* Optional Arguments
  -  -mi <br />
     The minimum number of fragments. Positions that have less fragments than this value are filtered out. default=the number of samples
  -  -o <br/>
     Output directory. All corrected bigwig files will be stored here. If the directory doesn't exist, cradle will make the directory. default=CRADLE_correctionResult.
  -  -p <br/>
     The number of cpus. default=(available cpus)/2
  -  -bl <br/>
     Text file that shows regions you want to filter out. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. ex) chr1\t1\t100
  -  -norm <br/>
     Whether normalization is needed for input bigwig files. Choose either 'True' or 'False'. default=True
  -  -generateNormBW <br />
     If you want to generate normalized observed bigwig files, type 'True' (only works when '-norm True'). If you don't want, type 'False'. default=False
  -  -rngSeed <br />
     Set the seed value for the RNG. This enables repeatable runs. default=None

### 3) callPeak
This command calls activated and repressed peaks with using corrected bigwig files as input.

Example of running callPeak:
```
cradle callPeak -ctrlbw ctrl1_corrected.bw ctrl2_corrected.bw ctrl3_corrected.bw
                -expbw exp1_corrected.bw exp2_corrected.bw exp3_corrected.bw
                -r /data/YoungSook/target_region.bed
                -fdr 0.05
                -o /data/YoungSook/CRADLE_peakCalling_result
```

* Required Arguments
  -  -ctrlbw <br />
      Ctrl bigwig files. Corrected bigwig files are recommended. Each file name should be spaced
  -  -expbw <br />
      Experimental bigwig files Corrected bigwig files are recommended.. Each file name should be spaced.
  -  -r <br />
      Text file that shows regions of analysis. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. ex) chr1\t100\t3000
  -  -fdr <br />
     FDR control <br/> <br/>

* Optional Arguments
  -  -o <br />
     Output directory. All corrected bigwig files will be stored here. If the directory doesn't exist, cradle will make the directory. default=CRADLE_peak_result.
  -  -bl <br />
     Text file that shows regions you want to filter out. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. ex) chr1\t1\t100
  -  -rbin <br />
     The size of a bin used for defining regions. rbin cannot be smaller than wbin. default = 300
  -  -wbin <br />
     The size of a bin used for testing differential activity. wbin cannot be larger than rbin. default = rbin/6
  -  -p <br/>
     The number of cpus. default=(available cpus)/2
  -  -d <br/>
     The minimum distance between peaks. Peaks distanced less than this value(bp) are merged. default=10
  -  -pl <br/>
     Minimum peak length. Peaks with smaller size than this value are filtered out. default=wbin
  -  -stat <br/>
     Choose a statistical testing: 't-test' for t-test and  'welch' for welch's t-test  default=t-test


### 3) Normalize
This command normalizes samples across different samples (accounting for sequencing depth) and different regions.
This command should be used for data that has uneven coverage resulting from any other reasons than biases.
For example, STARR-seq from BACs can have different coverage for each BAC region and overlapping BAC regions. This can cause correctBias or correctBias_stored to not efficiently model biases.

Example of running Normalize:
```
cradle normalize -ctrlbw ctrl1_corrected.bw ctrl2_corrected.bw ctrl3_corrected.bw
                -expbw exp1_corrected.bw exp2_corrected.bw exp3_corrected.bw
                -r /data/YoungSook/target_region.bed
                -o /data/YoungSook/CRADLE_normalize_result
```
* Required Arguments
  -  -ctrlbw <br />
      Ctrl bigwig files. Corrected bigwig files are recommended. Each file name should be spaced
  -  -expbw <br />
      Experimental bigwig files Corrected bigwig files are recommended.. Each file name should be spaced.
  -  -r <br />
      Text file that shows regions of analysis. Each line in the text file should have chromosome, start site, and end site that are tab-spaced. If you are suing BACs, please provide the coordinates of BACs without merging regions. ex) chr1\t100\t3000

* Optional Arguments
  -  -o <br />
     Output directory. All corrected bigwig files will be stored here. If the directory doesn't exist, cradle will make the directory. default=CRADLE_normalization.
  -  -p <br/>
     The number of cpus. default=(available cpus)/2


## Output files
### 1) correctBias and correctBias_stored.
   1) Corrected bigwigs files of which file name has '_corrected' in the suffix. The number of generated corrected bigwigs files will be the same as the total number of  bigwigs files used as input (this includes both control and experimental bigwigs).
   2) PNG files that shows fitting of the model with a subset of traning data. The number on the right bottom is Pearson's coefficient.


### 2) callPeak
You will get 'CRADLE_peak' as a result file which has the following format:
```
chr start   end Name    score   strand  effectSize  inputCount  outputCount -log(pvalue)    -log(qvalue)  cohen's D
chr1	70367085	70367210	chr1:70367085-70367210	.	.	133	-8	125	14.57	13.8 3.2
chr1	116853710	116853835	chr1:116853710-116853835	.	.	255	13	268	14.57	13.8 4.6
chr1	163672337	163672462	chr1:163672337-163672462	.	.	170	-2	168	14.57	13.8 12.3
chr10	80920118	80920243	chr10:80920118-80920243	.	.	93	-32	61	14.57	13.8 12.3
.
.
```
* The 1st-3rd columns(chr, start, end): genomic coordiantes
* The 4th,5th colum (score, strand): not applicable in CRADLE
* The 6th colum (effectSize):  effect size calculated by subtracting the mean of experimental read counts from the mean of control read counts.
* The 7th colum (inputCount):  the mean of control read counts.
* The 8th colum (outputCount):  the mean of experimental read counts.
* The 9-10th colum (-log(pvalue), -log(qvalue)):  -log10 of p value and q value. If a p value is zero, we used the maximum of -log(pvalue) values out of the total peaks. The same applies for q values.
* The 11th column: Cohen's D, standarized effect size. This column will have 'nan' values in the case where there is only one replicate in either -ctrlbw or -expbw.

### 3) Normalize
   Normalized bigwigs files of which file name has '_normalized' in the suffix. The number of generated corrected bigwigs files will be the same as the total number of  bigwigs files used as input (this includes both control and experimental bigwigs).


## How to download covariate files
We uploaded pre-computed covariates files for human genome (hg19, hg38). Those files are required to run "correctBias_stored"
1. Go to [synapse](http://www.synapse.org)
2. Register with synapse. (You cannot download the files unless you register)
3. Search covariate files with SynapseID, syn20369503.

## How to download human mappability and gquadruplex files
We liftover mappability files[2] and G-quadruplex files[3] from hg19 to hg38. You can download both hg19 and hg38 mappability files and G-quadruplex files.
1. Go to [synapse](http://www.synapse.org)
2. Register with synapse. (You cannot download the files unless you register)
3. Search covariate files with SynapseID, syn20369496.


## Tips on running CRADLE
* We strongly recommend using `correctBias_stored` when you have large regions because running `correctBias` might take a long time, especially when the fragment size is over 500. Small differences in fragment and sequenced lengths don't significantly affect correction power, so we recommend downloading covariate files from syanpse and runnning `correctBias_stored` if you can find fragment and sequenced lengths that are close to your data.

## Building a Singularity Image
[Singularity](https://www.sylabs.io/docs/) Is a container system created for use with scientific and research software. A [singularity image definition file](cradle_singularity.def) is included so users can build their own images.

## Using the Singularity Image
Using a singularity image should be much like using standard cradle. The main difference is that you may need to [bind](https://sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html) the directories containing input data are in if they aren't in your home directory. For example:

```sh
singularity run --bind /data cradle.sif correctBias_stored \
    -ctrlbw /data/ctrl_unnormalized.bw \
    -expbw /data/unnormalized/exp_unnormalized.bw \
    -r /data/ref_genome/regionfile_correctionModel \
    -biasType shear pcr map gquad \
    -genome /data/ref_genome/hg38/hg38.2bit \
    -covariDir /data/covariateFiles/hg38_fragLen1000_kmer50 \
    -mi 125 \
    -p 10 \
    -o corrected_bigwigs_50mer/ \
    -bl /data/ref_genome/hg38_filter_out.bed
```

However, if all your data is in a directory singularity mounts by default you can run the container like an execumtable:

```sh
./cradle.sif correctBias_stored \
    -ctrlbw ctrl_unnormalized.bw \
    -expbw unnormalized/exp_unnormalized.bw \
    -r ref_genome/regionfile_correctionModel \
    -biasType shear pcr map gquad \
    -genome ref_genome/hg38/hg38.2bit \
    -covariDir covariateFiles/hg38_fragLen1000_kmer50 \
    -mi 125 \
    -p 10 \
    -o corrected_bigwigs_50mer/ \
    -bl ref_genome/hg38_filter_out.bed
```

## References
1) DNAShape <br />
   Zhou T, Yang L, Lu Y, Dror I, Dantas Machado AC, Ghane T, Di Felice R, Rohs R.DNAshape: a method for the high-throughput prediction of DNA structural features on a genomic scale. Nucleic Acids Res. 2013 Jul;41(Web Server issue):W56-62. <br />
2) Mappability <br />
   Derrien T, Estellé J, Marco Sola S, Knowles DG, Raineri E, Guigó R, Ribeca P. Fast computation and applications of genome mappability. PLoS One. 2012;7(1):e30377. <br />
3) G-quadruplex sturcture <br/>
   Chambers VS, Marsico G, Boutell JM, Di Antonio M, Smith GP, Balasubramanian S. High-throughput sequencing of DNA G-quadruplex structures in the human genome.Nat Biotechnol. 2015 Aug;33(8):877-81.<br />

 ## Cite CRADLE
Kim YS, Johnson GD, Seo J, Barrera A, Cowart TN, Majoros WH, Ochoa A, Allen AS, Reddy TE. Correcting signal biases and detecting regulatory elements in STARR-seq data. Genome Res. 2021 May;31(5):877-889. doi: 10.1101/gr.269209.120. Epub 2021 Mar 15. PMID: 33722938; PMCID: PMC8092017.


