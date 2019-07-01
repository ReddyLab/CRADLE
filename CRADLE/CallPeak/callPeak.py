import os
import numpy as np
import multiprocessing
import argparse
import py2bit
import pyBigWig
import gc
import statsmodels.sandbox.stats.multicomp

from CRADLE.CallPeak import vari
from CRADLE.CallPeak import calculateRC

import sys ## for test


def getArgs():
	parser = argparse.ArgumentParser()

	### required
	requiredArgs = parser.add_argument_group('Required Arguments')
	
	requiredArgs.add_argument('-ctrlbw', help="Ctrl bigwig files. Corrected bigwig files are recommended. Each file name should be spaced. ex) -ctrlbw file1.bw file2.bw", nargs='+', required=True)

	requiredArgs.add_argument('-expbw', help="Experimental bigwig files. Corrected bigwig files are recommended. Each file name should be spaced. ex) -expbw file1.bw file2.bw", nargs='+', required=True)

	requiredArgs.add_argument('-l', help="Fragment length.", required=True)

	requiredArgs.add_argument('-r', help="Text file that shows regions of analysis. Each line in the text file should have chromosome, start site, and end site that are tab-spaced.", required=True)


	requiredArgs.add_argument('-fdr', help="FDR level", required=True)

	### optional  
        optionalArgs = parser.add_argument_group('Optional Arguments')

	optionalArgs.add_argument('-o', help="Output directoy")
	
	optionalArgs.add_argument('-bl', help="Blacklist regions")

	optionalArgs.add_argument('-rbin', help="The size of bin used for defining regions")

	optionalArgs.add_argument('-wbin', help="he size of bin used for testing differential activity")

	return parser


def run(args):

	###### INITIALIZE PARAMETERS
        print("======  INITIALIZING PARAMETERS ...\n")
	#args = getArgs().parse_args()
	vari.setGlobalVariables(args)	

	global numProcess
	numProcess = multiprocessing.cpu_count() -1
	if(numProcess > 28):
		numProcess = 28

	
	##### CALCULATE vari.FILTER_CUTOFF
	print("======  CALCULATING OVERALL VARIANCE FILTER CUTOFF ...")
	region_1stchr = np.array(vari.REGION)
	region_1stchr = region_1stchr[np.where(region_1stchr[:,0] == region_1stchr[0][0])].tolist()
	if(len(region_1stchr) < numProcess):
		pool = multiprocessing.Pool(len(region_1stchr))
	else:
		pool = multiprocessing.Pool(numProcess)

	result_filter = pool.map_async(calculateRC.getVariance, region_1stchr).get()
	pool.close()
	pool.join()

	var = []
	for i in range(len(result_filter)):
		if(result_filter[i] != None):
			var.extend(result_filter[i])

	vari.FILTER_CUTOFFS[0] = -1
	for i in range(1, len(vari.FILTER_CUTOFFS_THETAS)):
		vari.FILTER_CUTOFFS[i] = np.percentile(var, vari.FILTER_CUTOFFS_THETAS[i])
	vari.FILTER_CUTOFFS = np.array(vari.FILTER_CUTOFFS)

	print("Variance Cutoff: %s" % np.round(vari.FILTER_CUTOFFS))
	del pool, var, result_filter
	gc.collect()


	##### DEFINING REGIONS
	print("======  DEFINING REGIONS ...")
	# 1)  CALCULATE REGION_CUFOFF
	region_total = 0
	task_diff = []
	for region in vari.REGION:
		regionSize = int(region[2]) - int(region[1])
		region_total = region_total + regionSize 
		task_diff.append(region)		

		if(regionSize > 3* np.power(10, 8)):
			break

	if(len(task_diff) < numProcess):
		pool = multiprocessing.Pool(len(task_diff))
	else:
		pool = multiprocessing.Pool(numProcess)
	pool = multiprocessing.Pool(numProcess)
	result_diff = pool.map_async(calculateRC.getRegionCutoff, task_diff).get()
	pool.close()
	pool.join()

	diff = []
	for i in range(len(result_diff)):
		if(result_diff[i] != None):
			diff.extend(result_diff[i])

	vari.NULL_STD = np.sqrt(np.nanvar(diff))
	print("Null_std: %s" % vari.NULL_STD)
	vari.REGION_CUTOFF = np.percentile(np.array(diff), 99)
	print("Region cutoff: %s " % vari.REGION_CUTOFF)
	del pool, result_diff, diff, task_diff
	gc.collect()
	

	# 2)  DEINING REGIONS WITH 'vari.REGION_CUTOFF'
	pool = multiprocessing.Pool(numProcess)
	result_region = pool.map_async(calculateRC.defineRegion, vari.REGION).get()		
	pool.close()
        pool.join()
        gc.collect()
	

	#### tentative_for_verification
	output_filename = vari.OUTPUT_DIR + "/metaData1"
	output_stream = open(output_filename, "w")
	for i in range(len(result_region)):
		if(result_region[i] != None):
			output_stream.write(result_region[i] + "\n")
	output_stream.close()


	##### STATISTICAL TESTING FOR EACH REGION
	print("======  PERFORMING STAITSTICAL TESTING FOR EACH REGION ...")
	task_window = []
	for i in range(len(result_region)):
		if(result_region[i] != None):
			task_window.append(result_region[i])
	del result_region

	pool = multiprocessing.Pool(numProcess)
	result_ttest = pool.map_async(calculateRC.doWindowApproach, task_window).get()
	pool.close()
        pool.join()
	

        meta_filename = vari.OUTPUT_DIR + "/metaData_pvalues"
        meta_stream = open(meta_filename, "w")
        for i in range(len(result_ttest)):
                if(result_ttest[i] != None):
                        meta_stream.write(result_ttest[i] + "\n")
        meta_stream.close()
	del task_window, pool, result_ttest
	

	#meta_filename = vari.OUTPUT_DIR + "/metaData_pvalues"

	##### CHOOSING THETA 
	task_theta = [meta_filename]
	pool = multiprocessing.Pool(numProcess)
	result_theta = pool.map_async(calculateRC.selectTheta, task_theta).get()
	pool.close()
	pool.join()
	result_theta = result_theta[0]

	vari.THETA = result_theta[0]
	selectRegionNum = result_theta[1]
	totalRegionNum = result_theta[2]	

	##### FDR control
	print("======  CALLING PEAKS ...")
	vari.ADJ_FDR = ( vari.FDR * selectRegionNum ) / float(totalRegionNum)
	print("Selected Variance Theta: %s" % vari.THETA)
	print("Total number of regions: %s" % totalRegionNum)
	print("The number of selected regions: %s" % selectRegionNum)
	print("Newly adjusted cutoff: %s" % vari.ADJ_FDR)
	

	task_callPeak = []

	input_filename = meta_filename
	input_stream = open(input_filename)
	input_file = input_stream.readlines()

	for subFileIdx in range(len(input_file)):
		subfile_name = input_file[subFileIdx].split()[0]
		subfile_stream = open(subfile_name)
		subfile_file = subfile_stream.readlines()

		selectRegionIdx = []

		for regionIdx in range(len(subfile_file)):
			line = subfile_file[regionIdx].split()
			regionTheta = int(line[3])
			regionPvalue = float(line[4])
			
			if(regionTheta < vari.THETA):
				continue
			if(np.isnan(regionPvalue) == True):
				continue
			if(regionPvalue < vari.FDR):
				selectRegionIdx.extend([ regionIdx ])

		if(len(selectRegionIdx) != 0):
			task_callPeak.append([subfile_name, selectRegionIdx])
		#else:
			#print(subfile_name)
			#os.remove(subfile_name)
	
	output_filename = vari.OUTPUT_DIR + "/selected_filenames" 
	output_stream = open(output_filename, "w")
	for i in range(len(task_callPeak)):
		output_stream.write('\t'.join([str(x) for x in task_callPeak[i]]) + "\n")
	output_stream.close()

	pool = multiprocessing.Pool(numProcess)
	result_callPeak = pool.map_async(calculateRC.doFDRprocedure, task_callPeak).get()
	pool.close()
        pool.join()
	
	del pool, task_callPeak
	gc.collect()

	
	######## WRITE A RESULT FILE
	output_filename = vari.OUTPUT_DIR + "/CRADE_peaks"
	output_stream = open(output_filename, "w")

	for i in range(len(result_callPeak)):
		input_filename = result_callPeak[i]
		input_stream = open(input_filename)
		input_file = input_stream.readlines()

		for j in range(len(input_file)):
			temp = input_file[j].split()
			output_stream.write('\t'.join([str(x) for x in temp]) + "\n")
		input_stream.close()
		#os.remove(input_filename)
	output_stream.close()

	print("======= COMPLETED! ===========")
	print("The peak result was saved in %s" % vari.OUTPUT_DIR)

