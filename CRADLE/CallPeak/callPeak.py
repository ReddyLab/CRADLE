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



def run(args):

	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS ...\n")
	vari.setGlobalVariables(args)	

	
	##### CALCULATE vari.FILTER_CUTOFF
	print("======  CALCULATING OVERALL VARIANCE FILTER CUTOFF ...")
	region_1stchr = np.array(vari.REGION)
	region_1stchr = region_1stchr[np.where(region_1stchr[:,0] == region_1stchr[0][0])].tolist()
	if(len(region_1stchr) < vari.NUMPROCESS):
		pool = multiprocessing.Pool(len(region_1stchr))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)

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

	if(len(task_diff) < vari.NUMPROCESS):
		pool = multiprocessing.Pool(len(task_diff))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
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
	if(len(vari.REGION) < vari.NUMPROCESS):
		pool = multiprocessing.Pool(len(vari.REGION))
	else:	
		pool = multiprocessing.Pool(vari.NUMPROCESS)
	result_region = pool.map_async(calculateRC.defineRegion, vari.REGION).get()		
	pool.close()
	pool.join()
	gc.collect()
	

	##### STATISTICAL TESTING FOR EACH REGION
	print("======  PERFORMING STAITSTICAL TESTING FOR EACH REGION ...")
	task_window = []
	for i in range(len(result_region)):
		if(result_region[i] != None):
			task_window.append(result_region[i])
	del result_region

	if(len(task_window) < vari.NUMPROCESS):
		pool = multiprocessing.Pool(len(task_window))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
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
	

	##### CHOOSING THETA 
	task_theta = [meta_filename]
	pool = multiprocessing.Pool(1)
	result_theta = pool.map_async(calculateRC.selectTheta, task_theta).get()
	pool.close()
	pool.join()

	vari.THETA = result_theta[0][0]
	selectRegionNum = result_theta[0][1]
	totalRegionNum = result_theta[0][2]	


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
		else:
			os.remove(subfile_name)
	os.remove(meta_filename)	

	if(len(task_callPeak) < vari.NUMPROCESS):
		pool = multiprocessing.Pool(len(task_callPeak))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
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
		os.remove(input_filename)
	output_stream.close()

	print("======= COMPLETED! ===========")
	print("The peak result was saved in %s" % vari.OUTPUT_DIR)

