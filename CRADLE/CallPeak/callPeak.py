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


def mergePeaks(peak_result):

	## open bigwig files to calculate effect size
	ctrlBW = [0] * vari.CTRLBW_NUM
	expBW = [0] * vari.EXPBW_NUM

	for i in range(vari.CTRLBW_NUM):
		ctrlBW[i] = pyBigWig.open(vari.CTRLBW_NAMES[i])
	for i in range(vari.EXPBW_NUM):
		expBW[i] = pyBigWig.open(vari.EXPBW_NAMES[i])

	merged_peak = []

	pastChromo = peak_result[0][0]
	pastStart = int(peak_result[0][1])
	pastEnd = int(peak_result[0][2])
	pastEnrich = int(peak_result[0][3])
	pvalues = [float(peak_result[0][4])]

	merged_peak.append(peak_result[0])
	resultIdx = 0

	i = 1
	while(i < len(peak_result)):
		currChromo = peak_result[i][0]
		currStart = int(peak_result[i][1])
		currEnd = int(peak_result[i][2])
		currEnrich = int(peak_result[i][3])
		currpvalue = float(peak_result[i][4])

		if( (currChromo == pastChromo) and (currEnrich == pastEnrich) and ( (currStart-pastEnd) < vari.DISTANCE)):
			merged_peak[resultIdx][2] = currEnd
			pvalues.extend([ currpvalue ])
		else:
			## update the continuous regions
			merged_peak[resultIdx][4] = np.min(pvalues)
			regionChromo = merged_peak[resultIdx][0]
			regionStart = int(merged_peak[resultIdx][1])
			regionEnd = int(merged_peak[resultIdx][2])

			ctrlRC = []
			for rep in range(vari.CTRLBW_NUM):
				rc = np.nanmean(np.array(ctrlBW[rep].values(regionChromo, regionStart, regionEnd)))
				ctrlRC.extend([rc])
			ctrlRC_posMean = np.nanmean(ctrlRC)

			expRC = []
			for rep in range(vari.EXPBW_NUM):
				rc = np.nanmean(np.array(expBW[rep].values(regionChromo, regionStart, regionEnd)))
				expRC.extend([rc])
			expRC_posMean = np.nanmean(expRC)

			diff_pos = int(expRC_posMean - ctrlRC_posMean)
			merged_peak[resultIdx][5] = diff_pos

			## start a new region
			merged_peak.append(peak_result[i])
			pvalues = [currpvalue]
			resultIdx = resultIdx + 1

		if(i == (len(peak_result) -1)):
			merged_peak[resultIdx][4] = np.min(pvalues)
			regionChromo = merged_peak[resultIdx][0]
			regionStart = int(merged_peak[resultIdx][1])
			regionEnd = int(merged_peak[resultIdx][2])

			ctrlRC = []
			for rep in range(vari.CTRLBW_NUM):
				rc = np.nanmean(np.array(ctrlBW[rep].values(regionChromo, regionStart, regionEnd)))
				ctrlRC.extend([rc])
			ctrlRC_posMean = np.nanmean(ctrlRC)

			expRC = []
			for rep in range(vari.EXPBW_NUM):
				rc = np.nanmean(np.array(expBW[rep].values(regionChromo, regionStart, regionEnd)))
				expRC.extend([rc])
			expRC_posMean = np.nanmean(expRC)

			diff_pos = int(expRC_posMean - ctrlRC_posMean)
			merged_peak[resultIdx][5] = diff_pos

		pastChromo = currChromo
		pastStart = currStart
		pastEnd = currEnd
		pastEnrich = currEnrich

		i = i + 1

	for i in range(vari.CTRLBW_NUM):
		ctrlBW[i].close()
	for i in range(vari.EXPBW_NUM):
		expBW[i].close()

	return merged_peak


def filterSmallPeaks(peak_result):

	final_result = []
	for i in range(len(peak_result)):
		start = int(peak_result[i][1])
		end = int(peak_result[i][2])

		if( (end-start) >= vari.PEAKLEN ):
			final_result.append(peak_result[i])

	return final_result



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
	

	##### Applying the selected theta 
	input_filename = meta_filename
	input_stream = open(input_filename)
	input_file = input_stream.readlines()

	PVALUE_simes = []

	### Apply the selected thata to the data
	for subFileIdx in range(len(input_file)):
		subfile_name = input_file[subFileIdx].split()[0]
		subfile_stream = open(subfile_name)
		subfile_file = subfile_stream.readlines()

		for regionIdx in range(len(subfile_file)):
			line = subfile_file[regionIdx].split()
			regionTheta = int(line[3])
			regionPvalue = float(line[4])

			if(np.isnan(regionPvalue) == True):
				continue

			if(regionTheta >= vari.THETA):
				PVALUE_simes.extend([ regionPvalue ])
	
	PVALUE_group_bh = statsmodels.sandbox.stats.multicomp.multipletests(PVALUE_simes, alpha=vari.FDR, method='fdr_bh')[0]


	##### Selecting windows
	task_callPeak = []

	input_filename = meta_filename
	input_stream = open(input_filename)
	input_file = input_stream.readlines()

	groupPvalueIdx = 0
	for subFileIdx in range(len(input_file)):
		subfile_name = input_file[subFileIdx].split()[0]
		subfile_stream = open(subfile_name)
		subfile_file = subfile_stream.readlines()

		selectRegionIdx = []
		selectedIdx = 0

		for regionIdx in range(len(subfile_file)):
			line = subfile_file[regionIdx].split()
			regionTheta = int(line[3])
			regionPvalue = float(line[4])
			
			if(regionTheta < vari.THETA):
				continue
			if(np.isnan(regionPvalue) == True):
				continue
			if(PVALUE_group_bh[groupPvalueIdx+selectedIdx] == True):
				selectRegionIdx.extend([ regionIdx ])
			selectedIdx = selectedIdx + 1

		groupPvalueIdx = groupPvalueIdx + selectedIdx

		if(len(selectRegionIdx) != 0):
			task_callPeak.append([subfile_name, selectRegionIdx])
		else:
			os.remove(subfile_name)

	input_stream.close()
	os.remove(meta_filename)	

	if(len(task_callPeak) == 0):
		print("======= COMPLETED! ===========")
		print("There is no peak detected in %s." % vari.OUTPUT_DIR)
		return

	if(len(task_callPeak) < vari.NUMPROCESS):
		pool = multiprocessing.Pool(len(task_callPeak))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
	result_callPeak = pool.map_async(calculateRC.doFDRprocedure, task_callPeak).get()
	pool.close()
	pool.join()
	
	del pool, task_callPeak
	gc.collect()

	peak_result = []
	for i in range(len(result_callPeak)):
		input_filename = result_callPeak[i]
		input_stream = open(input_filename)
		input_file = input_stream.readlines()

		for j in range(len(input_file)):
			temp = input_file[j].split()
			peak_result.append(temp)
		input_stream.close()
		os.remove(input_filename)

	if(len(peak_result) == 0):
		print("======= COMPLETED! ===========")
		print("There is no peak detected in %s." % vari.OUTPUT_DIR)
		return

	######## WRITE A RESULT FILE
	if(vari.DISTANCE == 1):
		final_result = filterSmallPeaks(peak_result)

		numActi = 0
		numRepress = 0

		output_filename = vari.OUTPUT_DIR + "/CRADLE_peaks"
		output_stream = open(output_filename, "w")

		for i in range(len(final_result)):
			if(int(final_result[i][3]) == 1):
				numActi = numActi + 1
			else:
				numRepress = numRepress + 1

			output_stream.write('\t'.join([str(x) for x in final_result[i]]) + "\n")
		output_stream.close()
	else:
		merged_peaks = mergePeaks(peak_result)
		final_result = filterSmallPeaks(merged_peaks)

		numActi = 0
		numRepress = 0

		output_filename = vari.OUTPUT_DIR + "/CRADLE_peaks"
		output_stream = open(output_filename, "w")

		for i in range(len(final_result)):
			if(int(final_result[i][3]) == 1):
				numActi = numActi + 1
			else:
				numRepress = numRepress + 1

			output_stream.write('\t'.join([str(x) for x in final_result[i]]) + "\n")
		output_stream.close()

	print("======= COMPLETED! ===========")
	print("The peak result was saved in %s" % vari.OUTPUT_DIR)
	print("Total number of peaks: %s" % len(final_result))
	print("Activated peaks: %s" % numActi)
	print("Repressed peaks: %s" % numRepress)

	return
