import sys
import os
import numpy as np
import math
import time
import tempfile
import gc
import scipy
import py2bit
import pyBigWig
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import h5py
import random


from CRADLE.CorrectBiasStored import vari

cpdef performRegression(trainSet):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')
	
	faFile = py2bit.open(vari.FA)

	trainSet_new = []
	X_numRows = 0
	X_numCols = vari.COVARI_NUM + 1
	for trainIdx in range(len(trainSet)):
		chromo = trainSet[trainIdx][0]
		analysis_start = int(trainSet[trainIdx][1])
		analysis_end = int(trainSet[trainIdx][2])
		chromoEnd = int(faFile.chroms(chromo))

		frag_start = analysis_start - vari.FRAGLEN + 1
		frag_end = analysis_end + vari.FRAGLEN - 1  
		shear_start = frag_start - 2
		shear_end = frag_end + 2

		if(shear_start < 1):
			shear_start = 1
			frag_start = 3
			binStart = max(binStart, frag_start)
			analysis_start = max(analysis_start, frag_start)
		
		if(shear_end > chromoEnd):
			shear_end = chromoEnd
			frag_end = shear_end - 2
			binEnd = min(binEnd, (frag_end-1))
			analysis_end = min(analysis_end, frag_end)  # not included
	
		X_numRows = X_numRows + (analysis_end - analysis_start)
		trainSet_new.append([chromo, analysis_start, analysis_end])
	faFile.close()
	del trainSet, faFile

	#### Initialize COEF matrix
	COEFCTRL = np.zeros((vari.CTRLBW_NUM, (vari.COVARI_NUM+1)), dtype=np.float64)
	COEFEXP = np.zeros((vari.EXPBW_NUM, (vari.COVARI_NUM+1)), dtype=np.float64)

	#### Get X matrix 
	cdef double [:,:] X_view = np.ones((X_numRows, X_numCols), dtype=np.float64)

	cdef int row_ptr = 0
	cdef int pos
	cdef int j
	cdef int analysis_start_this
	cdef int analysis_end_this

	for trainIdx in range(len(trainSet_new)):
		chromo = trainSet_new[trainIdx][0]
		analysis_start_this = int(trainSet_new[trainIdx][1])
		analysis_end_this = int(trainSet_new[trainIdx][2])

		hdfFileName = vari.COVARI_DIR + "/" + vari.COVARI_Name + "_" + chromo + ".hdf5"
		f = h5py.File(hdfFileName, "r")

		pos = analysis_start_this
		while(pos < analysis_end_this):
			temp = f['covari'][pos-3] * vari.SELECT_COVARI
			temp = temp[np.isnan(temp) == False]

			j = 0
			while(j < vari.COVARI_NUM):
				X_view[(row_ptr+pos-analysis_start_this), j+1] = temp[j]
				j = j + 1
			pos = pos + 1
		row_ptr = row_ptr + (analysis_end_this - analysis_start_this)
		f.close()


	if(X_numRows < 50000):
		idx = np.array(list(range(X_numRows)))
	else:
		idx = np.random.choice(np.array(list(range(X_numRows))), 50000, replace=False)

	#### Get Y matrix
	cdef double [:] Y_view = np.zeros(X_numRows, dtype=np.float64)
	cdef int ptr
	cdef int posIdx

	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		
		ptr = 0
		for trainIdx in range(len(trainSet_new)):
			chromo = trainSet_new[trainIdx][0]
			analysis_start = int(trainSet_new[trainIdx][1])
			analysis_end = int(trainSet_new[trainIdx][2])

			rc = np.array(bw.values(chromo, analysis_start, analysis_end))
			rc[np.isnan(rc) == True] = float(0)
			rc = rc / vari.CTRLSCALER[rep]

			numPos = analysis_end - analysis_start
			posIdx = 0
			while(posIdx < numPos):
				Y_view[ptr+posIdx] = rc[posIdx]
				posIdx = posIdx + 1

			ptr = ptr + numPos
		bw.close()
		del rc

		#### do regression
		model = sm.GLM(np.array(Y_view).astype(int), np.array(X_view), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

		coef = model.params
		COEFCTRL[rep, ] = coef
		corr = np.corrcoef(model.fittedvalues, np.array(Y_view))[0, 1]
		corr = np.round(corr, 2)

		## PLOT
		maxi1 = np.nanmax(model.fittedvalues[idx])
		maxi2 = np.nanmax(np.array(Y_view)[idx])
		maxi = max(maxi1, maxi2)

		figName = vari.OUTPUT_DIR + "/ctrl_rep" + str(rep+1) + ".png"
		plt.plot(np.array(Y_view)[idx], model.fittedvalues[idx], color='g', marker='s', alpha=0.01)
		plt.text((maxi-25), 10, corr, ha='center', va='center')
		plt.xlabel("observed")
		plt.ylabel("predicted")
		plt.xlim(0, maxi)
		plt.ylim(0, maxi)
		plt.plot([0, maxi], [0, maxi], 'k-', color='r')
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig(figName)
		plt.close()
		plt.clf()

			
	for rep in range(vari.EXPBW_NUM):	
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])

		ptr = 0
		for trainIdx in range(len(trainSet_new)):
			chromo = trainSet_new[trainIdx][0]
			analysis_start = int(trainSet_new[trainIdx][1])
			analysis_end = int(trainSet_new[trainIdx][2])

			rc = np.array(bw.values(chromo, analysis_start, analysis_end))
			rc[np.isnan(rc) == True] = float(0)
			rc = rc / vari.EXPSCALER[rep]

			numPos = analysis_end - analysis_start
			posIdx = 0
			while(posIdx < numPos):
				Y_view[ptr+posIdx] = rc[posIdx]
				posIdx = posIdx +1

			ptr = ptr + numPos
		bw.close()
		del rc

		#### do regression
		model = sm.GLM(np.array(Y_view).astype(int), np.array(X_view), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

		coef = model.params
		COEFEXP[rep, ] = coef
		corr = np.corrcoef(model.fittedvalues, np.array(Y_view))[0, 1]
		corr = np.round(corr, 2)

		## PLOT
		maxi1 = np.nanmax(model.fittedvalues[idx])
		maxi2 = np.nanmax(np.array(Y_view)[idx])
		maxi = max(maxi1, maxi2)

		figName = vari.OUTPUT_DIR + "/exp_rep" + str(rep+1) + ".png"
		plt.plot(np.array(Y_view)[idx], model.fittedvalues[idx], color='g', marker='s', alpha=0.01)
		plt.text((maxi-25), 10, corr, ha='center', va='center')
		plt.xlabel("observed")
		plt.ylabel("predicted")
		plt.xlim(0, maxi)
		plt.ylim(0, maxi)
		plt.plot([0, maxi], [0, maxi], 'k-', color='r')
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig(figName)
		plt.close()
		plt.clf()

	return COEFCTRL, COEFEXP


cpdef correctReadCount(args):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	chromo = args[0]
	analysis_start = int(args[1])  # Genomic coordinates(starts from 1)
	analysis_end = int(args[2])

	faFile = py2bit.open(vari.FA)
	chromoEnd = int(faFile.chroms(chromo))
	faFile.close()

        ###### GENERATE A RESULT MATRIX 
	frag_start = analysis_start - vari.FRAGLEN + 1
	frag_end = analysis_end + vari.FRAGLEN - 1
	shear_start = frag_start - 2
	shear_end = frag_end + 2

	if(shear_start < 1):
		shear_start = 1
		frag_start = 3
		analysis_start = max(analysis_start, frag_start)
	if(shear_end > chromoEnd):
		shear_end = chromoEnd
		frag_end = shear_end - 2
		analysis_end = min(analysis_end, frag_end)


	## OUTPUT FILES 
	subfinalCtrl = [0] * vari.CTRLBW_NUM
	subfinalCtrlNames = [0] * vari.CTRLBW_NUM
	subfinalExp = [0] * vari.EXPBW_NUM
	subfinalExpNames = [0] * vari.EXPBW_NUM

	for i in range(vari.CTRLBW_NUM):
		subfinalCtrl[i] = tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=vari.OUTPUT_DIR, delete=False)
		subfinalCtrlNames[i] = subfinalCtrl[i].name
		subfinalCtrl[i].close()

	for i in range(vari.EXPBW_NUM):
		subfinalExp[i] = tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=vari.OUTPUT_DIR, delete=False)
		subfinalExpNames[i] = subfinalExp[i].name
		subfinalExp[i].close()

	###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > FILTERVALUE
	selectedIdx, highRC_idx, starts = selectIdx(chromo, analysis_start, analysis_end)

	if(len(selectedIdx) == 0):
		for i in range(vari.CTRLBW_NUM):
			os.remove(subfinalCtrlNames[i])
		for i in range(vari.EXPBW_NUM):
			os.remove(subfinalExpNames[i])

		return [ [None] * vari.CTRLBW_NUM, [None] * vari.EXPBW_NUM, chromo ]


	hdfFileName = vari.COVARI_DIR + "/" + vari.COVARI_Name + "_" + chromo + ".hdf5"
	f = h5py.File(hdfFileName, "r")

	for rep in range(vari.CTRLBW_NUM):	
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		rcArr = np.array(bw.values(chromo, analysis_start, analysis_end))
		rcArr[np.isnan(rcArr) == True] = float(0)
		rcArr = rcArr / vari.CTRLSCALER[rep]
		bw.close()

		prdvals = np.exp(np.nansum( (f['covari'][(analysis_start-3):(analysis_end-3)] * vari.SELECT_COVARI) * vari.COEFCTRL[rep, 1:], axis=1) + vari.COEFCTRL[rep, 0]) 
		prdvals[highRC_idx] = np.exp(np.nansum( (f['covari'][(analysis_start-3):(analysis_end-3)][highRC_idx] * vari.SELECT_COVARI) * vari.COEFCTRL_HIGHRC[rep, 1:], axis=1) + vari.COEFCTRL_HIGHRC[rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[selectedIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		if(len(idx[0]) > 0):
			tempStarts = np.delete(starts, idx)
			rcArr = np.delete(rcArr, idx)
			if(len(rcArr) > 0):
				writeBedFile(subfinalCtrlNames[rep], tempStarts, rcArr, analysis_end)
			else:
				os.remove(subfinalCtrlNames[rep])
				subfinalCtrlNames[rep] = None
		else:
			if(len(rcArr) > 0):
				writeBedFile(subfinalCtrlNames[rep], starts, rcArr, analysis_end)
			else:
				os.remove(subfinalCtrlNames[rep])
				subfinalCtrlNames[rep] = None


	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		rcArr = np.array(bw.values(chromo, analysis_start, analysis_end))
		rcArr[np.isnan(rcArr) == True] = float(0)
		rcArr = rcArr / vari.EXPSCALER[rep]
		bw.close()

		prdvals = np.exp(np.nansum( (f['covari'][(analysis_start-3):(analysis_end-3)] * vari.SELECT_COVARI) * vari.COEFEXP[rep, 1:], axis=1) + vari.COEFEXP[rep, 0])
		prdvals[highRC_idx] = np.exp(np.nansum( (f['covari'][(analysis_start-3):(analysis_end-3)][highRC_idx] * vari.SELECT_COVARI) * vari.COEFEXP_HIGHRC[rep, 1:], axis=1) + vari.COEFEXP_HIGHRC[rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[selectedIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		if(len(idx[0]) > 0):
			tempStarts = np.delete(starts, idx)
			rcArr = np.delete(rcArr, idx)
			if(len(rcArr) > 0):
				writeBedFile(subfinalExpNames[rep], tempStarts, rcArr, analysis_end)
			else:
				os.remove(subfinalExpNames[rep])
				subfinalExpNames[rep] = None
		else:
			if(len(rcArr) > 0):
				writeBedFile(subfinalExpNames[rep], starts, rcArr, analysis_end)
			else:
				os.remove(subfinalExpNames[rep])
				subfinalExpNames[rep] = None

	f.close()	

	return_array = [subfinalCtrlNames, subfinalExpNames, chromo]

	return return_array





cpdef selectIdx(chromo, analysis_start, analysis_end):
	ctrlRC = []
	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		
		temp = np.array(bw.values(chromo, analysis_start, analysis_end))
		temp[np.where(np.isnan(temp)==True)] = float(0)

		bw.close()
		ctrlRC.append(temp.tolist())

		if(rep == 0):
			rc_sum = temp
			highRC_idx = np.where(temp > vari.HIGHRC)[0]
		else:
			rc_sum = rc_sum + temp

	ctrlRC = np.nanmean(ctrlRC, axis=0)
	idx1 = np.where(ctrlRC > 0)[0].tolist()	
		
	expRC = []
	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])

		temp = np.array(bw.values(chromo, analysis_start, analysis_end))
		temp[np.where(np.isnan(temp)==True)] = float(0)

		bw.close()
		expRC.append(temp.tolist())

		rc_sum = rc_sum + temp

	expRC = np.nanmean(expRC, axis=0)
	idx2 = np.where(expRC > 0)[0].tolist()
	idx3 = np.where(rc_sum > vari.FILTERVALUE)[0].tolist()

	idx_temp = np.intersect1d(idx1, idx2)
	idx = np.intersect1d(idx_temp, idx3)

	if(len(idx) == 0):
		return np.array([]), np.array([]), np.array([])

	starts = np.array(list(range(analysis_start, analysis_end)))[idx]

	return idx, highRC_idx, starts



cpdef writeBedFile(subfileName, tempStarts, tempSignalvals, analysis_end):
	subfile = open(subfileName, "w")

	tempSignalvals = tempSignalvals.astype(int)
	numIdx = len(tempSignalvals)

	idx = 0
	prevStart = tempStarts[idx]
	prevRC = tempSignalvals[idx]
	line = [prevStart, (prevStart + vari.BINSIZE), prevRC]
	if(numIdx == 1):
		subfile.write('\t'.join([str(x) for x in line]) + "\n")
		subfile.close()
		return

	idx = 1
	while(idx < numIdx):
		currStart = tempStarts[idx]
		currRC = tempSignalvals[idx]

		if( (currStart == (prevStart + vari.BINSIZE)) and (currRC == prevRC) ):
			line[1] = currStart + vari.BINSIZE
			prevStart = currStart
			prevRC = currRC
			idx = idx + 1
		else:
			### End a current line
			subfile.write('\t'.join([str(x) for x in line]) + "\n")

			### Start a new line
			line = [currStart, (currStart+vari.BINSIZE), currRC]
			prevStart = currStart
			prevRC = currRC
			idx = idx + 1

		if(idx == numIdx):
			line[1] = analysis_end
			subfile.write('\t'.join([str(x) for x in line]) + "\n")
			subfile.close()
			break

	return




