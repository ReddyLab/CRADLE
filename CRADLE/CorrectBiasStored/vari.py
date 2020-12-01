import os
import math
import numpy as np
import sys
import multiprocessing
import pyBigWig

def setGlobalVariables(args):
	
	### input bigwig files
	setInputFiles(args.ctrlbw, args.expbw)
	setOutputDirectory(args.o)
	setCovariDir(args.biasType, args.covariDir, args.faFile)
	setAnlaysisRegion(args.r, args.bl)
	setFilterCriteria(args.mi)
	setNumProcess(args.p)
	setNormalization(args.norm, args.generateNormBW)

	return


def setInputFiles(ctrlbwFiles, expbwFiles):
	global CTRLBW_NAMES
	global EXPBW_NAMES
	
	global CTRLBW_NUM
	global EXPBW_NUM
	global SAMPLE_NUM

	global COEFCTRL
	global COEFEXP
	global COEFCTRL_HIGHRC
	global COEFEXP_HIGHRC

	global HIGHRC


	CTRLBW_NUM = len(ctrlbwFiles)
	EXPBW_NUM = len(expbwFiles)
	SAMPLE_NUM = CTRLBW_NUM + EXPBW_NUM

	CTRLBW_NAMES = [0] * CTRLBW_NUM
	for i in range(CTRLBW_NUM):
		CTRLBW_NAMES[i] = ctrlbwFiles[i]
	
	EXPBW_NAMES = [0] * EXPBW_NUM
	for i in range(EXPBW_NUM):
		EXPBW_NAMES[i] = expbwFiles[i]


	return


def setOutputDirectory(outputDir):
	global OUTPUT_DIR 

	if(outputDir == None):
		outputDir = os.getcwd() + "/CRADLE_correctionResult"
	
	if(outputDir[-1] == "/"):
		outputDir = outputDir[:-1]

	OUTPUT_DIR = outputDir

	dirExist = os.path.isdir(OUTPUT_DIR)
	if(dirExist == False):
		os.makedirs(OUTPUT_DIR)

	return

def setCovariDir(biasType, covariDir, faFile):
	global COVARI_DIR
	global COVARI_Name
	global SELECT_COVARI
	global FRAGLEN	
	global FA
	global COVARI_NUM
	global BINSIZE
	global COVARI_ORDER

	COVARI_DIR = covariDir

	if(os.path.isdir(COVARI_DIR) == False):
		print("Error! There is no covariate directory")
		sys.exit()

	if(COVARI_DIR[-1] == "/"):
		COVARI_DIR = COVARI_DIR[:-1]
	
	temp_str = COVARI_DIR.split('/')
	COVARI_Name = temp_str[len(temp_str)-1]
	temp_str = COVARI_Name.split("_")[1]
	FRAGLEN = int(temp_str.split("fragLen")[1])

	FA = faFile
	BINSIZE = 1

	SELECT_COVARI = np.array([np.nan] * 6)
	COVARI_NUM = 0
	
	COVARI_ORDER = ['Intercept']

	biasType = [x.lower() for x in biasType]

	for i in range(len(biasType)):
		if( (biasType[i] != 'shear') and (biasType[i] != 'pcr') and (biasType[i] != 'map') and (biasType[i] != 'gquad')):
			print("Error! Wrong value in -biasType. Only 'shear', 'pcr', 'map', 'gquad' are allowed")
			sys.exit()

	if('shear' in biasType):
		SELECT_COVARI[0] = 1
		SELECT_COVARI[1] = 1
		COVARI_NUM = COVARI_NUM + 2
		COVARI_ORDER.extend(["MGW_shear", "ProT_shear"])
	if('pcr' in biasType):
		SELECT_COVARI[2] = 1
		SELECT_COVARI[3] = 1
		COVARI_NUM = COVARI_NUM + 2
		COVARI_ORDER.extend(["Anneal_pcr", "Denature_pcr"])
	if('map' in biasType):
		SELECT_COVARI[4] = 1
		COVARI_NUM = COVARI_NUM + 1
		COVARI_ORDER.extend(["Map_map"])
	if('gquad' in biasType):
		SELECT_COVARI[5] = 1
		COVARI_NUM = COVARI_NUM + 1
		COVARI_ORDER.extend(["Gquad_gquad"])
	return



def setAnlaysisRegion(region, bl):
	global REGION

	REGION = []
	input_filename = region
	input_stream = open(input_filename)
	input_file = input_stream.readlines()

	for i in range(len(input_file)):
		temp = input_file[i].split()
		temp[1] = int(temp[1])
		temp[2] = int(temp[2])
		REGION.append(temp)
	input_stream.close()

	if(len(REGION) > 1):
		REGION = np.array(REGION)
		REGION = REGION[np.lexsort(( REGION[:,1].astype(int), REGION[:,0])  ) ]
		REGION = REGION.tolist()

		region_merged = []

		pos = 0
		pastChromo = REGION[pos][0]
		pastStart = int(REGION[pos][1])
		pastEnd = int(REGION[pos][2])
		region_merged.append([ pastChromo, pastStart, pastEnd])
		resultIdx = 0

		pos = 1
		while( pos < len(REGION) ):
			currChromo = REGION[pos][0]
			currStart = int(REGION[pos][1])
			currEnd = int(REGION[pos][2])

			if( (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd)):
				maxEnd = np.max([currEnd, pastEnd])
				region_merged[resultIdx][2] = maxEnd
				pos = pos + 1
				pastChromo = currChromo
				pastStart = currStart
				pastEnd = maxEnd
			else:
				region_merged.append([currChromo, currStart, currEnd])
				resultIdx = resultIdx + 1
				pos = pos + 1
				pastChromo = currChromo
				pastStart = currStart
				pastEnd = currEnd

		REGION = region_merged
	
	## BL	
	if(bl != None):  ### REMOVE BLACKLIST REGIONS FROM 'REGION'     
		bl_region_temp = []
		input_stream = open(bl)
		input_file = input_stream.readlines()
		for i in range(len(input_file)):
			temp = input_file[i].split()
			temp[1] = int(temp[1])
			temp[2] = int(temp[2])
			bl_region_temp.append(temp)

		## merge overlapping blacklist regions
		if(len(bl_region_temp) == 1):
			bl_region = bl_region_temp
			bl_region = np.array(bl_region)
		else:
			bl_region_temp = np.array(bl_region_temp)
			bl_region_temp = bl_region_temp[np.lexsort( ( bl_region_temp[:,1].astype(int), bl_region_temp[:,0] ) )]
			bl_region_temp = bl_region_temp.tolist()

			bl_region = []
			pos = 0
			pastChromo = bl_region_temp[pos][0]
			pastStart = int(bl_region_temp[pos][1])
			pastEnd = int(bl_region_temp[pos][2])
			bl_region.append([pastChromo, pastStart, pastEnd])
			resultIdx = 0

			pos = 1
			while( pos < len(bl_region_temp) ):
				currChromo = bl_region_temp[pos][0]
				currStart = int(bl_region_temp[pos][1])
				currEnd = int(bl_region_temp[pos][2])

				if( (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd)):
					bl_region[resultIdx][2] = currEnd
					pos = pos + 1
					pastChromo = currChromo
					pastStart = currStart
					pastEnd = currEnd
				else:
					bl_region.append([currChromo, currStart, currEnd])
					resultIdx = resultIdx + 1
					pos = pos + 1
					pastChromo = currChromo
					pastStart = currStart
					pastEnd = currEnd
			bl_region = np.array(bl_region)

		region_woBL = []
		for region in REGION:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			overlapped_bl = []
			## overlap Case 1 : A blacklist region completely covers the region. 
			idx = np.where( (bl_region[:,0] == regionChromo) & (bl_region[:,1].astype(int) <= regionStart) & (bl_region[:,2].astype(int) >= regionEnd) )[0]
			if(len(idx) > 0):
				continue

			## overlap Case 2 
			idx = np.where( (bl_region[:,0] == regionChromo) & (bl_region[:,2].astype(int) > regionStart) & (bl_region[:,2].astype(int) <= regionEnd) )[0]
			if(len(idx) > 0):
				overlapped_bl.extend( bl_region[idx].tolist() )

			## overlap Case 3
			idx = np.where( (bl_region[:,0] == regionChromo) & (bl_region[:,1].astype(int) >= regionStart) & (bl_region[:,2].astype(int) < regionEnd) )[0]
			if(len(idx) > 0):
				overlapped_bl.extend( bl_region[idx].tolist() )

			if(len(overlapped_bl) == 0):
				region_woBL.append(region)
				continue

			overlapped_bl = np.array(overlapped_bl)
			overlapped_bl = overlapped_bl[overlapped_bl[:,1].astype(int).argsort()]
			overlapped_bl = np.unique(overlapped_bl, axis=0)
			overlapped_bl = overlapped_bl[overlapped_bl[:,1].astype(int).argsort()]

			currStart = regionStart
			for pos in range(len(overlapped_bl)):
				blStart = int(overlapped_bl[pos][1])
				blEnd = int(overlapped_bl[pos][2])

				if( blStart <= regionStart ):
					currStart = blEnd
				else:
					if(currStart == blStart):
						currStart = blEnd
						continue

					region_woBL.append([ regionChromo, currStart, blStart ])
					currStart = blEnd

				if( (pos == (len(overlapped_bl)-1)) and (blEnd < regionEnd) ):
					if(blEnd == regionEnd):
						break
					region_woBL.append([ regionChromo, blEnd, regionEnd ])

		REGION = region_woBL
	
	# check if all chromosomes in the REGION in bigwig files
	bw = pyBigWig.open(CTRLBW_NAMES[0])
	region_final = []
	for regionIdx in range(len(REGION)):
		chromo = REGION[regionIdx][0]
		start = int(REGION[regionIdx][1])
		end = int(REGION[regionIdx][2])

		chromoLen = bw.chroms(chromo)
		if(chromoLen == None):
			continue
		if(end > chromoLen):
			REGION[regionIdx][2] = chromoLen
			if( chromoLen <= start ):
				continue
		region_final.append([chromo, start, end])
	bw.close()

	REGION = region_final


	return


def setFilterCriteria(minFrag):
	global FILTERVALUE

	if(minFrag == None):
		FILTERVALUE = SAMPLE_NUM
	else:
		FILTERVALUE = int(minFrag)

	return


def setScaler(scalerResult):
	global CTRLSCALER
	global EXPSCALER

	CTRLSCALER = [0] * CTRLBW_NUM
	EXPSCALER = [0] * EXPBW_NUM
	CTRLSCALER[0] = 1

	for i in range(1, CTRLBW_NUM):
		CTRLSCALER[i] = scalerResult[i-1]
	for i in range(EXPBW_NUM):
		EXPSCALER[i] = scalerResult[i+CTRLBW_NUM-1]
	
	return



def setNumProcess(numProcess):
	global NUMPROCESS
	
	system_cpus = int(multiprocessing.cpu_count())

	if(numProcess == None):
		NUMPROCESS = int(system_cpus / 2.0 )
		if(NUMPROCESS < 1):
			NUMPROCESS = 1
	else:
		NUMPROCESS = int(numProcess)

	if(NUMPROCESS > system_cpus):
		print("ERROR: You specified too many cpus! (-p). Running with the maximum cpus in the system")
		NUMPROCESS = system_cpus

	return


def setNormalization(norm, generateNormBW):
	global I_NORM
	global I_GENERATE_NormBW

	if(norm.lower() == 'false'):
		I_NORM = False
	else:
		I_NORM = True

	if(generateNormBW.lower() == 'false'):
		I_GENERATE_NormBW = False
	else:
		I_GENERATE_NormBW = True

	if((I_NORM == False) and (I_GENERATE_NormBW == True)):
		print("ERROR: I_NOMR should be 'True' if I_GENERATE_NormBW is 'True'")
		sys.exit()

	return
