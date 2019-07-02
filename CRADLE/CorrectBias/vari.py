import os
import math
import numpy as np
import sys
import multiprocessing

def setGlobalVariables(args):
	
	### input bigwig files
	setInputFiles(args.ctrlbw, args.expbw)
	setOutputDirectory(args.o)
	setBiasFiles(args)
	setFragLen(args.l)
	setAnlaysisRegion(args.r, args.bl)
	setFilterCriteria(args.mi)
	setBinSize(args.binSize)
	setNumProcess(args.p)

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

	OUTPUT_DIR = outputDir
	os.makedirs(OUTPUT_DIR)

	return


def setBiasFiles(args):
	global SHEAR   # indicator variable whether each type of bias will be corrected.
	global PCR
	global MAP
	global GQUAD
	global COVARI_NUM
	global FA

	SHEAR = 0
	PCR = 0
	MAP = 0
	GQUAD = 0

	FA = args.faFile

	COVARI_NUM = 0
	if('shear' in args.biasType):
		SHEAR = 1
		COVARI_NUM = COVARI_NUM + 2

		global MGW
		global PROT
		global N_MGW
		global N_PROT

		#######  MGW
		MGW = []

		input_filename = os.getcwd() + "/inputFiles/mgw"
		input_stream = open(input_filename)
		input_file = input_stream.readlines()
		for i in range(len(input_file)):
                	temp = input_file[i].split()
                	temp[1] = float(temp[1])
                	MGW.append(temp)
		input_stream.close()

		temp = np.array(MGW)
		N_MGW = temp[:,1].astype(float).min()		

		#######  ProT
		PROT = []

		input_filename = os.getcwd() + "/inputFiles/prot"
                input_stream = open(input_filename)
		input_file = input_stream.readlines()
                for i in range(len(input_file)):
                        temp = input_file[i].split()
                        temp[1] = float(temp[1])
                        PROT.append(temp)
		input_stream.close()

		temp = np.array(PROT)
		N_PROT = temp[:,1].astype(float).min()

	if('pcr' in args.biasType):
		PCR = 1
		COVARI_NUM = COVARI_NUM + 2 # ANNEAL & DENATURE

        	global GIBBS
		global ENTROPY
		global MIN_TM
		global MAX_TM
		global PARA1
		global PARA2
		global N_GIBBS

	        GIBBS = []

                input_filename = os.getcwd() + "/inputFiles/gibbs"
                input_stream = open(input_filename)
		input_file = input_stream.readlines()
                for i in range(len(input_file)):
                        temp = input_file[i].split()
                        temp[1] = float(temp[1])
                        GIBBS.append(temp)
		input_stream.close()

		ENTROPY = -0.02485
		temp = np.array(GIBBS)
	        MIN_TM = -0.12 / ENTROPY
        	MAX_TM = -2.7 / ENTROPY
        	PARA1 = (math.pow(10, 6) - math.exp(1)) / (math.pow(10, 6) - 1)
        	PARA2 =  math.pow(10, -6) / (1-PARA1)
        	N_GIBBS = np.median(temp[:,1].astype(float))

	if('map' in args.biasType):
		MAP = 1
		COVARI_NUM = COVARI_NUM + 1

		global MAPFILE
		global KMER

		if(args.mapFile == None):
			print("ERROR: No map file was specified !")
			sys.exit()
		if(args.kmer == None):
			print("ERROR: No kmer parameter was specified !")
                        sys.exit()
	
		MAPFILE = args.mapFile
		KMER = int(args.kmer)

	if('gquad' in args.biasType):
		GQUAD = 1
		COVARI_NUM = COVARI_NUM + 1
		
		global GQAUDFILE
		global GQAUD_MAX

		guadFileNum = len(args.gquadFile)
		
		if(guadFileNum == 0):
			print("ERROR: No g-quadruplex file was specified !")
                        sys.exit()

		GQAUDFILE = [0] * guadFileNum
		for i in range(guadFileNum):
			GQAUDFILE[i] = args.gquadFile[i]
		GQAUD_MAX = args.gquadMax

	return


def setFragLen(fragLen):
	global FRAGLEN
	FRAGLEN = int(fragLen)

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
                                region_merged[resultIdx][2] = currEnd
                                pos = pos + 1
                                pastChromo = currChromo
                                pastStart = currStart
                                pastEnd = currEnd
                        else:
                                region_merged.append([currChromo, currStart, currEnd])
                                resultIdx = resultIdx + 1
                                pos = pos + 1
                                pastChromo = currChromo
                                pastStart = currStart
                                pastEnd = currEnd

                REGION = region_merged

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
                        idx = np.where( (bl_region[:,0] == regionChromo) & (bl_region[:,1].astype(int) >= regionStart) & (bl_region[:,1].astype(int) < regionEnd) )[0]
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

	print("Anlaysis Region:")
        print(REGION)


	return


def setFilterCriteria(minFrag):
	global FILTERVALUE

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


def setBinSize(binSize):
	global BINSIZE

	BINSIZE = int(binSize)

	return

def setNumProcess(numProcess):
	global NUMPROCESS
	
	system_cpus = int(multiprocessing.cpu_count())

	if(numProcess == None):
		NUMPROCESS = int(system_cpus / 2.0 )
	else:
		NUMPROCESS = int(numProcess)

	if(NUMPROCESS > system_cpus):
		print("ERROR: You specified too many cpus!")
		sys.exit()

	return
