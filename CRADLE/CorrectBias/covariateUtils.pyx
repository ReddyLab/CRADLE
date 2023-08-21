# cython: language_level=3

import math

import numpy as np


cpdef find5merProb(mer5, globalVars):
	baseInfo = 0
	subtract = -1

	for i in range(5):
		if mer5[i]=='A' or mer5[i]=='a':
			baseInfo = baseInfo + np.power(4, 4-i) * 0
		elif mer5[i]=='C' or mer5[i]=='c':
			baseInfo = baseInfo + np.power(4, 4-i) * 1
		elif mer5[i]=='G' or mer5[i]=='g':
			baseInfo = baseInfo + np.power(4, 4-i) * 2
		elif mer5[i]=='T' or mer5[i]=='t':
			baseInfo = baseInfo + np.power(4, 4-i) * 3

		if i==0:
			subtract = baseInfo

	mgw = globalVars["mgw"][baseInfo][1]
	prot = globalVars["prot"][baseInfo][1]

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef findComple5merProb(mer5, globalVars):
	baseInfo = 0
	subtract = -1

	for i in range(5):
		if mer5[i]=='A' or mer5[i]=='a':
			baseInfo = baseInfo + np.power(4, i) * 3
		elif mer5[i]=='C' or mer5[i]=='c':
			baseInfo = baseInfo + np.power(4, i) * 2
		elif mer5[i]=='G' or mer5[i]=='g':
			baseInfo = baseInfo + np.power(4, i) * 1
		elif mer5[i]=='T' or mer5[i]=='t':
			baseInfo = baseInfo + np.power(4, i) * 0

		if i==0:
			subtract = baseInfo

	mgw = globalVars["mgw"][baseInfo][1]
	prot = globalVars["prot"][baseInfo][1]

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef edit5merProb(pastMer, oldBase, newBase, globalVars):
	baseInfo = pastMer
	baseInfo = baseInfo * 4
	subtract = -1

	## newBase
	if newBase=='A' or newBase=='a':
		baseInfo = baseInfo + 0
	elif newBase=='C' or newBase=='c':
		baseInfo = baseInfo + 1
	elif newBase=='G' or newBase=='g':
		baseInfo = baseInfo + 2
	elif newBase=='T' or newBase=='t':
		baseInfo = baseInfo + 3

	baseInfo = int(baseInfo)
	mgw = globalVars["mgw"][baseInfo][1]
	prot = globalVars["prot"][baseInfo][1]

	## subtract oldBase
	if oldBase=='A' or oldBase=='a':
		subtract = np.power(4, 4) * 0
	elif oldBase=='C' or oldBase=='c':
		subtract = np.power(4, 4) * 1
	elif oldBase=='G' or oldBase=='g':
		subtract = np.power(4, 4) * 2
	elif oldBase=='T' or oldBase=='t':
		subtract = np.power(4, 4) * 3

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef editComple5merProb(pastMer, oldBase, newBase, globalVars):
	baseInfo = pastMer
	baseInfo = int(baseInfo / 4)
	subtract = -1

	# newBase
	if newBase=='A' or newBase=='a':
		baseInfo = baseInfo + np.power(4, 4) * 3
	elif newBase=='C' or newBase=='c':
		baseInfo = baseInfo + np.power(4, 4) * 2
	elif newBase=='G' or newBase=='g':
		baseInfo = baseInfo + np.power(4, 4) * 1
	elif newBase=='T' or newBase=='t':
		baseInfo = baseInfo + np.power(4, 4) * 0

	baseInfo = int(baseInfo)
	mgw = globalVars["mgw"][baseInfo][1]
	prot = globalVars["prot"][baseInfo][1]

	## subtract oldBase
	if oldBase=='A' or oldBase=='a':
		subtract = 3
	elif oldBase=='C' or oldBase=='c':
		subtract = 2
	elif oldBase=='G' or oldBase=='g':
		subtract = 1
	elif oldBase=='T' or oldBase=='t':
		subtract = 0

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef findStartGibbs(seq, seqLen, globalVars):
	gibbs = 0
	subtract = -1

	for i in range(seqLen-1):
		dimer = seq[i:(i+2)].upper()
		if 'N' in dimer:
			gibbs = gibbs + globalVars["n_gibbs"]
		else:
			dimerIdx = 0
			for j in range(2):
				if dimer[j]=='A':
					dimerIdx = dimerIdx + np.power(4, 1-j) * 0
				elif dimer[j]=='C':
					dimerIdx = dimerIdx + np.power(4, 1-j) * 1
				elif dimer[j]=='G':
					dimerIdx = dimerIdx + np.power(4, 1-j) * 2
				elif dimer[j]=='T':
					dimerIdx = dimerIdx + np.power(4, 1-j) * 3
			gibbs = gibbs + globalVars["gibbs"][dimerIdx][1]

		if i==0:
			subtract = gibbs

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


cpdef editStartGibbs(oldDimer, newDimer, pastStartGibbs, globalVars):
	gibbs = pastStartGibbs
	subtract = -1

	# newDimer
	if 'N' in newDimer:
		gibbs = gibbs + globalVars["n_gibbs"]
	else:
		dimerIdx = 0
		for j in range(2):
			if newDimer[j]=='A':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 0
			elif newDimer[j]=='C':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 1
			elif newDimer[j]=='G':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 2
			elif newDimer[j]=='T':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 3
		gibbs = gibbs + globalVars["gibbs"][dimerIdx][1]

	## remove the old dimer for the next iteration
	if 'N' in oldDimer:
		subtract = globalVars["n_gibbs"]
	else:
		dimerIdx = 0
		for j in range(2):
			if oldDimer[j]=='A':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 0
			elif oldDimer[j]=='C':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 1
			elif oldDimer[j]=='G':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 2
			elif oldDimer[j]=='T':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 3
		subtract = globalVars["gibbs"][dimerIdx][1]

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


cpdef convertGibbs(gibbs, globalVars):
	tm = gibbs / (globalVars["entropy"]*(globalVars["fragLen"]-1))
	tm = (tm - globalVars["min_tm"]) / (globalVars["max_tm"] - globalVars["min_tm"])

	## anneal
	anneal = ( math.exp(tm) - globalVars["para1"] ) * globalVars["para2"]
	anneal = np.log(anneal)

	## denature
	tm = tm - 1
	denature = ( math.exp( tm*(-1) ) - globalVars["para1"] ) * globalVars["para2"]
	denature = math.log(denature)

	return anneal, denature


cpdef memoryView(value):
	cdef double [:] arrayView
	arrayView = value

	return arrayView