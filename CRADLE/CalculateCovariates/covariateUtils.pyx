# cython: language_level=3
# cython: profile=True

import math

import numpy as np

cimport cython

from functools import lru_cache
from CRADLE.CalculateCovariates import vari



cpdef find5merProb(mer5, globalVars):
	cdef int baseInfo = 0
	cdef int subtract = -1

	if mer5[0] == 65:
		pass  # baseInfo = baseInfo
	elif mer5[0] == 67:
		baseInfo += 256
	elif mer5[0] == 71:
		baseInfo += 512
	elif mer5[0] == 84:
		baseInfo += 768

	subtract = baseInfo

	if mer5[1] == 65:
		pass  # baseInfo = baseInfo
	elif mer5[1] == 67:
		baseInfo += 64
	elif mer5[1] == 71:
		baseInfo += 128
	elif mer5[1] == 84:
		baseInfo += 192

	if mer5[2] == 65:
		pass  # baseInfo = baseInfo
	elif mer5[2] == 67:
		baseInfo += 16
	elif mer5[2] == 71:
		baseInfo += 32
	elif mer5[2] == 84:
		baseInfo += 48

	if mer5[3] == 65:
		pass  # baseInfo = baseInfo
	elif mer5[3] == 67:
		baseInfo += 4
	elif mer5[3] == 71:
		baseInfo += 8
	elif mer5[3] == 84:
		baseInfo += 12

	if mer5[4] == 65:
		pass  # baseInfo = baseInfo
	elif mer5[4] == 67:
		baseInfo += 1
	elif mer5[4] == 71:
		baseInfo += 2
	elif mer5[4] == 84:
		baseInfo += 3


	mgw = globalVars["mgw"][baseInfo][1]
	prot = globalVars["prot"][baseInfo][1]

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef findComple5merProb(mer5, globalVars):
	cdef int baseInfo = 0
	cdef int subtract = -1

	if mer5[0] == 65:
		baseInfo += 3
	elif mer5[0] == 67:
		baseInfo += 2
	elif mer5[0] == 71:
		baseInfo += 1
	elif mer5[0] == 84:
		pass  # baseInfo = baseInfo

	subtract = baseInfo

	if mer5[1] == 65:
		baseInfo += 12
	elif mer5[1] == 67:
		baseInfo += 8
	elif mer5[1] == 71:
		baseInfo += 4
	elif mer5[1] == 84:
		pass  # baseInfo = baseInfo

	if mer5[2] == 65:
		baseInfo += 48
	elif mer5[2] == 67:
		baseInfo += 32
	elif mer5[2] == 71:
		baseInfo += 16
	elif mer5[2] == 84:
		pass  # baseInfo = baseInfo

	if mer5[3] == 65:
		baseInfo += 192
	elif mer5[3] == 67:
		baseInfo += 128
	elif mer5[3] == 71:
		baseInfo += 64
	elif mer5[3] == 84:
		pass  # baseInfo = baseInfo

	if mer5[4] == 65:
		baseInfo += 768
	elif mer5[4] == 67:
		baseInfo += 512
	elif mer5[4] == 71:
		baseInfo += 256
	elif mer5[4] == 84:
		pass  # baseInfo = baseInfo

	mgw = globalVars["mgw"][baseInfo][1]
	prot = globalVars["prot"][baseInfo][1]

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef edit5merProb(pastMer, oldBase, newBase, globalVars):
	cdef int baseInfo = pastMer * 4
	cdef int subtract = -1

	## newBase
	if newBase == 65: # A
		pass  # baseInfo = baseInfo
	elif newBase == 67: # C
		baseInfo += 1
	elif newBase == 71: # T
		baseInfo += 2
	elif newBase == 84: # G
		baseInfo += 3

	mgw = globalVars["mgw"][baseInfo][1]
	prot = globalVars["prot"][baseInfo][1]

	## subtract oldBase
	cdef int power = 256 # 4^4
	if oldBase == 65:
		subtract = 0  # power * 0
	elif oldBase == 67:
		subtract = power  # * 1
	elif oldBase == 71:
		subtract = power * 2
	elif oldBase == 84:
		subtract = power * 3

	cdef int nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef editComple5merProb(pastMer, oldBase, newBase, globalVars):
	cdef int baseInfo = pastMer
	baseInfo = baseInfo // 4
	cdef int subtract = -1

	cdef int power = 256 # 4^4
	# newBase
	if newBase == 65:
		baseInfo += power * 3
	elif newBase == 67:
		baseInfo += power * 2
	elif newBase == 71:
		baseInfo += power  # * 1
	elif newBase == 84:
		pass  # baseInfo = baseInfo

	mgw = globalVars["mgw"][baseInfo][1]
	prot = globalVars["prot"][baseInfo][1]

	## subtract oldBase
	if oldBase == 65:
		subtract = 3
	elif oldBase == 67:
		subtract = 2
	elif oldBase == 71:
		subtract = 1
	elif oldBase == 84:
		subtract = 0

	cdef int nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef editStartGibbs(oldDimer, newDimer, pastStartGibbs, globalVars):
	cdef int n_gibbs = globalVars["n_gibbs"]
	cdef float gibbs = pastStartGibbs
	cdef float subtract = -1
	cdef int dimerIdx = 0

	gibbs_nums = globalVars["gibbs"]

	# newDimer
	if 78 in newDimer: # N
		gibbs += n_gibbs
	else:
		dimerIdx = 0
		if newDimer[0] == 65:
			pass  # dimerIdx = dimerIdx
		elif newDimer[0] == 67:
			dimerIdx += 4
		elif newDimer[0] == 71:
			dimerIdx += 8
		elif newDimer[0] == 84:
			dimerIdx += 12

		if newDimer[1] == 65:
			pass  # dimerIdx = dimerIdx
		elif newDimer[1] == 67:
			dimerIdx += 1
		elif newDimer[1] == 71:
			dimerIdx += 2
		elif newDimer[1] == 84:
			dimerIdx += 3
		gibbs = gibbs + gibbs_nums[dimerIdx][1]

	## remove the old dimer for the next iteration
	if 78 in oldDimer:
		subtract = n_gibbs
	else:
		dimerIdx = 0
		if oldDimer[0] == 65:
			pass  # dimerIdx = dimerIdx
		elif oldDimer[0] == 67:
			dimerIdx += 4
		elif oldDimer[0] == 71:
			dimerIdx += 8
		elif oldDimer[0] == 84:
			dimerIdx += 12

		if oldDimer[1] == 65:
			pass  # dimerIdx = dimerIdx
		elif oldDimer[1] == 67:
			dimerIdx += 1
		elif oldDimer[1] == 71:
			dimerIdx += 2
		elif oldDimer[1] == 84:
			dimerIdx += 3
		subtract = gibbs_nums[dimerIdx][1]

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


cpdef convertGibbs(gibbs, globalVars):
	tm = gibbs / (globalVars["entropy"] * (globalVars["fragLen"] - 1))
	tm = (tm - globalVars["min_tm"]) / (globalVars["max_tm"] - globalVars["min_tm"])

	## anneal
	anneal = ( math.exp(tm) - globalVars["para1"] ) * globalVars["para2"]
	anneal = np.log(anneal)

	## denature
	tm -= 1
	denature = ( math.exp( tm*(-1) ) - globalVars["para1"] ) * globalVars["para2"]
	denature = math.log(denature)

	return anneal, denature
