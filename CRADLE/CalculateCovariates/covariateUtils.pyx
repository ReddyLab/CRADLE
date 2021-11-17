# cython: language_level=3

import math

import numpy as np

from functools import lru_cache
from CRADLE.CalculateCovariates import vari



@lru_cache(maxsize=1_024)
def find5merProb(mer5):
	cdef int baseInfo = 0
	cdef int subtract = -1

	if mer5[0] == 'A':
		pass  # baseInfo = baseInfo
	elif mer5[0] == 'C':
		baseInfo += 256
	elif mer5[0] == 'G':
		baseInfo += 512
	elif mer5[0] == 'T':
		baseInfo += 768

	subtract = baseInfo

	if mer5[1] == 'A':
		pass  # baseInfo = baseInfo
	elif mer5[1] == 'C':
		baseInfo += 64
	elif mer5[1] == 'G':
		baseInfo += 128
	elif mer5[1] == 'T':
		baseInfo += 192

	if mer5[2] == 'A':
		pass  # baseInfo = baseInfo
	elif mer5[2] == 'C':
		baseInfo += 16
	elif mer5[2] == 'G':
		baseInfo += 32
	elif mer5[2] == 'T':
		baseInfo += 48

	if mer5[3] == 'A':
		pass  # baseInfo = baseInfo
	elif mer5[3] == 'C':
		baseInfo += 4
	elif mer5[3] == 'G':
		baseInfo += 8
	elif mer5[3] == 'T':
		baseInfo += 12

	if mer5[4] == 'A':
		pass  # baseInfo = baseInfo
	elif mer5[4] == 'C':
		baseInfo += 1
	elif mer5[4] == 'G':
		baseInfo += 2
	elif mer5[4] == 'T':
		baseInfo += 3


	mgw = vari.MGW[baseInfo][1]
	prot = vari.PROT[baseInfo][1]

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


@lru_cache(maxsize=1_024)
def findComple5merProb(mer5):
	cdef int baseInfo = 0
	cdef int subtract = -1

	if mer5[0] == 'A':
		baseInfo += 3
	elif mer5[0] == 'C':
		baseInfo += 2
	elif mer5[0] == 'G':
		baseInfo += 1
	elif mer5[0] == 'T':
		pass  # baseInfo = baseInfo

	subtract = baseInfo

	if mer5[1] == 'A':
		baseInfo += 12
	elif mer5[1] == 'C':
		baseInfo += 8
	elif mer5[1] == 'G':
		baseInfo += 4
	elif mer5[1] == 'T':
		pass  # baseInfo = baseInfo

	if mer5[2] == 'A':
		baseInfo += 48
	elif mer5[2] == 'C':
		baseInfo += 32
	elif mer5[2] == 'G':
		baseInfo += 16
	elif mer5[2] == 'T':
		pass  # baseInfo = baseInfo

	if mer5[3] == 'A':
		baseInfo += 192
	elif mer5[3] == 'C':
		baseInfo += 128
	elif mer5[3] == 'G':
		baseInfo += 64
	elif mer5[3] == 'T':
		pass  # baseInfo = baseInfo

	if mer5[4] == 'A':
		baseInfo += 768
	elif mer5[4] == 'C':
		baseInfo += 512
	elif mer5[4] == 'G':
		baseInfo += 256
	elif mer5[4] == 'T':
		pass  # baseInfo = baseInfo

	mgw = vari.MGW[baseInfo][1]
	prot = vari.PROT[baseInfo][1]

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


@lru_cache(maxsize=4_096)
def edit5merProb(pastMer, oldBase, newBase):
	cdef int baseInfo = pastMer * 4
	cdef int subtract = -1

	## newBase
	if newBase == 'A':
		pass  # baseInfo = baseInfo
	elif newBase == 'C':
		baseInfo += 1
	elif newBase == 'G':
		baseInfo += 2
	elif newBase == 'T':
		baseInfo += 3

	mgw = vari.MGW[baseInfo][1]
	prot = vari.PROT[baseInfo][1]

	## subtract oldBase
	cdef int power = 256 # 4^4
	if oldBase == 'A':
		subtract = 0  # power * 0
	elif oldBase == 'C':
		subtract = power  # * 1
	elif oldBase == 'G':
		subtract = power * 2
	elif oldBase == 'T':
		subtract = power * 3

	cdef int nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


@lru_cache(maxsize=65_536)
def editComple5merProb(pastMer, oldBase, newBase):
	cdef int baseInfo = pastMer
	baseInfo = baseInfo // 4
	cdef int subtract = -1

	cdef int power = 256 # 4^4
	# newBase
	if newBase == 'A':
		baseInfo += power * 3
	elif newBase == 'C':
		baseInfo += power * 2
	elif newBase == 'G':
		baseInfo += power  # * 1
	elif newBase == 'T':
		pass  # baseInfo = baseInfo

	mgw = vari.MGW[baseInfo][1]
	prot = vari.PROT[baseInfo][1]

	## subtract oldBase
	if oldBase == 'A':
		subtract = 3
	elif oldBase == 'C':
		subtract = 2
	elif oldBase == 'G':
		subtract = 1
	elif oldBase == 'T':
		subtract = 0

	cdef int nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


@lru_cache(maxsize=1_024)
def findStartGibbs(seq, seqLen):
	gibbs = 0
	cdef int subtract = -1
	cdef int dimerIdx = 0

	for i in range(seqLen-1):
		dimer = seq[i:(i+2)].upper()
		if 'N' in dimer:
			gibbs = gibbs + vari.N_GIBBS
		else:
			dimerIdx = 0
			if dimer[0] == 'A':
				pass  # dimerIdx = dimerIdx
			elif dimer[0] == 'C':
				dimerIdx += 4
			elif dimer[0] == 'G':
				dimerIdx += 8
			elif dimer[0] == 'T':
				dimerIdx += 12

			if dimer[1] == 'A':
				pass  # dimerIdx = dimerIdx
			elif dimer[1] == 'C':
				dimerIdx += 1
			elif dimer[1] == 'G':
				dimerIdx += 2
			elif dimer[1] == 'T':
				dimerIdx += 3

			gibbs = gibbs + vari.GIBBS[dimerIdx][1]

		if i == 0:
			subtract = gibbs

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


@lru_cache(maxsize=4_096)
def editStartGibbs(oldDimer, newDimer, pastStartGibbs):
	gibbs = pastStartGibbs
	cdef int subtract = -1
	cdef int dimerIdx = 0

	# newDimer
	if 'N' in newDimer:
		gibbs += vari.N_GIBBS
	else:
		dimerIdx = 0
		if newDimer[0] == 'A':
			pass  # dimerIdx = dimerIdx
		elif newDimer[0] == 'C':
			dimerIdx += 4
		elif newDimer[0] == 'G':
			dimerIdx += 8
		elif newDimer[0] == 'T':
			dimerIdx += 12

		if newDimer[1] == 'A':
			pass  # dimerIdx = dimerIdx
		elif newDimer[1] == 'C':
			dimerIdx += 1
		elif newDimer[1] == 'G':
			dimerIdx += 2
		elif newDimer[1] == 'T':
			dimerIdx += 3
		gibbs = gibbs + vari.GIBBS[dimerIdx][1]

	## remove the old dimer for the next iteration
	if 'N' in oldDimer:
		subtract = vari.N_GIBBS
	else:
		dimerIdx = 0
		if oldDimer[0] == 'A':
			pass  # dimerIdx = dimerIdx
		elif oldDimer[0] == 'C':
			dimerIdx += 4
		elif oldDimer[0] == 'G':
			dimerIdx += 8
		elif oldDimer[0] == 'T':
			dimerIdx += 12

		if oldDimer[1] == 'A':
			pass  # dimerIdx = dimerIdx
		elif oldDimer[1] == 'C':
			dimerIdx += 1
		elif oldDimer[1] == 'G':
			dimerIdx += 2
		elif oldDimer[1] == 'T':
			dimerIdx += 3
		subtract = vari.GIBBS[dimerIdx][1]

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


@lru_cache(maxsize=1_024)
def convertGibbs(gibbs):
	tm = gibbs / (vari.ENTROPY*(vari.FRAGLEN-1))
	tm = (tm - vari.MIN_TM) / (vari.MAX_TM - vari.MIN_TM)

	## anneal
	anneal = ( math.exp(tm) - vari.PARA1 ) * vari.PARA2
	anneal = np.log(anneal)

	## denature
	tm -= 1
	denature = ( math.exp( tm*(-1) ) - vari.PARA1 ) * vari.PARA2
	denature = math.log(denature)

	return anneal, denature


cpdef memoryView(value):
	cdef double [:] arrayView
	arrayView = value

	return arrayView
