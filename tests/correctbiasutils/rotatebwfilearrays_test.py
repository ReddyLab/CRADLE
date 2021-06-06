from CRADLE.correctbiasutils import rotateBWFileArrays

def testRotateBWFileArrays():
	inputArray = [
	  # Results of Job 0
		[['ctrlFile0Temp0.msl', 'ctrlFile1Temp0.msl'], ['experiFile0Temp0.msl', 'experiFile1Temp0.msl'], ],
	  # Results of Job 1
		[['ctrlFile0Temp1.msl', 'ctrlFile1Temp1.msl'], ['experiFile0Temp1.msl', 'experiFile1Temp1.msl'], ],
	  # Results of Job 2
		[['ctrlFile0Temp2.msl', 'ctrlFile1Temp2.msl'], ['experiFile0Temp2.msl', 'experiFile1Temp2.msl'], ],
	]

	output = (
		[
			['ctrlFile0Temp0.msl', 'ctrlFile0Temp1.msl', 'ctrlFile0Temp2.msl'],
			['ctrlFile1Temp0.msl', 'ctrlFile1Temp1.msl', 'ctrlFile1Temp2.msl']
		],
		[
			['experiFile0Temp0.msl', 'experiFile0Temp1.msl', 'experiFile0Temp2.msl'],
			['experiFile1Temp0.msl', 'experiFile1Temp1.msl', 'experiFile1Temp2.msl']
		],
	)

	assert rotateBWFileArrays(inputArray, ['ctrlFile0', 'ctrlFile1'], ['experiFile0', 'experiFile1']) == output
