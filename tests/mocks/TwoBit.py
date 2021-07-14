class TwoBit:
	'''A mock for the py2bit data structure. Initialize with a dictionary of
	chromosome/sequence key/value pairs. E.g.:
	{
		"chr1": "actgtcgattcgctctcgatatagcatagctac",
		"chr2": "tctcgatcgctctcgcgctagagatccgag"
	 }
	'''
	def __init__(self, chromosomes):
		self.chromosomes = chromosomes

	def chroms(self, chrom=None):
		if chrom is None:
			return {k: len(v) for k, v in self.chromosomes.items()}
		else:
			sequence = self.chromosomes.get(chrom, None)
			return None if sequence is None else len(sequence)

	def sequence(self, chrom, start=None, end=None):
		sequence = self.chromosomes.get(chrom, None)
		if sequence is None:
			raise RuntimeError(f"Chromosome {chrom} doesn't exist in 2bit file.")

		if start is None and end is None:
			return sequence

		if start > end or start >= len(sequence) or start < 0:
			raise RuntimeError("Start is greater than end, less than 0, or past the end of the sequence.")
		if end > len(sequence):
			raise RuntimeError("End is greater than the length of the sequence.")

		return sequence[start:end]
