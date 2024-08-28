# Changelog

## Unreleased

## [0.30.1]

### Fixed

- CallPeak bug when calcuating analysis regions using a blacklist

## [0.30.0]

### Fixed

- CalculateCovariates didn't work if any bias types were left out

### Changed
- Major speed improvements (80% reduction) to covariate calculation when pcr is one of the bias types
- Brought covariate calculation code more in line with idiomatic python, making it easier to maintiain
- Minor speed improvements to peak calling
- Brought peak calling code more in line with idiomatic python, making it easier to maintain

## [0.29.0]

### Fixed

- CallPeak issue where mergePeak function would fail if no normalized bigwigs were
  included on the command line (they are optional).
- Issue in normalize code that broke getTrainSet. np.array_split not only splits an
  array, but also secretly changes the types in the array if it's heterogenous. Some ints
  were being converted to strings, and this made np.arange (and the author) very sad. Fortunately,
  our custom arraySplit code does no such nonsense.
- Issue in normalize code with getScaler. It expected the input to be a nx4 array, but it was being
  passed (correctly) an nx2 array.
- Adds `CRADLE/CalculateCovariates/covariateUtils.pyx` to setup.py list of ext_modules

### Changed

- Use np.int_ instead of np.long. np.long has been removed and the values should be nowhere near the
  max value of an int_ (an int64).

## [0.28.0]

### Changed

- Normalize command code changed to be more idiomatic python and more idiomatic numpy usage
- Tests added for normalize command
- Fixes some problems with region calculations in normalize command:

  * `mergeRegions` created 0-length overlaps. if two regions are next to
    each other they get merged, but they don't overlap. A zero-length
    overlap is created, which can cause a problem later when trying to
    get reads from the bigwig file.

  * region_overlap entries might have the wrong end.

    (line 80) `region_overlapped.append([currChromo, currStart, pastEnd])`

    is incorrect if currEnd is < pastEnd. Changed to

    `region_overlapped.append([currChromo, currStart, min(currEnd, pastEnd)])`

  * `excludeOverlapRegion` didn't fully separate region overlap cases.
    "overlap regions end inside the target region" and "overlap regions
    start inside the target region" can share some overlaps when one
    starts _and_ ends in the same region. This Fourth case is now
    handled separately.

  * in `excludeOverlapRegion` calculating non-overlapping regions had some subtle
    bugs.

## [0.27.0]

### Changed

- Default value for `correctBias` and `correctBias_stored` `-mi` argument is `5 * (number of samples)`. Previously is was just `number of samples`.

## [0.26.1]

### Fixed

- Typo of `ctrlWB` instead of `ctrlBW` in callPeak.py (https://github.com/ReddyLab/CRADLE/issues/91)
- C-style ternary operator instead of python-style in callPeak.py (https://github.com/ReddyLab/CRADLE/issues/91)
- Divide by zero error when calculating Cohen's D (https://github.com/ReddyLab/CRADLE/issues/92)
