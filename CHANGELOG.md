# Changelog

## [Unreleased]

### Changed

- Default value for `correctBias` and `correctBias_stored` `-mi` argument is `5 * (number of samples)`. Previously is was just `number of samples`.

## [0.26.1]

### Fixed

- Typo of `ctrlWB` instead of `ctrlBW` in callPeak.py (https://github.com/ReddyLab/CRADLE/issues/91)
- C-style ternary operator instead of python-style in callPeak.py (https://github.com/ReddyLab/CRADLE/issues/91)
- Divide by zero error when calculating Cohen's D (https://github.com/ReddyLab/CRADLE/issues/92)
