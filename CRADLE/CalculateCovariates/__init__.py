import warnings

### supress numpy nan-error message
warnings.filterwarnings('ignore', r'All-NaN slice encountered')
warnings.filterwarnings('ignore', r'Mean of empty slice')
