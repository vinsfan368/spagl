#!/usr/bin/env python
"""
HYPERPARAMS.py -- hyperparameters for the methods in spagl

"""
import numpy as np 

# Default supports for likelihood functions
DIFF_COEFS_DEFAULT = np.logspace(-2.0, 2.0, 151)
# LOC_ERRORS_DEFAULT = np.arange(0.0, 0.102, 0.002)
LOC_ERRORS_DEFAULT = np.arange(0.02, 0.0625, 0.0025)
HURST_PARS_DEFAULT = np.arange(0.05, 1.0, 0.05)

# When trajectories are too long, split them into smaller trajectories.
# *splitsize* defines the maximum trajectory length (in # jumps) before
# splitting.
DEFAULT_SPLITSIZE = 8

# For the fixed state sampler, the minimum number of pseudocounts to 
# use per state. This is a safety feature, intended to prevent aberrant
# detection of subpopulations when using low numbers of trajectories.
MIN_PSEUDOCOUNTS = 2.0

# The number of pseudocounts per state in the fixed state sampler, 
# expressed as a fraction of the total number of trajectories (after
# splitting).
DEFAULT_PSEUDOCOUNT_FRAC = 0.00001
