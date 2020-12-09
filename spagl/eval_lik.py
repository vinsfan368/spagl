#!/usr/bin/env python
"""
eval_lik.py -- evaluate a likelihood function on a set of 
trajectories

"""
import numpy as np 
import pandas as pd 

# Calculate jumps for a set of trajectories
from .utils import tracks_to_jumps 

# Split a set of trajectories into smaller trajectories
from .utils import split_jumps

# Raw likelihood functions
from .lik import (
    gamma_likelihood,
    rbme_likelihood,
    fbm_likelihood
)

# Available raw likelihood functions
LIKELIHOODS = {
    "gamma": gamma_likelihood,
    "rbme": rbme_likelihood,
    "fbme": fbme_likelihood
}

# Convert trajectories into a jump-based ndarray format
from .utils import tracks_to_jumps

# Correct for defocalization
from .defoc import defoc_corr

def eval_likelihood(tracks, likelihood="gamma", splitsize=4,
    max_jumps_per_track=None, start_frame=None, pixel_size_um=0.16,
    frame_interval=0.00748, scale_by_jumps=True, dz=None, **kwargs):
    """
    args
    ----
        tracks              :   pandas.DataFrame
        likelihood          :   str, "gamma", "rbme", or "fbm"
        splitsize           :   int, the length of the subsampled
                                trajectories in number of jumps
        max_jumps_per_track :   int, 
        frame_interval      :   float, frame interval in seconds
        scale_by_jumps      :   bool, scale the likelihoods for 
                                each trajectory by the number of 
                                jumps in that trajectory
        dz                  :   focal depth in microns
        kwargs              :   to the likelihood function

    returns
    -------
        (
            ndarray, the likelihood function for each trajectory;
            1D ndarray, the number of jumps in each trajectory;
            1D ndarray, the (potentially split) indices of each
                trajectory;
            1D ndarray, the original indices of each trajectory;
            tuple of 1D ndarray, the parameters corresponding to 
                axes 1 onward for the likelihood
        )

    """
    # Calculate all the vector jumps in this set of trajectories
    jumps = tracks_to_jumps(tracks, n_frames=1, start_frame=start_frame, 
        pixel_size_um=pixel_size_um)

    # If desired, split the trajectories into smaller trajectories
    orig_track_indices = jumps[:,1].astype(np.int64)
    if (not splitsize is None) and (not splitsize is np.inf):
        jumps[:,1] = split_jumps(jumps, splitsize=splitsize)

    # Calculate likelihoods for each trajectory
    L, n_jumps, track_indices, support = LIKELIHOODS[likelihood](jumps,
        frame_interval=frame_interval, max_jumps_per_track=max_jumps_per_track,
        **kwargs)

    # Account for defocalization
    if (not dz is None) and (not dz is np.inf):
        L = defoc_corr(L, support, likelihood=likelihood, 
            frame_interval=frame_interval, dz=dz)

    # Scale by the number of jumps in each trajectory, if desired
    if scale_by_jumps:
        L = (L.T * n_jumps).T 

    return L, n_jumps, track_indices, orig_track_indices, support
