#!/usr/bin/env python
"""
fss.py -- fixed state samplers for pure diffusion models

"""
import os
import sys

# Progress bar
from tqdm import tqdm 

# Numeric
import numpy as np 

# DataFrames
import pandas as pd 

# Digamma function
from scipy.special import digamma

# Evaluate likelihood functions on trajectories
from .eval_lik import eval_likelihood

# Defocalization probabilities
from .defoc import defoc_corr 

# The absolute minimum number of pseudocounts to use. Users
# should always work with large sample sizes - preferably 
# greater than 10000 trajectories. This is a safety feature 
# against aberrant state estimations
MIN_PSEUDOCOUNTS = 10.0

def fss(tracks, likelihood="rbme_marginal", splitsize=12, max_jumps_per_track=None,
    start_frame=None, pixel_size_um=0.16, frame_interval=0.00748, 
    dz=None, max_iter=1000, pseudocount_frac=0.005, convergence=0,
    **likelihood_kwargs):
    """
    Use a fixed state sampler (FSS) to estimate the underlying occupancies
    for a pure diffusive mixture, given a set of trajectories and a 
    particular diffusion model.

    args
    ----
        tracks          :   pandas.DataFrame, the set of input trajectories

        likelihood      :   str, the type of likelihood to calculate 

        splitsize       :   int. If *None*, the original trajectories are used.
                            If set, then trajectories are split into 
                            subtrajectories that have a maximum of *splitsize*
                            jumps.

        max_jumps_per_track :   int, the maximum number of jumps to consider
                            per trajectory.

        start_frame     :   int, disregard trajectories before this frame

        pixel_size_um   :   float, width of pixels in microns

        frame_interval  :   float, time between frames in seconds

        dz              :   float, focal depth in microns

        max_iter        :   int, the maximum number of iterations to do 

        pseudocount_frac:   float, the relative weight of the prior against 
                            the data, expressed as a fraction of the number of 
                            input trajectories. The prior is never allowed to 
                            have fewer than 2 pseudocounts per state

        convergence     :   float, convergence criterion for the posterior
                            Dirichlet parameter

        likelihood_kwargs   :   additional keyword arguments to the likelihood
                            function

    returns
    -------
        (
            R : ndarray, the posterior probabilities over the state 
                assignments. R[i,j,...] is the posterior probability for
                trajectory (i) to inhabit state (j, ...);

            n : ndarray, the parameter for the Dirichlet posterior 
                distribution over state occupancies;

            mean_occs : ndarray, the mean state occupancy under the 
                posterior distribution;

            n_jumps : 1D ndarray, the number of jumps in each trajectory;

            track_indices : 1D ndarray, the indices of each trajectory in
                the origin *tracks* DataFrame;

            support : tuple of ndarray, the parameter values corresponding
                to each bin
        )

    """
    # Evaluate the prior likelihood function on the set of trajectories
    L, n_jumps, track_indices, support = eval_likelihood(tracks,
        likelihood=likelihood, splitsize=splitsize, start_frame=start_frame,
        max_jumps_per_track=max_jumps_per_track, pixel_size_um=pixel_size_um,
        frame_interval=frame_interval, scale_by_jumps=False, dz=dz, 
        **likelihood_kwargs)

    # Posterior estimate over the state assignments
    L = L.T
    R = L.copy()

    # Axes in *R* corresponding to parameters for this likelihood
    par_indices = tuple([i for i in range(len(R.shape)-1)])   

    # Prior over state occupancies. Do not use fewer than *MIN_PSEUDOCOUNTS*
    # pseudocounts per state
    if likelihood_kwargs.get("verbose", False):
        print("\nCalculated concentration parameter: ", (R.shape[-1] * pseudocount_frac))
    pseudocounts = int(max(R.shape[-1] * pseudocount_frac, MIN_PSEUDOCOUNTS))
    prior = np.ones(R.shape[:-1], dtype=np.float64) * pseudocounts   

    # Previous Dirichlet prior estimate, to check for convergence
    m_prev = np.zeros(R.shape[:-1], dtype=np.float64)

    # Defocalization correction factors
    corr = np.zeros(L.shape[:-1], dtype=np.float64)

    # Iterate until convergence or until *max_iter* is reached
    for iter_idx in tqdm(range(max_iter)):

        # Update the posterior occupancy distribution (*n* is the 
        # parameter to a Dirichlet distribution over occupancies)
        n = (R * n_jumps).sum(axis=-1)
        m = n + prior 

        # Exponential of the expected log occupancies under the 
        # current posterior model
        exp_log_tau = np.exp(digamma(m))

        # Calculate posterior probabilities over the state assignments
        # and normalize over all states for each trajectory
        R = (L.T * exp_log_tau.T).T
        R = R / R.sum(axis=par_indices)

        # Check for convergence
        change = m - m_prev 
        if (np.abs(change) < convergence).all():
            break
        else:
            m_prev[:] = m[:]

    # Adjust posterior distribution to account for defocalization probabilities
    n = n.T 
    n = defoc_corr(n, support, likelihood=likelihood, 
        frame_interval=frame_interval, dz=dz)

    # Calculate the mean occupancies under the posterior model
    mean_occs = n / n.sum()

    return R.T, n, mean_occs, n_jumps, track_indices, support
