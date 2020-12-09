#!/usr/bin/env python
"""
lik.py -- trajectory likelihood functions

"""
import numpy as np 
import pandas as pd 

# Special functions for likelihood definitions
from scipy.special import gamma, gammainc, expi

# Calculate the sum of squared jumps for a set of trajectories
from .utils import (
    sum_squared_jumps
)

# Default supports for the likelihood functions
DIFF_COEFS_DEFAULT = np.logspace(-2.0, 2.0, 301)
LOC_ERRORS_DEFAULT = np.arange(0.0, 0.102, 0.002)
HURST_PARS_DEFAULT = np.arange(0.05, 1.0, 0.05)

def gamma_likelihood(jumps, diff_coefs=None, max_jumps_per_track=None,
    n_dim=2, frame_interval=0.00748, loc_error=0.035, mode="point"):
    """
    Gamma approximation to the likelihood of a regular Brownian motion
    with localization error.

    args
    ----
        jumps                   :   2D ndarray, trajectories in jumps format
                                    as returned by *tracks_to_jumps*
        diff_coefs              :   1D ndarray, the diffusion coefficient
                                    values at which to evaluate the likelihood
                                    function. If *None*, this is the default
                                    scheme.
        max_jumps_per_track     :   int, the maximum number of jumps to 
                                    include per trajectory
        n_dim                   :   int, the number of spatial dimensions
        frame_interval          :   float, seconds
        loc_error               :   float, root 1D localization variance in microns
        mode                    :   str, "point" or "binned", the likelihood
                                    approximation to use

    returns
    -------
        (
            2D ndarray of shape (n_tracks, n_bins), the likelihood of 
                each diffusion coefficient bin for this trajectory;

            1D ndarray of shape (n_tracks,), the number of jumps per
                trajectory;

            1D ndarray of shape (n_tracks,), the trajectory indices 
                in the original *tracks* dataframe corresponding to 
                each split trajectory in the output likelihood;

            (
                1D ndarray of shape (n_diff_coefs,), the support
                    in microns squared per second
            )
        )

    """
    # Default diffusion coefficient scheme
    if diff_coefs is None:
        diff_coefs = DIFF_COEFS_DEFAULT
    else:
        diff_coefs = np.asarray(diff_coefs)

    # If passed empty input, return empty output
    if jumps.shape[0] == 0:
        return np.zeros((0, diff_coefs.shape[0]), dtype=np.float64), \
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), \
            (diff_coefs,)

    # Localization variance
    le2 = loc_error ** 2

    # For each trajectory, compute the sum of squared jumps
    S = sum_squared_jumps(jumps, max_jumps_per_track=max_jumps_per_track)

    # Number of unique trajectories in this dataset
    n_tracks = S["trajectory"].nunique()

    # Pointwise-evaluated gamma likelihood
    if mode == "point":

        # Number of points at which to evaluate the likelihood
        K = diff_coefs.shape[0]

        # Alpha parameter governing the gamma likelihood
        alpha = np.asarray(S["n_jumps"] * n_dim / 2.0)

        # Sum of squared jumps in each trajectory
        sum_r2 = np.asarray(S["sum_sq_jump"])

        # Log likelihoods for each diffusion coefficient, given each trajectory
        L = np.zeros((n_tracks, K), dtype=np.float64)

        for j in range(K):
            phi = 4 * (diff_coefs[j] * frame_interval + le2)
            L[:,j] = -(sum_r2 / phi) - alpha * np.log(phi)

        # Stable conversion to likelihood
        L = (L.T - L.max(axis=1)).T 
        L = np.exp(L)

    # Integrated gamma likelihood
    elif mode == "binned":

        # Number of likelihood bins to use
        K = diff_coefs.shape[0] - 1

        # Log likelihoods for each diffusion coefficient, given each trajectory
        L = np.zeros((n_tracks, K), dtype=np.float64)

        # Divide trajectories into doublets and non-doublets
        doublets = np.asarray(S["n_jumps"] == 1)
        sum_r2_doublets = np.asarray(S.loc[doublets, "sum_sq_jump"])
        sum_r2_nondoublets = np.asarray(S.loc[~doublets, "sum_sq_jump"])
        alpha_doublets = np.asarray(S.loc[doublets, "n_jumps"] * n_dim / 2.0)

        for j in range(K):

            # Spatial variance at each bin edge
            V0 = 4 * (diff_coefs[j] * frame_interval + le2)
            V1 = 4 * (diff_coefs[j+1] * frame_interval + le2)

            # Deal with doublets
            L[doublets, j] = expi(-sum_r2_doublets/V0) - expi(-sum_r2_doublets/V1)

            # Deal with everything else
            L[~doublets, j] = (
                gammainc(alpha_nondoublets-1, sum_r2_nondoublets/V0) - \
                gammainc(alpha_nondoublets-1, sum_r2_nondoublets/V1)
            ) / gamma(alpha_nondoublets-1)

    # Normalize
    L = (L.T / L.sum(axis=1)).T 

    return L, np.asarray(S["n_jumps"]), np.asarray(S["trajectory"]), (diff_coefs,)

def rbme_likelihood(jumps, diff_coefs=None, loc_errors=None, max_jumps_per_track=10,
    frame_interval=0.00748):
    """
    Likelihood function for the jumps of a regular Brownian motion 
    with localization error. This likelihood is a function of the
    diffusion coefficient and the localization error, so the likelihoods
    are evaluated on a grid of (diffusion coef., loc. error) for each 
    trajectory. 

    args
    ----
        jumps                   :   2D ndarray
        diff_coefs              :   1D ndarray, the set of diffusion coefficients at 
                                    which to evaluate the likelihoods. If *None*, a 
                                    default scheme is used.
        loc_errors              :   1D ndarray, the set of localization errors at which
                                    to evaluate the likelihoods. If *None*, a 
                                    default scheme is used.
        max_jumps_per_track     :   int, maximum number of jumps to use
                                    from each trajectory
        n_dim                   :   int, the number of spatial dimensions
        frame_interval          :   float, seconds

    returns
    -------
        (
            3D ndarray of shape (n_tracks, n_diff_coefs, n_loc_errors),
                the likelihoods of each state for each trajectory.
                These likelihoods are normalized to sum to 1 across all
                states for that trajectory;
            1D ndarray of shape (n_tracks), the number of jumps per 
                trajectory;
            1D ndarray of shape (n_tracks), the indices of each trajectory
                in the original dataframe;
            (
                1D ndarray of shape (n_states), the diffusion coefficients
                    corresponding to each state;
                1D ndarray of shape (n_states), the localization error 
                    corresponding to each state
            )
        )

    """
    # If not passed, default to static diffusion coefficient / localization 
    # error schemes
    if diff_coefs is None:
        diff_coefs = DIFF_COEFS_DEFAULT
    if loc_errors is None:
        loc_errors = LOC_ERRORS_DEFAULT 

    # Get the size of the likelihood matrix for each trajectory
    diff_coefs = np.asarray(diff_coefs)
    loc_errors = np.asarray(loc_errors)
    nD = diff_coefs.shape[0]
    nLE = loc_errors.shape[0]

    # If passed empty input, return empty output
    if jumps.shape[0] == 0:
        return np.zeros((0, nD, nLE), dtype=np.float64), np.zeros(0, dtype=np.int64), \
            np.zeros(0, dtype=np.int64), (diff_coefs, loc_errors)

    # Number of spatial dimensions in this data
    n_cols = jumps.shape[1]
    n_dim = n_cols - 4
    assert n_dim == 2, "Only two dimensional trajectories supported by rbme_likelihood"

    # Convert the jumps to DataFrame format
    J_cols = ["track_length", "trajectory", "frame", "sum_sq_jump", "y", "x"]
    J = pd.DataFrame(jumps, columns=J_cols)

    # Number of jumps in each trajectry
    J = J.join(
        J.groupby("trajectory").size().rename("n_jumps"),
        on="trajectory"
    )
    n_jumps = np.asarray(J.groupby("trajectory").size())

    # Number of unique trajectories
    n_tracks = J["trajectory"].nunique()

    # Origin trajectory indices
    track_indices = np.asarray(J.groupby("trajectory").apply(lambda j: j.name)).astype(np.int64)

    # The size of the largest covariance matrix to make
    if (not max_jumps_per_track is None) or (max_jumps_per_track is np.inf):
        max_jumps_per_track = min(max_jumps_per_track, J["n_jumps"].max())
    else:
        max_jumps_per_track = J["n_jumps"].max()

    # Generate the RBME jump covariance matrix
    def make_cov(D, loc_error, n):
        """
        args
        ----
            D           :   float, diffusion coefficient in
                            squared microns per second
            loc_error   :   float, 1D localization error in microns
            n           :   int, the number of jumps

        returns
        -------
            2D ndarray of shape (n, n), the covariance matrix

        """
        le2 = loc_error ** 2
        C = 2 * (D * frame_interval + le2) * np.identity(n)
        for i in range(n-1):
            C[i,i+1] = -le2 
            C[i+1,i] = -le2 
        return C 

    # The output likelihood matrix
    log_L = np.zeros((n_tracks, nD, nLE), dtype=np.float64)

    # Iterate through the different trajectory lengths
    for l in range(1, max_jumps_per_track+1):

        # Get all of the trajectories with this number of jumps
        subtracks = np.asarray(J.loc[
            J["n_jumps"] == l,
            ["y", "x"]
        ])

        # The number of trajectories in this set
        n_match = subtracks.shape[0] // l

        # The y- and x- jumps of each trajectory
        y_jumps = subtracks[:,0].reshape((n_match, l)).T 
        x_jumps = subtracks[:,1].reshape((n_match, l)).T 

        for i, D in enumerate(diff_coefs):
            for j, le in enumerate(loc_errors):

                # Generate the covariance matrix 
                C = make_cov(D, le, l)

                # Inverse covariance matrix
                C_inv = np.linalg.inv(C)

                # Normalization factor (slogdet is the log determinant)
                norm_fac = l * np.log(2 * np.pi) + np.linalg.slogdet(C)[1]

                # Evaluate the log likelihood for the y- and x-jumps
                y_ll = (y_jumps * (C_inv @ y_jumps)).sum(axis=0)
                x_ll = (x_jumps * (C_inv @ x_jumps)).sum(axis=0)

                # Combine the axes components
                log_L[n_jumps==l, i, j] = -0.5 * (y_ll + x_ll) - norm_fac 

    # Normalize over all states for each trajectory
    L = np.zeros(log_L.shape, dtype=np.float64)
    for t in range(n_tracks):
        log_L[t,:,:] -= log_L[t,:,:].max()
        L[t,:,:] = np.exp(log_L[t,:,:])
        L[t,:,:] /= L[t,:,:].sum()
    del log_L 

    return L, n_jumps, track_indices, (diff_coefs, loc_errors)

def fbme_likelihood(jumps, diff_coefs=None, hurst_pars=None, max_jumps_per_track=10,
    frame_interval=0.00748, loc_error=0.035):
    """
    Likelihood function for a fractional Brownian motion with localization
    error, assuming that the localization error is known in advance. This
    is a function of the diffusion coefficient and the Hurst parameter, so 
    the likelihood is evaluated on a grid of (diff. coef., Hurst parameter)
    for each trajectory.

    args
    ----
        jumps                   :   2D ndarray
        diff_coefs              :   1D ndarray, the set of diffusion coefficients at 
                                    which to evaluate the likelihoods. If *None*, a 
                                    default scheme is used.
        hurst_pars              :   1D ndarray, the set of Hurst parameters at which
                                    to evaluate the likelihood. If *NOne*, a default
                                    scheme is used. 
        loc_error               ;   float, standard deviation of 1D normally distributed
                                    localization error in the experiment, assumed
                                    to be constant
        max_jumps_per_track     :   int, maximum number of jumps to use
                                    from each trajectory
        n_dim                   :   int, the number of spatial dimensions
        frame_interval          :   float, seconds

    returns
    -------
        (
            3D ndarray of shape (n_tracks, n_diff_coefs, n_hurst_pars),
                the likelihoods of each state for each trajectory.
                These likelihoods are normalized to sum to 1 across all
                states for that trajectory;
            1D ndarray of shape (n_tracks), the number of jumps per 
                trajectory;
            1D ndarray of shape (n_tracks), the indices of each trajectory
                in the original dataframe;
            (
                1D ndarray of shape (n_states), the diffusion coefficients
                    corresponding to each state;
                1D ndarray of shape (n_states), the Hurst parameters
                    corresponding to each state
            )
        )

    """
    # If not passed, default to static diffusion coefficient / Hurst
    # parameter schemes
    if diff_coefs is None:
        diff_coefs = DIFF_COEFS_DEFAULT
    if hurst_pars is None:
        hurst_pars = HURST_PARS_DEFAULT

    # Get the size of the likelihood matrix for each trajectory
    diff_coefs = np.asarray(diff_coefs)
    hurst_pars = np.asarray(hurst_pars)
    nD = diff_coefs.shape[0]
    nH = hurst_pars.shape[0]

    # If passed empty input, return empty output
    if jumps.shape[0] == 0:
        return np.zeros((0, nD, nH), dtype=np.float64), \
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), \
            (diff_coefs, hurst_pars)

    # 1D localization variance
    le2 = loc_error ** 2

    # Number of spatial dimensions in this data
    n_cols = jumps.shape[1]
    n_dim = n_cols - 4
    assert n_dim == 2, "Only two dimensional trajectories supported by rbme_likelihood"

    # Convert the jumps to DataFrame format
    J_cols = ["track_length", "trajectory", "frame", "sum_sq_jump", "y", "x"]
    J = pd.DataFrame(jumps, columns=J_cols)

    # Number of jumps in each trajectry
    J = J.join(
        J.groupby("trajectory").size().rename("n_jumps"),
        on="trajectory"
    )
    n_jumps = np.asarray(J.groupby("trajectory").size())

    # Number of unique trajectories
    n_tracks = J["trajectory"].nunique()

    # Origin trajectory indices
    track_indices = np.asarray(J.groupby("trajectory").apply(lambda j: j.name)).astype(np.int64)

    # The size of the largest covariance matrix to make
    if (not max_jumps_per_track is None) and (not max_jumps_per_track is np.inf):
        max_jumps_per_track = min(max_jumps_per_track, J["n_jumps"].max())
    else:
        max_jumps_per_track = J["n_jumps"].max()

    # Generate the RBME jump covariance matrix
    def make_cov(D, hurst, n):
        """
        args
        ----
            D           :   float, diffusion coefficient in
                            squared microns per second
            hurst       :   float, Hurst parameter
            n           :   int, the number of jumps

        returns
        -------
            2D ndarray of shape (n, n), the covariance matrix

        """
        h2 = hurst * 2
        T, S = (np.indices((n, n)) + 1)
        C = D * frame_interval * (
            np.power(np.abs(T - S + 1), h2) + 
            np.power(np.abs(T - S - 1), h2) - 
            2 * np.power(np.abs(T - S), h2)
        )
        C += (2 * le2 * np.identity(n))
        for i in range(n-1):
            C[i,i+1] -= le2 
            C[i+1,i] -= le2 
        return C 

    # The output likelihood matrix
    log_L = np.zeros((n_tracks, nD, nH), dtype=np.float64)

    # Iterate through the different trajectory lengths
    for l in range(1, max_jumps_per_track+1):

        # Get all of the trajectories with this number of jumps
        subtracks = np.asarray(J.loc[
            J["n_jumps"] == l,
            ["y", "x"]
        ])

        # The number of trajectories in this set
        n_match = subtracks.shape[0] // l

        # The y- and x- jumps of each trajectory
        y_jumps = subtracks[:,0].reshape((n_match, l)).T 
        x_jumps = subtracks[:,1].reshape((n_match, l)).T 

        for i, D in enumerate(diff_coefs):
            for j, H in enumerate(hurst_pars):

                # Generate the covariance matrix 
                C = make_cov(D, H, l)

                # Inverse covariance matrix
                C_inv = np.linalg.inv(C)

                # Normalization factor (slogdet is the log determinant)
                norm_fac = l * np.log(2 * np.pi) + np.linalg.slogdet(C)[1]

                # Evaluate the log likelihood for the y- and x-jumps
                y_ll = (y_jumps * (C_inv @ y_jumps)).sum(axis=0)
                x_ll = (x_jumps * (C_inv @ x_jumps)).sum(axis=0)

                # Combine the axes components
                log_L[n_jumps==l, i, j] = -0.5 * (y_ll + x_ll) - norm_fac 

    # Normalize over all states for each trajectory
    L = np.zeros(log_L.shape, dtype=np.float64)
    for t in range(n_tracks):
        log_L[t,:,:] -= log_L[t,:,:].max()
        L[t,:,:] = np.exp(log_L[t,:,:])
        L[t,:,:] /= L[t,:,:].sum()
    del log_L 

    return L, n_jumps, track_indices, (diff_coefs, hurst_pars)