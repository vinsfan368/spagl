#!/usr/bin/env python
"""
plot.py -- show the results of likelihood evaluation

"""
# Paths
import os
from glob import glob 
import sys

# Warnings, mostly when files cannot be found
import warnings

# Numeric
import numpy as np 
from scipy.ndimage import gaussian_filter 

# DataFrames
import pandas as pd 

# Plotting
import matplotlib.pyplot as plt 
import matplotlib.gridspec as grd 

# Set all fonts to Arial
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

# Place non-imshow axes directly above imshow axes in the presence
# of colorbars
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Evaluate likelihood functions on trajectories
from .eval_lik import eval_likelihood

# Default binning schemes
from .lik import (
    DIFF_COEFS_DEFAULT,
    LOC_ERRORS_DEFAULT,
    HURST_PARS_DEFAULT
)

# Defocalization terms
from .defoc import defoc_corr 

# Fixed state sampler
from .fss import fss 

# Load trajectories from a directory or files
from .utils import load_tracks

# Calculate trajectory length
from .utils import track_length

# Assign each localization an index relative to the first 
# localization in its respective trajectory
from .utils import assign_index_in_track 

# Default parameters
from .HYPERPARAMS import (
    DEFAULT_SPLITSIZE,
    DEFAULT_PSEUDOCOUNT_FRAC
)

########################
## PLOTTING UTILITIES ##
########################

def kill_ticks(axes, spines=False, grid=False):
    """
    Remove the ticks and/or splines and/or grid from a
    matplotlib Axes.

    args
    ----
        axes        :   matplotlib.axes.Axes
        spines      :   bool, also remove spines
        grid        :   boo, also remove the grid

    """
    axes.set_xticks([])
    axes.set_yticks([])
    if spines:
        for s in ['top', 'bottom', 'left', 'right']:
            axes.spines[s].set_visible(False)
    if grid:
        axes.grid(False)

def save_png(out_png, dpi=600):
    """
    Save a matplotlib.figure.Figure to a PNG.

    args
    ----
        out_png         :   str, out path
        dpi             :   int, resolution

    """
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    if sys.platform == "darwin":
        os.system("open {}".format(out_png))

def add_log_scale_imshow(axes, diff_coefs, fontsize=None, side="x"):
    """
    Add a log axis to a plot produced by matplotlib.axes.Axes.imshow.

    This is specifically used to show the log-values of the diffusion 
    coefficient corresponding to each (linear) x-axis point in some
    of the likelihood plots.

    args
    ----
        axes        :   matplotlib.axes.Axes
        diff_coefs  :   1D ndarray
        fontsize    :   int
        side        :   str, "x" or "y"

    returns
    -------
        None; modifies *axes* directly

    """
    diff_coefs = np.asarray(diff_coefs)
    K = diff_coefs.shape[0]
    d_min = min(diff_coefs)
    d_max = max(diff_coefs)

    # Linear range of the axes
    if side == "x":
        lim = axes.get_xlim()
    elif side == "y":
        lim = axes.get_ylim()

    # xlim = axes.get_xlim()
    lin_span = lim[1] - lim[0]

    # Determine the number of log-10 units (corresponding
    # to major ticks)
    log_diff_coefs = np.log10(diff_coefs)
    first_major_tick = int(log_diff_coefs[0])
    major_tick_values = [first_major_tick]
    c = first_major_tick
    while log_diff_coefs.max() > c:
        c += 1
        major_tick_values.append(c)
    n_major_ticks = len(major_tick_values)

    # Convert between the linear and log scales
    log_span = log_diff_coefs[-1] - log_diff_coefs[0]
    m = lin_span / log_span 
    b = lim[0] - m * log_diff_coefs[0]
    def convert_log_to_lin_coord(log_coord):
        return m * log_coord + b

    # Choose the location of the major ticks
    major_tick_locs = [convert_log_to_lin_coord(coord) \
        for coord in major_tick_values]

    # Major tick labels
    major_tick_labels = ["$10^{%d}$" % int(j) for j in major_tick_values]

    # Minor ticks 
    minor_tick_decile = np.log10(np.arange(1, 11))
    minor_tick_values = []
    for i in range(int(major_tick_values[0])-1, int(major_tick_values[-1])+2):
        minor_tick_values += list(minor_tick_decile + i)
    minor_tick_locs = [convert_log_to_lin_coord(v) for v in minor_tick_values]
    minor_tick_locs = [i for i in minor_tick_locs if ((i >= lim[0]) and (i <= lim[1]))]

    # Set the ticks
    if side == "x":
        axes.set_xticks(major_tick_locs, minor=False)
        axes.set_xticklabels(major_tick_labels, fontsize=fontsize)
        axes.set_xticks(minor_tick_locs, minor=True)
    elif side == "y":
        axes.set_yticks(major_tick_locs, minor=False)
        axes.set_yticklabels(major_tick_labels, fontsize=fontsize)
        axes.set_yticks(minor_tick_locs, minor=True)

def try_add_scalebar(axes, pixel_size, units="um", fontsize=8, location="lower left"):
    """
    If the *matplotlib_scalebar* package is available, add a 
    scalebar. Otherwise do nothing.

    args
    ----
        axes        :   matplotlib.axes.Axes
        pixel_size  :   float
        units       :   str
        fontsize    :   int
        location    :   str, "upper left", "lower left",
                        "upper right", or "lower right"

    returns
    -------
        None

    """
    try:
        from matplotlib_scalebar.scalebar import ScaleBar 
        scalebar = ScaleBar(pixel_size, units, location=location,
            frameon=False, color="w", font_properties={'size': fontsize})
        axes.add_artist(scalebar)
    except ModuleNotFoundError:
        pass 

#################################
## INDIVIDUAL LIKELIHOOD PLOTS ##
#################################

def gamma_likelihood_plot(tracks, diff_coefs=None, frame_interval=0.00748,
    pixel_size_um=0.16, loc_error=0.04, start_frame=None, dz=None, 
    log_x_axis=True, splitsize=DEFAULT_SPLITSIZE, d_err=True, ylim=None, axes=None,
    truncate_immobile_frac=False, out_png=None, out_csv=None):
    """
    Evaluate the gamma approximation to the regular Brownian motion likelihood
    function on one file, and show it in a plot.

    args
    ----
        tracks              :   pandas.DataFrame, trajectories

        diff_coefs          :   1D ndarray, the set of diffusion coefficients
                                in microns squared per second. If not set,
                                a default scheme is used.

        frame_interval      :   float, seconds

        pixel_size_um       :   float, microns

        loc_error           :   float, 1D standard deviation of the error in 
                                microns

        start_frame         :   int, disregard trajectories from this frame

        dz                  :   float, focal depth in microns

        log_x_axis          :   bool, do a log scale for the diffusion coefficient

        splitsize           :   int. If set, then each original trajectory is
                                broken into subtrajectories that are at most
                                *splitsize* jumps long.

        d_err               :   float, show the apparent diffusion coefficient
                                of a completely immobile object due to localization
                                error

        ylim                :   float, upper limit for the likelihood to show

        axes                :   matplotlib.axes.Axes. Created if not passed.

        truncate_immobile_frac: bool, truncate the plot's y-axis if the immobile
                                fraction is too large, to visualize the mobile
                                part. If *ylim* is set, then irrelevant.

        out_png             :   str, file to save figure to

        out_csv             :   str, file to the raw likelihoods to

    returns
    -------
        If *out_png* is set, saves the figure to that PNG. Otherwise returns
        (
            matplotlib.figure.Figure,
            matplotlib.axes.Axes
        )

    """
    fontsize = 7

    # Figure layout
    if axes is None:
        fig, axes = plt.subplots(figsize=(4, 1.5))

    # Calculate the likelihood of each diffusion coefficient for each
    # trajectory
    track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
        tracks, likelihood="gamma", splitsize=splitsize, start_frame=start_frame,
        pixel_size_um=pixel_size_um, frame_interval=frame_interval,
        scale_by_jumps=True, dz=None, diff_coefs=diff_coefs, loc_error=loc_error)
    diff_coefs = support[0]

    # Aggregate likelihoods across all trajectories
    agg_lik = track_likelihoods.sum(axis=0)

    # Correct for defocalization 
    if (not dz is None) and (not dz is np.inf):
        agg_lik = defoc_corr(agg_lik, support, likelihood="gamma",
            frame_interval=frame_interval, dz=dz)

    # Plot the density
    axes.plot(diff_coefs, agg_lik, color="k", linestyle='-', label=None)

    # Plot a vertical dotted line
    if d_err:
        _e = (loc_error ** 2) / frame_interval 
        axes.plot([_e, _e], [0, agg_lik.max()*1.3], color="k",
            linestyle='--', label="Loc error limit")

    # Log scale
    if log_x_axis:
        axes.set_xscale("log")

    # Axis labels
    axes.set_xlabel("Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)
    axes.set_ylabel("Gamma aggregate\nlikelihood", fontsize=fontsize)

    # Limit y-axis
    if not ylim is None:
        axes.set_ylim(ylim)
    else:
        if truncate_immobile_frac:
            upper_ylim = agg_lik[d>0.05].max() * 1.3
            axes.set_ylim((0, upper_ylim))
        else:
            axes.set_ylim((0, axes.get_ylim()[1]))
    axes.set_xlim((diff_coefs.min(), diff_coefs.max()))

    # Set tick font size
    axes.tick_params(labelsize=fontsize)

    # Save raw likelihoods, if desired
    if not out_csv is None:

        # Format output dataframe
        out_cols = ["diff_coef", "aggregate_likelihood"]
        df = pd.DataFrame(index=np.arange(len(agg_lik)), columns=out_cols)
        df["diff_coef"] = diff_coefs 
        df["aggregate_likelihood"] = agg_lik 
        df.to_csv(out_csv, index=False)

    # Save, if desired
    if not out_png is None:
        save_png(out_png, dpi=800)
    else:
        return fig, axes 

def rbme_likelihood_plot(tracks, diff_coefs=None, loc_errors=None,
    frame_interval=0.00748, start_frame=None, pixel_size_um=0.16, dz=None, 
    log_x_axis=True, log_y_axis=False, splitsize=DEFAULT_SPLITSIZE, cmap="viridis",
    vmax=None, vmax_perc=99, out_png=None, out_csv=None, show_iso_var=False,
    verbose=True):
    """
    Evaluate the likelihood function for regular Brownian motion with 
    localization error on some trajectories.

    args
    ----
        tracks              :   pandas.DataFrame, trajectories

        diff_coefs          :   1D ndarray, the set of diffusion coefficients
                                in microns squared per second. If not set,
                                a default scheme is used.

        loc_errors          :   1D ndarray, the set of localization errors 
                                at which to evaluate the likelihood (standard
                                deviations, in microns). If not set, a default
                                scheme is used.

        frame_interval      :   float, seconds

        pixel_size_um       :   float, microns

        start_frame         :   int, disregard trajectories from this frame

        dz                  :   float, focal depth in microns

        log_x_axis          :   bool, do a log scale for the diffusion coefficient

        log_y_axis          :   bool, do a log scale for the localization error

        splitsize           :   int. If set, then each original trajectory is
                                broken into subtrajectories that are at most
                                *splitsize* jumps long.

        cmap                :   str, color map to use

        out_png             :   str, file to save figure to

        out_csv             :   str, file to the raw likelihoods to

    returns
    -------
        If *out_png* is set, saves the figure to that PNG. Otherwise returns
        (
            matplotlib.figure.Figure,
            matplotlib.axes.Axes
        )

    """
    fontsize = 7

    # Figure layout
    x_ext = 4.0
    y_ext = 2.0
    extent = (0, x_ext, 0, y_ext)
    fig, axes = plt.subplots(figsize=(4, 2))

    # Calculate the likelihood of each diffusion coefficient for each
    # trajectory
    track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
        tracks, likelihood="rbme", splitsize=splitsize, start_frame=start_frame,
        pixel_size_um=pixel_size_um, frame_interval=frame_interval,
        scale_by_jumps=True, dz=None, diff_coefs=diff_coefs, loc_errors=loc_errors,
        verbose=verbose)
    diff_coefs = support[0]
    loc_errors = support[1]

    # Aggregate likelihoods across all trajectories
    agg_lik = track_likelihoods.sum(axis=0)

    # Correct for defocalization 
    if (not dz is None) and (not dz is np.inf):
        agg_lik = defoc_corr(agg_lik, support, likelihood="rbme",
            frame_interval=frame_interval, dz=dz)

    # Transpose, so that the diffusion coefficients are along the x-axis
    # and the localization errors are along the y-axis
    agg_lik = agg_lik.T 

    # Figure layout
    x_ext = 4.0
    y_ext = 2.0
    extent = (0, x_ext, 0, y_ext)
    fig, axes = plt.subplots(figsize=(4, 2))

    # Color map maximum
    if vmax is None:
        vmax = np.percentile(agg_lik, vmax_perc)

    # Plot the density
    S = axes.imshow(agg_lik, cmap=cmap, vmin=0, vmax=vmax, extent=extent, origin="lower")

    # Color scale
    cbar = plt.colorbar(S, ax=axes, shrink=0.6)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Aggregate likelihood", rotation=90, fontsize=fontsize)

    # Make the x-axis in terms of the log diffusion coefficient
    if log_x_axis:
        add_log_scale_imshow(axes, diff_coefs, fontsize=fontsize, side="x")
    else:
        n_labels = 7
        nD = len(diff_coefs)
        m = nD // n_labels 
        xticks = np.arange(nD) * x_ext / nD + x_ext / (nD * 2.0)
        axes.set_xticks(xticks)
        axes.set_xticklabels(np.asarray(diff_coefs)[m//2::m].round(2), fontsize=fontsize)

    # Make the y-axis in terms of the log localization error
    nLE = len(loc_errors)
    if log_y_axis:
        add_log_scale_imshow(axes, loc_errors, fontsize=fontsize, side="y")
    else:
        n_labels = 7
        y_indices = np.arange(nLE)
        m = nLE // n_labels
        yticks = np.arange(nLE) * y_ext / nLE + y_ext / (nLE * 2.0)
        yticks = yticks[m//2::m]
        axes.set_yticks(yticks)
        axes.set_yticklabels(np.asarray(loc_errors)[m//2::m].round(2), fontsize=fontsize)

    # Axis labels
    axes.set_xlabel("Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)
    axes.set_ylabel("Localization error ($\mu$m)", fontsize=fontsize)

    # Show iso-variance lines, if desired
    if show_iso_var:

        # The variance isotherms
        var_iso_values = [0.01, 0.03, 0.1, 0.3, 1.0]

        for const_var in var_iso_values:

            # Calculate iso-variance lines
            d_line = np.logspace(-2.0, 2.0, 30001)
            le2_line = const_var - d_line * frame_interval
            nonzero = np.logical_and(d_line > 0, le2_line > 0)

            d_line = d_line[nonzero]
            le2_line = le2_line[nonzero]

            # Transform into axis coordinates
            log_d_line = np.log10(d_line)
            log_le2_line = np.log10(le2_line)

            log_d_min = np.log10(min(diff_coefs))
            log_d_max = np.log10(max(diff_coefs))

            log_le2_min = np.log10(min(loc_errors))
            log_le2_max = np.log10(max(loc_errors))

            # Only include points that fall within the evaluated likelihood support
            include_D = np.logical_and(log_d_line >= log_d_min, log_d_line <= log_d_max)
            include_le2 = np.logical_and(log_le2_line >= log_le2_min, log_le2_line <= log_le2_max)
            include = np.logical_and(include_D, include_le2)
            log_d_line = log_d_line[include]
            log_le2_line = log_le2_line[include]

            u_log_d = x_ext * (log_d_line - log_d_min) / (log_d_max - log_d_min)
            u_log_le2 = y_ext * (log_le2_line - log_le2_min) / (log_le2_max - log_le2_min)

            axes.plot(u_log_d, u_log_le2, color='w', linestyle='--')

    # Save raw likelihoods, if desired
    if not out_csv is None:

        # Format output dataframe
        out_cols = ["diff_coef", "loc_error", "aggregate_likelihood"]
        M = len(diff_coefs) * len(loc_errors)
        indices = np.arange(M)
        df = pd.DataFrame(index=indices, columns=out_cols)
        for i, D in enumerate(diff_coefs):
            match = np.logical_and(indices >= (i*nLE), indices < ((i+1)*nLE))
            df.loc[match, "diff_coef"] = D 
            df.loc[match, "loc_error"] = loc_errors 
            df.loc[match, "aggregate_likelihood"] = agg_lik[:,i]
        df.to_csv(out_csv, index=False)

    # Save, if desired
    if not out_png is None:
        save_png(out_png, dpi=800)
        return agg_lik, n_jumps, orig_track_indices, support
    else:
        return fig, axes, agg_lik, n_jumps, orig_track_indices, support 

def fbme_likelihood_plot(tracks, diff_coefs=None, hurst_pars=None, loc_error=0.04,
    frame_interval=0.00748, start_frame=None, pixel_size_um=0.16, dz=None, 
    log_x_axis=True, splitsize=DEFAULT_SPLITSIZE, cmap="viridis", vmax=None, vmax_perc=99,
    out_png=None, out_csv=None):
    """
    Evaluate the likelihood function for regular Brownian motion with 
    localization error on some trajectories.

    args
    ----
        tracks              :   pandas.DataFrame, trajectories

        diff_coefs          :   1D ndarray, the set of diffusion coefficients
                                in microns squared per second. If not set,
                                a default scheme is used.

        hurst_pars          :   1D ndarray, the set of Hurst parameters
                                at which to evaluate the likelihood.
                                If not set, a default scheme is used.

        loc_error           :   float, 1D standard deviation of localization
                                error in microns

        frame_interval      :   float, seconds

        pixel_size_um       :   float, microns

        start_frame         :   int, disregard trajectories from this frame

        dz                  :   float, focal depth in microns

        log_x_axis          :   bool, do a log scale for the diffusion coefficient

        splitsize           :   int. If set, then each original trajectory is
                                broken into subtrajectories that are at most
                                *splitsize* jumps long.

        cmap                :   str, color map to use

        out_png             :   str, file to save figure to

        out_csv             :   str, file to the raw likelihoods to

    returns
    -------
        If *out_png* is set, saves the figure to that PNG. Otherwise returns
        (
            matplotlib.figure.Figure,
            matplotlib.axes.Axes
        )

    """
    fontsize = 7

    # Figure layout
    x_ext = 4.0
    y_ext = 2.0
    extent = (0, x_ext, 0, y_ext)
    fig, axes = plt.subplots(figsize=(4, 2))

    # Calculate the likelihood of each diffusion coefficient for each
    # trajectory
    track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
        tracks, likelihood="fbme", splitsize=splitsize, start_frame=start_frame,
        pixel_size_um=pixel_size_um, frame_interval=frame_interval,
        scale_by_jumps=True, dz=None, diff_coefs=diff_coefs, hurst_pars=hurst_pars,
        loc_error=loc_error)
    diff_coefs = support[0]
    hurst_pars = support[1]

    # Aggregate likelihoods across all trajectories
    agg_lik = track_likelihoods.sum(axis=0)

    # Correct for defocalization 
    if (not dz is None) and (not dz is np.inf):
        agg_lik = defoc_corr(agg_lik, support, likelihood="fbme",
            frame_interval=frame_interval, dz=dz)

    # Transpose, so that the diffusion coefficients are along the x-axis
    # and the Hurst parameters are along the y-axis
    agg_lik = agg_lik.T 

    # Figure layout
    x_ext = 4.0
    y_ext = 2.0
    extent = (0, x_ext, 0, y_ext)
    fig, axes = plt.subplots(figsize=(4, 2))

    # Color map maximum
    if vmax is None:
        vmax = np.percentile(agg_lik, vmax_perc)

    # Plot the density
    S = axes.imshow(agg_lik, cmap=cmap, vmin=0, vmax=vmax, extent=extent, origin="lower")

    # Color scale
    cbar = plt.colorbar(S, ax=axes, shrink=0.6)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Aggregate likelhood", rotation=90, fontsize=fontsize)

    # Make the x-axis in terms of the log diffusion coefficient
    if log_x_axis:
        add_log_scale_imshow(axes, diff_coefs, fontsize=fontsize, side="x")
    else:
        n_labels = 7
        nD = len(diff_coefs)
        m = nD // n_labels 
        D_pos = np.arange(nD) * x_ext / nD 
        x_ticks = D_pos[m//2::m]
        x_ticklabels = np.asarray(diff_coefs)[m//2::m].round(2)
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_ticklabels, fontsize=fontsize)

    # Make the y-axis in terms of the Hurst parameter
    n_labels = 7
    nH = len(hurst_pars)
    y_indices = np.arange(nH)
    m = nH // n_labels 
    yticks = np.arange(nH) * y_ext / nH + y_ext / (nH * 2.0)
    yticks = yticks[m//2::m]
    axes.set_yticks(yticks)
    axes.set_yticklabels(np.asarray(hurst_pars)[m//2::m].round(2), fontsize=fontsize)

    # Axis labels
    axes.set_xlabel("Modified diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)
    axes.set_ylabel("Hurst parameter", fontsize=fontsize)

    # Save raw likelihoods, if desired
    if not out_csv is None:

        # Format output dataframe
        out_cols = ["diff_coef", "hurst_par", "aggregate_likelihood"]
        M = len(diff_coefs) * len(hurst_pars)
        indices = np.arange(M)
        df = pd.DataFrame(index=indices, columns=out_cols)
        for i, D in enumerate(diff_coefs):
            match = np.logical_and(indices >= (i*nH), indices < ((i+1)*nH))
            df.loc[match, "diff_coef"] = D 
            df.loc[match, "hurst_par"] = hurst_pars 
            df.loc[match, "aggregate_likelihood"] = agg_lik[:,i]
        df.to_csv(out_csv, index=False)

    # Save, if desired
    if not out_png is None:
        save_png(out_png, dpi=800)
    else:
        return fig, axes 

###########################################################
## FILE-WISE, FRAME-WISE, or SPACE-WISE LIKELIHOOD PLOTS ##
###########################################################

def likelihood_by_frame(*track_csvs, likelihood="rbme_marginal", diff_coefs=None,
    frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.04, start_frame=0,
    interval=100, dz=None, splitsize=DEFAULT_SPLITSIZE, max_jumps_per_track=None, vmax=None,
    vmax_perc=99, log_y_axis=True, out_png=None, out_csv=None,
    normalize_by_frame_group=True, extent=(0, 7, 0, 1.5)):
    """
    Plot the gamma aggregated likelihood as a function of the frame in which 
    each trajectory was found.

    The trajectories are divided into groups. Each group comprises trajectories
    that were found in a specific frame range (for examples, frames 100 to 199). 
    The gamma likelihood is aggregated for each frame group separately. Additionally,
    the localization density (detections per frame) is plotted alongside the 
    result.

    args
    ----
        track_csvs          :   str or list of str, trajectory CSV file paths or a
                                directory with trajectory CSVs

        likelihood          :   str, "gamma" or "rbme_marginal"; the type of 
                                likelihood function to use

        diff_coefs          :   1D ndarray, the diffusion coefficients at which to 
                                evaluate the likelihood function in squared um per sec

        frame_interval      :   float, seconds

        pixel_size_um       :   float, microns

        loc_error           :   float, localization error in microns

        start_frame         :   int

        interval            :   int

        dz                  :   float, microns

        splitsize           :   int

        max_jumps_per_track :   int

        vmax                :   float

        vmax_perc           :   float

        log_y_axis          :   bool

        out_png             :   str

        out_csv             :   str

        normalize_by_frame_group    :   bool

    returns
    -------
        If *out_png* is specified, saves to that PNG. Otherwise returns
        (
            matplotlib.figure.Figure,
            matplotlib.axes.Axes
        )

    """
    if diff_coefs is None:
        diff_coefs = DIFF_COEFS_DEFAULT
        log_y_axis = True

    # Load all of the tracks in this directory
    tracks = load_tracks(*track_csvs, start_frame=start_frame, drop_singlets=True)
    n_files = tracks["dataframe_index"].nunique()

    # Coerce into ndarray
    diff_coefs = np.asarray(diff_coefs)

    # Calculate the likelihood for each diffusion coefficient for each trajectory
    if likelihood == "gamma":
        tracks_L, n_jumps, orig_track_indices, support = eval_likelihood(
            tracks, likelihood=likelihood, splitsize=splitsize, start_frame=start_frame,
            pixel_size_um=pixel_size_um, frame_interval=frame_interval,
            scale_by_jumps=True, dz=dz, diff_coefs=diff_coefs,
            max_jumps_per_track=max_jumps_per_track)
    elif likelihood == "rbme_marginal":
        tracks_L, n_jumps, orig_track_indices, support = eval_likelihood(
            tracks, likelihood=likelihood, splitsize=splitsize, start_frame=start_frame,
            pixel_size_um=pixel_size_um, frame_interval=frame_interval,
            scale_by_jumps=True, dz=dz, diff_coefs=diff_coefs, 
            max_jumps_per_track=max_jumps_per_track, verbose=True)

    # Map each trajectory back to the first frame in which it was found
    m = len(orig_track_indices)
    C = pd.DataFrame(index=np.arange(m), columns=["orig_track_index", "initial_frame"])
    C["orig_track_index"] = orig_track_indices
    initial_frames = tracks.groupby("trajectory", group_keys=orig_track_indices)["frame"].first()
    C["initial_frame"] = C["orig_track_index"].map(initial_frames)
    initial_frames = np.asarray(C["initial_frame"])

    # Divide the entire frame range into intervals
    intervals = np.arange(start_frame, tracks["frame"].max()+interval, interval)
    n_intervals = len(intervals) - 1

    # Localization density
    locs_per_frame = tracks.groupby("frame").size()
    loc_density = np.zeros(n_intervals, dtype=np.float64)

    # For each interval, sum the likelihoods
    intervals_L = np.zeros((n_intervals, tracks_L.shape[1]), dtype=np.float64)
    for j in range(n_intervals):
        start = intervals[j]
        stop = intervals[j+1]
        include = np.logical_and(initial_frames >= start, initial_frames < stop)
        intervals_L[j,:] = tracks_L[include,:].sum(axis=0)

        # Normalize
        if normalize_by_frame_group and (intervals_L[j,:].sum() > 0):
            intervals_L[j,:] /= intervals_L[j,:].sum()

        # Localization density in this frame interval (# detections / frame)
        include = np.logical_and(locs_per_frame.index >= start, locs_per_frame.index < stop)
        loc_density[j] = locs_per_frame[include].sum() / (interval * n_files)

    # Plot layout
    _, x_ext, _, y_ext = extent 
    fig, ax = plt.subplots(figsize=(x_ext, 2*y_ext))
    fontsize = 8
    if vmax is None:
        vmax = np.percentile(intervals_L, vmax_perc)

    # Plot the likelihoods
    intervals_L = intervals_L.T 
    S = ax.imshow(intervals_L, cmap="viridis", vmin=0, vmax=vmax, extent=extent, origin="lower")

    # Make a log y-axis
    if log_y_axis:
        add_log_scale_imshow(ax, diff_coefs, fontsize=fontsize, side="y")        

    # x-tick labels
    n_labels = 11
    space = n_intervals // n_labels 
    if max(intervals) > 10000:
        _x0 = min(intervals) // 1000
        _x1 = tracks["frame"].max() // 1000
        xticklabels = np.arange(_x0, _x1+1) * 1000
        xticks = np.arange(len(xticklabels)) * x_ext / len(xticklabels)
    else:
        xticks = (np.arange(n_intervals) * x_ext / n_intervals)[::space]
        xticklabels = intervals[:-1][::space]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize, rotation=90)

    # Axis labels
    ax.set_ylabel("Diffusion coefficient\n($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)
    ax.set_xlabel("Frame", fontsize=fontsize)

    # Plot localization density
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("top", size="50%", pad=0.12)
    ind = np.arange(n_intervals) * x_ext / n_intervals
    ax2.plot(ind, loc_density, color="k", linestyle="-")
    ax2.set_xticklabels([])
    ax2.set_ylabel("Detections\nper frame", fontsize=fontsize)
    ax2.set_xlim((0, x_ext))
    ax2.set_xticks(xticks)
    ax2.set_ylim((0, ax2.get_ylim()[1]))

    # Color bar 
    cbar = plt.colorbar(S, ax=ax, shrink=0.6)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Aggregate likelihood", rotation=90, fontsize=fontsize)

    # Save or return
    if not out_png is None:
        save_png(out_png, dpi=800)
    else:
        axes = np.array([ax, ax2])
        return fig, axes 

def likelihood_by_file(track_csvs, likelihood="rbme_marginal", group_labels=None,
    diff_coefs=None, frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.04,
    start_frame=None, dz=None, splitsize=DEFAULT_SPLITSIZE, max_jumps_per_track=None, vmax=None,
    vmax_perc=99, log_x_axis=True, label_by_file=False, scale_colors_by_group=False,
    track_csv_ext="trajs.csv", out_png=None, out_csv=None, verbose=True,
    scale_by_total_track_count=False, subplot_extent=(0, 7, 0, 2)):
    """
    Plot the gamma approximation to the regular Brownian likelihood for
    a collection of files in a single heat map. 

    note on supported kinds of *track_csvs*
    ---------------------------------------
        The user can pass three kinds of input to *track_csvs*:

        (1) A simple list of paths to trajectory CSVs. In this case, 
            a single subplot with rows corresponding to each CSV is produced.

            For example:

                track_csvs = ["some_tracks.csv", "some_more_tracks.csv"]

        (2) A list of lists of paths to trajectory CSVs. Each sublist 
            (or "file group") is plotted in a separate subplot. Optionally,
            the titles for each subplot can be set with *group_labels*.

            For example:

                track_csvs = [
                    ["experiment_A_tracks_rep1.csv", "experiment_A_tracks_rep2"],
                    ["experiment_B_tracks_rep1.csv", "experiment_B_tracks_rep2"]
                ]

                group_labels = ["Experiment A", "Experiment B")]

        (3) A list of paths to directories with trajectory CSVs. In this
            case, we find all of the trajectory CSVs in those directories,
            then proceed according to option (2).

            Exactly how to find trajectory CSVs in the target directories
            is determined by the *track_csv_ext* keyword argument, which sets
            the expected extension. For example, if we had

                - directory_for_experiment_A
                    - replicate_1_trajs.csv
                    - replicate_2_trajs.csv

                - directory_for_experiment_B
                    - replicate_1_trajs.csv
                    - replicate_2_trajs.csv

            then we could pass 

                track_csvs = ["directory_for_experiment_A", "directory_for_experiment_B"]

                group_labels = ["Experiment A", "Experiment B"]

                track_csv_ext = "trajs.csv"

    args
    ----
        track_csvs          :   list of file paths, list of directory paths,
                                or list of lists of file paths (see note above)

        likelihood          :   str, "gamma" or "rbme_marginal"; the type of 
                                likelihood function to use.

        group_labels        :   list of str, the label for each file group. If 
                                *track_csvs* is just a list of file paths, then
                                this is irrelevant.

        diff_coefs          :   1D ndarray, the set of diffusion coefficients 
                                on which to evaluate the liklihood. If *None*,
                                then the default scheme is used.

        frame_interval      :   float, acquisition interval in seconds

        pixel_size_um       :   float, size of pixels in microns

        loc_error           :   float, approximate 1D localization error (as a 
                                standard deviation) in microns

        start_frame         :   int, disregard trajectories before this frame

        dz                  :   float, focal depth in microns. If not set, then
                                the focal depth is assumed to be infinite.

        splitsize           :   int. If set, then trajectories are broken down 
                                into smaller trajectories that are each contiguous
                                parts of an original trajectory and have a maximum
                                of *splitsize* jumps.

        max_jumps_per_track :   int, maximum number of jumps to consider from 
                                any single trajectory. If using *splitsize*,
                                we recommend NOT setting this.

        vmax                :   float, color map maximum for likelihood heat maps.
                                If not set, estimated from the data.

        vmax_perc           :   float between 0 and 100, percentile of maximum
                                likelihood to use to scale the color map, if 
                                *vmax* is not set.

        log_x_axis          :   float, show a log scale for the x-axis

        label_by_file       :   bool, show name of the origin file for each set of 
                                trajectories

        scale_colors_by_group:  bool, scale color map separately for each group.
                                Has no effect if the user manually sets *vmax*.

        track_csv_ext       :   str, the extension of the trajectory CSVs to look
                                for, if passing a list of directories for *track_csvs*.
                                Otherwise ignored.

        out_png             :   str, output file to save the plot to. If not set,
                                then this function returns the figure and axes
                                for downstream manipulation.

        out_csv             :   str, file to save the raw likelihood matrices for
                                all files to. If not set, these are not saved.

        verbose             :   bool, show progress

        scale_by_total_track_count  :   bool, scale the color map for each file
                                        by the number of trajectories in that file

    returns
    -------
        If *out_png* is specified, saves to that PNG. Otherwise returns
        (
            matplotlib.figure.Figure,
            matplotlib.axes.Axes
        )

    """
    assert likelihood in ["gamma", "rbme_marginal"], \
        "likelihood must be one of `gamma`, `rbme_marginal`"

    if diff_coefs is None:
        diff_coefs = DIFF_COEFS_DEFAULT
        log_x_axis = True

    # Coerce into ndarray
    diff_coefs = np.asarray(diff_coefs)

    # Number of diffusion coefficient bins
    K = diff_coefs.shape[0]

    # Determine what kind of plot to make. There are three possibilities here:
    # (1) the user passes a simple list of file paths, in which case we make
    #     a single subplot where each row corresponds to one file;
    # (2) the user passes a list of lists of file paths, in which case we make 
    #     separate subplots for each sublist of file paths;
    # (3) the user passes a list of directories, in which case we find all of 
    #     the trajectory files in those directories and then proceed as (2).
    if (isinstance(track_csvs[0], str) and (os.path.isfile(track_csvs[0]))) or \
        (isinstance(track_csvs[0], list) and len(track_csvs) == 1):
        plot_type = 1
    elif isinstance(track_csvs[0], list):
        plot_type = 2
    elif isinstance(track_csvs[0], str) and os.path.isdir(track_csvs[0]):
        plot_type = 3
    else:
        raise RuntimeError("Argument to track_csvs not understood")

    # If using a list of directories, assemble the input into a list of 
    # lists of file paths
    if plot_type == 3:
        new_track_csvs = []
        for dirname in track_csvs:
            if os.path.isdir(dirname):
                new_track_csvs.append(glob(os.path.join(dirname, "*{}".format(track_csv_ext))))
        track_csvs = new_track_csvs
        plot_type = 2

    # Determine the type of plot to make. If the user passes a simple list of 
    # files, make a single subplot where each row corresponds to one file
    if plot_type == 1:

        # Get the subset of files whose paths exist
        for f in track_csvs:
            if not os.path.isfile(f):
                warnings.warn("WARNING: trajectory CSV {} does not exit".format(f))
        track_csvs = [f for f in track_csvs if os.path.isfile(f)]

        # Number of input files
        n = len(track_csvs)

        # If no trajectories remain, bail
        if n == 0:
            raise RuntimeError("Either passed an empty list of files, or " \
                "all files could not be found")

        # Calculate the likelihoods of each diffusion coefficient for each file
        L = np.zeros((n, K), dtype=np.float64)
        for i, track_csv in enumerate(track_csvs):
            tracks = pd.read_csv(track_csv)

            # Evaluate likelihoods
            if likelihood == "gamma":
                track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
                    tracks, likelihood="gamma", splitsize=splitsize, start_frame=start_frame,
                    pixel_size_um=pixel_size_um, frame_interval=frame_interval,
                    scale_by_jumps=True, dz=None, diff_coefs=diff_coefs, loc_error=loc_error,
                    max_jumps_per_track=max_jumps_per_track)
            elif likelihood == "rbme_marginal":
                track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
                    tracks, likelihood="rbme_marginal", splitsize=splitsize, start_frame=start_frame,
                    pixel_size_um=pixel_size_um, frame_interval=frame_interval,
                    scale_by_jumps=True, dz=None, diff_coefs=diff_coefs,
                    max_jumps_per_track=max_jumps_per_track, verbose=verbose)

            # Aggregate likelihoods across all trajectories in the dataset
            track_likelihoods = track_likelihoods.sum(axis=0)

            # Apply defocalization correction
            if (not dz is None) and (not dz is np.inf):
                track_likelihoods = defoc_corr(track_likelihoods, support, 
                    likelihood=likelihood, frame_interval=frame_interval, dz=dz)

            # Scale by the number of trajectories in this file, if desired
            if scale_by_total_track_count:
                track_likelihoods *= n_jumps.shape[0]

            L[i,:] = track_likelihoods

        # Sort by descending number of trajectories
        if scale_by_total_track_count:
            order = np.argsort(L.sum(axis=1))
            L = L[order,:]
            track_csvs = [track_csvs[i] for i in order]

        # Plot layout
        _, x_ext, _, y_ext = subplot_extent 
        fig, ax = plt.subplots(figsize=(x_ext, y_ext))
        if n == 1:
            ax = [ax]
        fontsize = 8
        if vmax is None:
            vmax = np.percentile(L, vmax_perc)

        # Make the primary plot
        S = ax.imshow(L, vmin=0, vmax=vmax, extent=(0, x_ext, 0, y_ext),
            origin="lower")

        # Color bar
        cbar = plt.colorbar(S, ax=ax, shrink=0.6)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label("Aggregate likelihood", rotation=90, fontsize=fontsize)

        # Make the x-axis into a log scale
        if log_x_axis:
            add_log_scale_imshow(ax, diff_coefs, fontsize=fontsize)

        # Use the actual names of the origin files for the y-axis labels,
        # if *label_by_file* is set, or just the word "File" otherwise
        if label_by_file:
            basenames = [os.path.basename(f) for f in track_csvs]
            ax.set_ylabel(None)
            yticks = np.arange(n) * y_ext / n + y_ext / (n * 2.0)
            ax.set_yticks(yticks)
            ax.set_yticklabels(basenames, fontsize=fontsize)
        else:
            ax.set_yticks([])
            ax.set_ylabel("Origin file", fontsize=fontsize)

        # Label the x-axis
        ax.set_xlabel("Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)", 
            fontsize=fontsize)

        # Axis title
        if (not group_labels is None):
            if isinstance(group_labels, list) and (len(group_labels) > 0):
                ax.set_title(group_labels[0], fontsize=fontsize)
            elif isinstance(group_labels, str):
                ax.set_title(group_labels, fontsize=fontsize)
        else:
            ax.set_title(None)

        # Save these likelihoods if desired
        if not out_csv is None:

            # Create output DataFrame. Each index of this dataframe 
            # corresponds to a single diffusion coefficient (or diffusion
            # coefficient bin) from one input file
            n_files, n_bins = L.shape
            M = n_files * n_bins 
            out_df_cols = ["origin_file", "diff_coef", "likelihood"]
            out_df = pd.DataFrame(index=np.arange(M), columns=out_df_cols)

            for i in range(n_files):

                # The set of indices in the output DataFrame corresponding
                # to this file
                start_bin = i * n_bins 
                stop_bin = (i + 1) * n_bins 
                match = out_df.index.isin(range(start_bin, stop_bin))

                # Update this part of the DataFrame
                out_df.loc[match, "origin_file"] = track_csvs[i]
                out_df.loc[match, "likelihood"] = L[i,:]
                out_df.loc[match, "diff_coef"] = diff_coefs 

            # Save
            out_df.to_csv(out_csv, index=False)

    # If the user instead passes a list of lists, where each sublist is a group
    # of files, plot each file group in a separate subplot
    elif plot_type == 2:

        # Determine the number of file groups
        n_groups = len(track_csvs)

        # Calculate the likelihood matrix for each group separately
        L_matrices = []
        for g in range(n_groups):

            # All files in this file group
            group_track_csvs = track_csvs[g]

            # Get the subset whose paths exist
            for f in group_track_csvs:
                if not os.path.isfile(f):
                    warnings.warn("WARNING: trajectory CSV {} does not exist".format(f))
            group_track_csvs = [f for f in group_track_csvs if os.path.isfile(f)]

            # Get the number of files, and bail if this is empty
            n = len(group_track_csvs)
            if n == 0:
                raise RuntimeError("Either passed an empty list of files, " \
                    "or all files in file group {} could not be found".format(g))

            # Likelihood matrix across all files in this file group
            L = np.zeros((n, K), dtype=np.float64)

            for i, track_csv in enumerate(group_track_csvs):
                tracks = pd.read_csv(track_csv)

                # Evaluate likelihoods of each diffusion coefficient for this
                # set of trajectories
                if likelihood == "gamma":
                    track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
                        tracks, likelihood="gamma", splitsize=splitsize, start_frame=start_frame,
                        pixel_size_um=pixel_size_um, frame_interval=frame_interval,
                        scale_by_jumps=True, dz=None, diff_coefs=diff_coefs, loc_error=loc_error,
                        max_jumps_per_track=max_jumps_per_track)
                elif likelihood == "rbme_marginal":
                    track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
                        tracks, likelihood="rbme_marginal", splitsize=splitsize, start_frame=start_frame,
                        pixel_size_um=pixel_size_um, frame_interval=frame_interval,
                        scale_by_jumps=True, dz=None, diff_coefs=diff_coefs,
                        max_jumps_per_track=max_jumps_per_track, verbose=verbose) 

                # Aggregate likelihoods across all trajectories in the dataset
                track_likelihoods = track_likelihoods.sum(axis=0)

                # Apply defocalization correction
                if (not dz is None) and (not dz is np.inf):
                    track_likelihoods = defoc_corr(track_likelihoods, support, 
                        likelihood=likelihood, frame_interval=frame_interval, dz=dz)

                # Scale color maps by the total number of trajectories in that file
                if scale_by_total_track_count:
                    track_likelihoods *= n_jumps.shape[0]

                L[i,:] = track_likelihoods

                if verbose:
                    print("\nFinished with file {} in group {}...\n".format(i+1 ,g+1))

            # Order the files by the number of trajectories
            if scale_by_total_track_count:
                order = np.argsort(L.sum(axis=1))
                L = L[order,:]
                track_csvs[g] = [group_track_csvs[i] for i in order]

            # Store a reference to this file group's likelihood matrix
            L_matrices.append(L)

        # Global color scalar
        if (not vmax is None):
            vmax = [vax for i in range(len(L_matrices))]
        elif (vmax is None) and scale_colors_by_group:
            vmax = [np.percentile(L, vmax_perc) for L in L_matrices]
        elif (vmax is None) and (not scale_colors_by_group):
            vmax = max([np.percentile(L, vmax_perc) for L in L_matrices])
            vmax = [vmax for i in range(len(L_matrices))]

        # Layout for file group plot
        _, x_ext, _, y_ext_subplot = subplot_extent
        y_ext = y_ext_subplot * n_groups

        # y_ext_subplot = 2.0
        # y_ext = 2.0 * n_groups 
        # x_ext = 7.0

        fig, ax = plt.subplots(n_groups, 1, figsize=(x_ext, y_ext), sharex=True)
        if n_groups == 1:
            ax = [ax]
        fontsize = 10

        # Make the primary plot for each file group
        for g, L in enumerate(L_matrices):
            S = ax[g].imshow(L, vmin=0, vmax=vmax[g], extent=(0, x_ext, 0, y_ext_subplot),
                origin="lower")

            # Color bar
            cbar = plt.colorbar(S, ax=ax[g], shrink=0.6)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label("Aggregate likelihood", rotation=90, fontsize=fontsize)

            # Make x-axis a log scale
            if log_x_axis:
                add_log_scale_imshow(ax[g], diff_coefs, fontsize=fontsize)

            # If desired, explicitly put the name of each file as the y-tick
            # label. 
            if label_by_file:
                group_track_csvs = track_csvs[g]
                basenames = [os.path.basename(f) for f in group_track_csvs]
                n_files = len(group_track_csvs)
                ax[g].set_ylabel(None)
                yticks = np.arange(n_files) * y_ext_subplot / n_files + \
                    y_ext_subplot / (n_files * 2.0)
                ax[g].set_yticks(yticks)
                ax[g].set_yticklabels(basenames, fontsize=fontsize)
            else:
                ax[g].set_yticks([])
                ax[g].set_ylabel("File", fontsize=fontsize)

            # Axis title
            if (not group_labels is None) and (g < len(group_labels)):
                ax[g].set_title(group_labels[g], fontsize=fontsize)
            else:
                ax[g].set_title("File group {}".format((g+1)), fontsize=fontsize)

        # x label
        ax[-1].set_xlabel("Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)", 
            fontsize=fontsize)

        # Save these likelihood matrices, if desired
        if not out_csv is None:

            n_groups = len(L_matrices)
            tot_n_files = sum([L.shape[0] for L in L_matrices])
            tot_n_bins = sum([L.shape[1] for L in L_matrices])

            # Output dataframe
            M = sum([L.shape[0] * L.shape[1] for L in L_matrices])
            out_df_cols = ["group_idx", "group_label", "diff_coef", "origin_file", "likelihood"]
            out_df = pd.DataFrame(index=np.arange(M), columns=out_df_cols)

            # Current DataFrame index 
            c = 0

            # Iterative through the file groups 
            for g, L in enumerate(L_matrices):

                # Number of files corresponding to this file group
                n_files, n_bins = L.shape

                for j in range(n_files):

                    # The set of output DataFrame indices corresponding to this file
                    match = out_df.index.isin(range(c, c+n_bins))

                    # The index of the corresponding file group
                    out_df.loc[match, "group_idx"] = g

                    # The label of the corresponding file group
                    if not group_labels is None:
                        out_df.loc[match, "group_label"] = group_labels[g]

                    # The diffusion coefficient corresponding to the likelihoods
                    out_df.loc[match, "diff_coef"] = diff_coefs

                    # The actual likelihood values
                    out_df.loc[match, "likelihood"] = L[j,:]

                    # The origin file
                    out_df.loc[match, "origin_file"] = track_csvs[g][j]

                    # Increment the index 
                    c += n_bins 

            # Save to the CSV
            out_df.to_csv(out_csv, index=False)

    # Save if desired
    if not out_png is None:
        save_png(out_png, dpi=800)
    else:
        return fig, ax 

def spatial_likelihood(track_csv, diff_coefs, likelihood="rbme_marginal", posterior=None, 
    frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.04, start_frame=None,
    dz=None, bin_size_um=0.05, filter_kernel_um=0.12, splitsize=DEFAULT_SPLITSIZE,
    max_jumps_per_track=None, vmax=None, vmax_perc=99, out_png=None,
    fontsize=10, normalize_by_loc_density=False, pos_cols=["y", "x"],
    normalize_diff_coefs_separately=True, count_by_jumps=False):
    """
    Visualize the likelihood of a set of diffusion coefficients as a 
    function of the spatial position of each trajectory. Additionally,
    this plots the localization density for comparison.

    args
    ----
        track_csv           :   str, path to a CSV with trajectories.
                                These should originate from a single
                                tracking movie.

        diff_coefs          :   1D ndarray, the set of diffusion
                                coefficients at which to evaluate the 
                                likelihood (in squared microns per sec)

        likelihood          :   str, "gamma" or "rbme_marginal"; the 
                                type of likelihood function to use.

        posterior           :   1D ndarray, the posterior probability 
                                for each diffusion coefficient. If *None*,
                                then we evaluate the probabilities under 
                                a uniform (noninformative) prior.

        frame_interval      :   float, seconds

        pixel_size_um       :   float, microns

        loc_error           :   float, microns (constant)

        start_frame         :   int, disregard trajectories before this
                                frame

        dz                  :   float, focal depth in microns

        bin_size_um         :   float, size of the bins to use when making
                                the spatial histogram, in microns

        filter_kernel_um    :   float, size of the uniform filter to use 
                                in microns

        splitsize           :   int. If trajectories have more jumps than
                                this, split them into smaller trajectories.

        max_jumps_per_track :   int

        vmax                :   float

        vmax_perc           :   float

        out_png             :   str, PNG to save to 

        fontsize            :   int

        normalize_by_loc_density    :   bool, normalize the result by 
                                the number of localizations in the local
                                neighborhood, so that the resulting color
                                reflects the probability to pick a molecule
                                with a given diffusion coefficient from that
                                area.

        normalize_diff_coefs_separately    :   bool, use a separate color
                                scale for each diffusion coefficient

        count_by_jumps      :   bool, weight the different likelihoods by 
                                jumps rather than points


    returns
    --------
        if *out_png* is specified, saves to that path as a PNG. 
        Otherwise returns
            (
                matplotlib.figure.Figure,
                matplotlib.axes.Axes
            )

    """
    M = len(diff_coefs) + 1
    diff_coef_indices = np.arange(len(diff_coefs))
    likelihood_cols = ["likelihood_%d" % j for j in diff_coef_indices]
    diff_coefs = np.asarray(diff_coefs)

    # Load trajectories
    T = pd.read_csv(track_csv)
    T = T.sort_values(by=["trajectory", "frame"])

    # Remove singlets
    tracks = track_length(T)
    tracks = tracks[tracks["track_length"] > 1]

    # Calculate likelihoods of each diffusion coefficient, given each 
    # trajectory
    if likelihood == "gamma":
        L, n_jumps, track_indices, support = eval_likelihood(
            tracks,
            diff_coefs=diff_coefs,
            likelihood="gamma",
            splitsize=splitsize,
            max_jumps_per_track=max_jumps_per_track,
            start_frame=start_frame,
            pixel_size_um=pixel_size_um,
            frame_interval=frame_interval,
            scale_by_jumps=False,
            dz=dz,
            loc_error=loc_error,
            mode="point"
        )
    elif likelihood == "rbme_marginal":
        L, n_jumps, track_indices, support = eval_likelihood(
            tracks,
            diff_coefs=diff_coefs,
            likelihood="rbme_marginal",
            splitsize=splitsize,
            max_jumps_per_track=max_jumps_per_track,
            start_frame=start_frame,
            pixel_size_um=pixel_size_um,
            frame_interval=frame_interval,
            scale_by_jumps=False,
            dz=dz,
            verbose=True,
        )       

    # Trajectory indices after splitting
    new_track_indices = np.arange(len(track_indices))

    # Scale by posterior state occupations, if desired
    if not posterior is None:
        posterior = np.asarray(posterior)
        L = L * posterior 

        # Renormalize over all diffusion coefficients for each trajectory
        L = (L.T / L.sum(axis=1)).T 

    # Format likelihoods as a dataframe indexed  by origin trajectory
    L = pd.DataFrame(L, columns=diff_coef_indices, index=track_indices)
    L["trajectory"] = track_indices
    L = L.groupby("trajectory")[diff_coef_indices].mean()

    # Count by jumps rather than by points. To do this, exclude the last
    # point in each trajectory 
    if count_by_jumps:
        tracks = assign_index_in_track(tracks)
        iit = np.asarray(tracks["index_in_track"])
        kill = np.concatenate((np.array([False]), (iit[1:] - iit[:-1]) < 0))
        tracks = tracks[~kill]

    # Map likelihoods back to origin trajectories
    for i in diff_coef_indices:
        tracks[likelihood_cols[i]] = tracks["trajectory"].map(L[i])

    # Only consider points for which the likelihood is defined
    tracks = tracks.loc[~pd.isnull(tracks[likelihood_cols[0]]), :]



    # Convert from pixels to um 
    tracks[pos_cols] = tracks[pos_cols] * pixel_size_um 
    T[pos_cols] = T[pos_cols] * pixel_size_um 

    # Determine the extent of the field of view
    y_min = tracks[pos_cols[0]].min() - 3.0 * pixel_size_um 
    y_max = tracks[pos_cols[0]].max() + 3.0 * pixel_size_um 
    x_min = tracks[pos_cols[1]].min() - 3.0 * pixel_size_um 
    x_max = tracks[pos_cols[1]].max() + 3.0 * pixel_size_um 
    y_span = y_max - y_min 
    x_span = x_max - x_min 

    # Bins for computing histogram
    bins_y = np.arange(y_min, y_max, bin_size_um)
    bins_x = np.arange(x_min, x_max, bin_size_um)
    n_bins_y = bins_y.shape[0] - 1
    n_bins_x = bins_x.shape[0] - 1
    filter_kernel = filter_kernel_um / bin_size_um 

    # Plot localization density
    loc_density = np.histogram2d(
        T[pos_cols[0]],
        T[pos_cols[1]],
        bins=(bins_y, bins_x)
    )[0].astype(np.float64)
    density = gaussian_filter(loc_density, filter_kernel)

    # Calculate a histogram of all localizations weighted by the
    # likelihood of each diffusion coefficient
    H = np.zeros((len(diff_coefs), n_bins_y, n_bins_x), dtype=np.float64)
    for i, diff_coef in enumerate(diff_coefs):

        # Calculate a histogram of all localizations weighted by the 
        # likelihood of this particular diffusion coefficient
        H[i,:,:] = np.histogram2d(
            tracks[pos_cols[0]],
            tracks[pos_cols[1]],
            bins=(bins_y, bins_x),
            weights=np.asarray(tracks[likelihood_cols[i]]),
        )[0].astype(np.float64)

        # Smooth
        H[i,:,:] = gaussian_filter(H[i,:,:], filter_kernel)

    # Normalize by the localization density, if desired. If doing
    # this, only show pixels that actually have enough nearby 
    # localizations to compute
    if normalize_by_loc_density:
        norm = H.sum(axis=0)
        nonzero = norm > 0.000001
        for i in range(len(diff_coefs)):
            H[i,:,:][nonzero] = H[i,:,:][nonzero] / norm[nonzero]
            H[i,:,:][~nonzero] = 0

    # Make a plot with the results
    if not out_png is None:

        # Plot layout
        if np.sqrt(M) % 1.0 == 0:
            r = int(np.sqrt(M))
            ny = nx = r 
        else:
            r = int(np.sqrt(M)) + 1
            nx = r 
            ny = M // r + 1

        fig, ax = plt.subplots(ny, nx, figsize=(nx*3, ny*3),
            sharex=True, sharey=True)

        # Plot localization density
        S = ax[0,0].imshow(density, cmap="gray", vmin=0, 
            vmax=np.percentile(density, 99))
        cbar = plt.colorbar(S, ax=ax[0,0], shrink=0.5)
        cbar.ax.tick_params(labelsize=fontsize)
        kill_ticks(ax[0,0])
        ax[0,0].set_title("Localization density", fontsize=fontsize)
        try_add_scalebar(ax[0,0], bin_size_um, "um", fontsize=fontsize)

        # If *normalize_diff_coefs_separately* is set, then use a 
        # separate color palette for each diffusion coefficient
        if normalize_diff_coefs_separately:
            vmax = [np.percentile(H[i,:,:], 99) for i in range(len(diff_coefs))]
        else:
            vmax = np.percentile(H, 99)
            vmax = [vmax for i in range(len(diff_coefs))]

        # Plot the likelihood maps
        for i, diff_coef in enumerate(diff_coefs):

            # The subplot to use
            j = i + 1 
            ax_x = j % nx 
            ax_y = j // nx 

            # Make an imshow plot
            S = ax[ax_y,ax_x].imshow(H[i,:,:], cmap="viridis", vmin=0, vmax=vmax[i])

            # Color bar
            cbar = plt.colorbar(S, ax=ax[ax_y, ax_x], shrink=0.5)
            cbar.ax.tick_params(labelsize=fontsize)

            # Title
            ax[ax_y,ax_x].set_title(
                "%.3f $\mu$m$^{2}$ s$^{-1}$" % diff_coef,
                fontsize=fontsize
            )

            # Scalebar
            try_add_scalebar(ax[ax_y,ax_x], bin_size_um, units="um", fontsize=fontsize)

            # Remove ticks
            kill_ticks(ax[ax_y,ax_x])

        # Make the rest of the subplots invisible
        for j in range(len(diff_coefs)+1, ny*nx):
            ax_x = j % nx 
            ax_y = j // nx 
            kill_ticks(ax[ax_y,ax_x], spines=True, grid=True)

            # Save
        save_png(out_png, dpi=1000)

    # Return the localization density and spatial likelihoods
    return density, H


###############################
## FIXED STATE SAMPLER PLOTS ##
###############################

def fss_plot(tracks, start_frame=None, pixel_size_um=0.16, frame_interval=0.00748,
    dz=None, max_iter=500, convergence=1.0e-8, splitsize=DEFAULT_SPLITSIZE, 
    max_jumps_per_track=None, pseudocount_frac=DEFAULT_PSEUDOCOUNT_FRAC,
    truncate_y_axis=True, verbose=True, out_png=None, out_csv=None):
    """
    This function estimates the underlying state distribution for a set of 
    trajectories using a fixed state sampler. The default settings are intended
    to be extremely conservative and robust, so change them at your peril.

    The resulting plot has three panels:

        (A) The aggregated RBME (regular Brownian motion with localization 
            error) likelihood function across all trajectories;

        (B) the posterior mean RBME state occupations as a function of both
            diffusion coefficient and localization error;

        (C) the posterior mean RBME state occupations marginalized on the 
            localization error

    note on usage
    -------------

        If *out_png* is set, makes the plot and saves to the indicated PNG.
        Otherwise, does not make a plot with th results.

        If *out_csv* is set, saves the resulting likelihood function and 
        posterior mean density to the CSV. Otherwise does not save the 
        results.

    args
    ----
        tracks                  :   pandas.DataFrame, a set of trajectories
                                    with the "trajectory", "frame", "y",
                                    and "x" columns;

        start_frame             :   int, disregard trajectories before this 
                                    frame;

        pixel_size_um           :   float, size of pixels in microns;

        frame_interval          :   float, the frame interval in seconds;

        dz                      :   float, focal depth in microns;

        max_iter                :   float, maximum number of iterations to do
                                    (unless we call convergence);

        convergence             :   float, criterion for convergence;

        splitsize               :   int. If trajectories have more jumps than
                                    this, split them into smaller trajectories;

        max_jumps_per_track     :   int, maximum number of jumps to include from
                                    any given trajectory;

        pseudocount_frac        :   float, the pseudocounts expressed as a 
                                    fraction of the total number of jumps in 
                                    the set of trajectories;

        truncate_y_axis         :   float, limit the y-axis so that the 
                                    free population is easier to see

        verbose                 :   bool, show progress;

        out_png                 :   str, output PNG for saving plots;

        out_csv                 :   str, output CSV for saving the state 
                                    likelihoods and posterior occupations

    returns
    -------
        (
            3D ndarray of shape (n_tracks, n_diff_coefs, n_loc_errors), the 
                posterior probability of each state for each trajectory;

            2D ndarray of shape (n_diff_coefs, n_loc_errors), the parameter
                for the posterior Dirichlet distribution over states;

            2D ndarray, the mean of the posterior distribution over states;

            3D ndarray of shape (n_tracks, n_diff_coefs, n_loc_errors), the
                naive likelihood of each state given each trajectory;

            1D ndarray of shape (n_tracks,), the number of jumps in each 
                trajectory;

            1D ndarray of shape (n_tracks,), the indices of each trajectory
                in the original dataframe;

            (
                1D ndarray of shape (n_diff_coefs,), the set of diffusion
                    coefficients for each state in squared microns per sec;

                1D ndarray of shape (n_lox_errors), the set of localization 
                    errors for each state in microns (root variance)
            )
        )

    """
    if verbose: print("Number of trajectories: {}".format(tracks['trajectory'].nunique()))

    # Run the fixed state sampler
    if verbose: print("Running the fixed state sampler...")
    R, n, posterior_mean, likelihood, n_jumps, track_indices, support = fss(
        tracks, pixel_size_um=pixel_size_um, likelihood="rbme", dz=dz,
        splitsize=splitsize, verbose=verbose, pseudocount_frac=pseudocount_frac,
        max_iter=max_iter, frame_interval=frame_interval
    )

    # Aggregate likelihood across all trajectories
    agg_lik = likelihood.sum(axis=0)
    agg_lik /= agg_lik.sum()

    # Marginalize the posterior mean on the localization error
    posterior_mean_marg = posterior_mean.sum(axis=1)
    posterior_mean_marg /= posterior_mean_marg.sum()

    # Make a plot of the result
    if not out_png is None:
        if verbose: print("Making plot...")

        # Plot layout
        fig, ax = plt.subplots(figsize=(6, 5))
        fontsize = 10
        gs = grd.GridSpec(3, 1, height_ratios=(2, 2, 1), width_ratios=None, hspace=0.75)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        ax = [ax0, ax1, ax2]

        # Color map scaling
        vmax0 = np.percentile(agg_lik, 99.5)
        vmax1 = np.percentile(posterior_mean, 99.5)

        # Plot the aggregate likelihood function
        p0 = ax[0].imshow(agg_lik.T, cmap="viridis", origin="lower", aspect="auto",
            vmin=0, vmax=vmax0)
        p1 = ax[1].imshow(posterior_mean.T, cmap="viridis", origin="lower", aspect="auto",
            vmin=0, vmax=vmax1)

        # Add log scales for the x-axis
        add_log_scale_imshow(ax[0], support[0], side="x")
        add_log_scale_imshow(ax[1], support[0], side="x")

        # y-ticks for localization error
        n_yticks = 7
        space = support[1].shape[0] // n_yticks
        if space > support[1].shape[0]:
            space = 1
        try:
            yticks = np.arange(support[1].shape[0])[::space]
            yticklabels = ['%.3f' % i for i in support[1][::space]]
        except:
            space = 1
            yticks = np.arange(support[1].shape[0])[::space]
            yticklabels = ['%.3f' % i for i in support[1][::space]]           
        for j in range(2):
            ax[j].set_yticks(yticks)
            ax[j].set_yticklabels(yticklabels, fontsize=fontsize)
            ax[j].set_ylabel("Localization error ($\mu$m)", fontsize=fontsize)

        # Show the posterior mean marginalized on localization error
        ax[2].plot(support[0], posterior_mean_marg, color='k')
        ax[2].set_xscale("log")
        ax[2].set_ylabel("Marginal\nposterior mean", fontsize=fontsize)
        ax[2].set_xlim((0.01, 100.0))
        if truncate_y_axis:
            ax[2].set_ylim((0, posterior_mean_marg[support[0]>0.05].max()*2.0))
        else:
            ax[2].set_ylim((0, max(posterior_mean_marg)))
        ax[2].set_xlabel("Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)

        # Subplot titles
        ax[0].set_title("Aggregated likelihood across all trajectories", fontsize=fontsize)
        ax[1].set_title("Posterior RBME mean", fontsize=fontsize)

        # Save, ignoring complaints from a call to matplotlib.pyplot.tight_layout()
        with warnings.catch_warnings(record=False) as w:
            warnings.simplefilter("ignore")
            save_png(out_png, dpi=800)

    # Save the output to a CSV
    if not out_csv is None:

        # CSV output prefix
        out_prefix = os.path.splitext(out_csv)[0]

        # Number of unique states
        M = support[0].shape[0] * support[1].shape[0]

        # Save the posterior mean as a function of both diffusion coefficient
        # and localization error
        out_df = pd.DataFrame(index=np.arange(M), columns=["diff_coef", "loc_error", "agg_lik", "posterior_mean"])
        out_df["diff_coef"] = np.tile(support[0], support[1].shape[0])
        out_df["loc_error"] = np.tile(support[1], support[0].shape[0])
        out_df["agg_lik"] = agg_lik.ravel()
        out_df["posterior_mean"] = posterior_mean.ravel()
        out_csv_rb = "{}_rbme_posterior.csv".format(out_prefix)
        out_df.to_csv(out_csv_rb, index=False)
        if verbose: print("Saved posterior mean to {}".format(out_csv_rb))

        # Save the posterior mean marginalized on the localization error
        n_dc = support[0].shape[0]
        out_df = pd.DataFrame(index=np.arange(n_dc), columns=["diff_coef", "posterior_mean"])
        out_df["diff_coef"] = support[0]
        out_df["posterior_mean"] = posterior_mean_marg 
        out_csv_rbm = "{}_rbme_marginal_posterior.csv".format(out_prefix)
        out_df.to_csv(out_csv_rbm, index=False)
        if verbose: print("Saved posterior mean to {}".format(out_csv_rbm))

    # Return the output
    return R, n, posterior_mean, likelihood, n_jumps, track_indices, support 

