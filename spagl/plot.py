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

# DataFrames
import pandas as pd 

# Plotting
import matplotlib.pyplot as plt 

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

# Load trajectories from a directory or files
from .utils import load_tracks

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

#################################
## INDIVIDUAL LIKELIHOOD PLOTS ##
#################################

def gamma_likelihood_plot(tracks, diff_coefs=None, frame_interval=0.00748,
    pixel_size_um=0.16, loc_error=0.04, start_frame=None, dz=None, 
    log_x_axis=True, splitsize=12, d_err=True, ylim=None, axes=None,
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
    log_x_axis=True, log_y_axis=False, splitsize=12, cmap="viridis",
    vmax=None, vmax_perc=99, out_png=None, out_csv=None):
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
        scale_by_jumps=True, dz=None, diff_coefs=diff_coefs, loc_errors=loc_errors)
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
    cbar.set_label("Aggregate likelhood", rotation=90, fontsize=fontsize)

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
    else:
        return fig, axes 

def fbme_likelihood_plot(tracks, diff_coefs=None, hurst_pars=None, loc_error=0.04,
    frame_interval=0.00748, start_frame=None, pixel_size_um=0.16, dz=None, 
    log_x_axis=True, splitsize=12, cmap="viridis", vmax=None, vmax_perc=99,
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



#################################
## CROSS-FILE LIKELIHOOD PLOTS ##
#################################

def gamma_likelihood_by_frame(*track_csvs, diff_coefs=None, frame_interval=0.00748,
    pixel_size_um=0.16, loc_error=0.04, start_frame=0, interval=100,
    dz=None, splitsize=12, max_jumps_per_track=None, vmax=None, vmax_perc=99,
    log_y_axis=True, out_png=None, out_csv=None, normalize_by_frame_group=True):
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
    tracks_L, n_jumps, orig_track_indices, support = eval_likelihood(
        tracks, likelihood="gamma", splitsize=splitsize, start_frame=start_frame,
        pixel_size_um=pixel_size_um, frame_interval=frame_interval,
        scale_by_jumps=True, dz=dz, diff_coefs=diff_coefs, loc_error=loc_error,
        max_jumps_per_track=max_jumps_per_track)

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
    y_ext = 2.0
    x_ext = 7.0
    extent = (0, x_ext, 0, y_ext)
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


def gamma_likelihood_by_file(track_csvs, group_labels=None, diff_coefs=None,
    frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.04, start_frame=None,
    dz=None, splitsize=12, max_jumps_per_track=None, vmax=None, vmax_perc=99,
    log_x_axis=True, label_by_file=False, scale_colors_by_group=False,
    track_csv_ext="trajs.csv", out_png=None, out_csv=None, verbose=True):
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
            track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
                tracks, likelihood="gamma", splitsize=splitsize, start_frame=start_frame,
                pixel_size_um=pixel_size_um, frame_interval=frame_interval,
                scale_by_jumps=True, dz=None, diff_coefs=diff_coefs, loc_error=loc_error,
                max_jumps_per_track=max_jumps_per_track)

            # Aggregate likelihoods across all trajectories in the dataset
            track_likelihoods = track_likelihoods.sum(axis=0)

            # Apply defocalization correction
            if (not dz is None) and (not dz is np.inf):
                track_likelihoods = defoc_corr(track_likelihoods, support, 
                    likelihood="gamma", frame_interval=frame_interval, dz=dz)

            L[i,:] = track_likelihoods

        # Plot layout
        y_ext = 2.0
        x_ext = 7.0
        fig, ax = plt.subplots(figsize=(x_ext, y_ext))
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
                track_likelihoods, n_jumps, orig_track_indices, support = eval_likelihood(
                    tracks, likelihood="gamma", splitsize=splitsize, start_frame=start_frame,
                    pixel_size_um=pixel_size_um, frame_interval=frame_interval,
                    scale_by_jumps=True, dz=None, diff_coefs=diff_coefs, loc_error=loc_error,
                    max_jumps_per_track=max_jumps_per_track)

                # Aggregate likelihoods across all trajectories in the dataset
                track_likelihoods = track_likelihoods.sum(axis=0)

                # Apply defocalization correction
                if (not dz is None) and (not dz is np.inf):
                    track_likelihoods = defoc_corr(track_likelihoods, support, 
                        likelihood="gamma", frame_interval=frame_interval, dz=dz)

                L[i,:] = track_likelihoods

                if verbose:
                    sys.stdout.write("Finished with file {} in group {}...\r".format(i+1 ,g+1))
                    sys.stdout.flush()

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
        y_ext_subplot = 2.0
        y_ext = 2.0 * n_groups 
        x_ext = 7.0
        fig, ax = plt.subplots(n_groups, 1, figsize=(x_ext, y_ext), sharex=True)
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

    if verbose: print("")

    # Save if desired
    if not out_png is None:
        save_png(out_png, dpi=800)
    else:
        return fig, ax 
