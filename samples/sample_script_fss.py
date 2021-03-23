#!/usr/bin/env python
"""
sample_script_fss.py -- sample use of fixed state samplers
with the RBME marginal likelihood function

"""
import os
import sys 
from glob import glob 
import matplotlib.pyplot as plt 

from spagl import load_tracks, fss_plot

# The directory with sample files
SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sample_track_csvs"
)

def save_png(out_png, dpi=800):
    """
    Save a plot to a file.

    """
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    if sys.platform == "darwin":
        os.system("open {}".format(out_png))

def sample_script_fss():
    """
    In addition to the raw aggregated likelihood functions,
    spagl supports a simple utility to infer the existence of 
    underlying states from the likelihood functions for a 
    collection of trajectories.

    Here, we use this tool to identify diffusing states in 
    HaloTag-NLS data and save the results to a PNG and CSV.

    """
    # Directory with target files
    target_dir = os.path.join(SAMPLE_DIR, "u2os_ht_nls_7.48ms")

    # Load all trajectories in these directories
    tracks = load_tracks(target_dir, start_frame=1000, drop_singlets=True)

    # Run the fixed state sampler
    R, n, posterior_mean, likelihood, n_jumps, track_indices, support = fss_plot(
        tracks,
        frame_interval=0.00748,   # frame interval in seconds
        pixel_size_um=0.16,       # size of pixels in microns
        dz=0.7,                   # focal depth in microns
        truncate_y_axis=True,     # clip the plot y-axis to make 
                                  # it easier to see free fraction
        verbose=True,
        out_png="sample_script_fss_out.png",
        out_csv="sample_script_fss_out.csv",
        max_iter=40,
    )

if __name__ == "__main__":
    sample_script_fss()
