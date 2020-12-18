#!/usr/bin/env python
"""
sample_script_rbme.py -- show the unmarginalized likelihood function
for regular Brownian motion with localization error, as a function 
of both diffusion coefficient and the magnitude of the localization 
error

"""
import os 
import numpy as np 

from spagl import load_tracks, rbme_likelihood_plot

SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sample_track_csvs"
)

def sample_script_rbme():
    """
    Show the full likelihood function for regular Brownian motion with
    localization error, including both diffusive and error components.

    Since the spatial variance for the jumps of a trajectory is due to 
    both diffusion and localization error, and the true localization 
    for SPT is typically not known exactly, this can be useful to tell
    exactly how much the profile of diffusion coefficients is likely
    to change for different values of the localization error.

    """
    # Directory with target files
    target_dir = os.path.join(SAMPLE_DIR, "u2os_ht_nls_7.48ms")

    # Load all trajectories in this directory
    tracks = load_tracks(target_dir, start_frame=1000, drop_singlets=True)

    # Evaluate the likelihood on a log-spaced grid of localization errors
    loc_errors = np.logspace(-3.0, 0.0, 51)

    # Make the plot
    rbme_likelihood_plot(
        tracks,
        loc_errors=loc_errors,
        frame_interval=0.00748,    # frame interval in seconds
        pixel_size_um=0.16,        # pixel size in microns
        dz=0.7,                    # focal depth in microns
        verbose=True,
        log_y_axis=True,
        out_png="sample_script_rbme_out.png"
    )

if __name__ == "__main__":
    sample_script_rbme()
