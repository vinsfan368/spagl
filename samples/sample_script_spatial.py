#!/usr/bin/env python
"""
sample_script_spatial.py -- plot the likelihood as a function of the 
spatial position of each trajectory

"""
import os
from glob import glob
import numpy as np 

from spagl import spatial_likelihood

# The directory with sample files
SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sample_track_csvs"
)

def sample_script_spatial():
    """
    Plot the diffusion coefficient likelihood as a spatial map.
    This can be useful to judge whether the diffusion coefficient
    in a cell is different from place to place.

    For this to be very informative, it typically requires a LARGE
    number of trajectories - for instance, >10000 per file.

    """
    # File to run on
    track_csv = os.path.join(SAMPLE_DIR, "u2os_rara_ht_7.48ms", "region_0_7ms_trajs.csv")

    # The set of diffusion coefficients at which to evaluate the likelihood
    diff_coefs = np.array([0.01, 0.1, 5.0, 20.0])

    spatial_likelihood(
        track_csv,
        diff_coefs, 
        frame_interval=0.00748,    # time between frames in seconds
        dz=0.7,                    # microscope focal depth in microns
        pixel_size_um=0.16,        # size of pixels in microns
        start_frame=0,             # disregard trajectories before this frame
        normalize_by_loc_density=False,
        out_png="sample_script_spatial_out.png",
    )

if __name__ == "__main__":
    sample_script_spatial()