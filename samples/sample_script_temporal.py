#!/usr/bin/env python
"""
sample_script_temporal.py

"""
import os 
from glob import glob 
from spagl import likelihood_by_frame

# The directory with sample files
SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sample_track_csvs"
)

def sample_script_temporal():
    """
    Plot the diffusion coefficient likelihood in a collection of 
    trajectory CSVs as a function of the frame in which each 
    trajectory was recorded. 

    This is useful to judge whether there are strong density effects
    on the likelihood. For example, SPT movies are often denser (more
    localizations per frame) at the beginning of the movie, which results
    in a higher proportion of tracking errors that can bias the 
    likelihood.

    """
    # Directory with target files
    target_dir = os.path.join(SAMPLE_DIR, "u2os_rara_ht_7.48ms")

    # Get all CSVs in this directory
    target_csvs = glob(os.path.join(target_dir, "*trajs.csv"))

    # Run the function
    likelihood_by_frame(
        *target_csvs,
        frame_interval=0.00748,         # frame interval in seconds
        pixel_size_um=0.16,             # pixel size in microns
        dz=0.7,                         # focal depth of microscope in microns
        interval=100,                   # frame group size
        start_frame=1000,               # starting frame
        normalize_by_frame_group=True,  # scale the color map separately 
                                        # for each frame group
        out_png="sample_script_temporal_out.png"
    )

if __name__ == "__main__":
    sample_script_temporal()