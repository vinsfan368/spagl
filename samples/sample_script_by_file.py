#!/usr/bin/env python
"""
sample_script_by_file.py -- plot the aggregate likelihood function for 
each of a collection of files alongside each other

"""
import os
from spagl import likelihood_by_file

# The directory with sample files
SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sample_track_csvs"
)

def sample_script_by_file():
    """
    Demonstrate the function

        spagl.plot.likelihood_by_file

    which plots the likelihood function by the individual file
    in a dataset. The result is a plot where each row corresponds
    to one file, showing a heat map of the diffusion coefficient.

    Note on input
    -------------

        likelihood_by_file() takes one of three possible inputs for 
        its first argument:

        (1) a list of files

        (2) a list of lists of files

        (3) a list of directories

    If (1), a single subplot will be generated where each row corresponds
    to the likelihood function evaluated on one of the files.

    If (2) or (3), multiple subplots will be generated for each group
    of files or directory of files.

    """
    # The set of target directories with trajectory CSVs
    target_dirs = [
        os.path.join(SAMPLE_DIR, "u2os_rara_ht_7.48ms"),
        os.path.join(SAMPLE_DIR, "u2os_ht_nls_7.48ms"),
    ]

    # Labels for each directory
    group_labels = ["RARA-HaloTag", "HaloTag-NLS"]

    # Plot the likelihood function by file
    likelihood_by_file(
        target_dirs,               # file groups
        group_labels=group_labels, # label for each file group
        frame_interval=0.00748,    # time between frames in sec
        dz=0.7,                    # microscope focal depth in microns
        pixel_size_um=0.16,        # pixel size in microns
        splitsize=20,              # maximum trajectory length
        track_csv_ext="trajs.csv", # extension for the trajectory CSVs
        start_frame=1000,          # disregard trajectories before this frame
        out_png="sample_script_by_file_out.png",
        verbose=True,
    )

if __name__ == "__main__":
    sample_script_by_file()
