#!/usr/bin/env python
"""
setup.py

"""
import setuptools

setuptools.setup(
    name="spagl",
    version="1.0",
    packages=setuptools.find_packages(),
    author="Alec Heckert",
    author_email="aheckert@berkeley.edu",
    description="Aggregate likelihood functions for pure diffusive mixtures in single particle tracking"
)
