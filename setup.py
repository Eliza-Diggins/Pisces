#!/usr/bin/env python
"""
Setup file for Pisces module.

Written by: Eliza Diggins
Last Updated: 01/04/24
"""
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

print(os.path)
# @@ CYTHON UTILITIES @@ #
# All of the cython extensions for the package have to be added
# here to ensure that they are accessible and installed on use.
rejection_sampling_utils = Extension(
    "pisces.utilities.math_utils._sampling_opt",
    sources=["pisces/utilities/math_utils/_sampling_opt.pyx"],
    language="c",
    libraries=["m"],
    include_dirs=[np.get_include()],
)

# Read the readme from /Pisces/README.rst. As long as we are in the
# project directory, this should be accessible directly in path.
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup function
setup(
    name="pisces",
    version="0.0.1",
    description="Multipurpose astrophysical modeling in Python.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Eliza C. Diggins",
    author_email="eliza.diggins@utah.edu",
    url="https://github.com/eliza-diggins/pisces",
    setup_requires=[
        "numpy",
        "cython",
    ],  # Ensure numpy and cython are installed before setup
    download_url="https://github.com/eliza-diggins/pisces/tarbar/0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy<2",
        "scipy",
        "cython",
        "matplotlib",
        "tqdm",
        "ruamel.yaml",
        "h5py",
        "sympy",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    include_package_data=True,
    ext_modules=cythonize([rejection_sampling_utils]),
    python_requires=">=3.6",
)
