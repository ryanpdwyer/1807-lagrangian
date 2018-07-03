# Lagrangian and impedance spectroscopy analysis and figures



[![DOI](https://zenodo.org/badge/95897198.svg)](https://zenodo.org/badge/latestdoi/95897198)

This repository contains the analysis and figures for the paper "Lagrangian and impedance spectroscpoy treatements 

## Installation

Binary packages:

    numpy
    scipy
    matplotlib
    sympy
    pandas
    cython
    h5py
    numba

Remaining packages:

    pip install -r requirements.txt


## Generating figures

Run the command

    bash generate-figs.sh

which will evaluate all of the Jupyter notebooks in the `fig_scripts` directory.

## Theoretical results

For cases where the results in the paper were supported by simulations or symbolic computation, the appropriate analysis

## Conda environment

The conda environment used to run this analysis can be reproduced using the commands below.

    conda create -n 1807-lagrangian python=2.7 numpy scipy matplotlib sympy ipython jupyter cython pandas h5py numba
    source activate 1807-lagrangian 
    pip install tqdm sigutils pint json-tricks
    pip install -e "git+https://github.com/marohngroup/kpfm.git#egg=kpfm"
    pip install -e "bundled-dependencies/ffta"