# finder-grid-checks
Flier Finder and Holiday Finder checks for gridded bathymetry data. Adapted from QC Tools.

# Installation

Assumes a miniconda Python distribution has been installed.

    git clone https://github.com/ausseabed/finder-grid-checks
    cd finder-grid-checks

    conda create -y -n findergc python=3.7
    conda activate findergc

    pip install -r requirements.txt
    conda install -y -c conda-forge --file requirements_conda.txt

# Tests

Unit tests can be run with the following command line

    python -m pytest -s --cov=ausseabed.findergc tests/
