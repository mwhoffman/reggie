# reggie

A Python package for Bayesian regression.

The goal of this package is to be a relatively self-contained python package
for Bayesian regression problems. The predominant focus of this package, for
the time-being is on Gaussian Process (GP) models, loosely based on Carl
Rasmussen's `GPML` toolbox in matlab, however the focus has shifted slightly as
we have tried to generalize some of these methods.

[![Build Status][travis-shield]][travis]
[![Coverage Status][coveralls-shield]][coveralls]

[travis]: https://travis-ci.org/mwhoffman/reggie
[coveralls]: https://coveralls.io/r/mwhoffman/reggie
[travis-shield]: https://img.shields.io/travis/mwhoffman/reggie.svg?style=flat
[coveralls-shield]: https://img.shields.io/coveralls/mwhoffman/reggie.svg?style=flat


## Installation

The easiest way to install this package is by running

    pip install -r https://github.com/mwhoffman/reggie/raw/master/requirements.txt
    pip install git+https://github.com/mwhoffman/reggie.git

which will install the package and any of its dependencies. Once the package is
installed the included demos can be run directly via python. For example, by
running

    python -m reggie.demos.basic

A full list of demos can be viewed [here](reggie/demos).
