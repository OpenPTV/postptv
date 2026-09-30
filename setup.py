# -*- coding: utf-8 -*-
"""
Installation script for the Flowtracks package.

@author: yosef
"""

import os
from glob import glob

from setuptools import find_packages, setup

# Metadata (name, version, dependencies, python_requires, classifiers, ...)
# lives in pyproject.toml's [project] table, which setuptools treats as
# authoritative. Only what pyproject.toml can't express stays here.
setup(
    packages=find_packages(),
    data_files=[('flowtracks-examples', [f for f in glob('examples/*') if os.path.isfile(f)])],
    scripts=['scripts/analyse_fhdf.py'],
    # NOTE: console_scripts live in pyproject.toml [project.scripts], which
    # setuptools treats as authoritative; defining entry_points here as well
    # would be silently ignored (setuptools warns "`scripts` defined outside
    # of `pyproject.toml` is ignored").
)
