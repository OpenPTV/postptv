# -*- coding: utf-8 -*-
"""
Installation script for the Flowtracks package.

@author: yosef
"""

import os
from setuptools import setup, find_packages
from glob import glob

# Metadata (name, version, dependencies, python_requires, classifiers, ...)
# lives in pyproject.toml's [project] table, which setuptools treats as
# authoritative. Only what pyproject.toml can't express stays here.
setup(
    packages=find_packages(),
    data_files=[('flowtracks-examples', [f for f in glob('examples/*') if os.path.isfile(f)])],
    scripts=['scripts/analyse_fhdf.py'],
    entry_points={
        'console_scripts': [
            'postptv-combine = flowtracks.combine:main',
        ],
    },
)
