# -*- coding: utf-8 -*-
"""
Installation script for the Flowtracks package.

@author: yosef
"""

from distutils.core import setup
from glob import glob

setup(name='flowtracks',
    version='1.0',
    description='Library for handling of PTV trajectory database.',
    author='Yosef Meller',
    author_email='yosefmel2post.tau.ac.il',
    url='https://github.com/OpenPTV/postptv',
    packages=['flowtracks'],
    data_files=[('flowtracks-examples', glob('examples/*'))],
    scripts=['scripts/analyse_fhdf.py']
)
