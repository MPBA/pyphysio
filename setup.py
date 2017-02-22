# coding=utf-8
from setuptools import setup
from os.path import dirname

pkg_path = dirname(__file__)


version = 'beta-released'

setup(
    name='pyphysio',
    version=version,
    author='FBK - MPBA',
    license='GNU GPL version 3',
    requires=[
        'pandas (>= 0.13.1)',
        'numpy (>= 1.7.1)',
        'scipy (>= 0.12.0)',
        'spectrum'
    ],

    package_dir={'pyphysio': 'pyphysio'},
    packages=['pyphysio']#, 'pyphysio.filters', 'pyphysio.tools', 'pyphysio.estimators', 'pyphysio.indicators', 'pyphysio.segmentation']
)
