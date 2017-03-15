# coding=utf-8
from distutils.core import setup

setup(
    name='pyphysio',
    packages=['pyphysio',
              'pyphysio.estimators',
              'pyphysio.filters',
              'pyphysio.indicators',
              'pyphysio.segmentation',
              'pyphysio.tools',
              'pyphysio.tests',
              ],
    package_data={'pyphysio.tests': ['data/*']},
    version='0.9',
    description='Python library for physiological signals analysis (IBI & HRV, ECG, BVP, EDA, RESP...)',
    author='MPBA FBK',
    author_email='bizzego@fbk.eu',
    url='https://sites.google.com/site/pyhrvlib/pyhrv',
    download_url='https://github.com/peterldowns/mypackage/archive/0.1.tar.gz',
    keywords=['eda', 'gsr', 'ecg', 'bvp', 'signal', 'analysis', 'physiological', 'pyhrv', 'hrv'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'spectrum',
    ],
    requires=[
        'pytest',
    ],
)
