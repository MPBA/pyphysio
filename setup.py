# coding=utf-8
from distutils.core import setup
from os.path import join, dirname

pkg_path = dirname(__file__)

README = join(pkg_path, 'README.md')
rm = open(README)
lines = rm.readlines()
description = ''.join(lines[:3])
long_description = ''.join(lines[:4])
rm.close()

vh = open('version')
lines = vh.readlines()
version = lines[-1].rstrip('\n').rstrip('\r')
vh.close()
cl = ['Intended Audience :: Science/Research', 'License :: OSI Approved :: GNU General Public License (GPL)',
      'Natural Language :: English', 'Operating System :: POSIX :: Linux', 'Operating System :: MacOS',
      'Operating System :: Microsoft :: Windows', 'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Bio-Informatics', ["Development Status :: 2 - Pre-Alpha",
                                                             "Development Status :: 3 - Alpha",
                                                             "Development Status :: 4 - Beta",
                                                             "Development Status :: 5 - Production/Stable",
                                                             "Development Status :: 6 - Mature"][
          int(version.split(".")[2])]]
setup(
    name='PyHRV',
    version=version,
    url='https://github.com/MPBA/pyHRV',
    description=description,
    long_description=long_description,
    keywords='HRV, heart, rate, variability, analysis, galaxy, project',
    author='FBK - MPBA',
    author_email='albattisti@fbk.eu',
    license='GNU GPL version 3',
    download_url='https://github.com/MPBA/pyHRV/archive/master.zip',
    classifiers=cl,
    requires=[
        'pandas (>= 0.13.1)',
        'numpy (>= 1.7.1)',
        'scipy (>= 0.12.0)',
        'spectrum'
    ],

    package_dir={'pyPhysio': 'pyPhysio'},
    packages=['pyPhysio', 'pyPhysio.features', 'pyPhysio.segmentation', 'pyPhysio.galaxy', 'pyPhysio.example_data']
)
