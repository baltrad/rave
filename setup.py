#!/bin/env python
#
# $Id: setup.py,v 1.2 2006/12/18 09:34:07 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2005-
#                All rights reserved.
#
# $Log: setup.py,v $
# Revision 1.2  2006/12/18 09:34:07  dmichels
# *** empty log message ***
#
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
Distutils setup file for RAVE.
"""
import sys, os
from distutils.core import setup, Extension

NAME = "RAVE"
VERSION = "2.0"  # should be the same as the CVS version
DESCRIPTION = "Radar Analysis and Visualization Environment"
AUTHOR = "Daniel Michelson", "daniel.michelson@smhi.se"
HOMEPAGE = "http://nordrad.net/"

MODULES = []


## USGS PROJ4
#
# Add prefix to USGS PROJ4 manually to this list if setup fails.
prefixes = ['/usr/local', '/usr', '/']

INCLUDE_DIRS, LIBRARY_DIRS = [], []

import numpy
INCLUDE_DIRS.append(os.path.join(os.path.split(numpy.__file__)[0],
                                 'core/include/numpy'))
LIBRARY_DIRS.append(os.path.join(os.path.split(numpy.__file__)[0],
                                 'lib'))

for p in prefixes:
    header = os.path.join(p, 'include/projects.h')
    if os.path.isfile(header):
        INCLUDE_DIRS.append(p+'/include')

for p in prefixes:
    lib = os.path.join(p, 'lib/libproj.a')
    if os.path.isfile(lib):
        LIBRARY_DIRS.append(p+'/lib')

if len(INCLUDE_DIRS) == 0 and len(LIBRARY_DIRS) == 0:
    print '\tCould not find USGS PROJ4.'
    print '\tManually add its path to "prefixes" and try again.'
    sys.exit()

LIBRARIES = ['proj']

MODULES.append(
    Extension(
        "_proj", ["modules/_proj.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

## RAVE modules
#
MODULES.append(
    Extension(
        "_h5rad", ["modules/h5rad.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

INCLUDE_DIRS.append('./librave/transform')
INCLUDE_DIRS.append('./librave/pyapi')
LIBRARY_DIRS.append('./librave/transform')
LIBRARY_DIRS.append('./librave/pyapi')
LIBRARIES.append('ravetransform')
LIBRARIES.append('ravepyapi')

MODULES.append(
    Extension(
        "_helpers", ["modules/helpers.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )
MODULES.append(
    Extension(
        "_ctoc", ["modules/ctoc.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )
MODULES.append(
    Extension(
        "_composite", ["modules/composite.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_ptop", ["modules/ptop.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_polarvolume", ["modules/polarvolume.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

# build!

if __name__ == "__main__":
    import glob, distutils.sysconfig

    rroot = os.getenv('RAVEROOT')
    if rroot is None:
	rroot="/opt/rave"
	print "No RAVEROOT environment variable set. Will use /opt/rave\n"
    rlib = rroot + '/Lib'

    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR[0],
        author_email=AUTHOR[1],
        description=DESCRIPTION,
        url=HOMEPAGE,
        packages=[""],  # one entry for each subdirectory
        extra_path = rlib,  # all modules go here
        package_dir={"": "Lib"},  # all Python modules are here
        data_files=[(rroot+'/config', glob.glob('config/*.xml')),
                    (rroot+'/include', glob.glob('librave/*.h')),
                    (rroot+'/lib', glob.glob('librave/*.so')),
                    (rroot+'/bin', glob.glob('bin/*[!CVS]*')),
                    (rroot+'/tmp', []),
                    (rroot, ['rave.xbm', 'Copyright'])],
        ext_modules = MODULES
        )

    # distutils doesn't allow flexible planting of .pth files,
    # so we have to create it under extra_path and then ship it to
    # where the default interpreter will find it.
    
    # Only perform this operation during installation
    isinstalling = 0
    for item in sys.argv:
        if item=="install":
            isinstalling = 1
    if isinstalling:
        source = rlib + '.pth'
        dest = distutils.sysconfig.get_python_lib() + '/rave.pth'
        print "Moving %s to %s" % (source, dest)
        try:
            os.rename(source, dest)  # doesn't work in some environments
        except:
            import shutil
            shutil.copyfile(source, dest)
            os.unlink(source)
