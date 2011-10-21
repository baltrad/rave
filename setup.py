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
import re, string
from distutils.core import setup, Extension

NAME = "RAVE"
VERSION = "2.0"  # should be the same as the CVS version
DESCRIPTION = "Radar Analysis and Visualization Environment"
AUTHOR = "Daniel Michelson", "daniel.michelson@smhi.se"
HOMEPAGE = "http://baltrad.eu/"

MODULES = []


## USGS PROJ4
#
# Add prefix to USGS PROJ4 manually to this list if setup fails.
PROJ4PREFIXES = ['/usr/local', '/usr', '/']
prefixes = ['/usr/local', '/usr', '/']

INCLUDE_DIRS, LIBRARY_DIRS = [], []

import numpy
INCLUDE_DIRS.append(os.path.join(os.path.split(numpy.__file__)[0],
                                 'core/include/numpy'))
LIBRARY_DIRS.append(os.path.join(os.path.split(numpy.__file__)[0],
                                 'lib'))

LIBRARIES = []

# Determine hlhdf and the other parts (using the same hlhdf information that was available
# when compiling librave
def get_param_value(pname,lines):
  result = None
  for line in lines:
    g = re.match(pname+"=[ \t]*([^$]+)", line)
    if g != None:
      result = string.strip(g.group(1))
      break
  return result

def extract_hlhdf_info(filename):
  deffp = open(filename)
  lines = deffp.readlines()
  incdir = get_param_value("HLHDF_INCLUDE_DIR",lines)
  libdir = get_param_value("HLHDF_LIB_DIR",lines)
  hldefmk = get_param_value("HLHDF_HLDEF_MK_FILE",lines)

  gg = re.match("([ \t]*-I)(.*)", incdir)
  if gg != None:
    incdir = gg.group(2)
  gg = re.match("([ \t]*-L)(.*)", libdir)
  if gg != None:
    libdir = gg.group(2)

  if os.path.isfile(incdir+"/hlhdf.h") and \
     os.path.isfile(libdir+"/libhlhdf.so") and \
     os.path.isfile(hldefmk):
    return (incdir,libdir,hldefmk)
  else:
    raise IOError, "Failed to determine hlhdf settings"

def get_szinfo_from_hlhdf(defmk):
  fp = open(defmk)
  lines = fp.readlines()
  gotsz = get_param_value("GOT_SZ_COMPRESS",lines)
  szinc = get_param_value("SZLIB_INCDIR", lines)
  szlib = get_param_value("SZLIB_LIBDIR", lines)
  if gotsz == "no":
    return None
  else:
    return (szinc,szlib)

def get_zlibinfo_from_hlhdf(defmk):
  fp = open(defmk)
  lines = fp.readlines()
  zlinc = get_param_value("ZLIB_INCDIR", lines)
  zllib = get_param_value("ZLIB_LIBDIR", lines)
  return (zlinc,zllib)

def get_hdf5info_from_hlhdf(defmk):
  fp = open(defmk)
  lines = fp.readlines()
  hdfinc = get_param_value("HDF5_INCDIR", lines)
  hdflib = get_param_value("HDF5_LIBDIR", lines)
  gg = re.match("([ \t]*-I)(.*)", hdfinc)
  if gg != None:
    hdfinc = gg.group(2)
  gg = re.match("([ \t]*-L)(.*)", hdflib)
  if gg != None:
    hdflib = gg.group(2)  
  return (hdfinc,hdflib)

def get_expat_info_from_def(defmk):
  fp = open(defmk)
  lines = fp.readlines()
  expatsup=get_param_value("EXPAT_SUPPRESSED", lines)
  expatinc=get_param_value("EXPAT_INCLUDE_DIR", lines)
  expatlib=get_param_value("EXPAT_LIB_DIR", lines)
  if expatsup == None or expatsup == "":
    expatsup = "yes"
    
  gg = re.match("([ \t]*-I)(.*)", expatinc)
  if gg != None:
    expatinc = gg.group(2)
  gg = re.match("([ \t]*-L)(.*)", expatlib)
  if gg != None:
    expatlib = gg.group(2)
  return (expatsup, expatinc, expatlib)

def get_proj4_dirs():
  proot = os.getenv("PROJ4ROOT")
  if proot is None:
    princ = os.getenv("PROJ4INC")
    prlib = os.getenv("PROJ4LIB")
  else:
    princ = "%s/include"%proot
    prlib = "%s/lib"%proot

  if princ is None or prlib is None:
    for p in PROJ4PREFIXES:
      if princ is None:
        header = os.path.join(p, 'include/projects.h')
        if os.path.isfile(header):
          princ = p+'/include'
          break
    for p in PROJ4PREFIXES:
      if prlib is None:
        library = os.path.join(p, 'lib/libproj.so')
        if not os.path.isfile(library):
          library = os.path.join(p, 'lib/libproj.a')
        if os.path.isfile(library):
          prlib = p+'/lib'
          break

  return princ,prlib

# Utility function for replacing configuration definitions
# var = the name of the environment variable
# lines = the contents of rave_defines.py read with readlines()
# existing = the part of the (existing) definition that doesn't change
def replace_def(var, lines, existing='%s\n'):
    for l in range(len(lines)):
        if len(lines[l]) > len(var):
            if lines[l][:len(var)] == var:
                lines[l] = var + ' = ' + existing % os.getenv(var)

# Continuing ...
incdir,libdir,hldefmk = extract_hlhdf_info("./librave/def.mk")
INCLUDE_DIRS.append(incdir)
LIBRARY_DIRS.append(libdir)

szinfo = get_szinfo_from_hlhdf(hldefmk)
zlibinfo = get_zlibinfo_from_hlhdf(hldefmk)
hdfinfo = get_hdf5info_from_hlhdf(hldefmk)

if hdfinfo[0] != None and hdfinfo[0] != "":
  INCLUDE_DIRS.append(hdfinfo[0])
if hdfinfo[1] != None and hdfinfo[1] != "":
  LIBRARY_DIRS.append(hdfinfo[1])

if szinfo != None:
  if szinfo[0] != None and szinfo[0] != "":
    INCLUDE_DIRS.append(szinfo[0])
  if szinfo[1] != None and szinfo[1] != "":
    LIBRARY_DIRS.append(szinfo[1])

if zlibinfo[0] != None and zlibinfo[0] != "":
  INCLUDE_DIRS.append(zlibinfo[0])
if zlibinfo[1] != None and zlibinfo[1] != "":
  LIBRARY_DIRS.append(zlibinfo[1])

projincdir,projlibdir = get_proj4_dirs()

if projincdir=='' and projlibdir=='':
    print '\tCould not find USGS PROJ4.'
    sys.exit()  

if projincdir != '':
    INCLUDE_DIRS.append(projincdir)

if projlibdir != '':
    LIBRARY_DIRS.append(projlibdir)

expatsuppressed,expatinc,expatlib = get_expat_info_from_def("./librave/def.mk")

if expatsuppressed == "no":
  if expatinc != '':
    INCLUDE_DIRS.append(expatinc)
  if expatlib != '':
    LIBRARY_DIRS.append(expatlib)
    
LIBRARIES.append('proj')
LIBRARIES.append("hlhdf")
LIBRARIES.append("hdf5")

if expatsuppressed == "no":
  LIBRARIES.append("expat")
  
LIBRARIES.append("z")
  
print "LIBDIR: " + `LIBRARY_DIRS`
if szinfo != None:
  LIBRARIES.append("sz")

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
# We need to insert libraries so that they are placed correctly during build
LIBRARIES.insert(0, 'ravepyapi')
LIBRARIES.insert(0, 'ravetransform')

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
        "_mean", ["modules/mean.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_rave", ["modules/rave.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_polarnav", ["modules/pypolarnav.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_projection", ["modules/pyprojection.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_polarscan", ["modules/pypolarscan.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_polarscanparam", ["modules/pypolarscanparam.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_polarvolume", ["modules/pypolarvolume.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_cartesian", ["modules/pycartesian.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_raveio", ["modules/pyraveio.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_transform", ["modules/pytransform.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_area", ["modules/pyarea.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_pycomposite", ["modules/pycomposite.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_cartesianvolume", ["modules/pycartesianvolume.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_ravefield", ["modules/pyravefield.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

MODULES.append(
    Extension(
        "_radardef", ["modules/pyradardefinition.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

if expatsuppressed == "no":
  MODULES.append(
    Extension(
      "_projectionregistry", ["modules/pyprojectionregistry.c"],
      include_dirs=INCLUDE_DIRS,
      library_dirs=LIBRARY_DIRS,
      libraries=LIBRARIES
    )
  )

  MODULES.append(
    Extension(
      "_arearegistry", ["modules/pyarearegistry.c"],
      include_dirs=INCLUDE_DIRS,
      library_dirs=LIBRARY_DIRS,
      libraries=LIBRARIES
    )
  )

MODULES.append(
    Extension(
        "_detectionrange", ["modules/pydetectionrange.c"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES
        )
    )

INCLUDE_DIRS.append('./librave/scansun')
LIBRARY_DIRS.append('./librave/scansun')
LIBRARIES.insert(0, 'ravescansun')

MODULES.append(
    Extension(
        "_scansun", ["modules/pyscansun.c"],
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
                    (rroot+'/include/python', glob.glob('modules/*.h')),
                    (rroot+'/lib', glob.glob('librave/*.so')),
                    (rroot+'/mkf', ['librave/def.mk']),
                    (rroot+'/bin', glob.glob('bin/*[!CVS]*')),
                    (rroot+'/etc', glob.glob('etc/*')),
                    (rroot+'/tmp', []),
                    (rroot, ['rave.xbm','COPYING','COPYING.LESSER','LICENSE'])],
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
      try:
        shutil.copyfile(source, dest)
        os.unlink(source)
      except:
        print "Failed to plant rave.pth file in your Python distribution's site-packages directory. This may not matter, but you should probably check..."

    # Replace default definitions with ones given at build time
    fd = open(rlib + '/rave_defines.py')
    defs = fd.readlines()
    fd.close()
    if os.getenv('PGF_HOST'): 
      replace_def('PGF_HOST', defs, '"%s"\n')
    if os.getenv('PGF_PORT'): 
      replace_def('PGF_PORT', defs)
    if os.getenv('DEX_SPOE'): 
      replace_def('DEX_SPOE', defs, '"http://%s/BaltradDex/dispatch.htm"\n')
    if os.getenv('CENTER_ID'): 
      replace_def('CENTER_ID', defs, '"ORG:%s"\n')    
    fd = open(rlib + '/rave_defines.py', 'w')
    for d in defs:
      fd.write(d)
    fd.close()
