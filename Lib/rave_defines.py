#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave_defines.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2005-
#                All rights reserved.
#
# $Log: rave_defines.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""


Arguments:

Returns:
"""
import os
import rave_defines

## PATHS
#
RAVEROOT = os.path.split(os.path.split(rave_defines.__file__)[0])[0]
if not RAVEROOT: RAVEROOT = '..'
RAVELIB =  RAVEROOT + '/Lib'
RAVECONFIG = RAVEROOT + '/config'
RAVEDB = RAVEROOT + '/db'
RAVEBIN = RAVEROOT + '/bin'
# Can't include RAVETEMP from rave_tempfile here.

LIBRAVEINCLUDE = RAVEROOT + '/include'
LIBRAVELIB = RAVEROOT + '/lib'

RAVEICON = RAVEROOT + '/rave.xbm'

# RAVE version
RAVE_VERSION = '2.0'

# HDF5 Information model version
H5RAD_VERSION = 'H5rad 2.0'
H5RAD_VERSIONS = ('H5rad 1.2', 'H5rad 2.0')

# ODIM_H5 version
ODIM_VERSION = 'ODIM_H5/V2_0'

# Default text encoding
ENCODING = 'iso-8859-1'

# Default compression to use for DATASET nodes
COMPRESSION = "zlib" # use "none" (or None), "zlib", or "szip"

# Compression level for ZLIB (0-9)
COMPRESSION_ZLIB_LEVEL = 6 # default

# Mapping between Numeric and HDF5 dataset types
ARRAYTYPES = {'b':'char', 'B':'uchar', 'I':'int',
              'L':'long', 'f':'float', 'd':'double'}

# Default gain and offset values for linear transformation between raw and dBZ
GAIN = 0.4
OFFSET = -30.0


if __name__ == "__main__":
    print __doc__
