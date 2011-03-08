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
RAVEETC = RAVEROOT + '/etc'
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
UTF8 = 'UTF-8'

# Default compression to use for DATASET nodes
COMPRESSION = "zlib" # use "none" (or None), "zlib", or "szip"

# Compression level for ZLIB (0-9)
COMPRESSION_ZLIB_LEVEL = 6 # default

# Mapping between Numeric and HDF5 dataset types
ARRAYTYPES = {'b':'char', 'B':'uchar', 'I':'int', 'h':'short',
              'L':'long', 'f':'float', 'd':'double'}

# Default gain and offset values for linear transformation between raw and dBZ
GAIN = 0.4
OFFSET = -30.0

# XML-RPC server variables
PIDFILE = os.path.join(RAVEETC, 'rave_pgf_server.pid')
HOST = 'localhost'
PORT = 8085
STDOE = os.path.join(RAVEETC, 'rave_pgf_stdout_stderr.log')

DEX_SPOE = 'http://localhost:8084/BaltradDex/dispatch.htm'
DEX_CHANNEL = 'default_products'
DEX_USER = 'default'

# The originating center id, used to indicate where a product has been generated.
CENTER_ID = 'ORG:82' # Change this if your country is not Sweden.

GENREG  = 'generate-registry'  # root registry tag
REGFILE = os.path.join(RAVEETC, 'rave_pgf_registry.xml')  # registry file

QFILE = os.path.join(RAVEETC, 'rave_pgf_queue.xml')  # queue file
PGF_TAG = 'bltgenerate'  # used for sending files to the DEX

LOG_ID = "PGF-Logger"  # identifier of the logger instance
LOGFILE     = os.path.join(RAVEETC, "rave_pgf.log")
LOGFILESIZE = 5000000  # 5 Mb each
LOGFILES    = 5


if __name__ == "__main__":
    print __doc__
