'''
Copyright (C) 2005 - Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.

'''
## Contains a variety of definitions used by RAVE

## @file
## @author Daniel Michelson, SMHI
## @date 2011-06-27

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
RAVE_VERSIONS = ('2.0')

# HDF5 Information model version
H5RAD_VERSION = 'H5rad 2.0'
H5RAD_VERSIONS = ('H5rad 1.2', 'H5rad 2.0', 'H5rad 2.1')

# ODIM_H5 version
ODIM_VERSION = 'ODIM_H5/V2_0'
ODIM_VERSIONS = ('ODIM_H5/V2_0', 'ODIM_H5/V2_1')

# Default text encoding
ENCODING = 'iso-8859-1'
UTF8 = 'UTF-8'

# Default compression to use for DATASET nodes
# szip is illegal with ODIM_H5
COMPRESSION = "zlib" # use "none" (or None), "zlib", or "szip"

# Compression level for ZLIB (0-9)
COMPRESSION_ZLIB_LEVEL = 6 # default

# Mapping between Numeric and HDF5 dataset types
ARRAYTYPES = {'b':'char', 'B':'uchar', 'I':'int', 'h':'short',
              'L':'long', 'f':'float', 'd':'double'}

# Default gain and offset values for linear transformation between raw and dBZ
GAIN = 0.4
OFFSET = -30.0

# Projection and area registries
PROJECTION_REGISTRY = os.path.join(RAVECONFIG, 'projection_registry.xml')
AREA_REGISTRY = os.path.join(RAVECONFIG, 'area_registry.xml')

# XML-RPC server variables
PIDFILE = os.path.join(RAVEETC, 'rave_pgf_server.pid')
PGF_HOST = 'localhost'
PGF_PORT = 8085
STDOE = os.path.join(RAVEETC, 'rave_pgf_stdout_stderr.log')

DEX_SPOE = 'http://localhost:8084/BaltradDex/dispatch.htm'
DEX_CHANNEL = 'default_products'
DEX_USER = 'rave_pgf'

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

ODIM_SOURCE_FILE = os.path.join(RAVECONFIG, 'odim_source.xml')


if __name__ == "__main__":
    print __doc__
