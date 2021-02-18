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

import sys, os, datetime
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
H5RAD_VERSIONS = ('H5rad 1.2', 'H5rad 2.0', 'H5rad 2.1', 'H5rad 2.2', 'H5rad 2.3')

# ODIM_H5 version
ODIM_VERSION = 'ODIM_H5/V2_0'
ODIM_VERSIONS = ('ODIM_H5/V2_0', 'ODIM_H5/V2_1', 'ODIM_H5/V2_2', 'ODIM_H5/V2_3')

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

# Default Z-R coefficients, legacy from BALTEX Working Group on Radar
ZR_A = 200.0
ZR_b = 1.5

# Gauge adjustment - migrated from NORDRAD2
GADJUST_STATFILE = RAVEETC + '/gadjust.stat'
DEFAULTA = 0.323868068019
DEFAULTB = -0.00107776407064
DEFAULTC = 1.77500903316e-05
MERGETERMS = 20  # how many 12-hour SYNOP terms to merge: 10 days.
TIMELIMIT_CLIMATOLOGIC_COEFF = 48 # how many hours back in time we can use generated gra coefficients before using the climatologic variant

# SAF-NWC MSG CT filter
CT_FTEMPLATE = "SAFNWC_MSG?_CT___%s_FES_________.h5"
CTPATH = "/opt/baltrad/MSG_CT"
CTDELTA = datetime.timedelta(minutes=15)
CT_MAX_DELTAS = 3  # look backwards in time for ct_max_deltas * ctdelta

# Statistics
TFILE = RAVECONFIG + "/t-critical.pickle"
TFILE_TEMPLATE = RAVECONFIG + "/t-critical.txt"

# Projection and area registries
PROJECTION_REGISTRY = os.path.join(RAVECONFIG, 'projection_registry.xml')
AREA_REGISTRY = os.path.join(RAVECONFIG, 'area_registry.xml')

# XML-RPC server variables
PIDFILE = os.path.join(RAVEETC, 'rave_pgf_server.pid')
PGF_HOST = 'localhost'
PGF_PORT = 8085
PGFs = 4
STDOE = os.path.join(RAVEETC, 'rave_pgf_stdout_stderr.log')

DEX_SPOE = 'http://localhost:8084/BaltradDex'
DEX_CHANNEL = 'default_products'
DEX_USER = 'rave_pgf'

DEX_NODENAME = 'localhost'
DEX_PRIVATEKEY = None

BDB_CONFIG_FILE = None

# The originating center id, used to indicate where a product has been generated.
CENTER_ID = 'ORG:82' # Change this if your country is not Sweden.

GENREG  = 'generate-registry'  # root registry tag
REGFILE = os.path.join(RAVEETC, 'rave_pgf_registry.xml')  # registry file

QFILE = os.path.join(RAVEETC, 'rave_pgf_queue.xml')  # queue file
PGF_TAG = 'bltgenerate'  # used for sending files to the DEX

# Logging - little of this is relevant if SysLog is used or the OS rotates the logs.
LOGID = 'PGF[rave.baltrad.eu]'
LOGPORT = 8089
LOGFILE     = os.path.join(RAVEETC, "rave_pgf.log") # Default logger is to syslog.
LOGFILESIZE = 5000000  # 5 Mb each
LOGFILES    = 5
LOGFACILITY = "local3"
LOGLEVEL = "info"
LOGPIDFILE = os.path.join(RAVEETC, 'rave_pgf_log_server.pid')
SYSLOG_FORMAT = "%(name)s: %(levelname)-8s %(message)s"
LOGFILE_FORMAT = "%(asctime)-15s %(levelname)-8s %(message)s"
if sys.platform == "darwin":
    SYSLOG = "/var/run/syslog"
else:
    SYSLOG = "/dev/log"
    
LOGGER_TYPE="syslog" # Can be stdout or logfile but logfile might result in unordered log entries since there will be more than one process writing to same file

ODIM_SOURCE_FILE = os.path.join(RAVECONFIG, 'odim_source.xml')

QUALITY_REGISTRY=os.path.join(RAVEETC, 'rave_pgf_quality_registry.xml')

RAVE_TILE_REGISTRY=os.path.join(RAVEETC, 'rave_tile_registry.xml')

# Max number of processes to use when performing the composite tiling. If None, then
# the number of processes will be set to number of tiles or less depending on how many
# cores that are available. If number of cores > 1, then there will always be one core
# left for handling the result.
RAVE_TILE_COMPOSITING_PROCESSES=None

# Max number of process to use when executing the quality controls. Default is 4 but this
# should probably be tuned depending on how many files that needs to be quality controled
# and number of available cores. 
RAVE_QUALITY_CONTROL_PROCESSES=4

# If the quality fields should be reprocessed or not if the input already contains a relevant how/task
# quality field. 
RAVE_PGF_QUALITY_FIELD_REPROCESSING=False

# If the compositing should utilize azimuthal navigation or not (how/astart, how/startazA)
#
RAVE_PGF_AZIMUTHAL_NAVIGATION=True

# If the compositing should use lazy loading or not
#
RAVE_PGF_COMPOSITING_USE_LAZY_LOADING=False

# If the compositing should use quantity preload when using lazy loading or not
#
RAVE_PGF_COMPOSITING_USE_LAZY_LOADING_PRELOADS=False

# What algorithm that should be used when performing QI-total
#
QITOTAL_METHOD = "minimum"

# Directory specifying the output scansun path
RAVESCANSUN_OUT = None

# Rave IO default writing version
#RaveIO_ODIM_Version_2_2 = 2,        /**< ODIM 2.2 */
#RaveIO_ODIM_Version_2_3 = 3         /**< ODIM 2.3, The default version */
RAVE_IO_DEFAULT_VERSION = 2

if __name__ == "__main__":
    print(__doc__)
