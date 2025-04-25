/*
Copyright (C) 2025 - Swedish Meteorological and Hydrological Institute (SMHI)

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

*/
#ifndef RAVE_DEFINES_H
#define RAVE_DEFINES_H

//## Contains a variety of definitions used by RAVE

//## @file
//## @author Daniel Michelson, SMHI
//## @date 2011-06-27


//## PATHS
//#
extern "C" {
#include "rave_types.h"
}
#include <string>
#include <vector>
#include <map>

extern std::string _RAVEROOT;

// _RAVEROOT set in radarcomp_main.cpp at execution time.
// Will be empty in compile time so _RAVEROOT must be added before for example RAVECONFIG to get av valid path.
const std::string RAVEROOT=_RAVEROOT;

const std::string RAVELIB = RAVEROOT + "/Lib";
const std::string RAVECONFIG = RAVEROOT + "/config";
const std::string RAVEDB = RAVEROOT + "/db";
const std::string RAVEBIN = RAVEROOT + "/bin";
const std::string RAVEETC = RAVEROOT + "/etc";

//# Can't include RAVETEMP from rave_tempfile here.

const std::string LIBRAVEINCLUDE = RAVEROOT + "/include";
const std::string LIBRAVELIB = RAVEROOT + "/lib";

const std::string RAVEICON = RAVEROOT + "/rave.xbm";

//# RAVE version
const std::string RAVE_VERSION = "2.0";
const std::vector<std::string> RAVE_VERSIONS = {"2.0"};

//# HDF5 Information model version
const std::string H5RAD_VERSION = "H5rad 2.0";
const std::vector<std::string> H5RAD_VERSIONS = {"H5rad 1.2", "H5rad 2.0", "H5rad 2.1", "H5rad 2.2", "H5rad 2.3"};

//# ODIM_H5 version
const std::string ODIM_VERSION = "ODIM_H5/V2_0";
const std::vector<std::string> ODIM_VERSIONS = {"ODIM_H5/V2_0", "ODIM_H5/V2_1", "ODIM_H5/V2_2", "ODIM_H5/V2_3"};

//# Default text encoding
const std::string ENCODING = "iso-8859-1";
const std::string UTF8 = "UTF-8";

//# Default compression to use for DATASET nodes
//# szip is illegal with ODIM_H5
const std::string COMPRESSION = "zlib";// # use "none" (or None), "zlib", or "szip"

//# Compression level for ZLIB (0-9)
const int COMPRESSION_ZLIB_LEVEL = 6; //# default

//# Mapping between Numeric and HDF5 dataset types
const std::map<std::string,std::string> ARRAYTYPES {{"b","char"}, {"B","uchar"}, {"I","int"}, {"h","short"},{"L","long"}, {"f","float"}, {"d","double"}};

//# Default gain and offset values for linear transformation between raw and dBZ
const float GAIN = 0.4;
const float OFFSET = -30.0;

struct gain_offset {
    float gain;
    float offset;
    RaveDataType data_type;
    float nodata;
    float undetect;
};



const std::map<std::string,struct gain_offset> FACTORY_GAIN_OFFSET_TABLE { {"DBZH",{GAIN, OFFSET, RaveDataType::RaveDataType_UCHAR, 255.0, 0.0}},{"RATE",{1.0, 0.0, RaveDataType::RaveDataType_DOUBLE, -1.0, 0.0}}};



//# Default Z-R coefficients, legacy from BALTEX Working Group on Radar
const float ZR_A = 200.0;
const float ZR_b = 1.5;

//# Gauge adjustment - migrated from NORDRAD2
const std::string GADJUST_STATFILE = RAVEETC + "/gadjust.stat";
const float DEFAULTA = 0.323868068019;
const float DEFAULTB = -0.00107776407064;
const float DEFAULTC = 1.77500903316e-05;
const int MERGETERMS = 20;  //# how many 12-hour SYNOP terms to merge: 10 days.
const int TIMELIMIT_CLIMATOLOGIC_COEFF = 48; //# how many hours back in time we can use generated gra coefficients before using the climatologic variant

//# SAF-NWC MSG CT filter, not used
/*
CT_FTEMPLATE = "SAFNWC_MSG?_CT___%s_FES_________.h5"
CTPATH = "/opt/baltrad/MSG_CT"
CTDELTA = datetime.timedelta(minutes=15)
CT_MAX_DELTAS = 3  # look backwards in time for ct_max_deltas * ctdelta
*/

//# Statistics
const std::string TFILE = RAVECONFIG + "/t-critical.pickle";
const std::string TFILE_TEMPLATE = RAVECONFIG + "/t-critical.txt";

//# Projection and area registries
const std::string PROJECTION_REGISTRY = RAVECONFIG + "/projection_registry.xml";
const std::string AREA_REGISTRY = RAVECONFIG + "/area_registry.xml";

//# XML-RPC server variables
const std::string PIDFILE = RAVEETC + "/rave_pgf_server.pid";
const std::string PGF_HOST = "localhost";
const int PGF_PORT = 8085;
const int PGFs = 4;
const std::string STDOE = RAVEETC + "/rave_pgf_stdout_stderr.log";

const std::string DEX_SPOE = "http://localhost:8084/BaltradDex";
const std::string DEX_CHANNEL = "default_products";
const std::string DEX_USER = "rave_pgf";

const std::string DEX_NODENAME = "localhost";
const std::string DEX_PRIVATEKEY;

const std::string BDB_CONFIG_FILE;

//# The originating center id, used to indicate where a product has been generated.
const std::string CENTER_ID = "ORG:82"; // # Change this if your country is not Sweden.

const std::string GENREG  = "generate-registry"; //  # root registry tag
const std::string REGFILE = RAVEETC + "/rave_pgf_registry.xml"; //  # registry file

const std::string QFILE = RAVEETC + "/rave_pgf_queue.xml"; //  # queue file
const std::string PGF_TAG = "bltgenerate"; //  # used for sending files to the DEX

//# Logging - little of this is relevant if SysLog is used or the OS rotates the logs.
const std::string LOGID = "PGF[rave.baltrad.eu]";
const int LOGPORT = 8089;
const std::string LOGFILE = RAVEETC + "/rave_pgf.log"; //# Default logger is to syslog.
const int LOGFILESIZE = 5000000;  //# 5 Mb each
const int LOGFILES    = 5;
const std::string LOGFACILITY = "local3";
const std::string LOGLEVEL = "info";
const std::string LOGPIDFILE = RAVEETC + "/rave_pgf_log_server.pid";
const std::string SYSLOG_FORMAT = "%(name)s: %(levelname)-8s %(message)s";
const std::string LOGFILE_FORMAT = "%(asctime)-15s %(levelname)-8s %(message)s";

const std::string SYSLOG = "/dev/log";
    
const std::string LOGGER_TYPE="syslog"; // # Can be stdout or logfile but logfile might result in unordered log entries since there will be more than one process writing to same file

const std::string ODIM_SOURCE_FILE = RAVECONFIG + "/odim_source.xml";

const std::string QUALITY_REGISTRY= RAVEETC + "/rave_pgf_quality_registry.xml";

const std::string RAVE_TILE_REGISTRY=RAVEETC + "/rave_tile_registry.xml";

//# The name of the composite generator filter file containing the settings for factories.
const std::string COMPOSITE_GENERATOR_FILTER_FILENAME = RAVECONFIG + "/composite_generator_filter.xml";

//# Where the generator property file can be found
const std::string COMPOSITE_GENERATOR_PROPERTY_FILE = RAVECONFIG + "/rave_properties.json";

//# The location where the cluttermaps can be found when using ACQVA
//# The names of the cluttermaps should be in the format <nod>.h5, for example
//# seang.h5, ...
const std::string ACQVA_CLUTTERMAP_DIR = "/var/lib/baltrad/rave/acqva/cluttermap";

//# Max number of processes to use when performing the composite tiling. If None, then
//# the number of processes will be set to number of tiles or less depending on how many
//# cores that are available. If number of cores > 1, then there will always be one core
//# left for handling the result.
const int RAVE_TILE_COMPOSITING_PROCESSES=0;

//# Timeout in seconds when waiting for a tile to be completed. If no timeout is specified (None)
//# the wait will be indefinite. However, the recommended timeout is somewhere between
//# 1 and 15 minutes depending on the load you are expecting. The reason for this timeout
//# is that if the process creating a tile crashes (like OOM) the complete PGF will hang
//# for ever. Defined in seconds!
const int RAVE_TILE_COMPOSITING_TIMEOUT= 290;

//# If a tile is missing due to a timeout it is possible to either allow that behavior and
//# ignore the problem like if all files was missing or else let a runtime error be thrown
//# which will result in a missing composite.
const bool RAVE_TILE_COMPOSITING_ALLOW_MISSING_TILES=false;

//# Maximum number of tasks that a single worker process in a multiprocessing pool is allowed to execute
//# before it is terminated and replaced by a new process.
//# Used for managing resource utilization by preventing long-running processes.
//# Adjust this setting based on your workload characteristics and resource availability.
//# See: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool
//# Max tasks per worker must be a positive integer (>0) or None. If None it means that
//# the worker will work indefinitely.
const int RAVE_MULTIPROCESSING_MAX_TASKS_PER_WORKER=-1;

//# Max number of process to use when executing the quality controls. Default is 4 but this
//# should probably be tuned depending on how many files that needs to be quality controled
//# and number of available cores.
const int RAVE_QUALITY_CONTROL_PROCESSES=4;

//# If the quality fields should be reprocessed or not if the input already contains a relevant how/task
//# quality field.
const bool RAVE_PGF_QUALITY_FIELD_REPROCESSING=false;

//# If the compositing should utilize azimuthal navigation or not (how/astart, how/startazA)
//#
const bool RAVE_PGF_AZIMUTHAL_NAVIGATION=true;

//# If the compositing should use lazy loading or not
//#
const bool RAVE_PGF_COMPOSITING_USE_LAZY_LOADING=false;

//# If the compositing should use quantity preload when using lazy loading or not
//#
const bool RAVE_PGF_COMPOSITING_USE_LAZY_LOADING_PRELOADS=false;

//# What algorithm that should be used when performing QI-total
//#
const std::string QITOTAL_METHOD = "minimum";

//# Directory specifying the output scansun path
const std::string RAVESCANSUN_OUT;

//# Rave IO default writing version
//#RaveIO_ODIM_Version_2_2 = 2,        /**< ODIM 2.2 */
//#RaveIO_ODIM_Version_2_3 = 3         /**< ODIM 2.3 */
//#RaveIO_ODIM_Version_2_4 = 4         /**< ODIM 2.4, The default version */
const int RAVE_IO_DEFAULT_VERSION = 2;

#endif
