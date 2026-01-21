'''
Copyright (C) 2018- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Plugin for exporting cf-convention files.
## Register in pgf with
## --name=eu.baltrad.beast.ExportCF
##
## @file
## @author Anders Henja, SMHI
## @date 2018-01-24

import _cartesian, _cartesianvolume
import _rave
import _raveio
import string
import rave_tempfile
import odim_source
import math, datetime
import rave_pgf_quality_registry
import logging
import rave_pgf_logger

logger = rave_pgf_logger.create_logger()

ravebdb = None
try:
    import rave_bdb

    ravebdb = rave_bdb.rave_bdb()
except:
    pass


## Creates a dictionary from a rave argument list
# @param arglist the argument list
# @return a dictionary
def arglist2dict(arglist):
    result = {}
    for i in range(0, len(arglist), 2):
        result[arglist[i]] = arglist[i + 1]
    return result


##
# Converts a string into a number, either int or float
# @param sval the string to translate
# @return the translated value
# @throws ValueError if value not could be translated
#
def strToNumber(sval):
    try:
        return float(sval)
    except ValueError as e:
        return int(sval)


##
# Exports a CF convention file
#
def generate(files, arguments):
    mpname = multiprocessing.current_process().name
    entertime = time.time()
    logger.info(f"[{mpname}] rave_pgf_cf_exporter_plugin.generate: Enter.")

    args = arglist2dict(arguments)

    if not _rave.isCFConventionSupported():
        logger.info(
            f"[{mpname}] rave_pgf_cf_exporter_plugin.generate: CF Conventions is not supported, ignoring export"
        )
        return None

    if len(files) != 1:
        raise AttributeError("Must provide one file to export")

    if not "filename" in args.keys():
        raise AttributeError("Must specify name of file to export")

    obj = None
    if ravebdb != None:
        obj = ravebdb.get_rave_object(files[0])
    else:
        rio = _raveio.open(files[0])
        obj = rio.object

    if not _cartesianvolume.isCartesianVolume(obj) and not _cartesian.isCartesian(obj):
        raise AttributeError("Must call plugin with cartesian products")

    filename = args["filename"]

    rio = _raveio.new()
    rio.object = obj
    rio.file_format = _raveio.RaveIO_FileFormat_CF
    rio.save(filename)

    exectime = int((time.time() - entertime) * 1000)
    logger.info(f"[{mpname}] rave_pgf_cf_exporter_plugin.generate: Exit. Generated in {exectime}.")

    return None


if __name__ == "__main__":
    import sys

    args = ["filename", sys.argv[2]]
    fname = sys.argv[1]
    generate([fname], args)
