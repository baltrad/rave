'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Plugin for generating a volume that is initiated from the beast
## framework.
## Register in the RAVE PGF with: % pgf_registry -a -H http://<host>:<port>/RAVE
## --name=eu.baltrad.beast.generatevolume --strings=source,date,time,anomaly-qc -m rave_pgf_volume_plugin -f generate
## -d 'Polar volume generation from individual scans'
##

## @file
## @author Anders Henja, SMHI
## @date 2011-01-19

# Standard python libs:
import re
import string

# Module/Project:
import _rave
import _raveio
import _polarvolume
import _polarscan
import _polarscanparam
import rave_tempfile
import odim_source
import rave_pgf_quality_registry
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY
from rave_defines import RAVE_IO_DEFAULT_VERSION

from rave_defines import CENTER_ID

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


## Generates a volume
# @param files the list of files to be used for generating the volume
# @param args the dictionary containing the arguments
# @return the volume
def generateVolume(files, args):
    if len(files) <= 0:
        raise AttributeError("Volume must consist of at least 1 scan")

    firstscan = False
    volume = _polarvolume.new()
    volume.date = args['date']
    volume.time = args['time']

    #'longitude', 'latitude', 'height', 'time', 'date', 'source'

    for fname in files:
        scan = None
        if ravebdb != None:
            scan = ravebdb.get_rave_object(fname)
        else:
            scan = _raveio.open(fname).object
        if firstscan == False:
            firstscan = True
            volume.longitude = scan.longitude
            volume.latitude = scan.latitude
            volume.height = scan.height
            volume.beamwidth = scan.beamwidth
        volume.addScan(scan)

    volume.source = scan.source  # Recycle the last input, it won't necessarily be correct ...
    odim_source.CheckSource(volume)  # ... so check it!
    return volume


##
# Performs a quality control sequence on a volume
# @param volume: the volume to perform the quality controls on
# @param detectors: the detectors that should be run on the volume
#
def perform_quality_control(volume, detectors, qc_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY):
    for d in detectors:
        p = rave_pgf_quality_registry.get_plugin(d)
        if p != None:
            volume = p.process(volume, True, qc_mode)
            if isinstance(volume, tuple):
                volume, algorithm = volume[0], volume[1]
    return volume


## Creates a volume
# @param files the list of files to be used for generating the volume
# @param arguments the arguments defining the volume
# @return a temporary h5 file with the volume
def generate(files, arguments):
    args = arglist2dict(arguments)

    quality_control_mode = QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY

    volume = generateVolume(files, args)

    fileno, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")

    if "anomaly-qc" in args.keys():
        detectors = args["anomaly-qc"].split(",")
    else:
        detectors = []

    if "qc-mode" in args.keys():
        quality_control_mode = args["qc-mode"]

    volume = perform_quality_control(volume, detectors, quality_control_mode.lower())

    ios = _raveio.new()
    ios.object = volume
    ios.filename = outfile
    ios.version = RAVE_IO_DEFAULT_VERSION
    ios.save()

    return outfile
