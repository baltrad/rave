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
## Plugin for applying quality control on a volume that is initiated from the beast
## framework.
## Register in the RAVE PGF with: % pgf_registry -a -H http://<host>:<port>/RAVE
## --name=eu.baltrad.beast.applyqc_plugin --strings=source,date,time,anomaly-qc -m rave_pgf_apply_qc -f generate
## -d 'Apply quality controls on a polar volume'
##

## @file
## @author Mats Vernersson, SMHI
## @date 2016-04-15

import _raveio
import string
import rave_tempfile
import rave_pgf_quality_registry
import rave_pgf_logger
import rave_util

logger = rave_pgf_logger.create_logger()

ravebdb = None
try:
  import rave_bdb
  ravebdb = rave_bdb.rave_bdb()
except:
  pass

## Creates a dictionary from a rave argument list
#@param arglist the argument list
#@return a dictionary
def arglist2dict(arglist):
  result={}
  for i in range(0, len(arglist), 2):
    result[arglist[i]] = arglist[i+1]
  return result

##
# Performs a quality control sequence on a volume
# @param volume: the volume to perform the quality controls on
# @param detectors: the detectors that should be run on the volume
#
def perform_quality_control(volume, detectors):
  for d in detectors:
    p = rave_pgf_quality_registry.get_plugin(d)
    if p != None:
      volume = p.process(volume)
      if isinstance(volume,tuple):
        volume, _ = volume[0],volume[1]
  return volume

def generate_new_volume_with_qc(original_file, args):
  logger.debug("Generating new volume with quality controls applied.")
  
  if ravebdb != None:
    volume = ravebdb.get_rave_object(original_file)
  else:
    volume = _raveio.open(original_file).object
    
  if "remove-malfunc" in args.keys():
    try:
      if args["remove-malfunc"].lower() in ["true", "yes", "y", "1"]:
        logger.debug("Checking volume for malfunc tags. Will remove scans, or complete volume, if marked malfunc.")
        volume = rave_util.remove_malfunc(volume)
        if volume == None:
          logger.info("Malfunc volume! Since option 'remove_malfunc' is set, no new volume with QC applied will be generated!")
          return None
    except:
      pass
  
  if "anomaly-qc" in args.keys():
    detectors = string.split(args["anomaly-qc"], ",")
  else:
    detectors = []

  volume = perform_quality_control(volume, detectors)
  
  logger.debug("Quality controls applied on new volume: %s", (",".join(detectors)))

  new_time = args.get('time')
  if new_time:
    volume.time = new_time
    
  new_date = args.get('date')
  if new_date:
    volume.date = new_date
  
  return volume
      
## Creates a new volume based on the incoming with quality controls applied to it
#@param files a list of files to apply quality controls on. currently assume only one file 
#@param arguments the arguments defining what quality controls to apply
#@return a temporary h5 file with the volume
def generate(files, arguments):
  args = arglist2dict(arguments)
  
  logger.debug("rave_pgf_apply_qc_plugin called with arguments: %s", args)

  # should only be one file
  fname = files[0]
  
  volume = generate_new_volume_with_qc(fname, args)
  
  if volume == None:
    logger.info("No volume with QC applied could be generated!")
    return None

  _, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")
  
  ios = _raveio.new()
  ios.object = volume
  ios.filename = outfile
  ios.save()
  
  logger.info("Generated new volume with QC applied.")
  
  return outfile
  
