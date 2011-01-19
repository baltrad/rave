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
## Register in pgf with
## --name=eu.baltrad.beast.generatevolume --strings=source,date,time -m rave_pgf_volume_plugin -f generate
##

## @file
## @author Anders Henja, SMHI
## @date 2011-01-19

import _rave
import _raveio
import _polarvolume
import _polarscan
import _polarscanparam
import string
import rave_tempfile

from rave_defines import CENTER_ID


## Creates a dictionary from a rave argument list
#@param arglist the argument list
#@return a dictionary
def arglist2dict(arglist):
  result={}
  for i in range(0, len(arglist), 2):
    result[arglist[i]] = arglist[i+1]
  return result

## Generates a volume
#@param files the list of files to be used for generating the volume
#@param args the dictionary containing the arguments
#@return the volume
def generateVolume(files, args):
  if len(files) <=0:
    raise AttributeError, "Volume must consist of at least 1 scan"

  firstscan=False  
  volume = _polarvolume.new()
  volume.date = args['date']
  volume.time = args['time']
  
  #'longitude', 'latitude', 'height', 'time', 'date', 'source'

  for fname in files:
    rio = _raveio.open(fname)
    if firstscan == False:
      firstscan = True
      volume.longitude = rio.object.longitude
      volume.latitude = rio.object.latitude
      volume.height = rio.object.height
      volume.source = "%s,CMT:%s"%(rio.object.source,args['source'])
    volume.addScan(rio.object)
  
  return volume

## Creates a volume
#@param files the list of files to be used for generating the volume
#@param arguments the arguments defining the volume
#@return a temporary h5 file with the volume
def generate(files, arguments):
  args = arglist2dict(arguments)
  
  volume = generateVolume(files, args)
  
  fileno, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")
  
  ios = _raveio.new()
  ios.object = volume
  ios.filename = outfile
  ios.save()
  
  return outfile
  
  