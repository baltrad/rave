'''
Copyright (C) 2013- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Plugin for clutter management using hit accumulations: HAC
## This plugin is intended for real-time incrementation of the HAC files.
## Filtering is better done through the use of a quality plugin.
## Register in the RAVE PGF with: % pgf_registry -a -H http://<host>:<port>/RAVE
## --name=eu.opera.odc.hacincrementor -m odc_hac_pgf_plugin -f generate
## -d 'Hit-accumulation incrementor'
##

## @file
## @author Daniel Michelson, SMHI
## @date 2013-01-25

import _raveio
import odc_hac

ravebdb = None
try:
  import rave_bdb
  ravebdb = rave_bdb.rave_bdb()
except:
  pass

## Creates a dictionary from a RAVE argument list
# @param arglist the argument list
# @return a dictionary
def arglist2dict(arglist):
  result={}
  for i in range(0, len(arglist), 2):
    result[arglist[i]] = arglist[i+1]
  return result


## Increments HAC file(s) for the given object.
# @param files the list of files to be used for generating the volume. Should be only one, normally.
# @param arguments the arguments defining the volume. Should be empty in this first version; the desired quantity can be added later.
def generate(files, arguments):
  args = arglist2dict(arguments)
  
  for fname in files:
    obj = None
    if ravebdb != None:
      obj = ravebdb.get_rave_object(fname)
    else:
      obj = _raveio.open(fname).object

    odc_hac.hacIncrement(obj)
  
  return None
