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
## Plugin for generating a composite that is initiated from the beast
## framework.
## Register in pgf with
## --name=eu.baltrad.beast.generatecomposite --strings=area,quantity,method,date,time --floats=height
## -m rave_pgf_composite_plugin -f generate
##

## @file
## @author Anders Henja, SMHI
## @date 2010-10-15

import _cartesian
import _pycomposite
import _rave
import _area
import _projection
import _raveio
import area
import string
import rave_tempfile

from rave_defines import CENTER_ID, GAIN, OFFSET


## Creates a dictionary from a rave argument list
#@param arglist the argument list
#@return a dictionary
def arglist2dict(arglist):
  result={}
  for i in range(0, len(arglist), 2):
    result[arglist[i]] = arglist[i+1]
  return result

## Creates a composite
#@param files the list of files to be used for generating the composite
#@param arguments the arguments defining the composite
#@return a temporary h5 file with the composite
def generate(files, arguments):
  args = arglist2dict(arguments)
  
  generator = _pycomposite.new()
  
  a = area.area(args["area"])
  p = a.pcs
  pyarea = _area.new()
  pyarea.id = a.Id
  pyarea.xsize = a.xsize
  pyarea.ysize = a.ysize
  pyarea.xscale = a.xscale
  pyarea.yscale = a.yscale
  pyarea.extent = a.extent
  pyarea.projection = _projection.new(p.id, p.name, string.join(p.definition, ' '))

  for fname in files:
    rio = _raveio.open(fname)
    generator.add(rio.object)
    
  generator.quantity = "DBZH"

  if "quantity" in args.keys():
    generator.quantity = args["quantity"]
  
  method = "pcappi"
  if "method" in args.keys():
    method = args["method"]
  if method == "ppi":
    generator.product = _rave.Rave_ProductType_PPI
  elif method == "cappi":
    generator.product = _rave.Rave_ProductType_CAPPI
  else:
    generator.product = _rave.Rave_ProductType_PCAPPI

  generator.height = 1000.0
  if "height" in args.keys():
    generator.height = args["height"]
    
  generator.time = args["time"]
  generator.date = args["date"]
  generator.gain = GAIN
  generator.offset = OFFSET
  result = generator.nearest(pyarea)
  
  # Fix so that we get a valid place
  plc = result.source
  result.source = "%s,CMT:%s"%(CENTER_ID,plc)
  
  fileno, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")
  
  ios = _raveio.new()
  ios.object = result
  ios.filename = outfile
  ios.save()
  
  return outfile
  
  