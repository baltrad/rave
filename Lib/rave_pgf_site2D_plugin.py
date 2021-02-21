'''
Copyright (C) 2014- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Plugin for scanning a polar volume for sun hits, using the RAVE product 
## generation framework.
## Register in the RAVE PGF with: % pgf_registry -a -H http://<host>:<port>/RAVE --name=eu.baltrad.rave.site2D --strings=area,pcsid,quantity,product,gf,ctfilter --floats=scale,prodpar,range --sequences=qc -m rave_pgf_site2D_plugin -f generate -d '2-D single-site Cartesian product generator'

## @file
## @author Daniel Michelson, SMHI
## @date 2014-04-01
import string
import datetime
import rave_tempfile
import logging
import rave_pgf_logger
import rave_dom_db
import rave_util
import _raveio, _rave

from compositing import compositing

from rave_defines import CENTER_ID, GAIN, OFFSET
from rave_defines import RAVE_IO_DEFAULT_VERSION

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
# Converts a string into a number, either int or float
# @param sval the string to translate
# @return the translated value
# @throws ValueError if value not could be translated
#
def strToNumber(sval):
  try:
    return float(sval)
  except ValueError:
    return int(sval)


## Performs
# @param files list containing a single file string.
# @arguments list containing arguments for the generator
# @return temporary H5 file containing the generated product
## Creates a composite
#@param files the list of files to be used for generating the composite
#@param arguments the arguments defining the composite
#@return a temporary h5 file with the composite
def generate(files, arguments):
  args = arglist2dict(arguments)
  
  comp = compositing(ravebdb)
  if len(files) != 1:
    raise AttributeError("Input files list must contain only one file string")

  comp.filenames = files
  
  if "anomaly-qc" in args.keys():
    comp.detectors = args["anomaly-qc"].split(",")

  if "qc-mode" in args.keys():
    comp.set_quality_control_mode_from_string(args["qc-mode"])
    
  if "ignore-malfunc" in args.keys():
    try:
      if args["ignore-malfunc"].lower() in ["true", "yes", "y", "1"]:
        comp.ignore_malfunc = True
    except:
      pass

  comp.quantity = "DBZH"
  if "quantity" in args.keys():
    comp.quantity = args["quantity"]
  comp.gain = GAIN
  comp.offset = OFFSET

  comp.set_product_from_string("pcappi")
  if "method" in args.keys():
    comp.set_product_from_string(args["method"].lower())

  comp.height = 1000.0
  comp.elangle = 0.0
  comp.range = 200000.0

  if "prodpar" in args.keys():
    comp.prodpar = args["prodpar"]

  if "range" in args.keys() and comp.product == _rave.Rave_ProductType_PMAX:
    comp.range = float(args["range"])
  
  if "pcsid" in args.keys():
    comp.pcsid = args["pcsid"]
  
  if "xscale" in args.keys():
    comp.xscale = float(args["xscale"])
    
  if "yscale" in args.keys():
    comp.yscale = float(args["yscale"])
  
  #if options.gf: Activate gap filling for rule
  #  comp.applygapfilling = True
  
  # Optional cloud-type residual non-precip filter
  if "ctfilter" in args:
    if eval(args["ctfilter"]):
      comp.applyctfilter = True
  
  if "applygra" in args:
    comp.applygra = True
  if "zrA" in args:
    comp.zr_A = float(args["zrA"])
  if "zrb" in args:
    comp.zr_b = float(args["zrb"])
  
  if "pcsid" in args:
    comp.pcsid = args["pcsid"]
    comp.xscale = float(args["xscale"])
    comp.yscale = float(args["yscale"])
  
  areaid = None
  if "area" in args:
    areaid = args["area"]
  
  comp.use_site_source = True
  
  result = comp.generate(None, None, areaid)
  
  if result == None:
    logger.info("No site2D-composite could be generated.")
    return None

  result.objectType = _rave.Rave_ObjectType_IMAGE 
  
  fileno, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")
  
  rio = _raveio.new()
  rio.object = result
  rio.filename = outfile
  rio.version = RAVE_IO_DEFAULT_VERSION
  rio.save()

  return outfile


if __name__ == '__main__':
    pass
