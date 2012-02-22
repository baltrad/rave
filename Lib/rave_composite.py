'''
Copyright (C) 2012- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Python interface to composite generation functionality.
#  This module is based on the PGF plugin. For the time
#  being we will live with some code duplication.

## @file
## @author Anders Henja and Daniel Michelson, SMHI
## @date 2012-01-21

import _cartesian
import _pycomposite
import _rave
import _area
import _projection
import _raveio
import _polarvolume
import _polarscan
import _transform
import rave_area
import string
import rave_tempfile
import odim_source
import math
import rave_pgf_quality_registry

from rave_defines import CENTER_ID, GAIN, OFFSET


## Generates a composite
#@param list of objects (usually polar volumes or scans) in memory used to generate the composite
#@param args tuple containing variable-length number of arguments
#@return a composite object in memory
def generate(in_objects, **args):  
  generator = _pycomposite.new()

  a = rave_area.area(args["area"])
  p = a.pcs
  pyarea = _area.new()
  pyarea.id = a.Id
  pyarea.xsize = a.xsize
  pyarea.ysize = a.ysize
  pyarea.xscale = a.xscale
  pyarea.yscale = a.yscale
  pyarea.extent = a.extent
  pyarea.projection = _projection.new(p.id, p.name, string.join(p.definition, ' '))

  if "qc" in args.keys():
      detectors = string.split(args["qc"], ",")
  else:
      detectors = []

  nodes = ""
  qfields = []  # The quality fields we want to get as a result in the composite

  for d in detectors:
    p = rave_pgf_quality_registry.get_plugin(d)
    if p != None:
      qfields.extend(p.getQualityFields())

  for obj in in_objects:

    if len(nodes):
      nodes += ",'%s'" % odim_source.NODfromSource(obj)
    else:
      nodes += "'%s'" % odim_source.NODfromSource(obj)
    
    for d in detectors:
      p = rave_pgf_quality_registry.get_plugin(d)
      if p != None:
        obj = p.process(obj)
        na = p.algorithm()
        if generator.algorithm == None and na != None: # Try to get the generator algorithm != None 
          generator.algorithm = na

    generator.add(obj)

  quantity = "DBZH"
  gain = GAIN
  offset = OFFSET
  if "quantity" in args.keys():
    quantity = args["quantity"].upper()
  if "gain" in args.keys():
    gain = args["gain"]
  if "offset" in args.keys():
    offset = args["offset"]

  generator.addParameter(quantity, gain, offset)
  
  product = "pcappi"
  if "product" in args.keys():
    product = args["product"].lower()

  if product == "ppi":
    generator.product = _rave.Rave_ProductType_PPI
  elif product == "cappi":
    generator.product = _rave.Rave_ProductType_CAPPI
  else:
    generator.product = _rave.Rave_ProductType_PCAPPI

  generator.height = 1000.0
  generator.elangle = 0.0
  if "prodpar" in args.keys():
    if generator.product in [_rave.Rave_ProductType_CAPPI, _rave.Rave_ProductType_PCAPPI]:
      try:
        generator.height = args["prodpar"]
      except ValueError,e:
        pass
    elif generator.product in [_rave.Rave_ProductType_PPI]:
      try:
        v = args["prodpar"]
        generator.elangle = v * math.pi / 180.0
      except ValueError,e:
        pass

  generator.selection_method = _pycomposite.SelectionMethod_NEAREST
  if "method" in args.keys():
    if args["method"].upper() == "NEAREST_RADAR":
      generator.selection_method = _pycomposite.SelectionMethod_NEAREST
    elif args["method"].upper() == "HEIGHT_ABOVE_SEALEVEL":
      generator.selection_method = _pycomposite.SelectionMethod_HEIGHT

  generator.date = obj.date # First guess: date of last input object
  generator.time = obj.time # A bit risky if nominal times of input data are different
  if "date" in args.keys() and args["date"] is not None:
      generator.time = args["time"]
  if "time" in args.keys() and args["time"] is not None:
      generator.date = args["date"]

  result = generator.nearest(pyarea, qfields)  # Might want to rename this method...

  # Optional gap filling
  if eval(args["gf"]):
      t = _transform.new()
      gap_filled = t.fillGap(result)
      result.setData(gap_filled.getData())      
  
  # Fix so that we get a valid place for /what/source and /how/nodes 
  plc = result.source
  result.source = "%s,CMT:%s"%(CENTER_ID,plc)
  result.addAttribute('how/nodes', nodes)
  
  return result
