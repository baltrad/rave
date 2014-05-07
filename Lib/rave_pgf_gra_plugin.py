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
## Plugin for generating the gra coefficients that is initiated from the beast
## framework.
## Register in pgf with
## --name=eu.baltrad.beast.GenerateGraCoefficients
## --strings=date,time,quantity,distancefield
## --floats=zra,zrb --ints=hours,N,accept -m rave_pgf_acrr_plugin -f generate
##
## The ACRR generation is executed by specifying a number of composites/images
## with the same resolution and containing the same parameter (quantity) and
## a quality field specifying the distance to the radar.
##
## Then the gra coefficient algorithm will create an acrr product and after that
## match synop-entries with corresponding points in the acrr product and generate
## the coefficients.
##
## acrr = new acrr
## for each file in files:
##   acrr.sum (file.parameter(quantity), zrA, zrB)
## result = acrr.accumulate(accept, N, hours)
##
## accept is the percent of N that is allowed to be nodata for each observation
## N is the expected number of fields to be used in the accumulation
## hours is the number of hours the accumulation covers
## 
## @file
## @author Anders Henja, SMHI
## @date 2013-08-08

import _cartesian, _cartesianvolume
import _acrr
import _rave
import _raveio
import string
import rave_tempfile
import odim_source
import math, datetime
import rave_pgf_quality_registry
import logging
import rave_pgf_logger
from gadjust import gra, obsmatcher
from gadjust.gra import gra_coefficient
import odim_source
import rave_dom_db

from rave_defines import CENTER_ID, GAIN, OFFSET, MERGETERMS

logger = rave_pgf_logger.rave_pgf_syslog_client()
  
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
    return int(sval)
  except ValueError, e:
    return float(sval)

## Creates a composite
#@param files the list of files to be used for generating the composite
#@param arguments the arguments defining the composite
#@return a temporary h5 file with the composite
def generate(files, arguments):
  args = arglist2dict(arguments)
  
  zr_a = 200.0
  zr_b = 1.6
  quantity = "DBZH"
  accept = 0.0
  distancefield = "se.smhi.composite.distance.radar"
  interval = 12
  N = 13
  adjustmentfile = None
  
  #Accept is the required limit for how many nodata-pixels that are allowed in order for the
  #data to be accumulated
  #If we expect to have 10 observations and an accept limit of 20, then it can be 0, 1 or 2 observations
  #with nodata for that position.
  
  etime = args["time"]
  edate = args["date"]
  
  acrrproduct = None
  
  if "zra" in args.keys():
    zr_a = strToNumber(args["zra"])
  if "zrb" in args.keys():
    zr_b = strToNumber(args["zrb"])
  if "quantity" in args.keys():
    quantity = args["quantity"]
  if "accept" in args.keys():
    accept = strToNumber(args["accept"]) / 100.0 
  if "distancefield" in args.keys():
    distancefield = args["distancefield"]
  if "interval" in args.keys():
    interval = strToNumber(args["interval"])
  if "N" in args.keys():
    N = strToNumber(args["N"])
  if "adjustmentfile" in args.keys():
    adjustmentfile = args["adjustmentfile"]
    
  if distancefield == "eu.baltrad.composite.quality.distance.radar":
    distancefield = "se.smhi.composite.distance.radar"
  
  acrr = _acrr.new()
  acrr.nodata = -1.0
  acrr.undetect = 0.0
  acrr.quality_field_name = distancefield

  for fname in files:
    obj = None
    if ravebdb != None:
      obj = ravebdb.get_rave_object(fname)
    else:
      rio = _raveio.open(fname)
      obj = rio.object

    if _cartesianvolume.isCartesianVolume(obj):
      obj = obj.getImage(0)

    if not _cartesian.isCartesian(obj):
      raise AttributeError, "Must call plugin with cartesian products"

    if acrrproduct == None:
      acrrproduct = _cartesian.new()
      acrrproduct.xscale = obj.xscale
      acrrproduct.yscale = obj.yscale
      acrrproduct.areaextent = obj.areaextent
      acrrproduct.projection = obj.projection
      acrrproduct.product = obj.product
      acrrproduct.source = obj.source
      acrrproduct.time = etime
      acrrproduct.date = edate

    if obj.xscale != acrrproduct.xscale or obj.yscale != acrrproduct.yscale or \
       obj.projection.definition != acrrproduct.projection.definition:
      raise AttributeError, "Scale or projdef inconsistancy for used area"

    par = obj.getParameter(quantity)
    if par == None:
      logger.warn("Could not find parameter (%s) for %s %s"%(quantity, obj.date, obj.time))
    else:
      if par.getQualityFieldByHowTask(distancefield) != None:
        acrr.sum(par, zr_a, zr_b)

  # accept, N, hours
  acrrparam = acrr.accumulate(accept, N, interval)
  acrrproduct.addParameter(acrrparam)
  
  db = rave_dom_db.create_db_from_conf()
  
  matcher = obsmatcher.obsmatcher(db)
  
  points = matcher.match(acrrproduct, offset=interval, quantity="ACRR", how_task=distancefield)
  if len(points) == 0:
    logger.warn("Could not find any matching observations")
    return None
  
  logger.info("Matched %d points between acrr product and observation db"%len(points))
  db.merge(points)

  d = acrrproduct.date
  t = acrrproduct.time

  tlimit = datetime.datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(t[0:2]), int(t[2:4]), int(t[4:6]))
  tlimit = tlimit - datetime.timedelta(hours=interval*MERGETERMS)
  dlimit = datetime.datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(t[0:2]), int(t[2:4]), int(t[4:6]))
  dlimit = dlimit - datetime.timedelta(hours=12*MERGETERMS)
  
  db.delete_grapoints(dlimit) # We don't want any points older than 12 hour * MERGETERMS back in time
  
  points = db.get_grapoints(tlimit); # Get all gra points newer than interval*MERGETERMS hours back in time
  logger.info("Using %d number of points for calculating the gra coefficients"%len(points))
  
  if adjustmentfile != None:
    significant, npoints, loss, r, sig, corr_coeff, a, b, c, m, dev = gra.generate(points, edate, etime, adjustmentfile)
  else:
    significant, npoints, loss, r, sig, corr_coeff, a, b, c, m, dev = gra.generate(points, edate, etime)

  # Also store the coefficients in the database so that we can search for them when applying the coefficients
  NOD = odim_source.NODfromSource(acrrproduct)
  if not NOD:
    NOD = ""
  grac = gra_coefficient(NOD, acrrproduct.date, acrrproduct.time, significant, npoints, loss, r, sig, corr_coeff, a, b, c, float(m), float(dev))
  db.merge(grac)

  return None
