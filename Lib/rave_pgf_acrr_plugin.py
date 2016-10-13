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
## Plugin for generating the acrr product generation that is initiated from the beast
## framework.
## Register in pgf with
## --name=eu.baltrad.beast.GenerateAcrr
## --strings=date,time,quantity,distancefield
## --floats=zra,zrb --ints=hours,N,accept -m rave_pgf_acrr_plugin -f generate
##
## The ACRR generation is executed by specifying a number of composites/images
## with the same resolution and containing the same parameter (quantity) and
## a quality field specifying the distance to the radar.
##
## Then, the acrr generation takes place by first adding each field and as a final
## step accumulating the result.
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

import _cartesian
import _cartesianvolume
import _acrr
import _rave
import _raveio
import _gra
import string
import datetime
import rave_tempfile
import odim_source
import math
import rave_pgf_quality_registry
import logging
import rave_pgf_logger

from rave_defines import CENTER_ID, GAIN, OFFSET
from rave_defines import DEFAULTA, DEFAULTB, DEFAULTC

logger = rave_pgf_logger.create_logger()

ravebdb = None
try:
  import rave_bdb
  ravebdb = rave_bdb.rave_bdb()
except:
  pass

import rave_dom_db

## Creates a dictionary from a rave argument list
#@param arglist the argument list
#@return a dictionary
def arglist2dict(arglist):
  result={}
  for i in range(0, len(arglist), 2):
    result[arglist[i]] = arglist[i+1]
  return result


##
# Returns the backup coefficients to use. First the newest coefficient between
# dt - maxage <= found <= dt is located. If none is found, then the climatologic
# coefficients are used instead.
#
def get_backup_gra_coefficient(db, agedt, nowdt):
  try:
    coeff = db.get_newest_gra_coefficient(agedt, nowdt)
    if coeff and not math.isnan(coeff.a) and not math.isnan(coeff.b) and not math.isnan(coeff.c):
      logger.info("Reusing gra coefficients from %s %s"%(coeff.date, coeff.time))
      return coeff.significant, coeff.points, coeff.loss, coeff.r, coeff.r_significant, coeff.corr_coeff, coeff.a, coeff.b, coeff.c, coeff.mean, coeff.stddev
  except Exception, e:
    logger.exception("Failed to aquire coefficients")

  logger.warn("Could not aquire coefficients newer than %s, defaulting to climatologic"%agedt.strftime("%Y%m%d %H:%M:%S"))
  return "False", 0, 0, 0.0, "False", 0.0, DEFAULTA, DEFAULTB, DEFAULTC, 0.0, 0.0

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
  hours = 1
  N = 5
  applygra = False
  
  #Accept is the required limit for how many nodata-pixels that are allowed in order for the
  #data to be accumulated
  #If we expect to have 10 observations and an accept limit of 20, then it can be 0, 1 or 2 observations
  #with nodata for that position.
  
  etime = args["time"]
  edate = args["date"]
  
  img = None
  
  if "zra" in args.keys():
    zr_a = float(args["zra"])
  if "zrb" in args.keys():
    zr_b = float(args["zrb"])
  if "quantity" in args.keys():
    quantity = args["quantity"]
  if "accept" in args.keys():
    accept = float(args["accept"]) / 100.0 
  if "distancefield" in args.keys():
    distancefield = args["distancefield"]
  if "hours" in args.keys():
    hours = int(args["hours"])
  if "N" in args.keys():
    N = int(args["N"])
  if args.has_key("applygra"):
    applygra = True
  
  if distancefield == "eu.baltrad.composite.quality.distance.radar":
    distancefield = "se.smhi.composite.distance.radar"

  pdatetime = datetime.datetime.strptime(edate+etime, "%Y%m%d%H%M%S") - datetime.timedelta(minutes=60 * hours)
  sdate = pdatetime.strftime("%Y%m%d")
  stime = pdatetime.strftime("%H%M00")
  
  acrr = _acrr.new()
  acrr.nodata = -1.0
  acrr.undetect = 0.0
  acrr.quality_field_name = distancefield

  nodes = None

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

    if img == None:
      img = _cartesian.new()
      img.xscale = obj.xscale
      img.yscale = obj.yscale
      img.areaextent = obj.areaextent
      img.projection = obj.projection
      img.product = _rave.Rave_ProductType_RR
      img.source = obj.source
      img.time = etime
      img.date = edate
      img.startdate = sdate
      img.starttime = stime
      img.enddate = edate
      img.endtime = etime

    if obj.hasAttribute("how/nodes") and nodes == None:
      nodes = obj.getAttribute("how/nodes")

    if obj.xscale != img.xscale or obj.yscale != img.yscale or \
      obj.projection.definition != img.projection.definition:
      raise AttributeError, "Scale or projdef inconsistancy for used area"

    par = obj.getParameter(quantity)
    if par == None:
      logger.warn("Could not find parameter (%s) for %s %s"%(quantity, obj.date, obj.time))
    else:
      if par.getQualityFieldByHowTask(distancefield) != None:
        acrr.sum(par, zr_a, zr_b)

  # accept, N, hours
  result = acrr.accumulate(accept, N, hours)

  fileno, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")
  if nodes != None:
    img.addAttribute("how/nodes", nodes)

  img.addParameter(result)  
  
  #logger.info("Apply gra: %s"%`applygra`)
  if applygra:
    db = rave_dom_db.create_db_from_conf()
    dt = datetime.datetime(int(edate[:4]), int(edate[4:6]), int(edate[6:]), int(etime[:2]), int(etime[2:4]), 0)
    dt = dt - datetime.timedelta(seconds=3600 * 12) # 12 hours back in time for now..
    
    gra = _gra.new()
    gra.A = DEFAULTA
    gra.B = DEFAULTB
    gra.C = DEFAULTC
    gra.zrA = zr_a
    gra.zrb = zr_b
    
    grac = db.get_gra_coefficient(dt)
    if grac != None and not math.isnan(grac.a) and not math.isnan(grac.b) and not math.isnan(grac.c):
      logger.debug("Using gra coefficients from database, quantity: %s"%quantity)
      gra.A = grac.a
      gra.B = grac.b
      gra.C = grac.c
    else:
      logger.info("Could not find coefficients for given time, trying to get aged or climatologic coefficients")
      nowdt = datetime.datetime(int(edate[:4]), int(edate[4:6]), int(edate[6:]), int(etime[:2]), int(etime[2:4]), 0)
      agedt = nowdt - datetime.timedelta(seconds=3600 * 48) # 2 days back
      sig,pts,loss,r,rsig,corr,gra.A,gra.B,gra.C,mean,dev = get_backup_gra_coefficient(db, agedt, nowdt)
      
    dfield = result.getQualityFieldByHowTask(distancefield)
    
    gra_field = gra.apply(dfield, result)
      
    gra_field.quantity = result.quantity + "_CORR"
    img.addParameter(gra_field)

  ios = _raveio.new()
  ios.object = img
  ios.filename = outfile
  ios.save()

  return outfile
