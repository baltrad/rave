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
##
# Matcher for extracting observations that matches the provided acrr composite product
#
import datetime, math
import _rave
from gadjust import grapoint
from Proj import dr, rd
import rave_pgf_logger
## 
# @file
# @author Anders Henja, SMHI
# @date 2013-12-11
logger = rave_pgf_logger.rave_pgf_logger_client()

##
# Class for extracting relevant observations that are covered by the provided area
#
class obsmatcher(object):
  def __init__(self, domdb):
    self.db = domdb

  ## Extract pixel values matching the location of each point in a composite.
  # @param points list of point instances
  # @param image composite Cartesian instance. TODO: generalize?
  # @param quantity string quantity, most often "ACRR"
  # @param image_idx int index of the image to read, should be 0
  # @param how_task string task identifier of the distance quality field
  
  ## Extract observations that are covered by the provided image. The image must have a distance quality field
  # and a relevant quantity (for example ACRR).
  # @param image - the cartesian image.
  # @param offset - the hours back in time we are interested of
  # @param quantity - the quantity we want to use, default is ACRR
  # @param how_task - the name of the distance field. Default is se.smhi.composite.distance.radar
  def match(self, image, offset=12, quantity="ACRR", how_task="se.smhi.composite.distance.radar"):
    ul,lr=image.getExtremeLonLatBoundaries()
    d = image.date
    t = image.time
    image.defaultParameter=quantity
    
    sdt = datetime.datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(t[0:2]), int(t[2:4]), int(t[4:6]))
    edt = sdt
    sdt = sdt - datetime.timedelta(hours=offset)

    obses = self.db.get_observations_in_bbox(ul[0]*rd, ul[1]*rd, lr[0]*rd, lr[1]*rd, sdt, edt)
    
    result = []
    for obs in obses:
      t, v = image.getConvertedValueAtLonLat((obs.longitude*dr, obs.latitude*dr))
      if t in [_rave.RaveValueType_DATA, _rave.RaveValueType_UNDETECT]:
        d = image.getQualityValueAtLonLat((obs.longitude*dr, obs.latitude*dr), how_task)
        if d != None:
          if obs.liquid_precipitation > 0.0 and v > 0.0:
            result.append(grapoint.grapoint.from_observation(t, v, d, obs))

    return result 
