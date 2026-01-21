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
# @file
# @author Anders Henja, SMHI
# @date 2013-12-11

# Standard python libs:
import datetime
import math

# Third-party:

# In house project libs:

# Module/Project:
import _rave
from gadjust import grapoint
from Proj import dr, rd
import rave_pgf_logger

logger = rave_pgf_logger.rave_pgf_syslog_client()


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
    # @param acc_period - the hours back in time we are interested of
    # @param quantity - the quantity we want to use, default is ACRR
    # @param how_task - the name of the distance field. Default is se.smhi.composite.distance.radar
    # @param offset_hours - as default matching is performed from nominal time until now, if offset_hours > 0 then matching is performed
    # between nominal time and nominal time + offset_hours
    def match(self, image, acc_period=12, quantity="ACRR", how_task="se.smhi.composite.distance.radar", offset_hours=0):
        ul, lr = image.getExtremeLonLatBoundaries()
        distance = image.date
        time = image.time
        image.defaultParameter = quantity

        sdt = datetime.datetime(
            int(distance[:4]), int(distance[4:6]), int(distance[6:8]), int(time[0:2]), int(time[2:4]), int(time[4:6])
        )
        edt = datetime.datetime.now()
        if offset_hours > 0:
            edt = sdt + datetime.timedelta(hours=offset_hours)
        # We want all observations reported from nominaltime (sdt) until now
        # edt = sdt
        # sdt = sdt - datetime.timedelta(hours=offset)

        obses = self.db.get_observations_in_bbox(ul[0] * rd, ul[1] * rd, lr[0] * rd, lr[1] * rd, sdt, edt)

        xpts = 0
        xptst = 0
        xptsq = 0
        xptsv = 0

        result = []
        for obs in obses:
            if obs.accumulation_period == acc_period:
                xpts = xpts + 1
                time, value = image.getConvertedValueAtLonLat((obs.longitude * dr, obs.latitude * dr))
                if time in [_rave.RaveValueType_DATA, _rave.RaveValueType_UNDETECT]:
                    xptst = xptst + 1
                    distance = image.getConvertedQualityValueAtLonLat((obs.longitude * dr, obs.latitude * dr), how_task)
                    # distance is in unit meters in the product, for grapoints it should be stored in unit km. Thus, we convert below
                    distance = distance / 1000.0
                    if distance != None:
                        xptsq = xptsq + 1
                        if obs.liquid_precipitation >= grapoint.MIN_GMM and value >= grapoint.MIN_RMM:
                            xptsv = xptsv + 1
                            result.append(grapoint.grapoint.from_observation(time, value, distance, obs))

        logger.info("obses = %d, xpts=%d, xptst = %d, xptsq = %d, xptsv = %d" % (len(obses), xpts, xptst, xptsq, xptsv))

        return result
