#!/usr/bin/env python
'''
Copyright (C) 2014- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''

## Identification and removal of residual non-precipitation echoes using the
## Meteosat Second Generation Cloud-Type product: CTFILTER

## @file
## @author Daniel Michelson, SMHI
## @date 2014-03-26
# Standard python libs:
import sys
import os
import glob
import datetime

# In house project libs:
import _pyhl

# Module/Project:
import _rave, _raveio
import _cartesian, _cartesianparam
import _projection
import _ctfilter
from rave_defines import CT_FTEMPLATE, CTPATH, CTDELTA, CT_MAX_DELTAS


NODENAMES = [
    "/XGEO_UP_LEFT",
    "/YGEO_UP_LEFT",
    "/XGEO_LOW_RIGHT",
    "/YGEO_LOW_RIGHT",
    "/PROJECTION",
    "/CT",
    "/CT/N_LINES",
    "/CT/N_COLS",
]


## MSG CT file reader
# @param filename string of the input MSG CT file
# @return Cartesian object representing the CT product
def readCT(filename):
    ct, cp = _cartesian.new(), _cartesianparam.new()
    
    nodelist = _pyhl.read_nodelist(filename)
    for n in NODENAMES:
        nodelist.selectNode(n)
    nodelist.fetch()
    
    ct.defaultParameter = "CT"
    ct.projection = _projection.new("MSG", "no description", nodelist.getNode("/PROJECTION").data())
    
    cp.setData(nodelist.getNode("/CT").data())
    
    ysize = nodelist.getNode("/CT/N_LINES").data()
    xsize = nodelist.getNode("/CT/N_COLS").data()
    ULx = nodelist.getNode("/XGEO_UP_LEFT").data()
    ULy = nodelist.getNode("/YGEO_UP_LEFT").data()
    LRx = nodelist.getNode("/XGEO_LOW_RIGHT").data()
    LRy = nodelist.getNode("/YGEO_LOW_RIGHT").data()
    yscale = (ULy - LRy) / ysize
    xscale = (LRx - ULx) / xsize
    xoffset, yoffset = xscale / 2, yscale / 2  # Offsets to adjust LL and UR corners
    LLx = LRx - (xsize * xscale) - xoffset
    LLy = ULy - (ysize * yscale) - yoffset
    URx = LRx + xoffset
    URy = ULy + yoffset
    ct.areaextent = (LLx, LLy, URx, URy)  # Differs ~5 cm from PPS and PyTROLL
    ct.xscale, ct.yscale = xscale, yscale
    cp.quantity = "CT"
    cp.gain, cp.offset = 1.0, 0.0
    cp.nodata, cp.undetect = -1.0, 0.0
    ct.addParameter(cp)
    
    return ct


## MSG CT file finder. Looks for the existence of CT files that match
# the given input image product
# @param prod Cartesian image or composite product
# @return string if a matching filename is found, otherwise None
def getMatchingCT(prod):
    prodt = datetime.datetime.strptime(prod.date + prod.time, "%Y%m%d%H%M%S")
    
    if 0 < prodt.minute < 15:
        prodt -= datetime.timedelta(minutes=prodt.minute)
    elif 15 < prodt.minute < 20:
        prodt -= datetime.timedelta(minutes=prodt.minute - 15)
    elif 30 < prodt.minute < 45:
        prodt -= datetime.timedelta(minutes=prodt.minute - 30)
    elif 45 < prodt.minute <= 59:
        prodt -= datetime.timedelta(minutes=prodt.minute - 45)
    
    for dt in range(CT_MAX_DELTAS):
        mydt = prodt - (dt * CTDELTA)
        globstr = os.path.join(CTPATH, CT_FTEMPLATE % mydt.strftime("%Y%m%d%H%M"))
        fstrs = glob.glob(globstr)
        
        if len(fstrs) == 1:
            if os.path.getsize(fstrs[0]):
                return fstrs[0]


## MSG CT file filter generator.
# @param prod Cartesian image or composite product
# @param quantity string representing the quantity to process
# @return True upon success, False upon failure, None if abandoned due to lack of CT
def ctFilter(prod, quantity="DBZH"):
    if not prod.hasParameter(quantity):
        raise AttributeError("Input product has no quantity called %s" % quantity)
    prod.defaultParameter = quantity
    ct_filename = getMatchingCT(prod)
    
    if ct_filename:
        ct = readCT(ct_filename)
        ct.defaultParameter = "CT"
        ct.addAttribute("how/task_args", os.path.split(ct_filename)[1])
        
        ret = _ctfilter.ctFilter(prod, ct)
        if ret is True:
            ct_qfield = prod.getParameter(quantity).getQualityFieldByHowTask("se.smhi.quality.ctfilter")
            ct_qfield.addAttribute("how/task_args", os.path.split(ct_filename)[1])
        
        return ret


if __name__ == "__main__":
    pass
