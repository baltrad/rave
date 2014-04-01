#!/usr/bin/env python
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
## Command-line site-specific 2-D Cartesian product generation

## @file
## @author Daniel Michelson, SMHI
## @date 2011-11-25

import sys
import _rave, _raveio
import rave_area
import odim_source
import rave_composite
import rave_pgf_quality_registry


## Mother function for generating 2-D single-site Cartesian products from input
#  polar data.
# @param rio RAVE I/O container returned from _raveio.open()
# @param kwargs variable-length dictionary containing arguments
# @return rio RAVE I/O container containing an output IMAGE object
def site2D(rio, **kwargs):
    obj = rio.object
    if rio.objectType == _rave.Rave_ObjectType_PVOL:
        # Assert ascending volume, assuming the scan with the longest range will be the one with the longest surface distance
        scan = obj.getScanWithMaxDistance()
    elif rio.objectType == rave.Rave_ObjectType_SCAN:
        scan = obj
    else:
        raise IOError, "Input file %s is not a polar volume or scan" % fstr

    if "area" not in kwargs.keys():
        A = rave_area.MakeSingleAreaFromSCAN(scan, kwargs["pcsid"],
                                             kwargs["scale"], kwargs["scale"])
        areaid = odim_source.NODfromSource(obj) + "_%s" % kwargs["pcsid"]
        A.Id, A.name = areaid, "on-the-fly area definition"
        rave_area.register(A)
        kwargs["area"] = areaid

    comp = rave_composite.generate([obj], **kwargs)

    rio = _raveio.new()
    rio.object = comp
    rio.object.objectType = _rave.Rave_ObjectType_IMAGE
    return rio


if __name__ == "__main__":
    pass
