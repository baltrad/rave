'''
Copyright (C) 1997 - Swedish Meteorological and Hydrological Institute (SMHI)

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
## PROJ.4 interface
# History:
# 1997-06-15 fl   Created
# 1997-06-17 fl   Added helpers
# 1998-06-02 dm   Added more helpers
# 1998-06-09 dm   Modified the constructor to simplify automation
# 2005-09-29 dm   Tidied up
# 2011-06-29 dm   Doxygenified for BALTRAD

## @file
## @author Daniel Michelson, SMHI, based on work originally contracted to Fredrik Lundh
## @date 2011-06-29
# Standard python libs:
import math

# Module/Project:
import _proj

## Exception
error = _proj.error


## Projection class
class Proj:
    ## Initializer
    # @param args List of PROJ.4 arguments
    def __init__(self, args):  # args is a list of arguments as strings
        self._proj = _proj.proj(args)
        
        # delegate methods
        self.proj = self._proj.proj
        self.invproj = self._proj.invproj


## Helper
dmstor = _proj.dmstor

## degrees to radians
dr = math.pi / 180.0


## Convenience function for converting a tuple of angles expressed in degrees to radians
# @param ll tuple of angles expressed in degrees
# @returns tuple of angles expressed in radians
def d2r(ll):
    return ll[0] * dr, ll[1] * dr


## radians to degrees
rd = 180.0 / math.pi


## Convenience function for converting a tuple of angles expressed in radians to degrees
# @param xy tuple of angles expressed in radians
# @returns tuple of angles expressed in degrees
def r2d(xy):
    return xy[0] * rd, xy[1] * rd


## Function for converting a lon/lat coordinate pair to a pair of PCS (projection-specific) coordinates
# @param ll variable-length tuple containing lon/lat coordinate pairs
# @param pcs_id string identifier of the projection to use. Check registered projections using 'projection_rgistry'
# @returns tuple of PCS XY coordinate pairs
def c2s(indata, pcs_id):
    import rave_projection
    
    p = rave_projection.pcs(pcs_id)  # pcs_id = "ps60n", for example
    outdata = []
    for ll in indata:
        outdata.append(p.proj(d2r(ll)))
    return outdata


## Function for converting a PCS (projection-specific) XY coordinate pair to a lon/lat coordinate pair
# @param xy tuple of PCS XY coordinates
# @param pcs_id string identifier of the projection to use. Check registered projections using 'projection_rgistry'
# @returns tuple of lon/lat coordinates
def s2c(indata, pcs_id):
    import rave_projection
    
    p = rave_projection.pcs(pcs_id)
    outdata = []
    for ll in indata:
        outdata.append(r2d(p.invproj(ll)))
    return outdata


## Function for converting real resolution on the Earth's surface (meters) to
# projection-specific resolution. Based on Petr Novak's work reported to OPERA.
# @param lat float latitude position
# @param scale float surface resolution on the Earth in meters
# @returns float representing the resolution expressed in projection-specific space
def ScaleResolutionFromReal(lat, scale):
    return scale * 1 / math.cos(lat * dr)


## Function for converting projection-specific resolution real resolution on the
# Earth's surface (meters). Based on Petr Novak's work reported to OPERA.
# @param lat float latitude position
# @param scale float resolution in projection-specific space
# @returns float surface resolution on the Earth in meters
def RealResolutionFromScale(lat, scale):
    return scale / 1 * math.cos(lat * dr)
