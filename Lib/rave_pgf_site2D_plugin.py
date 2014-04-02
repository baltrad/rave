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

import os
import _raveio
import rave_tempfile
import rave_site2D


## Creates a dictionary from a RAVE argument list
# @param arglist the argument list
# @return a dictionary
def arglist2dict(arglist):
  result={}
  for i in range(0, len(arglist), 2):
    result[arglist[i]] = arglist[i+1]
  return result


## Performs
# @param files list containing a single file string.
# @arguments list containing arguments for the generator
# @return temporary H5 file containing the generated product
def generate(files, arguments):
    if len(files) == 1:
        kwargs = arglist2dict(arguments)

        rio = _raveio.open(files[0])

        rio = rave_site2D.site2D(rio, **kwargs)
        
        fileno, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")

        rio.save(outfile)
        rio.save("/Users/baltrad/Documents/ERAD-2014/OSS/Py-ART/odim_h5_reader/bobbe.h5")

        return outfile
    raise AttributeError, "Input files list must contain only one file string"


if __name__ == '__main__':
    pass
