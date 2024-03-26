#!/usr/bin/env python
'''
Copyright (C) 2018 The Crown (i.e. Her Majesty the Queen in Right of Canada)

This file is an add-on to RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE and this software are distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.

'''
##
#  Regenerates fixtures used for unit testing


## 
# @file
# @author Daniel Michelson, Environment and Climate Change Canada
# @date 2019-06-04

import _raveio
import _dr_qc



if __name__=="__main__":
    rio = _raveio.open('WKR_201209090100_POLPPI.h5')
    scan = rio.object
    _dr_qc.drDeriveParameter(scan, 0.0)
    rio.object = scan
    rio.save('WKR_201209090100_POLPPI_DR.h5')
    _dr_qc.drSpeckleFilter(scan, 'DBZH', 3, 3, 35.0, -12.0)
    rio.object = scan
    rio.save('WKR_201209090100_POLPPI_DRSF.h5')
