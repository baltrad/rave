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

## Encodes/decodes ODIM quantities to/from hex for the purposes of creating
#  simple file strings representing which quantities are in the contents.

## @file
## @author Daniel Michelson, SMHI
## @date 2014-10-19

import sys, os
from copy import deepcopy as copy
from rave_defines import RAVECONFIG
import xml.etree.ElementTree as ET
import numpy as np


QUANTFILE = os.path.join(RAVECONFIG, "odim_quantities.xml")
QUANTITIES = []
initialized = 0
bitl = list(np.zeros((128,), np.uint8))  # A 64-element long list of unsigned bytes. Used as an intermediate information holder.

use_long_type=False
if sys.version_info < (3,):
    use_long_type=True

## Initializes QUANTITIES by reading content from XML file.
#  This is done once, after which the QUANTITIES are available in memory.
def init():
    global initialized, QUANTITIES
    if initialized: return
    
    C = ET.parse(QUANTFILE)
    QUANTS = C.getroot()

    for q in QUANTS:
        QUANTITIES.append(q.text)

    QUANTITIES = tuple(QUANTITIES)

    initialized = 1


## Initialize
init()


## Digs out all the quantities in a SCAN
# @param PolarScan object
# @returns list of quantity strings
def qFromScan(scan):
    return scan.getParameterNames()


## Digs out all the quantities in a PVOL
# @param PolarVolume object
# @returns list of quantity strings
def qFromPvol(pvol):
    quants = []
    for i in range(pvol.getNumberOfScans()):
        scan = pvol.getScan(i)
        squants = qFromScan(scan)
        for q in squants:
            if q not in quants:
                quants.append(q)
    return quants


## Digs out the quantities from a given file string
#  Assumes the hex string's location is fixed.
# @param string file name
# @returns list of quantity strings
def qFromFstr(fstr):
    path, fstr = os.path.split(fstr)
    stuff = fstr.split('.')[-2:]  # Get the last two parts only
    h = stuff[0].split('_')[-1:][0]
    return hex2q(h)


## Converts a list of quantities to a hex string
# @param list of quantity strings
# @returns hex string
def q2hex(quants):
    b = copy(bitl)
    for q in quants:
        if q in QUANTITIES:
          i = QUANTITIES.index(q)
          b[i] = 1
    bstr = hex(bitl2long(b))
    if bstr[-1] == "L":
      bstr = bstr[:-1]
    return bstr


## Converts a hex string to a list of quantities
# @param hex string
# @returns list of quantity strings
def hex2q(h):
    q = []
    bits = long2bits(hex2long(h))
    for i in range(len(bits)):
        if bits[i] == "1":
            q.append(QUANTITIES[i])
    return q


## Bit list to long integer
# @param list of 8-bit bytes, each representing one bit in a 64-bit integer
# @returns long integer
def bitl2long(bitl):
    out = 0
    if use_long_type: #work around for python 2.7 / 3 difference
      out = long(0)
      
    for bit in bitl:
        out = (out << 1) | int(bit)
    return out


## Long integer to bit string
# @param long integer
# @returns list of 8-bit bytes, each representing one bit in a 128-bit integer
def long2bits(l):
    return format(l, '128b')


## hex string to long integer
# @param hex string
# @returns long integer
def hex2long(h):
    if use_long_type: #work around for python 2.7 / 3 difference
      return int(long(h+"L", 16))
    return int(h, 16)

if __name__ == "__main__":
    pass
