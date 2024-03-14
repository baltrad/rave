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
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''

## Contains lookup table of critical values for Student's T-test of means.

## @file
## @author Daniel Michelson, SMHI
## @date 2013-01-01
# Standard python libs:
import sys
import os
import pickle

# Module/Project:
from rave_defines import TFILE, TFILE_TEMPLATE


## Look-up table reader
# @param fstr file string
# @return dictionary containing T values
def readT(fstr=TFILE):
    fd = open(fstr)
    return pickle.load(fd)


## One-off to create the pickle file
# @param fstr file string
def convert2dict(fstr=TFILE_TEMPLATE):
    fd = open(fstr)
    LINES = fd.readlines()
    fd.close()
    
    TVALUES = {}
    ONETAIL = {}
    TWOTAIL = {}
    
    for line in LINES:
        l = line.split()
        df = int(l[0])  # degrees of freedom
        # These variable names are for two-tailed values of alpha
        t90, t95, t98, t99, t995, t998, t999 = (
            float(l[1]),
            float(l[2]),
            float(l[3]),
            float(l[4]),
            float(l[5]),
            float(l[6]),
            float(l[7]),
        )
        
        TWOTAIL[df] = {
            "0.1"  : t90,
            "0.05" : t95,
            "0.02" : t98,
            "0.01" : t99,
            "0.005": t995,
            "0.002": t998,
            "0.001": t999
        }
        ONETAIL[df] = {
            "0.05"  : t90,
            "0.025" : t95,
            "0.01"  : t98,
            "0.005" : t99,
            "0.0025": t995,
            "0.001" : t998,
            "0.0005": t999,
        }
    
    TVALUES["ONETAIL"] = ONETAIL
    TVALUES["TWOTAIL"] = TWOTAIL
    
    fd = open(TFILE, 'w')
    pickle.dump(TVALUES, fd)
    fd.close()


TTABLE = None


##
# Call this function to get the translation table instead of going directly at TTABLE.
# Will read TTABLE once and then return the cached copy
def getTTABLE():
    global TTABLE
    if TTABLE is None:
        TTABLE = readT()
    return TTABLE


if __name__ == "__main__":
    pass
