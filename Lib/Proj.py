#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: Proj.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Authors: Daniel Michelson, based on work contracted to Fredrik Lundh
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                1997-
#                All rights reserved.
# history:
# 97-06-15 fl   Created
# 97-06-17 fl   Added helpers
# 98-06-02 dm   Added more helpers
# 98-06-09 dm   Modified the constructor to simplify automation
# 05-09-29 dm   Tidied up
#
# $Log: Proj.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
Proj.py - USGS PROJ4 interface
"""
import _proj
import math

#
# exception

error = _proj.error

#
# class

class Proj:

    def __init__(self, args):  # args is a list of arguments as strings

        self._proj = _proj.proj(args)

        # delegate methods
        self.proj = self._proj.proj
        self.invproj = self._proj.invproj


#
# helpers

dmstor = _proj.dmstor

# degrees to radians
dr = math.pi / 180.0
def d2r(ll):
    return ll[0] * dr, ll[1] * dr

# radians to degrees
rd = 180.0 / math.pi
def r2d(xy):
    return xy[0] * rd, xy[1] * rd

# for translating a (list of) tuple(s) of long/lats to surface coords.
def c2s(indata, pcs_id):
    import pcs
    p = pcs.pcs(pcs_id) # pcs_id = "ps60n", for example
    outdata = []
    for ll in indata:
	outdata.append(p.proj(d2r(ll)))
    return outdata

# for translating a (list of) tuple(s) of surface coords to long/lat in degrees
def s2c(indata, pcs_id):
    import pcs
    p = pcs.pcs(pcs_id)
    outdata = []
    for ll in indata:
	outdata.append(r2d(p.invproj(ll)))
    return outdata

