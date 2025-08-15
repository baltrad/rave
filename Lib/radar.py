#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: radar.py,v 1.2 2006/12/18 09:34:16 dmichels Exp $
#
# Authors: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2006-
#                All rights reserved.
#
# $Log: radar.py,v $
# Revision 1.2  2006/12/18 09:34:16  dmichels
# *** empty log message ***
#
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
radar.py

Module for defining polar areas and radar configurations based on the
configuration information for specified radars.
"""
# Standard python libs:
import os
import string

# Module/Project:
import rave_xml
from rave_defines import RAVECONFIG, ENCODING

RADARS = os.path.join(RAVECONFIG, '*radars.xml')


# --------------------------------------------------------------------
# Registries

_registry = {}


def keys():
    return _registry.keys()


def items():
    return _registry.items()


# --------------------------------------------------------------------
# Initialization

initialized = 0


class RADAR(rave_xml.xmlmap):
    def __init__(self):
        pass

    def fromCommon(self, common):
        """
        Maps attributes from RADAR instance 'common' which don't exist
        in 'self'.
        """
        for a in dir(common):
            if not hasattr(self, a):
                setattr(self, a, getattr(common, a))


def init():
    import glob
    from xml.etree import ElementTree

    global initialized
    if initialized:
        return

    for fstr in glob.glob(RADARS):
        E = ElementTree.parse(fstr)

        common = RADAR()
        common.getArgs(E.find('common').find('radardef'))

        p = E.find('cartesian')  # Common radar-specific projection info
        if p is not None:
            proj = RADAR()
            proj.getArgs(p)

        for e in E.findall('radar'):
            this = RADAR()
            this.Id = e.get('id')
            this.name = e.find('description').text.encode(ENCODING)
            this.getArgs(e.find('radardef'))

            this.validate(["Id", "place", "lon", "lat", "height"])
            this.fromCommon(common)

            _registry[this.Id] = this

            # Intialize radar-specific cartesian areas
            if p is not None:
                MakeCartesianArea(proj, this)

    initialized = 1


# --------------------------------------------------------------------
# Object factories


def radar(Id):
    if type(Id) != str:
        raise KeyError("Argument 'Id' not a string")
    return _registry[Id]


def register(P):
    P.validate(["Id", "place", "lon", "lat", "height", "rays", "bins", "angles", "xsize", "beamwidth"])
    _registry[P.Id] = P


def MakeCartesianArea(proj, rad):
    from copy import deepcopy
    import pcs, area

    a = area.AREA()
    a.name = rad.place.decode(ENCODING) + " specific projection"

    xsizes, max_ranges = deepcopy(rad.xsize), deepcopy(proj.max_range)

    if type(rad.xsize) is list:
        # xsizes must be in the same order as max_ranges: must correspond
        for i in range(len(rad.xsize)):
            rad.xsize, proj.max_range = xsizes[i], max_ranges[i]
            MakeCartesianArea(proj, rad)
        rad.xsize, proj.max_range = xsizes, max_ranges
    else:
        a.xsize = a.ysize = proj.size
        a.xscale = a.yscale = proj.max_range * 1000 * 2 / proj.size
        ex = 1000 * proj.max_range
        a.extent = (-ex, -ex, ex - a.xscale, ex - a.yscale)
        a.Id = rad.Id + "_%i" % int(proj.max_range)
        projtype = '+proj=' + str(proj.proj.decode(ENCODING))
        radius = '+R=' + str(proj.R)
        lon = '+lon_0=' + str(rad.lon)
        lat = '+lat_0=' + str(rad.lat)
        projid = rad.Id + "_" + proj.proj.decode(ENCODING)
        definition = [projtype, radius, lat, lon]
        name = "%s %s" % (rad.place, proj.proj)
        pcs.define(projid, name, definition)
        # print "projid=%s, name=%s, definition=%s"%(projid,name,definition)
        a.pcs = projid
        area._registry[a.Id] = a


# --------------------------------------------------------------------
# INITIALIZE
init()

if __name__ == "__main__":
    print(__doc__)
