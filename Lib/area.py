#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: area.py,v 1.2 2006/12/18 09:34:16 dmichels Exp $
#
# Authors: Daniel Michelson, based on work contracted to Fredrik Lundh
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                1998-
#                All rights reserved.
#
# $Log: area.py,v $
# Revision 1.2  2006/12/18 09:34:16  dmichels
# *** empty log message ***
#
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
area.py

Module for defining geographical areas (cartographic surfaces) using the
Proj module.

The area definitions are loaded from configuration file(s) given
by the AREAS variable.
"""

# Standard python libs:
import os, string

# Module/Project:
import pcs
import rave_xml
from rave_defines import RAVECONFIG, ENCODING

AREAS = os.path.join(RAVECONFIG, '*areas.xml')


# --------------------------------------------------------------------
# Registry

_registry = {}


def keys():
    return _registry.keys()


def items():
    return _registry.items()


# --------------------------------------------------------------------
# Initialization

initialized = 0


class AREA(rave_xml.xmlmap):
    def __init__(self):
        pass

    def fromCommon(self, common):
        """
        Maps attributes from AREA instance 'common' which don't exist in 'self'
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

    for fstr in glob.glob(AREAS):
        E = ElementTree.parse(fstr)

        for e in E.findall('area'):
            this = AREA()
            this.Id = e.get('id')
            this.name = e.find('description').text.encode(ENCODING)
            this.getArgs(e.find('areadef'))
            if hasattr(this, 'size'):
                this.xsize = this.ysize = this.size
            if hasattr(this, 'scale'):
                this.xscale = this.yscale = this.scale

            register(this)

    initialized = 1


# --------------------------------------------------------------------
# Object factory


def area(Id):
    if type(Id) != str:
        raise KeyError("Argument 'Id' not a string")
    return _registry[Id]


def register(A):
    A.validate(["Id", "name", "pcs", "extent", "xsize", "ysize", "xscale", "yscale"])
    A.pcs = pcs.pcs(str(A.pcs.decode(ENCODING)))
    _registry[A.Id] = A


# --------------------------------------------------------------------
# INITIALIZE
init()
import radar  # initialize radar-specific areas last due to cross-dependency


# --------------------------------------------------------------------
# HELPER for defining new areas based on existing ones


# Input:
# in_areaid - a string containing the input area identifier
# maxR - maximum range in METERS, as a float!
# azimuths - number of azimuth gates per scan (typically 420 or 360)
#            DOUBLE this for better accuracy
# scale - horizonal resolution in METERS, as a float!
# pcsid - a string containing the output projection identifier
def area_from_area(in_areaid, maxR, azimuths, scale, pcsid):
    import re
    import rave
    import pcs, Proj
    import numpy
    from xml.etree.ElementTree import Element, SubElement
    from rave_IO import prettyprint

    dr = numpy.pi / 180.0

    minlon = 10e100
    maxlon = -10e100
    minlat = 10e100
    maxlat = -10e100

    in_area = area(in_areaid)

    azres = 360.0 / azimuths  # DOUBLE azimuths argument for better accuracy
    az = 0.5 * azres  # Start properly: half an aziumuth gate from north
    while az < 360.0:
        alpha = az * dr

        hlon = maxR * numpy.sin(alpha)
        hlat = maxR * numpy.cos(alpha)

        herec = Proj.s2c([(hlon, hlat)], in_area.pcs)
        thislon, thislat = Proj.c2s(herec, pcsid)[0]

        if thislon < minlon:
            minlon = thislon
        if thislon > maxlon:
            maxlon = thislon
        if thislat < minlat:
            minlat = thislat
        if thislat > maxlat:
            maxlat = thislat

        az += azres

    # Expand to nearest pixel
    dx = (maxlon - minlon) / scale
    dx = (1.0 - (dx - int(dx))) / 2.0 * scale
    if dx > 0.0:
        maxlon += 2 * dx
    dy = (maxlat - minlat) / scale
    dy = (1.0 - (dy - int(dy))) / 2.0 * scale
    if dy > 0.0:
        maxlat += 2 * dy

    xsize = int(round((maxlon - minlon) / scale, 0))
    ysize = int(round((maxlat - minlat) / scale, 0))

    A = Element("area")
    A_Id = in_areaid.split('_')[0] + '_' + pcsid
    A.set('id', A_Id)
    description = SubElement(A, 'description')
    description.text = "Based on " + in_areaid
    areadef = SubElement(A, 'areadef')
    arg = makearg(areadef, 'pcs', pcsid)
    arg = makearg(areadef, 'xsize', str(xsize), 'int')
    arg = makearg(areadef, 'ysize', str(ysize), 'int')
    arg = makearg(areadef, 'scale', str(scale), 'float')
    arg = makearg(areadef, 'extent', "%f, %f, %f, %f" % (minlon, minlat, maxlon - dx, maxlat - dy), 'sequence')
    prettyprint(A)


#    return A_Id, xsize, ysize, scale, (minlon, minlat, maxlon, maxlat)


def makearg(parent, id, text, Type=None):
    from xml.etree.ElementTree import SubElement

    arg = SubElement(parent, 'arg')
    arg.set('id', id)
    if Type:
        arg.set('type', Type)
    arg.text = text
    return arg


if __name__ == "__main__":
    print(__doc__)
