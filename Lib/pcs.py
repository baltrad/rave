#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: pcs.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson, based on work contracted to Fredrik Lundh
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                1998-
#                All rights reserved.
#
# $Log: pcs.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
pcs.py

Projection Coordinate System

Module for defining projections using the Proj module.

The projection definitions are loaded from configuration file(s) given
by the PROJECTIONS variable.

TODO: Restructure this module like the 'area' and 'radar' modules.
"""
import os
from types import StringType
import Proj
from rave_defines import RAVECONFIG, ENCODING

PROJECTIONS = os.path.join(RAVECONFIG, '*projections.xml')


# --------------------------------------------------------------------
# Initialization

initialized = 0

def init():
    import glob
    from xml.etree import ElementTree

    global initialized
    if initialized: return

    for fstr in glob.glob(PROJECTIONS):    

        E = ElementTree.parse(fstr)

        for e in E.findall('projection'):
            Id = e.get('id')
            name = e.find('description').text.encode(ENCODING)
            definition = e.find('projdef').text.split()
            define(Id, name, definition)

    initialized = 1


# --------------------------------------------------------------------
# Registry

class interface:
    name = None
    def proj(ll):
        pass
    def invproj(sxy):
        pass

_registry = {}

def register(id, pcs):
    # validate
    for attr in ["name", "proj", "invproj"]:
        if not hasattr(pcs, attr):
            raise AttributeError, "object lacks required attribute " + attr
    pcs.id = id
    _registry[id] = pcs

def keys():
    return _registry.keys()

def items():
    return _registry.items()


# --------------------------------------------------------------------
# Object factory

def pcs(Id):
    if type(Id) != StringType:
        raise KeyError, "Argument 'Id' not a string"
    return _registry[Id]


# --------------------------------------------------------------------
# USGS PROJ 4.3.3 and higher interface (requires Proj.py, _proj.so)

class usgs:
    def __init__(self, name, definition):
        try:
            import Proj
        except ImportError:
            raise ImportError, "Module Proj is missing: "\
                  "check python configuration"
        self.name = name
        self.definition = definition
        self.instance = Proj.Proj(definition)

    def proj(self, xy):
        return self.instance.proj(xy)

    def invproj(self, xy):
        return self.instance.invproj(xy)

    def tostring(self):
        o = ''
        for s in self.definition:
            if len(o): o += " " + s
            else: o = s
        return o


# --------------------------------------------------------------------
# Register function for pre-defined pcs definitions in PROJECTIONS

def define(id, name, definition):
    p = usgs(name, definition)
    register(id, p)


# --------------------------------------------------------------------
# INITIALIZE
init()


if __name__ == "__main__":
    import pcs # cannot use myself, due to recursive import
    for id in pcs.keys():
        p = pcs.pcs(id)
        print id, p.name, p
