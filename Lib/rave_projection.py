#!/usr/bin/env python
'''
Copyright (C) 1998 - Swedish Meteorological and Hydrological Institute (SMHI)

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
## Module for defining projections using the Proj module.
# Reads projection definitions from XML using the _projectionregistry
# module. Based on pcs.py

## @file
## @author Daniel Michelson, SMHI, based on work originally contracted to Fredrik Lundh
## @date 2011-06-27

import os, string
from types import StringType
import Proj
import _projection, _projectionregistry
from rave_defines import RAVECONFIG, UTF8, PROJECTION_REGISTRY

## There's only one official projection registry, but this module allows 
# greater flexibility as long as files use the same naming convention.
PROJECTIONS = os.path.join(RAVECONFIG, '*projection_registry.xml')


# --------------------------------------------------------------------
# Initialization

initialized = 0


## Initializer
def init():
    import glob
    from xml.etree import ElementTree

    global initialized
    if initialized: return

    for fstr in glob.glob(PROJECTIONS):

        E = ElementTree.parse(fstr)

        for e in E.findall('projection'):
            Id = e.get('id')
            description = e.find('description').text.encode(UTF8)
            definition = e.find('projdef').text.split()
            define(Id, description, definition)

    initialized = 1


# --------------------------------------------------------------------
# Registry

## Object for interfacing with PROJ.4 
class interface:
    name = None

    ## Forward projection
    # @param ll Longitude/latitude coordinate pair
    def proj(ll):
        pass

    ## Inverse projection
    # @param sxy X/Y projection-specific coordinate pair
    def invproj(sxy):
        pass

## Empty registry to be filled
_registry = {}

## Adds a projection to the registry 
# @param id Projection identifier (string)
# @param pcs Projection object as returned by the usgs class 
def register(id, pcs):
    # validate
    for attr in ["name", "proj", "invproj"]:
        if not hasattr(pcs, attr):
            raise AttributeError, "object lacks required attribute " + attr
    pcs.id = id
    # Ridiculous hack for trimming whacked XML strings from rave_simple_xml.c. Should be deprecated down the line.  
    if pcs.name[:7] == '\n      ' and pcs.name[-5:] == '\n    ': pcs.name = pcs.name[7:len(pcs.name)-5]
    _registry[id] = pcs

## Returns a list of keys in the registry
def keys():
    return _registry.keys()

## Returns a list of tuples containing key:item pairs in the registry
# where the key is the projection identifier and the item is its usgs object.
def items():
    return _registry.items()


## Object factory. PCS = Projection Coordinate System
# @param Id Projection identifier (string)
# @returns Projection object corresponding with the input identifier
def pcs(Id):
    if type(Id) != StringType:
        raise KeyError, "Argument 'Id' not a string"
    return _registry[Id]


## PROJ 4.7.0 and higher interface (requires Proj.py, _proj.so)
class usgs:
    ## Initializer
    # @param description string containing this projection's identifier
    # @param definition PROJ.4 string containing the projection definition 
    def __init__(self, description, definition):
        try:
            import Proj
        except ImportError:
            raise ImportError, "Module Proj is missing: "\
                  "check Python configuration"
        self.name = description
        self.definition = definition
        self.instance = Proj.Proj(definition)

    ## Forward projection
    # @param xy tuple containing lon/lat coordinate pair
    # @returns tuple containing X/Y in projection-specific coordinates
    def proj(self, xy):
        return self.instance.proj(xy)

    ## Inverse projection
    # @param tuple containing X/Y in projection-specific coordinates
    # @returns xy tuple containing lon/lat coordinate pair
    def invproj(self, xy):
        return self.instance.invproj(xy)

    ## String representation of a projection definition
    # @ returns string representation of the projection
    def tostring(self):
        o = ''
        for s in self.definition:
            if len(o): o += " " + s
            else: o = s
        return o


## Register utility function for pre-defined projection definitions in PROJECTIONS
# @param id string containing this projection's identifier
# @param description string containing a free-text description of this projection
# @param definition PROJ.4 string containing the projection definition 
def define(id, description, definition):
    p = usgs(description, definition)
    register(id, p)


# --------------------------------------------------------------------
# INITIALIZE
init()


## Adds a projection to the registry.
# This is a bit overworked, but it bridges the gap between old and new interfaces.
# If a previous instance of a projection with the same identifier exists, it is overwritten.
# @param id String containing the identifier of the new projection
# @param description String containing a description of the new projection
# @param definition PROJ.4 string containing the new projection's definition
# @param filename Full path to the XML file containing the projection registry 
def add(id, description, definition, filename=PROJECTION_REGISTRY):
    reg = _projectionregistry.load(filename)
    reg.removeByName(id)  # Is silent if entry doesn't exist
    reg.add(_projection.new(id, description, definition))
    reg.write(filename)


## Removes a projection from the registry
# @param id String containing the identifier of the projection to remove
# @param filename Full path to the XML file containing the projection registry
def remove(id, filename=PROJECTION_REGISTRY):
    reg = _projectionregistry.load(filename)
    reg.removeByName(id)
    reg.write(filename)


## Writes the contents of the registry to file.
# This is a bit overworked, but it bridges the gap between old and new interfaces.
# @param filename Complete path of the XML file to which to write the contents of the registry.
def write(filename=PROJECTION_REGISTRY):
    check = []  # Used to avoid duplicate entries
    new_registry = _projectionregistry.new()
    for p in keys():
        if p not in check:
            tmp = pcs(p)
            new_registry.add(_projection.new(tmp.id, tmp.name, 
                                             string.join(tmp.definition)))
            check.append(p)
        else:
            print "Duplicate entry for id %s. Ignored." % p
    new_registry.write(filename)


## Prints the identifier, description, and PROJ.4 definition to stdout
# @param id The projection's string identifier 
def describe(id):
    p = _registry[id]
    print "%s -\t%s" % (id, p.name)
    print "\t%s" % string.join(p.definition)


if __name__ == "__main__":
    import rave_projection # cannot use myself, due to recursive import
    for id in rave_projection.keys():
        describe(id)
