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
## Module for defining geographical areas (cartographic surfaces) using the
# Proj and rave_projection module.
# The area definitions are loaded from configuration file(s) given
# by the AREAS variable.

## @file
## @author Daniel Michelson, SMHI, based on work originally contracted to Fredrik Lundh
## @date 2011-06-28

import os, string
import rave_projection, _arearegistry, _area, _projection, Proj
import rave_xml
from rave_defines import RAVECONFIG, UTF8, AREA_REGISTRY
import _polarscan, _polarvolume

## There's only one official area registry, but this module allows 
# greater flexibility as long as files use the same naming convention.
AREAS = os.path.join(RAVECONFIG, '*area_registry.xml')


## Empty registry to be filled
_registry = {}

## Returns a list of keys in the registry
def keys():
    return _registry.keys()

## Returns a list of tuples containing key:item pairs in the registry
# where the key is the area's identifier and the item is its object.
def items():
    return _registry.items()


# --------------------------------------------------------------------
# Initialization

initialized = 0

## Area object
class AREA(rave_xml.xmlmap):
    ## Dummy initializer
    def __init__(self):
        pass

    ## Maps attributes from AREA instance 'common' which don't exist in 'self'
    ## @param common attribute
    def fromCommon(self, common):
         for a in dir(common):
            if not hasattr(self, a):
                setattr(self, a, getattr(common, a))


## Initializer
def init():
    import glob
    from xml.etree import ElementTree

    global initialized
    if initialized: return

    for fstr in glob.glob(AREAS):

        E = ElementTree.parse(fstr)

        for e in E.findall('area'):
            this = AREA()
            this.Id = e.get('id')
            this.name = e.find('description').text.encode(UTF8)
            this.getArgs(e.find('areadef'))
            if hasattr(this, 'size'): this.xsize = this.ysize = this.size
            if hasattr(this, 'scale'): this.xscale = this.yscale = this.scale

            register(this)

    initialized = 1


## Returns the area instance corresponding with the given identifier
# @param Id String identifier of the desired area
# @returns an AREA instance representing the desired area 
def area(Id):
    if not isinstance(Id, str):
        raise KeyError("Argument 'Id' not a string")
    return _registry[Id]


## Registers a new AREA instance
# @param A AREA instance
def register(A):
    A.validate(["Id", "name", "pcs", "extent",
                "xsize", "ysize", "xscale", "yscale"])

    # Ridiculous hack for trimming whacked XML strings from rave_simple_xml.c. Should be deprecated down the line.  
    if A.pcs[:9] == '\n        ' and A.pcs[-7:] == '\n      ': A.pcs = A.pcs[9:len(A.pcs)-7]
    if A.name[:7] == '\n      ' and A.name[-5:] == '\n    ': A.name = A.name[7:len(A.name)-5]
    
    # Switch to strings
    if isinstance(A.pcs, bytes): A.pcs = A.pcs.decode()
    if isinstance(A.name, bytes): A.name = A.name.decode()
    if isinstance(A.Id, bytes): A.Id = A.Id.decode()
    
    # Likewise, rave_simplexml.c doesn't write argument types, so we must enforce them here.
    if isinstance(A.xsize, str): A.xsize = int(A.xsize)
    if isinstance(A.ysize, str): A.ysize = int(A.ysize)
    if isinstance(A.xscale, str): A.xscale = float(A.xscale)
    if isinstance(A.yscale, str): A.yscale = float(A.yscale)
    if isinstance(A.extent, str): A.extent = make_tuple(A.extent)

    A.pcs = rave_projection.pcs(A.pcs)
    _registry[A.Id] = A


## Convenience function for converting a text representation of a tuple to a tuple.
# @param text Input string
# @returns a tuple containing the converted string 
def make_tuple(text):
    import re
    text = re.sub(" ", "", text)  # weed out spacebars
    L = []
    for item in text.split(','):
        L.append(eval(item))
    return tuple(L)


# --------------------------------------------------------------------
# INITIALIZE
init()


## Convenience function for creating XML
# @param parent parent XML element
# @param id string identifier for the tag
# @param text string containing a test representation of the SubElement's contents
# @param Type can be any of 'float', 'int', or 'sequence'  
# @returns the formatted SubElement
def makearg(parent, id, text, Type=None):
    from xml.etree.ElementTree import SubElement
    arg = SubElement(parent, 'arg')
    arg.set('id', id)
    if Type: arg.set('type', Type)
    arg.text = text
    return arg


# --------------------------------------------------------------------
# Using new APIs

## Adds a new area to the registry. If an existing entry with the same identifier exists,
# it is overwritten.
# @param id string identifier of this area
# @param description string free-text description of this area
# @param projection_id string identifier of the projection used to define this area
# @param extent tuple of floats giving the PCS coordinates of the lower-left and upper-right pixels
#               in the form (LLlon, LLlat, URlon, URlat). Note that the PCS coordinates of the
#               UR pixel are for the lower-left corner of the upper-right pixel.
# @param xsize int number of pixels in the X dimension
# @param ysize int number of pixels in the Y dimension
# @param xscale float X scale in PCS space (commonly expressed in meters)
# @param yscale float Y scale in PCS space (commonly expressed in meters)
# @param filename Full path to the XML file containing the registry
def add(id, description, projection_id, extent, xsize, ysize, xscale, yscale, filename=AREA_REGISTRY):
    reg = _arearegistry.load(filename)
    reg.removeByName(id)  # Is silent if entry doesn't exist
    a = _area.new()

    a.id, a.description, a.pcsid = id, description, projection_id
    a.extent = extent
    a.xsize, a.ysize = xsize, ysize
    a.xscale, a.yscale = xscale, yscale
    p = rave_projection.pcs(a.pcsid)
    a.projection = _projection.new(p.id, p.name, string.join(p.definition))

    reg.add(a)
    reg.write(filename)


## Removes an area from the registry
# @param id String containing the identifier of the area to remove
# @param filename Full path to the XML file containing the area registry
def remove(id, filename=AREA_REGISTRY):
    reg = _arearegistry.load(filename)
    reg.removeByName(id)
    reg.write(filename)


## Writes the contents of the registry to file.
# This is a bit overworked, but it bridges the gap between old and new interfaces.
# @param filename Complete path of the XML file to which to write the contents of the registry.
def write(filename=AREA_REGISTRY):
    check = []  # Used to avoid duplicate entries
    new_registry = _arearegistry.new()
    for k, i in items():
        if k not in check:
            tmp = _area.new()

            tmp.id, tmp.description, tmp.pcsid = i.id, i.name, i.pcs.id
            tmp.extent = i.extent
            tmp.xsize, tmp.ysize = i.xsize, i.ysize
            tmp.xscale, tmp.yscale = i.xscale, i.yscale
            
            new_registry.add(tmp)

            check.append(k)
        else:
            print("Duplicate entry for id %s. Ignored." % k)
    new_registry.write(filename)


## Prints an area's characteristics to stdout
# @param id The area's string identifier 
def describe(id):
    a = _registry[id]
    (LL_lon, LL_lat), (UR_lon, UR_lat), (UL_lon, UL_lat), (LR_lon, LR_lat) = MakeCornersFromExtent(id)
    print("%s -\t%s" % (id, a.name))
    print("\tprojection identifier = %s" % a.pcs.id)
    print("\textent = %f, %f, %f, %f" % a.extent)
    print("\txsize = %i, ysize = %i" % (a.xsize, a.ysize))
    print("\txscale = %f, yscale = %f" % (a.xscale, a.yscale))
    print("\tSouth-west corner lon/lat: %f, %f" % (LL_lon, LL_lat))
    print("\tNorth-west corner lon/lat: %f, %f" % (UL_lon, UL_lat))
    print("\tNorth-east corner lon/lat: %f, %f" % (UR_lon, UR_lat))
    print("\tSouth-east corner lon/lat: %f, %f" % (LR_lon, LR_lat))


## Calculates the corner coordinates in lon/lat based on an area's extent.
# NOTE: the corners in lon/lat are the true outside corners of each pixel,
# whereas the extent always represents the position of the lower-left corner
# of each corner pixel.
# @param id string identifying the area
# @returns tuple of tuples containing floats with lon/lat coordinates for 
# lower-left, upper-right, upper-left, and lower-right corner coordinates
def MakeCornersFromExtent(id):
    a = _registry[id]
    extent = a.extent

    p = Proj.Proj(a.pcs.definition)

    LL_lon, LL_lat = Proj.r2d(p.invproj((extent[0], extent[1])))
    UR_lon, UR_lat = Proj.r2d(p.invproj((extent[2]+a.xscale,
                                         extent[3]+a.yscale)))
    UL_lon, UL_lat = Proj.r2d(p.invproj((extent[0], extent[3]+a.yscale)))
    LR_lon, LR_lat = Proj.r2d(p.invproj((extent[2]+a.xscale, extent[1])))
    return (LL_lon, LL_lat), (UR_lon, UR_lat), (UL_lon, UL_lat), (LR_lon, LR_lat)


## Convenience function that automatically derives a new area from several input 
# ODIM_H5 polar volume or scan files. Mandatory single input file will give an area
# for that single site. If more files are given, the derived area will represent 
# the coverage of all these radars. Depending on the characteristics of the given
# projection, different radars will determine the north, south, east, and west
# edges of the derived area.
# @param files List of file strings of input ODIM_H5 files
# @param proj_id identifier string of the projection to use for this area
# @param xscale float Horizontal X-dimension resolution in projection-specific units (commonly meters)  
# @param yscale float Horizontal Y-dimension resolution in projection-specific units (commonly meters)  
# @returns Don't know
def MakeAreaFromPolarFiles(files, proj_id='llwgs84', xscale=2000.0, yscale=2000.0):
    import _rave, _raveio
    
    areas = []
    for fstr in files:
        io = _raveio.open(fstr)
        if io.objectType == _rave.Rave_ObjectType_PVOL:
        # Assert ascending volume, assuming the scan with the longest range will be the one with the longest surface distance
            scan = io.object.getScanWithMaxDistance()
        elif io.objectType == _rave.Rave_ObjectType_SCAN:
            scan = io.object
        else:
            raise IOError("Input file %s is not a polar volume or scan" % fstr)

        
        io.close()
        areas.append(MakeSingleAreaFromSCAN(scan, proj_id, xscale, yscale))
    
    minx =  10e100
    maxx = -10e100
    miny =  10e100
    maxy = -10e100

    for a in areas:
        if a.extent[0] < minx: minx = a.extent[0]
        if a.extent[1] < miny: miny = a.extent[1]
        if a.extent[2] > maxx: maxx = a.extent[2]
        if a.extent[3] > maxy: maxy = a.extent[3]

    # Expand to nearest pixel - buffering by one pixel was done in MakeSingleAreaFromSCAN
    dx = (maxx-minx) / xscale
    dx = (1.0-(dx-int(dx))) * xscale
    if dx < xscale:
        minx -= dx
    dy = (maxy-miny) / yscale
    dy = (1.0-(dy-int(dy))) * yscale
    if dy < yscale:
        miny -= dy

    xsize = int(round((maxx-minx)/xscale, 0))
    ysize = int(round((maxy-miny)/yscale, 0))

    A = AREA()
    A.xsize, A.ysize, A.xscale, A.yscale = xsize, ysize, xscale, yscale
    A.extent = minx, miny, maxx, maxy
    A.pcs = proj_id

    return A

## Convenience function that automatically derives a new area from several input 
# ODIM_H5 polar volume or scan objects. Mandatory single input object will give an area
# for that single site. If more objects are given, the derived area will represent 
# the coverage of all these radars. Depending on the characteristics of the given
# projection, different radars will determine the north, south, east, and west
# edges of the derived area.
# @param files List of polar objects
# @param proj_id identifier string of the projection to use for this area
# @param xscale float Horizontal X-dimension resolution in projection-specific units (commonly meters)  
# @param yscale float Horizontal Y-dimension resolution in projection-specific units (commonly meters)  
# @returns Don't know
def MakeAreaFromPolarObjects(objects, proj_id='llwgs84', xscale=2000.0, yscale=2000.0):
    import _rave, _raveio
    
    areas = []
    for o in objects:
      if _polarvolume.isPolarVolume(o):
        scan = o.getScanWithMaxDistance()
      elif _polarscan.isPolarScan(o):
        scan = o
      else:
        raise IOError("Input object is not a polar scan or volume")
      
      areas.append(MakeSingleAreaFromSCAN(scan, proj_id, xscale, yscale))
    
    minx =  10e100
    maxx = -10e100
    miny =  10e100
    maxy = -10e100

    for a in areas:
        if a.extent[0] < minx: minx = a.extent[0]
        if a.extent[1] < miny: miny = a.extent[1]
        if a.extent[2] > maxx: maxx = a.extent[2]
        if a.extent[3] > maxy: maxy = a.extent[3]

    # Expand to nearest pixel - buffering by one pixel was done in MakeSingleAreaFromSCAN
    dx = (maxx-minx) / xscale
    dx = (1.0-(dx-int(dx))) * xscale
    if dx < xscale:
        minx -= dx
    dy = (maxy-miny) / yscale
    dy = (1.0-(dy-int(dy))) * yscale
    if dy < yscale:
        miny -= dy

    xsize = int(round((maxx-minx)/xscale, 0))
    ysize = int(round((maxy-miny)/yscale, 0))

    A = AREA()
    A.xsize, A.ysize, A.xscale, A.yscale = xsize, ysize, xscale, yscale
    A.extent = minx, miny, maxx, maxy
    A.pcs = proj_id

    return A


## Helper for defining new areas.
# @param scan PolarScanCore object
# @param pcsid string containing the output projection identifier
# @param xscale float Horizontal X-dimension resolution in projection-specific units (commonly meters)  
# @param yscale float Horizontal Y-dimension resolution in projection-specific units (commonly meters)  
# @returns an XML Element
def MakeSingleAreaFromSCAN(scan, pcsid, xscale, yscale):
    import numpy
    from xml.etree.ElementTree import Element, SubElement
    import _polarnav

    pn = _polarnav.new()
    pn.lon0, pn.lat0, pn.alt0 = scan.longitude, scan.latitude, scan.height
    maxR = scan.nbins * scan.rscale
    nrays = scan.nrays * 2  # Doubled for greater accuracy

    minx =  10e100
    maxx = -10e100
    miny =  10e100
    maxy = -10e100

    azres = 360.0/nrays
    #az = 0.5*azres  # Start properly: half an azimuth gate from north
    az = 0.0  # Let's not and say we did ...
    while az < 360.0:
        latr, lonr = pn.daToLl(maxR, az*Proj.dr)
        herec = lonr*Proj.rd, latr*Proj.rd

        thislon, thislat = Proj.c2s([herec], pcsid)[0]

        if thislon < minx: minx = thislon
        if thislon > maxx: maxx = thislon
        if thislat < miny: miny = thislat
        if thislat > maxy: maxy = thislat

        az+=azres

    # Expand to nearest pixel and buffer by one just to be sure
    dx = (maxx-minx) / xscale
    dx = (1.0-(dx-int(dx))) * xscale
    if dx < xscale:
        minx -= xscale + dx
        maxx += xscale
    dy = (maxy-miny) / yscale
    dy = (1.0-(dy-int(dy))) * yscale
    if dy < yscale:
        miny -= yscale + dy
        maxy += yscale

    xsize = int(round((maxx-minx)/xscale, 0))
    ysize = int(round((maxy-miny)/yscale, 0))

    A = AREA()
    A.xsize, A.ysize, A.xscale, A.yscale = xsize, ysize, xscale, yscale
    A.extent = minx, miny, maxx, maxy
    A.pcs = pcsid

    return A


if __name__ == "__main__":
    import rave_area # cannot use myself, due to recursive import
    for id in rave_area.keys():
        describe(id)

