#!/usr/bin/env python
'''
Copyright (C) 2011- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Various look-ups of the /what/source attribute

## @file
## @author Daniel Michelson, SMHI
## @date 2011-09-06

import sys
from rave_defines import UTF8, ODIM_SOURCE_FILE
import xml.etree.ElementTree as ET


## Dictionaries containing look-ups for identifiers except CTY.
# WMO, RAD, PLC, and SOURCE all use the NOD as the look-up
# NOD uses the WMO number as the look-up 
# CCCC will be the same for all radars from a given country, so there'll be
# a lot of redundancy, but this is needed to create Odyssey file strings.
NOD, WMO, RAD, PLC, CCCC = {None:None}, {None:None}, {None:None}, {None:None}, {None:None}
SOURCE = {None:None}


initialized = 0

## Initializer. Reads XML and puts the values into dictionaries.
# Because all radars must have a NOD identifier, the other identifiers are looked up
# based on NOD.
def init():
    global initialized
    if initialized: return
    
    O = ET.parse(ODIM_SOURCE_FILE)
    DB = O.getroot()
    for country in list(DB):
        cccc, org = country.attrib["CCCC"], country.attrib["org"]
        for radar in list(country):
            nod = radar.tag
            wmo, rad, plc = radar.attrib['wmo'], radar.attrib['rad'], radar.attrib['plc']
            WMO[nod] = wmo
            RAD[nod] = rad
            PLC[nod] = plc
            CCCC[nod] = cccc
            if wmo != "00000": 
                NOD[wmo] = nod
                SOURCE[nod] = u"WMO:%s,NOD:%s,RAD:%s,PLC:%s" % (wmo, nod, rad, plc)
                #SOURCE[nod] = u"WMO:%s,NOD:%s,RAD:%s,ORG:%s,PLC:%s" % (wmo, nod, rad, org, plc)
            else:
                SOURCE[nod] = u"NOD:%s,RAD:%s,PLC:%s" % (nod, rad, plc)
                #SOURCE[nod] = u"NOD:%s,RAD:%s,ORG:%s,PLC:%s" % (nod, rad, org, plc)
    initialized = 1


init()


## One-time function, so read radar-db from text file.
# @param filename string of the input file to read
# @ param xmlfile string of the output XML file to write. Defaults to \ref rave_defines.ODIM_SOURCE_FILE 
def text2Element(filename, xmlfile=ODIM_SOURCE_FILE):
    import __builtin__
    from rave_IO import prettyprint

    E = R = None

    fd = open(filename)
    LINES = fd.readlines()
    LINES = LINES[0].split("\r")
    fd.close()

    ALL = ET.Element("radar-db", {"author" : "Daniel Michelson"})
    
    for L in LINES:
        l = L.split('\t')
        if len(l) == 3:
            if E:
                ALL.append(E)
            org, rad_prefix, country = l
            E = ET.Element(country, {"org":org})
        elif len(l) == 5:
            CCCC, nr, wmo, nod, plc = l
            E.attrib["CCCC"] = CCCC
            R = ET.Element(nod, {"wmo":wmo, "rad":"%s%s" % (rad_prefix,nr), 
                                 "plc":plc.decode('utf-8')})
            E.append(R)
        else:
            print "FAILED to process %s" % l

    ALL.append(E)

    fd = __builtin__.open(xmlfile, 'w')
    sys.stdout = fd
    print "<?xml version='1.0' encoding='%s'?>" % UTF8
    prettyprint(ALL)
    fd.close()
    sys.stdout = sys.__stdout__
    

class ODIM_Source:
    ## Initializer
    # @param src string containing a '/what/source' attribute
    def __init__(self, src=None):
        self.source = src
        self.wmo = self.nod = self.rad = self.plc = self.org = self.cty = self.cmt = None
        if self.source: self.split_source()

    ## Splits the input string into identifier values        
    def split_source(self):
        split = self.source.split(',')
        for s in split:
            prefix, value = s.split(':')
            prefix = prefix.lower()  # safety precaution in case someone changes case in their files
            if   prefix == 'wmo': self.wmo = value  # Keep this as a string!
            elif prefix == 'rad': self.rad = value
            elif prefix == 'plc': self.plc = value.decode(UTF8)
            elif prefix == 'nod': self.nod = value
            elif prefix == 'org': self.org = value
            elif prefix == 'cty': self.cty = value
            elif prefix == 'cmt': self.cmt = value 


## Convenience function. Gets the NOD identifier from /what/source .
# Assumes that the NOD is there or can be looked up based on the WMO identifier.
# If WMO isn't there either, then a 'n/a' (not available) is returned.
# @param obj input SCAN or PVOL object
# @return the NOD identifier or 'n/a'
def NODfromSource(obj):
  S = ODIM_Source(obj.source)
  if S.nod: return S.nod
  else:
    try:
      return NOD[S.wmo]
    except KeyError:
      return 'n/a'


## Convenience function. Checks and, if necessary, reformats a complete /what/source attribute .
# Doesn't return anything. 
# @param inobj input SCAN or PVOL object
def CheckSource(inobj):
    S = ODIM_Source(inobj.source)
    if not S.nod:
        try:
            S.nod = NOD[S.wmo]
            inobj.source = SOURCE[S.nod].encode(UTF8)
        except:
            pass


if __name__ == "__main__":
    pass
