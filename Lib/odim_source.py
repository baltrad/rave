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

# Hack to come around string handling differences between py3 and py27.
# We might consider go all-in on unicode handling but that will require some time.
use_source_encoding=False
if sys.version_info < (3,):
    use_source_encoding=True
    

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
            CCCC[nod] = cccc
            keys = radar.attrib.keys()
            if "wmo" in keys: 
                wmo = radar.attrib["wmo"]
                WMO[nod] = wmo
            else:
                wmo = None
            if "rad" in keys: 
                rad = radar.attrib["rad"]
                RAD[nod] = rad
            else:
                rad = None
            if "plc" in keys: 
                plc = radar.attrib["plc"]
                PLC[nod] = plc
            else:
                plc = None
            if wmo not in ("00000", None): 
                NOD[wmo] = nod
                SOURCE[nod] = u"NOD:%s" % nod
                if wmo: SOURCE[nod] += ",WMO:%s" % wmo
                if rad: SOURCE[nod] += ",RAD:%s" % rad
                if plc: SOURCE[nod] += ",PLC:%s" % plc
            else:
                SOURCE[nod] = u"NOD:%s" % nod
                if rad: SOURCE[nod] += ",RAD:%s" % rad
                if org and wmo!=None: SOURCE[nod] += ",ORG:%s" % org
                if plc: SOURCE[nod] += ",PLC:%s" % plc
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
            print("FAILED to process %s" % l)

    ALL.append(E)

    fd = __builtin__.open(xmlfile, 'w')
    sys.stdout = fd
    print("<?xml version='1.0' encoding='%s'?>" % UTF8)
    prettyprint(ALL)
    fd.close()
    sys.stdout = sys.__stdout__
    

class ODIM_Source:
    ## Initializer
    # @param src string containing a '/what/source' attribute
    def __init__(self, src=None):
        self.source = src
        if not isinstance(self.source,bytes) and self.source is not None:
          self.source = bytes(self.source, UTF8)
        self.wmo = self.nod = self.rad = self.plc = self.org = self.cty = self.cmt = None
        if self.source: self.split_source()

    ## Splits the input string into identifier values        
    def split_source(self):
        split = self.source.split(b',')
        for s in split:
            prefix, value = s.split(b':')
            prefix = prefix.lower()  # safety precaution in case someone changes case in their files
            if   prefix == b'wmo': self.wmo = value.decode(UTF8)  # Keep this as a string!
            elif prefix == b'rad': self.rad = value.decode(UTF8)
            elif prefix == b'plc': self.plc = value.decode(UTF8)
            elif prefix == b'nod': self.nod = value.decode(UTF8)
            elif prefix == b'org': self.org = value.decode(UTF8)
            elif prefix == b'cty': self.cty = value.decode(UTF8)
            elif prefix == b'cmt': self.cmt = value.decode(UTF8) 


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
            if SOURCE[S.nod] is not None:
              if use_source_encoding:
                inobj.source = SOURCE[S.nod].encode(UTF8)
              else:
                inobj.source = SOURCE[S.nod]
        except Exception as e:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    pass
