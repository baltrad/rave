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

## Real-time management of RADVOL configuration options.

## @file
## @author Daniel Michelson, SMHI
## @date 2014-04-02

# Standard python libs:
import sys
import os
import copy
from xml.etree import ElementTree as ET

# Module/Project:
import rave_defines
import odim_source

## Contains site-specific argument settings
CONFIG_FILE = os.path.join(rave_defines.RAVECONFIG, "radvol_params.xml")

initialized = 0

ARGS = {}  # Empty dictionary to be filled with site-specific options/arguments

# Parameters that are integers. All others are floats.
INTS = (
    "DBZHtoTH",
    "BROAD_QIOn",
    "BROAD_QCOn",
    "SPIKE_QIOn",
    "SPIKE_QCOn",
    "SPIKE_AAzim",
    "SPIKE_AVarAzim",
    "SPIKE_ABeam",
    "SPIKE_AVarBeam",
    "SPIKE_BAzim" "NMET_QIOn", ###FIXME?
    "NMET_QCOn",
    "SPECK_QIOn",
    "SPECK_QCOn",
    "BLOCK_QIOn",
    "BLOCK_QCOn",
    "ATT_QIOn",
    "ATT_QCOn",
)


## Initializes the ARGS dictionary by reading content from XML file
def init():
    global initialized
    if initialized:
        return
    
    C = ET.parse(CONFIG_FILE)
    OPTIONS = C.getroot()
    
    default = OPTIONS.find("default")
    default_opts = options()
    for e in list(default):
        if e.tag in INTS:
            default_opts.__setattr__(e.tag, int(e.text))
        else:
            default_opts.__setattr__(e.tag, float(e.text))
    ARGS["default"] = default_opts
    
    for site in list(OPTIONS):
        if site.tag != "default":
            opts = copy.deepcopy(default_opts)
            
            for e in list(site):
                if e.tag in INTS:
                    opts.__setattr__(e.tag, int(e.text))
                else:
                    opts.__setattr__(e.tag, float(e.text))
            
            ARGS[site.tag] = opts
    initialized = 1


## Generic object used to organize options and argument values to radvol.
class options(object):
    pass


## Based on the /what/source attribute, find site-specific options/arguments
# @param inobj input SCAN or PVOL object
# @return options object. If the look-up fails, then default options are returned
def get_options(inobj):
    odim_source.CheckSource(inobj)
    S = odim_source.ODIM_Source(inobj.source)
    try:
        return copy.deepcopy(ARGS[S.nod])
    except KeyError:
        return copy.deepcopy(ARGS["default"])


def proof():
    elems = ET.parse(CONFIG_FILE).getroot()
    d = elems.find("default")
    e = d.find("SPIKE_QIOn")
    SPIKE_QIOn = int(e.text)


## Initialize
init()

if __name__ == "__main__":
    pass
