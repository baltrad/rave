'''
Copyright (C) 2014- Swedish Meteorological and Hydrological Institute (SMHI)

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
##
# A registry for keeping track on areas that are divided into a number of tiles for faster processing.
#
# This is no interactive registry, instead you will have to modify the xml file'manually.
#
# <?xml version='1.0' encoding='UTF-8'?>
# <rave-tile-registry>
#   <area id="bltgmaps_2000"> <!-- maps to an area definition in area_registry.xml
#     <!-- 971337.728807, 7196461.17902, 3015337.72881, 11028461.179 -->
#     <tile id="1" extent="971337.728807,7196461.17902,1993337.7288084999,9112461.1790100001" /> <!-- 0, 0 -->
#     <tile id="2" extent="1993337.7288084999,7196461.17902,3015337.72881,9112461.1790100001" /> <!-- 1, 0 -->
#     ....
#   </area>
# </rave-tile-registry>

## 
# @file
# @author Anders Henja, SMHI
# @date 2011-11-04

#from rave_defines import TILE_REGISTRY
import xml.etree.ElementTree as ET
import _area
import rave_pgf_logger

logger = rave_pgf_logger.rave_pgf_syslog_client()

TILE_REGISTRY="slask.xml"

_initialized = False
_registry = {}

class tiledef(object):
  def __init__(self, id, extent):
    self.id = id
    self.extent = tuple(extent)

  def __repr__(self):
    return "<tildef id='%s', extent=%s>"%(self.id, `self.extent`)

##
# Get all the tiled areas belonging to the specified area. The area has to reside in the area_registry in order
# for this area to be registered.
# @param a the AreaCore (_area) instance
# @return: a list of tiled area definitions
def get_tiled_areas(a):
  if not _registry.has_key(a.id):
    raise KeyError, "No such area (%s) with tiles defined"%a.id
  tiledareas=[]
  
  totalx, totaly = 0, 0
  
  for ta in _registry[a.id]:
    pyarea = _area.new()
    pyarea.id = "%s_%s"%(a.id, ta.id)
    pyarea.xsize = int(round((ta.extent[2] - ta.extent[0]) / a.xscale))
    pyarea.ysize = int(round((ta.extent[3] - ta.extent[1]) / a.yscale))
    totalx = totalx + pyarea.xsize
    totaly = totaly + pyarea.ysize
    pyarea.xscale = a.xscale
    pyarea.yscale = a.yscale
    pyarea.projection = a.projection
    pyarea.extent = ta.extent
    tiledareas.append(pyarea)

  return tiledareas    

##
# Initializes the registry by reading the xml file with the plugin
# definitions.
#
def init():
  global _initialized
  if _initialized: return
  import imp
    
  O = ET.parse(TILE_REGISTRY)
  registry = O.getroot()
  for adef in list(registry):
    aid = adef.attrib["id"]
    tiles=[]
    for tdef in list(adef):
      extent = [float(a.strip()) for a in tdef.attrib["extent"].split(",")]
      td = tiledef(tdef.attrib["id"], extent)
      tiles.append(td)
    _registry[aid] = tiles

  _initialized = True

##
# Load the registry
init()

