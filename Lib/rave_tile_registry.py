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
import area_registry

from rave_defines import RAVE_TILE_REGISTRY

logger = rave_pgf_logger.rave_pgf_syslog_client()

_initialized = False
_registry = {}

my_area_registry = area_registry.area_registry()

class tiledef(object):
  def __init__(self, id, extent):
    self.id = id
    self.extent = tuple(extent)

  def __repr__(self):
    return "<tile id=\"%s\", extent=\"%f,%f,%f,%f\">"%(self.id, self.extent[0],self.extent[1],self.extent[2],self.extent[3])

##
# Returns if the area id is registered in the tile registry or not
# @param aid: the area identifier (e.g. swegmaps_2000)
# @return if the area id has got a tile definition or not
def has_tiled_area(aid):
  return _registry.has_key(aid)

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
# Creates the tiles for the specified area with the specified number of tiles. The tiles should be a 
# tuple containing (nr of x-tiles, nr of y-tiles).
# @param a: the name of the area as defined in the area registry
# @param tiles: a tuple defining the tile definition (nr of x-tiles, nr of y-tiles)
# @return the tile definition
def generate_tiles_for_area(aid, tiles):
  if len(tiles) != 2 or tiles[0] == 0 or tiles[1] == 0:
    raise ValueError, "tiles should be a tuple (nr of x-tiles, nr of y-tiles)"
  a = my_area_registry.getarea(aid)
  xdistance = a.extent[2]-a.extent[0]
  ydistance = a.extent[3]-a.extent[1]
  xtilesize = xdistance / float(tiles[0])
  ytilesize = ydistance / float(tiles[1])
  
  ulY = a.extent[3]
  
  result=[]
  for y in range(tiles[1]):
    ulY_end = ulY - ytilesize
    ulX = a.extent[0]
    for x in range(tiles[0]):
      ulX_end = ulX + xtilesize
      result.append(tiledef("%d_%d"%(y,x), (ulX,ulY_end,ulX_end,ulY)))
      ulX = ulX_end
    ulY = ulY_end
  return result

##
# Creates an appropriate tile registry definition for the specified area with the
# specified number of tiles. The tiles should be a tuple containing (nr of x-tiles, nr of y-tiles).
# @param a: the name of the area as defined in the area registry
# @param tiles: a tuple defining the tile definition (nr of x-tiles, nr of y-tiles)
# @return the tile definition
def create_tile_definition_for_area(aid, tiles):
  tiledefs = generate_tiles_for_area(aid, tiles)
  print "<area id=\"%s\"><!-- maps to an area definition in area_registry.xml -->"%aid
  for t in tiledefs:
    print "  %s"%t
  print "</area>"

##
# Initializes the registry by reading the xml file with the plugin
# definitions.
#
def init():
  global _initialized
  if _initialized: return
  import imp
    
  O = ET.parse(RAVE_TILE_REGISTRY)
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

if __name__=="__main__":
  create_tile_definition_for_area("swegmaps_2000", (2,2))
