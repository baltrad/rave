'''
Copyright (C) 2012- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Area registry functionality for beeing able to provide the user with proper
# rave py-c style area definitions. 
# This registry will first try out the c-api registries for area and projections.
# If this can not be found, then it will try the area registries and see if there
# is anything there. 

from rave_defines import AREA_REGISTRY, PROJECTION_REGISTRY
import _rave, _area, _projection, area
import string

##
# Area registry wrapper
class area_registry(object):
  _registry = {}
  _arearegistryconfig=AREA_REGISTRY
  _projregistryconfig=PROJECTION_REGISTRY
  
  ##
  # Default sonstructor
  def __init__(self, areg=AREA_REGISTRY, preg=PROJECTION_REGISTRY):
    self._arearegistryconfig = areg
    self._projregistryconfig = preg
    self.reload()

  ##
  # Reloads everything from the area registry
  #
  def reload(self):
    self._registry = {}
    if not _rave.isXmlSupported():
      return
    import _arearegistry, _projectionregistry
    projreg = _projectionregistry.load(self._projregistryconfig)
    areareg = _arearegistry.load(self._arearegistryconfig,projreg)
    len = areareg.size()
    for i in range(len):
      a = areareg.get(i)
      self._registry[a.id] = a

  ##
  # Loads the spcific area if it can be found in the projection registry
  #
  def _loadarea(self, areaid):
    if not _rave.isXmlSupported():
      return
    import _arearegistry, _projectionregistry
    projreg = _projectionregistry.load(self._projregistryconfig)
    areareg = _arearegistry.load(self._arearegistryconfig,projreg)
    try:
      foundarea = areareg.getByName(areaid)
      self._registry[foundarea.id] = foundarea
    except IndexError, e:
      pass     

  ##
  # Returns the wanted area if found, otherwise an exception will be thrown
  #
  def getarea(self, areaid):
    if not self._registry.has_key(areaid):
      self._loadarea(areaid)

    if self._registry.has_key(areaid):
      result = self._registry[areaid]
    else:
      a = area.area(areaid)
      p = a.pcs
      pyarea = _area.new()
      pyarea.id = a.Id
      pyarea.xsize = a.xsize
      pyarea.ysize = a.ysize
      pyarea.xscale = a.xscale
      pyarea.yscale = a.yscale
      pyarea.extent = a.extent
      pyarea.projection = _projection.new(p.id, p.name, string.join(p.definition, ' '))
      self._registry[pyarea.id] = pyarea
      result = pyarea
      
    return result
