'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

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
# A registry for managing different quality-based algorithms for processing
# and ensuring that specific quality information is added to the resulting
# composite.
#
# This is no interactive registry, instead you will have to modify the xml file
# in COMPOSITE_QUALITY_REGISTRY manually.
#
# <?xml version='1.0' encoding='UTF-8'?>
# <rave-pgf-composite-quality-registry>
#   <quality-plugin name="ropo" class="ropo_pgf_composite_quality_plugin" />
#   <quality-plugin name="rave-overshooting" class="rave_overshooting_quality_plugin" />
# </rave-pgf-composite-quality-registry>

## 
# @file
# @author Anders Henja, SMHI
# @date 2011-11-04

from rave_defines import QUALITY_REGISTRY
import xml.etree.ElementTree as ET

_initialized = False
_registry = {}

##
# Initializes the registry by reading the xml file with the plugin
# definitions.
#
def init():
  global _initialized
  if _initialized: return
  import imp
    
  O = ET.parse(QUALITY_REGISTRY)
  registry = O.getroot()
  for plugin in list(registry):
    name, module, c = plugin.attrib["name"], plugin.attrib["module"], plugin.attrib["class"]
    fd, pathname, description = imp.find_module(module)
    fmodule = imp.load_module(module, fd, pathname, description)
    try:
      fd.close()  # File descriptor is returned open, so close it.
    except:
      pass
    inst = getattr(fmodule, c)
    _registry[name] = inst()
  
  _initialized = True

##
# Adds a plugin to the registry. Used for debugging
# and testing purposes.
#
def add_plugin(name, plug):
  _registry[name] = plug
   

##
# Load the registry
init()


##
# Return the plugin with the given name or None if no such
# plugin exists.  
def get_plugin(name):
  if _registry.has_key(name):
    return _registry[name]
  return None


    