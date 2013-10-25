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
# A manager for modifying the quality based registry. Note, this is not intended for runtime
# usage but is ment to be used by an installation script or configuration tool.
#
# <?xml version='1.0' encoding='UTF-8'?>
# <rave-pgf-composite-quality-registry>
#   <quality-plugin name="ropo" class="ropo_pgf_composite_quality_plugin" />
#   <quality-plugin name="rave-overshooting" class="rave_overshooting_quality_plugin" />
# </rave-pgf-composite-quality-registry>

## 
# @file
# @author Anders Henja, SMHI
# @date 2013-10-24

from rave_defines import QUALITY_REGISTRY, UTF8, PGF_TAG
import xml.etree.ElementTree as ET

class rave_pgf_quality_registry_mgr(ET._ElementInterface):
  header = None
  def __init__(self, filename = QUALITY_REGISTRY, encoding=UTF8):
    self.tail='\n  '
    self.text='\n  '
    self.tag = "rave-pgf-composite-quality-registry"
    self.attrib = {}  # This is needed even if it's empty
    self._children = []
    self.encoding = encoding
    self.setheader(self.encoding)
    if filename:
      self.read(filename)
    else:
      raise ValueError, "Must specify a valid quality registry file name"
    
  def setheader(self, encoding):
    self.header = """<?xml version="1.0" encoding="%s"?>""" % encoding

  ## Formats the object and its contents to an XML string.
  # Suggestion: add line breaks and indentation.
  # @return string XML representation of this message.
  def tostring(self):
    return "%s\n%s" % (self.header, ET.tostring(self))

  ## Formats the object from an XML message string.
  # @param msg string XML representation of the message.
  def fromstring(self, msg):
    e = ET.fromstring(msg)
    self.attrib = e.attrib.copy()
    self._children = e._children

  ## Writes an XML message to file.
  # @param filename string of the file to write.
  def save(self, filename):
    fd = open(filename, 'w')
    fd.write(self.tostring())
    fd.close()

  ## Reads a message from XML file.
  # @param filename string of the XML file to read.
  def read(self, filename):
    this = ET.parse(filename)
    root = this.getroot()
    self.attrib = root.attrib.copy()
    self._children = root._children
    if len(self._children) > 0:
      self.tail = self._children[len(self._children) - 1].tail

    
  ## Adds a plugin to the registry. 
  # @param name - the presentation name of the plugin
  # @param module - the module where the plugin class is defined
  # @param cls - the plugin class, should be a subclass of rave_quality_plugin
  def add_plugin(self, name, module, cls):
    e = ET._ElementInterface("quality-plugin", {})
    e.set("name", name)
    e.set("module", module)
    e.set("class", cls)
    e.tail = '\n'
    if len(self._children) > 0:
      e.tail = self._children[len(self._children)-1].tail
      if len(self._children) > 1:
        self._children[len(self._children)-1].tail = self._children[len(self._children)-2].tail
    self._children.append(e)

  ## Remove a plugin
  # @param name - the name of the plugin to remove
  def remove_plugin(self, name):
    for i in range(len(self._children)):
      if self._children[i].get("name") == name:
        t = self._children[i].tail
        del self._children[i]
        if i > 0:
          self._children[i-1].tail = t
        return
  
  ## Returns if this registry has the specified plugin
  # @param name - the name of the plugin
  # @returns True if the registry has the plugin otherwise False
  def has_plugin(self, name):
    for i in range(len(self._children)):
      if self._children[i].get("name") == name:
        return True
    return False
  
if __name__=="__main__":
  p = rave_pgf_quality_registry_mgr()
  p.add_plugin("nisse", "nisses_modul", "nisses_klass")
  print p.tostring()
  p.remove_plugin("nisse")
  print p.tostring()
  
  