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
## Functionality for managing XML in BALTRAD messages.

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-12

#import xml.etree.ElementTree as ET
#from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET

from rave_defines import UTF8, PGF_TAG

##
# Element
class BltXMLElement(ET.Element):
  def __init__(self, tag=None, attrib={}):
    super(BltXMLElement, self).__init__(tag, attrib)

##
# Base element
class BltXML(object):
  def __init__(self, tag=PGF_TAG, encoding=UTF8, filename=None, msg=None):
    self.tag = tag
    self.encoding=encoding
    self.filename=filename
    self.element = BltXMLElement(tag)
    self.header = """<?xml version="1.0" encoding="%s"?>""" % encoding
    if self.filename:
      self.read(self.filename)
    elif msg:
      self.fromstring(msg)

  ## Sets encoding attribute of this instance.
  # @param encoding string, the character encoding used,
  # should probably be UTF-8.
  def setencoding(self, encoding):
    self.encoding = encoding

  ## Formats the object and its contents to an XML string.
  # Suggestion: add line breaks and indentation.
  # @return string XML representation of this message.
  def tostring(self, doindent=True):
    if doindent:
      self.indent()
    s = "%s\n%s" % (self.header, ET.tostring(self.element).decode(self.encoding))
    return s
        
  ## Formats the object from an XML message string.
  # @param msg string XML representation of the message.
  def fromstring(self, msg):
    self.element = ET.fromstring(msg)
  
  ## Indents self
  #
  def indent(self, elem=None, level=0):
    i = "\n" + level*"  "
    if elem == None:
      elem = self.element

    if len(elem):
      if not elem.text or not elem.text.strip():
        elem.text = i + "  "
      if not elem.tail or not elem.tail.strip():
        elem.tail = i
        for elem in elem:
          self.indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
          elem.tail = i
    else:
      if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

  ## Writes an XML message to file.
  # @param filename string of the file to write.
  def save(self, filename=None, doindent=True):
    outfile=self.filename
    if filename is not None:
      outfile=filename
    if outfile is None:
      raise Exception("Can not use None as filename")
    fd = open(outfile, "w")
    fd.write(self.tostring(doindent))
    fd.close()

  ## Reads a message from XML file.
  # @param filename string of the XML file to read.
  def read(self, filename=None):
    element = BltXMLElement()
    infile = self.filename
    if filename is not None:
      infile = filename
    if infile is None:
      raise Exception("Can not read non-specified file")
    efile = ET.parse(infile)
    tag = efile.getroot().tag
    element.tag = tag
    element.extend(list(efile.getroot()))
    # Don't modify self until everything is read
    self.element = element
    self.tag = tag

  def append(self, el):
    self.element.append(el)

  def find(self, name):
    return self.element.find(name)
  
  def remove(self, element):
    self.element.remove(element)
  
  def set(self, key, value):
    self.element.set(key, value)
  
  def subelement(self, key):
    return ET.SubElement(self.element, key)

  def getelement(self):
    return self.element
  
if __name__ == "__main__":
    print(__doc__)
