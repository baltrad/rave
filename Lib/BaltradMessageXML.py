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

import xml.etree.ElementTree as ET
from rave_defines import ENCODING, PGF_TAG


## Main container object for manipulating XML. Although this object has a few
# of its own methods, the object's internal heirarchy can contain either
# BltXML or Element objects, most conveniently the latter.
# Based almost entirely on Fredrik Lundh's ElementTree.
class BltXML(ET._ElementInterface):

    ## Constructor
    # @param tag string, the root tag of this object.
    # @param encoding string, the character encoding used,
    # should probably be UTF-8.
    # @param filename string, file from which to read message.
    # @param msg string, message in memory to format to message.
    def __init__(self, tag=PGF_TAG, encoding=ENCODING, filename=None, msg=None):
        self.tag = tag
        self.attrib = {}  # This is needed even if it's empty
        self._children = []
        self.encoding = encoding
        self.setheader(self.encoding)
        if filename:
            self.read(filename)
        elif msg:
            self.fromstring(msg)


    ## Sets encoding attribute of this instance.
    # @param encoding string, the character encoding used,
    # should probably be UTF-8.
    def setencoding(self, encoding):
        self.encoding = encoding


    ## Sets the standard XML header for this instance.
    # @param encoding string, the character encoding used,
    # should probably be UTF-8.
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


## Convenience function for reading XML files.
# @param filename string of the input filename
def read(filename):
    pass


## baltrad_frame message generator for passing data messages to the DEX.
# This convenience function generates only the XML envelope for the message.
# @param sender string, the identity of the sender
# @param channel string, the channel identifier
# @param name string, the file name
# @param tag string, should always be 'baltrad_frame' for these messages
# @param encoding string, should probably always be 'UTF-8'
# @return string XML envelope
def MakeBaltradFrameXML(sender, channel, name,
                        tag='baltrad_frame',encoding=ENCODING):
    this = BltXML(tag=tag, encoding=encoding)
    h, c = ET.Element("header"), ET.Element("content")
    h.set("mimetype", "multipart/form-data")
    h.set("sender", sender)
    this.append(h)
    c.set("channel", channel)
    c.set("name", name)
    c.set("type", "file")
    this.append(c)
    return this.tostring()
    


if __name__ == "__main__":
    print __doc__
