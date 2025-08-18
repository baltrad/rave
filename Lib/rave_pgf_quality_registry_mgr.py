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

# Standard python libs:
from rave_defines import QUALITY_REGISTRY, UTF8, PGF_TAG
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


class rave_pgf_quality_registry_mgr_element(Element):
    def __init__(self):
        super(rave_pgf_quality_registry_mgr_element, self).__init__("rave-pgf-quality-registry")
        self.tail = "\n"
        self.text = "\n  "

    ## Adds a plugin to the registry.
    # @param name - the presentation name of the plugin
    # @param module - the module where the plugin class is defined
    # @param cls - the plugin class, should be a subclass of rave_quality_plugin
    def add_plugin(self, name, module, cls):
        e = Element("quality-plugin", {})
        e.set("class", cls)
        e.set("module", module)
        e.set("name", name)
        e.tail = "\n"
        if len(self) > 0:
            e.tail = list(self)[len(list(self)) - 1].tail
            if len(list(self)) > 1:
                list(self)[len(list(self)) - 1].tail = list(self)[len(list(self)) - 2].tail
        self.append(e)

    ## Remove a plugin
    # @param name - the name of the plugin to remove
    def remove_plugin(self, name):
        for i in range(len(list(self))):
            if self[i].get("name") == name:
                t = self[i].tail
                del self[i]
                if i > 0:
                    self[i - 1].tail = t
                return

    ## Returns if this registry has the specified plugin
    # @param name - the name of the plugin
    # @returns True if the registry has the plugin otherwise False
    def has_plugin(self, name):
        for i in range(len(self)):
            if self[i].get("name") == name:
                return True
        return False


class rave_pgf_quality_registry_mgr:
    def __init__(self, filename=QUALITY_REGISTRY, encoding=UTF8):
        self.filename = filename
        self.encoding = encoding
        self.read(self.filename)
        self.header = """<?xml version="1.0" encoding="%s"?>""" % encoding

    ## Formats the object and its contents to an XML string.
    # Suggestion: add line breaks and indentation.
    # @return string XML representation of this message.
    def tostring(self):
        s = "%s\n%s" % (self.header, ET.tostring(self.element).decode("utf-8"))
        return s

    ## Formats the object from an XML message string.
    # @param msg string XML representation of the message.
    def fromstring(self, msg):
        self.element = ET.fromstring(msg)

    ## Writes an XML message to file.
    # @param filename string of the file to write.
    def save(self, filename):
        fd = open(filename, "w")
        fd.write(self.tostring())
        fd.close()

    ## Reads a message from XML file.
    # @param filename string of the XML file to read.
    def read(self, filename):
        element = rave_pgf_quality_registry_mgr_element()
        efile = ET.parse(filename)
        element.extend(list(efile.getroot()))
        self.element = element

    ## Adds a plugin to the registry.
    # @param name - the presentation name of the plugin
    # @param module - the module where the plugin class is defined
    # @param cls - the plugin class, should be a subclass of rave_quality_plugin
    def add_plugin(self, name, module, cls):
        self.element.add_plugin(name, module, cls)

    ## Remove a plugin
    # @param name - the name of the plugin to remove
    def remove_plugin(self, name):
        self.element.remove_plugin(name)

    ## Returns if this registry has the specified plugin
    # @param name - the name of the plugin
    # @returns True if the registry has the plugin otherwise False
    def has_plugin(self, name):
        return self.element.has_plugin(name)


if __name__ == "__main__":
    p = rave_pgf_quality_registry_mgr()
    p.add_plugin("nisse", "nisses_modul", "nisses_klass")
    print(p.tostring())
    p.remove_plugin("nisse")
    print(p.tostring())
