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
## Functionality for managing the registry containing algorithms in the
# product generation framework.

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-20

import os, string
from rave_defines import RAVECONFIG
import BaltradMessageXML
from xml.etree import ElementTree as ET
from rave_defines import GENREG, REGFILE, UTF8


## Registry containing product generation algorithm information.
class PGF_Registry(BaltradMessageXML.BltXML):

    ## Constructor
    # @param tag string of the root Element, should always be GENREG
    # @param encoding string character encoding name, should probably always
    # be UTF-8
    # @param filename string, optional file name from which to read registry.
    # @param msg string, optional XML string containing registry to parse.
    def __init__(self, tag=GENREG, encoding=UTF8, filename=None, msg=None):
      super(PGF_Registry, self).__init__(tag, encoding, filename, msg)

    ## Creates a registry entry for a given algorithm.
    # @param name string algorithm's name.
    # @param module string name of the module to import.
    # @param function string name of the function to run in the module.
    # @param Help string containing explanatory text for this registry entry.
    # @param strings string of comma-separated argument names that are strings.
    # @param ints string of comma-separated argument names that are integers.
    # @param floats string of comma-separated argument names that are floats.
    # @param seqs string of comma-separated argument names that are sequences.
    # @return nothing, the Element is appended to the registry.
    def register(self, name, module, function, Help="",
                 strings="", ints="", floats="", seqs=""):
        e = BaltradMessageXML.BltXML(name)
        e.set("module", module)
        e.set("function", function)
        e.set("help", Help)
        a = ET.SubElement(e, "arguments")
        if len(strings) > 0: a.set("strings", strings)
        if len(ints) > 0: a.set("ints", ints)
        if len(floats) > 0: a.set("floats", floats)
        if len(seqs) > 0: a.set("seqs", seqs)

        self.append(e)
        self.save(REGFILE)


    ## Checks whether an algorithm is registered.
    # @param algorithm string name of the algorithm to check.
    # @return boolean True or False.
    def is_registered(self, algorithm):
        if self.find(algorithm): return True
        else: return False


    ## De-registers an algorithm from the registry.
    # @param name string, name of the item to de-register.
    def deregister(self, name):
        e = self.find(name)
        if e:
            self.remove(e)
        self.save(REGFILE)


    ## @return string a help text comprising the names of each registered
    # algorithm and its descriptive text.
    def Help(self, name):
        if name == None:
            s = "\nRegistered algorithms in RAVE:\n"
            for c in self._children:
                s += "%-15s %s\n" % (c.tag, c.get('help'))
            return s
        else:
            s = "\nAlgorithm: %s" % name
            c = self.find(name)
            if not c:
                return s + " is not registered."
            a = c.find("arguments")
            s += "\nDescription: %s\n" % c.get('help')
            s += "Module: %s\n" % c.get('module')
            s += "Function: %s\n" % c.get('function')
            s += "Argument names:\n"
            s += "\tStrings: %s\n" % a.get('strings')
            s += "\tInts: %s\n" % a.get('ints')
            s += "\tFloats: %s\n" % a.get('floats')
            s += "\tSequences: %s\n" % a.get('seqs')
            return s



if __name__ == "__main__":
    print(__doc__)
