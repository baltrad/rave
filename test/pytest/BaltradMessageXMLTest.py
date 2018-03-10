'''
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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
------------------------------------------------------------------------*/

Tests the python version of the area registry.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2012-03-16
'''
import unittest
import string
import os
import BaltradMessageXML
from xml.etree.ElementTree import Element, SubElement

class BaltradMessageXMLTest(unittest.TestCase):
  TEMPORARY_FILE="baltrad_message_xml_test.xml"
  TEMPORARY_FILE2="baltrad_message_xml_test2.xml"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
    if os.path.isfile(self.TEMPORARY_FILE2):
      os.unlink(self.TEMPORARY_FILE2)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
    if os.path.isfile(self.TEMPORARY_FILE2):
      os.unlink(self.TEMPORARY_FILE2)

  def testRead(self):
    self.writeTempFile("""<?xml version="1.0" encoding="UTF-8"?>
<slask>
  <list>
    <e name="1" />
    <e name="2" />
  </list>
</slask>      
""")
    classUnderTest = BaltradMessageXML.BltXML()
    classUnderTest.read(self.TEMPORARY_FILE)
    
    self.assertEqual("slask", classUnderTest.tag)
    self.assertEqual("list", list(classUnderTest.element)[0].tag)
    self.assertEqual(2, len(list(list(classUnderTest.element)[0])))
    self.assertEqual("e", list(list(classUnderTest.element)[0])[0].tag)
    self.assertEqual("e", list(list(classUnderTest.element)[0])[1].tag)
    self.assertEqual("1", list(list(classUnderTest.element)[0])[0].attrib['name'])
    self.assertEqual("2", list(list(classUnderTest.element)[0])[1].attrib['name'])

    classUnderTest.save(self.TEMPORARY_FILE2)
    with open(self.TEMPORARY_FILE2) as fp:
      txt = fp.read()
    self.assertEqual("""<?xml version="1.0" encoding="UTF-8"?>
<slask>
  <list>
    <e name="1" />
    <e name="2" />
  </list>
</slask>
""", txt)
    
  def testWrite(self):
    classUnderTest = BaltradMessageXML.BltXML()
    el = SubElement(classUnderTest.element, "list", {})
    n1 = SubElement(el, "e", {"name":"1"})
    n2 = SubElement(el, "e", {"name":"2"})
    classUnderTest.save(self.TEMPORARY_FILE)
    
    with open(self.TEMPORARY_FILE) as fp:
      txt = fp.read()
    self.assertEqual("""<?xml version="1.0" encoding="UTF-8"?>
<bltgenerate>
  <list>
    <e name="1" />
    <e name="2" />
  </list>
</bltgenerate>
""", txt)

  def writeTempFile(self, txt):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
    fp = open(self.TEMPORARY_FILE, "w")
    fp.write(txt)
    fp.close()
