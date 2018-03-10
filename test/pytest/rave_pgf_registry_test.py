'''
Copyright (C) 2018 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the python version of the pgf registry xml handling

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2018-03-10
'''
import unittest
import string
import os
import rave_pgf_registry

class rave_pgf_registry_test(unittest.TestCase):
  TEMPORARY_FILE="rave_pgf_registry_test.xml"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def testRead(self):
    txt = """<?xml version="1.0" encoding="UTF-8"?>
<generate-registry>
  <se.somefunc.1 function="generate" help="help text" module="rave_plugin">
    <arguments floats="zra,zrb" ints="a,b,c" strings="str" />
  </se.somefunc.1>
  <se.somefunc.2 function="generate2" help="help text2" module="rave_plugin2">
    <arguments floats="zraz,zrbz" ints="az,bz,cz" strings="strz" />
  </se.somefunc.2>
</generate-registry>
"""    
    self.writeTempFile(txt)
    
    classUnderTest = rave_pgf_registry.PGF_Registry()
    classUnderTest.read(self.TEMPORARY_FILE)
    self.assertEqual(txt, classUnderTest.tostring())
    
    el = classUnderTest.find("se.somefunc.1")
    self.assertTrue(el != None)
    self.assertEqual("generate", el.attrib["function"])
    self.assertEqual("rave_plugin", el.attrib["module"])
    self.assertEqual("help text", el.attrib["help"])
    self.assertEqual("zra,zrb", el[0].attrib["floats"])
    self.assertEqual("a,b,c", el[0].attrib["ints"])
    self.assertEqual("str", el[0].attrib["strings"])
    
    el = classUnderTest.find("se.somefunc.2")
    self.assertTrue(el != None)
    self.assertEqual("generate2", el.attrib["function"])
    self.assertEqual("rave_plugin2", el.attrib["module"])
    self.assertEqual("help text2", el.attrib["help"])
    self.assertEqual("zraz,zrbz", el[0].attrib["floats"])
    self.assertEqual("az,bz,cz", el[0].attrib["ints"])
    self.assertEqual("strz", el[0].attrib["strings"])
    

  def writeTempFile(self, txt):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
    fp = open(self.TEMPORARY_FILE, "w")
    fp.write(txt)
    fp.close()
