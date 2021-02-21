'''
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the py projection registry module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2010-12-15
'''
import unittest

import _projection
import _rave
import string
import os

from xml.etree import ElementTree

class PyProjectionRegistryTest(unittest.TestCase):
  FIXTURE="fixtures/fixture_projections.xml"
  TEMPORARY_FILE="pyprojtest.xml"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def test_new(self):
    if not _rave.isXmlSupported():
      return
    import _projectionregistry
    obj = _projectionregistry.new()
    
    self.assertNotEqual(-1, str(type(obj)).find("ProjectionRegistryCore"))

  def test_load(self):
    if not _rave.isXmlSupported():
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    self.assertTrue(registry != None)
    
    self.assertEqual(5, registry.size())
    self.assertEqual("llwgs84", registry.get(0).id)
    self.assertEqual("rack", registry.get(1).id)
    self.assertEqual("ps14e60n", registry.get(2).id)
    self.assertEqual("laea20e60n", registry.get(3).id)
    self.assertEqual("rot10w30s", registry.get(4).id)

  def test_getByName(self):
    if not _rave.isXmlSupported():
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    self.assertTrue(registry != None)

    self.assertEqual("llwgs84", registry.getByName("llwgs84").id)
    self.assertEqual("rack", registry.getByName("rack").id)
    self.assertEqual("ps14e60n", registry.getByName("ps14e60n").id)
    self.assertEqual("laea20e60n", registry.getByName("laea20e60n").id)
    self.assertEqual("rot10w30s", registry.getByName("rot10w30s").id)

  def test_add(self):
    if not _rave.isXmlSupported():
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    newproj = _projection.new("testid", "something", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    registry.add(newproj)
    self.assertEqual(6, registry.size())
    self.assertEqual("testid", registry.getByName("testid").id)
    self.assertEqual("something", registry.getByName("testid").description)
    
  def test_remove(self):
    if not _rave.isXmlSupported():
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    registry.remove(2)
    self.assertEqual(4, registry.size())
    try:
      registry.getByName("ps14e60n")
      self.fail("Expected IndexError")
    except IndexError:
      pass

  def test_removeByName(self):
    if not _rave.isXmlSupported():
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    registry.removeByName("ps14e60n")
    self.assertEqual(4, registry.size())
    try:
      registry.getByName("ps14e60n")
      self.fail("Expected IndexError")
    except IndexError:
      pass

  def test_write(self):
    if not _rave.isXmlSupported():
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    newproj = _projection.new("testid", "something", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    registry.add(newproj)
    registry.write(self.TEMPORARY_FILE)

    tree = ElementTree.parse(self.TEMPORARY_FILE)
    projs = tree.findall("projection")
    self.assertEqual(6, len(projs))
    self.assertEqual("testid", projs[5].get('id'))
    self.assertEqual("something", projs[5].find("description").text.strip())
    self.assertEqual("+proj=latlong +ellps=WGS84 +datum=WGS84", projs[5].find("projdef").text.strip())

  def test_write_2(self):
    if not _rave.isXmlSupported():
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    newproj = _projection.new("testid", "something", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    registry.add(newproj)
    registry.write(self.TEMPORARY_FILE)

    nreg = _projectionregistry.load(self.TEMPORARY_FILE);
    
    self.assertEqual(6, nreg.size())
    self.assertEqual("testid", nreg.get(5).id)
    self.assertEqual("something", nreg.get(5).description)
    self.assertEqual("+proj=latlong +ellps=WGS84 +datum=WGS84", nreg.get(5).definition)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()