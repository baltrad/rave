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
SKIP_TESTS=1
try:
  import _projectionregistry
  SKIP_TESTS=0
except:
  print "  Skipping projection registry tests!!"

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
    if SKIP_TESTS == 1:
      return
    import _projectionregistry
    obj = _projectionregistry.new()
    
    isok = string.find(`type(obj)`, "ProjectionRegistryCore")
    self.assertNotEqual(-1, isok)

  def test_load(self):
    if SKIP_TESTS == 1:
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    self.assertTrue(registry != None)
    
    self.assertEquals(5, registry.size())
    self.assertEquals("llwgs84", registry.get(0).id)
    self.assertEquals("rack", registry.get(1).id)
    self.assertEquals("ps14e60n", registry.get(2).id)
    self.assertEquals("laea20e60n", registry.get(3).id)
    self.assertEquals("rot10w30s", registry.get(4).id)

  def test_getByName(self):
    if SKIP_TESTS == 1:
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    self.assertTrue(registry != None)

    self.assertEquals("llwgs84", registry.getByName("llwgs84").id)
    self.assertEquals("rack", registry.getByName("rack").id)
    self.assertEquals("ps14e60n", registry.getByName("ps14e60n").id)
    self.assertEquals("laea20e60n", registry.getByName("laea20e60n").id)
    self.assertEquals("rot10w30s", registry.getByName("rot10w30s").id)

  def test_add(self):
    if SKIP_TESTS == 1:
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    newproj = _projection.new("testid", "something", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    registry.add(newproj)
    self.assertEquals(6, registry.size())
    self.assertEquals("testid", registry.getByName("testid").id)
    self.assertEquals("something", registry.getByName("testid").description)
    
  def test_remove(self):
    if SKIP_TESTS == 1:
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    registry.remove(2)
    self.assertEquals(4, registry.size())
    try:
      registry.getByName("ps14e60n")
      self.fail("Expected IndexError")
    except IndexError,e:
      pass

  def test_removeByName(self):
    if SKIP_TESTS == 1:
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    registry.removeByName("ps14e60n")
    self.assertEquals(4, registry.size())
    try:
      registry.getByName("ps14e60n")
      self.fail("Expected IndexError")
    except IndexError,e:
      pass

  def test_write(self):
    if SKIP_TESTS == 1:
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    newproj = _projection.new("testid", "something", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    registry.add(newproj)
    registry.write(self.TEMPORARY_FILE)

    tree = ElementTree.parse(self.TEMPORARY_FILE)
    projs = tree.findall("projection")
    self.assertEquals(6, len(projs))
    self.assertEquals("testid", projs[5].get('id'))
    self.assertEquals("something", string.strip(projs[5].find("description").text))
    self.assertEquals("+proj=latlong +ellps=WGS84 +datum=WGS84", string.strip(projs[5].find("projdef").text))

  def test_write_2(self):
    if SKIP_TESTS == 1:
      return
    import _projectionregistry    
    registry = _projectionregistry.load(self.FIXTURE)
    newproj = _projection.new("testid", "something", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    registry.add(newproj)
    registry.write(self.TEMPORARY_FILE)

    nreg = _projectionregistry.load(self.TEMPORARY_FILE);
    
    self.assertEquals(6, nreg.size())
    self.assertEquals("testid", nreg.get(5).id)
    self.assertEquals("something", nreg.get(5).description)
    self.assertEquals("+proj=latlong +ellps=WGS84 +datum=WGS84", nreg.get(5).definition)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()