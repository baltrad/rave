'''
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the py rave value module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-02-14
'''
import unittest
import os
import _ravevalue, _raveproperties, _odimsources, _rave
import string
import math

class PyRavePropertiesTest(unittest.TestCase):
  FIXTURE_WITHOUT_SOURCES="fixtures/rave_properties_without_sources.json"
  FIXTURE_WITH_SOURCES="fixtures/rave_properties_with_sources.json"

  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _raveproperties.new()
    isobj = str(type(obj)).find("RavePropertiesCore")
    self.assertNotEqual(-1, isobj)

  def test_empty(self):
    obj = _raveproperties.new()
    self.assertEqual(0, obj.size())

  def test_string_value(self):
    classUnderTest = _raveproperties.new()
    classUnderTest.set("this.property", "this value")
    self.assertTrue(classUnderTest.hasProperty("this.property"))
    self.assertEqual("this value", classUnderTest.get("this.property"))

  def test_long_value(self):
    classUnderTest = _raveproperties.new()
    classUnderTest.set("this.property", 12)
    self.assertTrue(classUnderTest.hasProperty("this.property"))
    self.assertEqual(12, classUnderTest.get("this.property"))

  def test_double_value(self):
    classUnderTest = _raveproperties.new()
    classUnderTest.set("this.property", 12.1)
    self.assertTrue(classUnderTest.hasProperty("this.property"))
    self.assertEqual(12.1, classUnderTest.get("this.property"), 4)

  def test_string_array(self):
    classUnderTest = _raveproperties.new()
    classUnderTest.set("this.property", ["1", "2", "3"])
    self.assertTrue(classUnderTest.hasProperty("this.property"))
    self.assertTrue(set(["1","2","3"]) == set(classUnderTest.get("this.property")))

  def test_long_array(self):
    classUnderTest = _raveproperties.new()
    classUnderTest.set("this.property", [1,2,3])
    self.assertTrue(classUnderTest.hasProperty("this.property"))
    self.assertTrue(set([1,2,3]) == set(classUnderTest.get("this.property")))

  def test_double_array(self):
    classUnderTest = _raveproperties.new()
    classUnderTest.set("this.property", [1.1,2.2,3.3])
    self.assertTrue(classUnderTest.hasProperty("this.property"))
    self.assertEqual(3, len(classUnderTest.get("this.property")))
    self.assertEqual(1.1, classUnderTest.get("this.property")[0], 4)
    self.assertEqual(2.2, classUnderTest.get("this.property")[1], 4)
    self.assertEqual(3.3, classUnderTest.get("this.property")[2], 4)

  def test_handle_properties(self):
    classUnderTest = _raveproperties.new()
    classUnderTest.set("property.1", "hello")
    classUnderTest.set("property.2", 1)
    classUnderTest.set("property.3", 1.2)
    classUnderTest.set("property.4", [1,2,3])
    self.assertEqual(4, classUnderTest.size())
    classUnderTest.remove("property.2")
    self.assertEqual(3, classUnderTest.size())
    self.assertFalse(classUnderTest.hasProperty("property.2"))
    classUnderTest.remove("property.3")
    self.assertEqual(2, classUnderTest.size())
    self.assertFalse(classUnderTest.hasProperty("property.3"))
    classUnderTest.remove("property.1")
    self.assertEqual(1, classUnderTest.size())
    self.assertFalse(classUnderTest.hasProperty("property.1"))
    classUnderTest.remove("property.4")
    self.assertEqual(0, classUnderTest.size())

  def test_sources(self):
    classUnderTest = _raveproperties.new()
    self.assertTrue(classUnderTest.sources is None)
    classUnderTest.sources = _odimsources.new()
    self.assertTrue(classUnderTest.sources is not None)

  def test_load_properties_with_sources(self):
    if _rave.isXmlSupported() and _rave.isJsonSupported():
      result = _raveproperties.load(self.FIXTURE_WITH_SOURCES)
      self.assertTrue(result.hasProperty("rave.acqva.featuremap.dir"))
      self.assertEqual("/var/lib/baltrad/rave/acqva/featuremap", result.get("rave.acqva.featuremap.dir"))
      self.assertTrue(result.hasProperty("rave.rate.zr.coefficients"))
      self.assertEqual(200.0, result.get("rave.rate.zr.coefficients")["sella"][0], 4)
      self.assertEqual(1.6, result.get("rave.rate.zr.coefficients")["sekrn"][1], 4)
      self.assertTrue(result.sources is not None)

  def test_load_properties_without_sources(self):
    if _rave.isXmlSupported() and _rave.isJsonSupported():
      result = _raveproperties.load(self.FIXTURE_WITHOUT_SOURCES)
      self.assertTrue(result.hasProperty("rave.acqva.featuremap.dir"))
      self.assertEqual("/var/lib/baltrad/rave/acqva/featuremap", result.get("rave.acqva.featuremap.dir"))
      self.assertTrue(result.hasProperty("rave.rate.zr.coefficients"))
      self.assertEqual(200.0, result.get("rave.rate.zr.coefficients")["sella"][0], 4)
      self.assertEqual(1.6, result.get("rave.rate.zr.coefficients")["sekrn"][1], 4)
      self.assertTrue(result.sources is None)
