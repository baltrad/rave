'''
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the py composite arguments module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2024-12-13
'''
import unittest
import os
import _compositearguments
import _area
import _projection
import _polarscan
import _polarvolume
import _rave
import string
import math

class PyCompositeArgumentsTest(unittest.TestCase):
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _compositearguments.new()
    
    iscorrect = str(type(obj)).find("CompositeArgumentsCore")
    self.assertNotEqual(-1, iscorrect)

  def test_isCompositeArguments(self):
    obj = _compositearguments.new()
    self.assertEqual(True, _compositearguments.isCompositeArguments(obj))
    
    self.assertEqual(False, _compositearguments.isCompositeArguments("ABC"))

  def test_area(self):
    obj = _compositearguments.new()
    self.assertTrue(obj.area is None)
    obj.area = _area.new()
    self.assertTrue(_area.isArea(obj.area))
    try:
      obj.area = "a2"
      fail("Expected TypeError")
    except TypeError:
      pass
    obj.area = None
    self.assertTrue(obj.area is None)

  def test_product(self):
    obj = _compositearguments.new()
    self.assertTrue(obj.product is None)
    obj.product = "CAPPI"
    self.assertEqual("CAPPI", obj.product)
    obj.product = "NISSE"
    self.assertEqual("NISSE", obj.product)
    obj.product = None
    self.assertTrue(obj.product is None)
    try:
      obj.product = 123
      fail("Expected ValueError")
    except ValueError:
      pass

  def test_time(self):
    obj = _compositearguments.new()
    self.assertEqual(None, obj.time)
    obj.time = "200500"
    self.assertEqual("200500", obj.time)
    obj.time = None
    self.assertEqual(None, obj.time)

  def test_time_badValues(self):
    obj = _compositearguments.new()
    values = ["10101", "1010101", "1010ab", "1010x0", "abcdef", 123456]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError:
        pass

  def test_date(self):
    obj = _compositearguments.new()
    self.assertEqual(None, obj.date)
    obj.date = "20050101"
    self.assertEqual("20050101", obj.date)
    obj.date = None
    self.assertEqual(None, obj.date)

  def test_date_badValues(self):
    obj = _compositearguments.new()
    values = ["200910101", "2001010", "200a1010", 20091010]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError:
        pass

  def test_height(self):
    obj = _compositearguments.new()
    self.assertAlmostEqual(1000.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEqual(1.0, obj.height, 4)
     
  def test_range(self):
    obj = _compositearguments.new()
    self.assertAlmostEqual(500000.0, obj.range, 4)
    obj.range = 1.0
    self.assertAlmostEqual(1.0, obj.range, 4)

  def test_elangle(self):
    obj = _compositearguments.new()
    self.assertAlmostEqual(0.0, obj.elangle, 4)
    obj.elangle = 1.0
    self.assertAlmostEqual(1.0, obj.elangle, 4)

  def test_strategy(self):
    obj = _compositearguments.new()
    self.assertTrue(obj.strategy is None)
    obj.strategy = "mystrategy"
    self.assertEqual("mystrategy", obj.strategy)
    obj.strategy = None
    self.assertTrue(obj.strategy is None)

  def test_arguments(self):
    obj = _compositearguments.new()
    obj.addArgument("interpolation_method", "3d")
    obj.addArgument("interpolation_range", 300.0)
    obj.addArgument("interpolation_height", 300)
    self.assertEqual("3d", obj.getArgument("interpolation_method"))
    self.assertAlmostEqual(300.0, obj.getArgument("interpolation_range"), 4)
    self.assertEqual(300, obj.getArgument("interpolation_height"))

  def test_arguments_notfound(self):
    obj = _compositearguments.new()
    try:
      self.assertEqual("3d", obj.getArgument("interpolation"))
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

    obj.addArgument("interpolation_method", "3d")
    try:
      self.assertEqual("3d", obj.getArgument("interpolation"))
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    
  def test_parameter(self):
    obj = _compositearguments.new()
    try:
      obj.getParameter("DBZH")
    except KeyError:
      pass
    self.assertEqual(0, obj.getParameterCount())
    obj.addParameter("DBZH", 1.0, 0.0)
    self.assertEqual(1, obj.getParameterCount())
    obj.addParameter("TH", 2.0, 1.0)
    self.assertEqual(2, obj.getParameterCount())
    (gain, offset, datatype, nodata, undetect) = obj.getParameter("DBZH")
    self.assertAlmostEqual(1.0, gain, 4)
    self.assertAlmostEqual(0.0, offset, 4)
    self.assertEqual(_rave.RaveDataType_UCHAR, datatype)
    self.assertAlmostEqual(255.0, nodata, 4)
    self.assertAlmostEqual(0.0, undetect, 4)


    (quantity, gain, offset, datatype, nodata, undetect) = obj.getParameterAtIndex(0)
    self.assertEqual("DBZH", quantity)
    self.assertAlmostEqual(1.0, gain, 4)
    self.assertAlmostEqual(0.0, offset, 4)
    self.assertEqual(_rave.RaveDataType_UCHAR, datatype)
    self.assertAlmostEqual(255.0, nodata, 4)
    self.assertAlmostEqual(0.0, undetect, 4)

    (gain, offset, datatype, nodata, undetect) = obj.getParameter("TH")
    self.assertAlmostEqual(2.0, gain, 4)
    self.assertAlmostEqual(1.0, offset, 4)
    self.assertEqual(_rave.RaveDataType_UCHAR, datatype)
    self.assertAlmostEqual(255.0, nodata, 4)
    self.assertAlmostEqual(0.0, undetect, 4)

    (quantity, gain, offset, datatype, nodata, undetect) = obj.getParameterAtIndex(1)
    self.assertEqual("TH", quantity)
    self.assertAlmostEqual(2.0, gain, 4)
    self.assertAlmostEqual(1.0, offset, 4)
    self.assertEqual(_rave.RaveDataType_UCHAR, datatype)
    self.assertAlmostEqual(255.0, nodata, 4)
    self.assertAlmostEqual(0.0, undetect, 4)

  def test_addObject_scan(self):
    obj = _compositearguments.new()
    s1 = _polarscan.new()
    s1.date="20240101"
    obj.addObject(s1)
    self.assertEqual(1, obj.getNumberOfObjects())

  def test_addObject_volume(self):
    obj = _compositearguments.new()
    s1 = _polarvolume.new()
    s1.date="20240101"
    obj.addObject(s1)
    self.assertEqual(1, obj.getNumberOfObjects())

  def test_objects(self):
    obj = _compositearguments.new()
    self.assertEqual(0, obj.getNumberOfObjects())
    s1 = _polarscan.new()
    s1.date="20240101"
    obj.addObject(s1)
    self.assertEqual(1, obj.getNumberOfObjects())
    self.assertEqual("20240101", obj.getObject(0).date)
    s2 = _polarscan.new()
    s2.date="20240102"
    obj.addObject(s2)
    self.assertEqual(2, obj.getNumberOfObjects())
    self.assertEqual("20240102", obj.getObject(1).date)
  
  def test_addQualityFlag(self):
    obj = _compositearguments.new()
    self.assertEqual(0, obj.getNumberOfQualityFlags())
    obj.addQualityFlag("se.smhi.something")
    self.assertEqual(1, obj.getNumberOfQualityFlags())
    obj.addQualityFlag("se.smhi.something.else")
    self.assertEqual(2, obj.getNumberOfQualityFlags())

  def test_setQualityFlags(self):
    obj = _compositearguments.new()
    obj.setQualityFlags(["se.smhi.something", "se.smhi.something.else"])
    self.assertEqual(2, obj.getNumberOfQualityFlags())
    self.assertEqual("se.smhi.something", obj.getQualityFlagAt(0))
    self.assertEqual("se.smhi.something.else", obj.getQualityFlagAt(1))

  def test_getQualityFlagAt(self):
    obj = _compositearguments.new()
    obj.addQualityFlag("se.smhi.something")
    obj.addQualityFlag("se.smhi.something.else")
    self.assertEqual(2, obj.getNumberOfQualityFlags())
    self.assertEqual("se.smhi.something", obj.getQualityFlagAt(0))
    self.assertEqual("se.smhi.something.else", obj.getQualityFlagAt(1))

  def test_removeQualityFlagAt(self):
    obj = _compositearguments.new()
    obj.addQualityFlag("se.smhi.something")
    obj.addQualityFlag("se.smhi.something.else")
    obj.removeQualityFlagAt(0)
    self.assertEqual(1, obj.getNumberOfQualityFlags())
    self.assertEqual("se.smhi.something.else", obj.getQualityFlagAt(0))

  def test_removeQualityFlag(self):
    obj = _compositearguments.new()
    obj.addQualityFlag("se.smhi.something")
    obj.addQualityFlag("se.smhi.something.else")
    obj.removeQualityFlag("se.smhi.something")
    self.assertEqual(1, obj.getNumberOfQualityFlags())
    self.assertEqual("se.smhi.something.else", obj.getQualityFlagAt(0))
    obj.removeQualityFlag("se.smhi.something.else")
    self.assertEqual(0, obj.getNumberOfQualityFlags())
