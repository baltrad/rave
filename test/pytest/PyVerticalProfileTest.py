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

Tests the verticalprofile module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2012-08-24

@co-author Ulf Nordh (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2017-10-27. Updated code with new fields for vertical profiles.
'''
import unittest
import os
import _verticalprofile
import _ravefield
import string
import numpy
import math

class PyVerticalProfileTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _verticalprofile.new()
    self.assertNotEqual(-1, str(type(obj)).find("VerticalProfileCore"))

  def test_isVerticalProfile(self):
    obj = _verticalprofile.new()
    param = _ravefield.new()
    self.assertTrue(_verticalprofile.isVerticalProfile(obj))
    self.assertFalse(_verticalprofile.isVerticalProfile(param))

  def test_setGetLongitude(self):
    obj = _verticalprofile.new()
    self.assertAlmostEqual(0.0, obj.longitude, 4)
    obj.longitude = 1.0
    self.assertAlmostEqual(1.0, obj.longitude, 4)

  def test_setGetLatitude(self):
    obj = _verticalprofile.new()
    self.assertAlmostEqual(0.0, obj.latitude, 4)
    obj.latitude = 1.0
    self.assertAlmostEqual(1.0, obj.latitude, 4)

  def test_setGetHeight(self):
    obj = _verticalprofile.new()
    self.assertAlmostEqual(0.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEqual(1.0, obj.height, 4)

  def test_setGetInterval(self):
    obj = _verticalprofile.new()
    self.assertAlmostEqual(0.0, obj.interval, 4)
    obj.interval = 1.0
    self.assertAlmostEqual(1.0, obj.interval, 4)
    
  def test_setGetMinheight(self):
    obj = _verticalprofile.new()
    self.assertAlmostEqual(0.0, obj.minheight, 4)
    obj.minheight = 1.0
    self.assertAlmostEqual(1.0, obj.minheight, 4)

  def test_setGetMaxheight(self):
    obj = _verticalprofile.new()
    self.assertAlmostEqual(0.0, obj.maxheight, 4)
    obj.maxheight = 1.0
    self.assertAlmostEqual(1.0, obj.maxheight, 4)
    
  def test_setGetstartTime(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.starttime)
    obj.starttime = "000000"
    self.assertEqual("000000", obj.starttime, 4)
    
  def test_setGetendTime(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.endtime)
    obj.endtime = "000000"
    self.assertEqual("000000", obj.endtime, 4)
    
  def test_setGetstartDate(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.startdate)
    obj.startdate = "20171103"
    self.assertEqual("20171103", obj.startdate, 4)
    
  def test_setGetendDate(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.enddate)
    obj.enddate = "20171103"
    self.assertEqual("20171103", obj.enddate, 4)
    
  def test_setGetProduct(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.product)
    obj.product = "VP"
    self.assertEqual("VP", obj.product, 4)

  def test_setGetLevels(self):
    obj = _verticalprofile.new()
    self.assertEqual(0, obj.getLevels())
    obj.setLevels(1)
    self.assertEqual(1, obj.getLevels(), 4)

  def test_setLevels_withField(self):
    obj = _verticalprofile.new()
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    f.addAttribute("what/quantity", "ff")
    obj.addField(f)
    self.assertEqual(10, obj.getLevels())
    try:
      obj.setLevels(1)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(10, obj.getLevels())
  
  def test_setGetFF(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getFF())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setFF(f)
    result = obj.getFF()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("ff", result.getAttribute("what/quantity"))
 
  def test_setGetFFDev(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getFFDev())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setFFDev(f)
    result = obj.getFFDev()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("ff_dev", result.getAttribute("what/quantity"))
    
  def test_setGetW(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getW())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setW(f)
    result = obj.getW()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("w", result.getAttribute("what/quantity"))

  def test_setGetWDev(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getWDev())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setWDev(f)
    result = obj.getWDev()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("w_dev", result.getAttribute("what/quantity"))

  def test_setGetDD(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getDD())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setDD(f)
    result = obj.getDD()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("dd", result.getAttribute("what/quantity"))

  def test_setGetDDDev(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getDDDev())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setDDDev(f)
    result = obj.getDDDev()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("dd_dev", result.getAttribute("what/quantity"))

  def test_setGetDiv(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getDiv())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setDiv(f)
    result = obj.getDiv()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("div", result.getAttribute("what/quantity"))

  def test_setGetDivDev(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getDivDev())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setDivDev(f)
    result = obj.getDivDev()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("div_dev", result.getAttribute("what/quantity"))

  def test_setGetDef(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getDef())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setDef(f)
    result = obj.getDef()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("def", result.getAttribute("what/quantity"))

  def test_setGetDefDev(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getDefDev())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setDefDev(f)
    result = obj.getDefDev()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("def_dev", result.getAttribute("what/quantity"))

  def test_setGetAD(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getAD())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setAD(f)
    result = obj.getAD()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("ad", result.getAttribute("what/quantity"))

  def test_setGetADDev(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getADDev())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setADDev(f)
    result = obj.getADDev()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("ad_dev", result.getAttribute("what/quantity"))

  def test_setGetDBZ(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getDBZ())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setDBZ(f)
    result = obj.getDBZ()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("dbz", result.getAttribute("what/quantity"))

  def test_setGetDBZDev(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getDBZDev())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setDBZDev(f)
    result = obj.getDBZDev()
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("dbz_dev", result.getAttribute("what/quantity"))

  def test_setGetHGHT(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getHGHT())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setHGHT(f)
    result = obj.getHGHT()
    self.assertAlmostEquals(1.0, result.getAttribute("how/this"), 4)
    self.assertEquals("HGHT", result.getAttribute("what/quantity"))
    
  def test_setGetUWND(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getUWND())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setUWND(f)
    result = obj.getUWND()
    self.assertAlmostEquals(1.0, result.getAttribute("how/this"), 4)
    self.assertEquals("UWND", result.getAttribute("what/quantity"))
    
  def test_setGetVWND(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getVWND())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setVWND(f)
    result = obj.getVWND()
    self.assertAlmostEquals(1.0, result.getAttribute("how/this"), 4)
    self.assertEquals("VWND", result.getAttribute("what/quantity"))
    
  def test_setGetNV(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getNV())
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setNV(f)
    result = obj.getNV()
    self.assertAlmostEquals(1.0, result.getAttribute("how/this"), 4)
    self.assertEquals("n", result.getAttribute("what/quantity"))

  def test_addField(self):
    obj = _verticalprofile.new()
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    f.addAttribute("what/quantity", "ff")
    obj.addField(f)
    result = obj.getField("ff")
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("ff", result.getAttribute("what/quantity"))
    self.assertEqual(10, obj.getLevels())

  def test_addField_withLevels_preset(self):
    obj = _verticalprofile.new()
    obj.setLevels(10)
    f = _ravefield.new()
    f.setData(numpy.zeros((11,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    f.addAttribute("what/quantity", "ff")
    try:
      obj.addField(f)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    result = obj.getField("ff")
    self.assertEqual(None, result)

  def test_addField_differentSize(self):
    obj = _verticalprofile.new()
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    f.addAttribute("what/quantity", "ff")
    obj.addField(f)
    f2 = _ravefield.new()
    f2.setData(numpy.zeros((11,1), numpy.uint8))
    f2.addAttribute("how/this", 2.0)
    f2.addAttribute("what/quantity", "ff_dev")

    try:
      obj.addField(f2)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    result = obj.getField("ff")
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    result2 = obj.getField("ff_dev")
    self.assertEqual(None, result2)

  def test_addField_tooHighXsize(self):
    obj = _verticalprofile.new()
    f = _ravefield.new()
    f.setData(numpy.zeros((10,10), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    f.addAttribute("what/quantity", "ff")
    try:
      obj.addField(f)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(None, obj.getField("ff"))

  def test_addField_dev_bird(self):
    obj = _verticalprofile.new()
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    f.addAttribute("what/quantity", "dev_bird")
    obj.addField(f)
    result = obj.getField("dev_bird")
    self.assertAlmostEqual(1.0, result.getAttribute("how/this"), 4)
    self.assertEqual("dev_bird", result.getAttribute("what/quantity"))
    self.assertEqual(10, obj.getLevels())
    
  def test_addField_no_quantity(self):
    obj = _verticalprofile.new()
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    try:
      obj.addField(f)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_getFields(self):
    obj = _verticalprofile.new()
    f = _ravefield.new()
    f.setData(numpy.zeros((10,1), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    f.addAttribute("what/quantity", "ff")
    obj.addField(f)

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((10,1), numpy.uint8))
    f2.addAttribute("how/this", 2.0)
    f2.addAttribute("what/quantity", "ff_dev")
    obj.addField(f2)
    
    result = obj.getFields()
    self.assertEqual(2, len(result))
    if result[0].getAttribute("what/quantity") == "ff":
      self.assertEqual("ff_dev", result[1].getAttribute("what/quantity"))
    elif result[0].getAttribute("what/quantity") == "ff_dev":
      self.assertEqual("ff", result[1].getAttribute("what/quantity"))
    else:
      self.fail("Unexpected combination of quantities")

  def test_addAttribute(self):
    obj = _verticalprofile.new()
    obj.addAttribute("how/astr", "astr")
    obj.addAttribute("how/int", 10)
    obj.addAttribute("how/double", 10.2)
    self.assertEqual("astr", obj.getAttribute("how/astr"))
    self.assertEqual(10, obj.getAttribute("how/int"))
    self.assertAlmostEqual(10.2, obj.getAttribute("how/double"), 4)
    
  def test_add_how_array_attribute_double(self):
    obj = _verticalprofile.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.float32))
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertAlmostEqual(0.0, result[0], 2)
    self.assertAlmostEqual(3.0, result[3], 2)
    self.assertAlmostEqual(5.0, result[5], 2)
    self.assertAlmostEqual(9.0, result[9], 2)
  
  def test_hasAttribute(self):
    obj = _verticalprofile.new()
    obj.addAttribute("how/something", 1.0)
    obj.addAttribute("how/something2", "jupp")
    self.assertEqual(True, obj.hasAttribute("how/something"))
    self.assertEqual(True, obj.hasAttribute("how/something2"))
    self.assertEqual(False, obj.hasAttribute("how/something3"))
    try:
      obj.hasAttribute(None)
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_getAttributeNames(self):
    obj = _verticalprofile.new()
    obj.addAttribute("how/something", 1.0)
    obj.addAttribute("how/something2", "jupp")
    result = obj.getAttributeNames()
    self.assertTrue(set(result) == set(["how/something", "how/something2"]))

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()
