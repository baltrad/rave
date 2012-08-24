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
    self.assertNotEqual(-1, string.find(`type(obj)`, "VerticalProfileCore"))

  def test_isVerticalProfile(self):
    obj = _verticalprofile.new()
    param = _ravefield.new()
    self.assertTrue(_verticalprofile.isVerticalProfile(obj))
    self.assertFalse(_verticalprofile.isVerticalProfile(param))

  def test_setGetLongitude(self):
    obj = _verticalprofile.new()
    self.assertAlmostEquals(0.0, obj.longitude, 4)
    obj.longitude = 1.0
    self.assertAlmostEquals(1.0, obj.longitude, 4)

  def test_setGetLatitude(self):
    obj = _verticalprofile.new()
    self.assertAlmostEquals(0.0, obj.latitude, 4)
    obj.latitude = 1.0
    self.assertAlmostEquals(1.0, obj.latitude, 4)

  def test_setGetHeight(self):
    obj = _verticalprofile.new()
    self.assertAlmostEquals(0.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEquals(1.0, obj.height, 4)

  def test_setGetInterval(self):
    obj = _verticalprofile.new()
    self.assertAlmostEquals(0.0, obj.interval, 4)
    obj.interval = 1.0
    self.assertAlmostEquals(1.0, obj.interval, 4)
    
  def test_setGetMinheight(self):
    obj = _verticalprofile.new()
    self.assertAlmostEquals(0.0, obj.minheight, 4)
    obj.minheight = 1.0
    self.assertAlmostEquals(1.0, obj.minheight, 4)

  def test_setGetMaxheight(self):
    obj = _verticalprofile.new()
    self.assertAlmostEquals(0.0, obj.maxheight, 4)
    obj.maxheight = 1.0
    self.assertAlmostEquals(1.0, obj.maxheight, 4)

  def test_setGetLevels(self):
    obj = _verticalprofile.new()
    self.assertEquals(0, obj.getLevels())
    obj.setLevels(1)
    self.assertEquals(1, obj.getLevels(), 4)
  
  def test_setGetFF(self):
    obj = _verticalprofile.new()
    self.assertTrue(None == obj.getFF())
    f = _ravefield.new()
    f.setData(numpy.zeros((1,10), numpy.uint8))
    f.addAttribute("how/this", 1.0)
    obj.setFF(f)
    result = obj.getFF()
    self.assertAlmostEquals(1.0, result.getAttribute("how/this"), 4)
    
if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()