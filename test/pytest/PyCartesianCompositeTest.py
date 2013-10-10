'''
Copyright (C) 2013 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the cartesian composite module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2013-10-10
'''
import unittest
import os
import _cartesian
import _cartesiancomposite
import _rave
import _area
import _projection
import _raveio
import math
import string
import numpy

class PyCartesianCompositeTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _cartesiancomposite.new()
    self.assertNotEqual(-1, string.find(`type(obj)`, "CartesianCompositeCore"))

  def test_date(self):
    obj = _cartesiancomposite.new()
    self.assertEquals(None, obj.date)
    obj.date = "20120101"
    self.assertEquals("20120101", obj.date)
    obj.date = None
    self.assertEquals(None, obj.date)

  def test_time(self):
    obj = _cartesiancomposite.new()
    self.assertEquals(None, obj.time)
    obj.time = "100000"
    self.assertEquals("100000", obj.time)
    obj.time = None
    self.assertEquals(None, obj.time)
    
  def test_quantity(self):
    obj = _cartesiancomposite.new()
    self.assertEquals("DBZH", obj.quantity)
    obj.quantity = "TH"
    self.assertEquals("TH", obj.quantity)
    try:
      obj.quantity = None
      self.fail("Expected ValueError")
    except ValueError, e:
      pass
    self.assertEquals("TH", obj.quantity)

  def test_offset(self):
    obj = _cartesiancomposite.new()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    obj.offset = 2.5
    self.assertAlmostEquals(2.5, obj.offset, 4)
  
  def test_gain(self):
    obj = _cartesiancomposite.new()
    self.assertAlmostEquals(1.0, obj.gain, 4)
    obj.gain = 2.0
    self.assertAlmostEquals(2.0, obj.gain, 4)

  #def test_attribute_visibility(self):
  #  attrs = ['height', 'range', 'elangle', 'product', 'date', 'time', 'selection_method']
  #  obj = _pycomposite.new()
  #  alist = dir(obj)
  #  for a in attrs:
  #    self.assertEquals(True, a in alist)
  