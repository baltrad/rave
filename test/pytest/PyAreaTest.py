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

Tests the py area module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-12-10
'''
import unittest
import os
import _area
import string
import math

class PyAreaTest(unittest.TestCase):
  
  def setUp(self):
    pass

  def tearDown(self):
    pass
  
  def test_new(self):
    obj = _area.new()
    
    isarea = string.find(`type(obj)`, "AreaCore")
    self.assertNotEqual(-1, isarea)
    
  def test_xsize(self):
    obj = _area.new()
    self.assertEquals(0, obj.xsize)
    obj.xsize = 10
    self.assertEquals(10, obj.xsize)

  def test_xsize_typeError(self):
    obj = _area.new()
    self.assertEquals(0, obj.xsize)
    try:
      obj.xsize = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.xsize)

  def test_ysize(self):
    obj = _area.new()
    self.assertEquals(0, obj.ysize)
    obj.ysize = 10
    self.assertEquals(10, obj.ysize)

  def test_ysize_typeError(self):
    obj = _area.new()
    self.assertEquals(0, obj.ysize)
    try:
      obj.ysize = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.ysize)

  def test_xscale(self):
    obj = _area.new()
    self.assertAlmostEquals(0.0, obj.xscale, 4)
    obj.xscale = 10.0
    self.assertAlmostEquals(10.0, obj.xscale, 4)

  def test_xscale_typeError(self):
    obj = _area.new()
    self.assertAlmostEquals(0.0, obj.xscale, 4)
    try:
      obj.xscale = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.xscale, 4)

  def test_yscale(self):
    obj = _area.new()
    self.assertAlmostEquals(0.0, obj.yscale, 4)
    obj.yscale = 10.0
    self.assertAlmostEquals(10.0, obj.yscale, 4)

  def test_yscale_typeError(self):
    obj = _area.new()
    self.assertAlmostEquals(0.0, obj.yscale, 4)
    try:
      obj.yscale = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.yscale, 4)
