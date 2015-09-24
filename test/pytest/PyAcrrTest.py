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

Tests the py acrr module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2012-06-01
'''
import unittest
import os
import _acrr
import _cartesianparam, _ravefield, _rave
import string
from numpy import array, reshape, uint8
import math

class PyAcrrTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass
  
  def test_new(self):
    obj = _acrr.new()
    isacrr = string.find(`type(obj)`, "AcrrCore")
    self.assertNotEqual(-1, isacrr)

  def test_attribute_visibility(self):
    attrs = ['nodata', 'undetect', 'quality_field_name']
    acrr = _acrr.new()
    alist = dir(acrr)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_nodata(self):
    obj = _acrr.new()
    self.assertAlmostEquals(-1.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEquals(10.0, obj.nodata, 4)

  def test_undetect(self):
    obj = _acrr.new()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    obj.undetect = 11.0
    self.assertAlmostEquals(11.0, obj.undetect, 4)

  def test_quality_field_name(self):
    obj = _acrr.new()
    self.assertEquals("se.smhi.composite.distance.radar", obj.quality_field_name)
    obj.quality_field_name = "se.smhi.composite.distance.radar2"
    self.assertEquals("se.smhi.composite.distance.radar2", obj.quality_field_name)

  def test_before_initalization(self):
    obj = _acrr.new()
    self.assertEquals(False, obj.isInitialized())
    self.assertEquals(None, obj.getQuantity())

  def test_after_initialization(self):
    p1 = _cartesianparam.new()
    p1.quantity = "DBZH"
    p1.setData(reshape(array((255,111,0,111), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = 255.0, 0.0, 0.5, -32.5

    d1 = _ravefield.new()
    d1.addAttribute("what/gain", 2000.0)
    d1.setData(reshape(array((0,100,75,50), uint8), (2,2)))
    d1.addAttribute("how/task", "se.smhi.composite.distance.radar")

    p1.addQualityField(d1)
    
    obj = _acrr.new()
    obj.sum(p1, 200.0, 1.6)
    
    self.assertEquals(True, obj.isInitialized())
    self.assertEquals("DBZH", obj.getQuantity())

  def test_initialize_without_qualityfield(self):
    p1 = _cartesianparam.new()
    p1.quantity = "DBZH"
    p1.setData(reshape(array((255,111,0,111), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = 255.0, 0.0, 0.5, -32.5

    obj = _acrr.new()
    try:
      obj.sum(p1, 200.0, 1.6)
      self.fail("Expected IOError")
    except IOError, e:
      pass
    
    self.assertEquals(False, obj.isInitialized())
    self.assertEquals(None, obj.getQuantity())

  def test_sum_different_quantity(self):
    p1 = _cartesianparam.new()
    p1.quantity = "DBZH"
    p1.setData(reshape(array((255,111,0,111), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = 255.0, 0.0, 0.5, -32.5

    d1 = _ravefield.new()
    d1.addAttribute("what/gain", 2000.0)
    d1.setData(reshape(array((0,100,75,50), uint8), (2,2)))
    d1.addAttribute("how/task", "se.smhi.composite.distance.radar")

    p1.addQualityField(d1)
 
    p2 = _cartesianparam.new()
    p2.quantity = "TH"
    p2.setData(reshape(array((255,111,0,111), uint8), (2,2)))
    p2.nodata, p2.undetect, p2.gain, p2.offset = 255.0, 0.0, 0.5, -32.5

    d2 = _ravefield.new()
    d2.addAttribute("what/gain", 2000.0)
    d2.setData(reshape(array((0,100,75,50), uint8), (2,2)))
    d2.addAttribute("how/task", "se.smhi.composite.distance.radar")

    p2.addQualityField(d2)
    
    obj = _acrr.new()
    obj.sum(p1, 200.0, 1.6)
    try:
      obj.sum(p2, 200.0, 1.6)
      self.fail("Expected IOError")
    except IOError, e:
      pass
    
    self.assertEquals(True, obj.isInitialized())
    self.assertEquals("DBZH", obj.getQuantity())

  def test_accumulate(self):
    p1 = _cartesianparam.new()
    p1.quantity = "DBZH"
    p1.setData(reshape(array((255,111,0,111), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = 255.0, 0.0, 0.5, -32.5
    
    d1 = _ravefield.new()
    d1.addAttribute("what/gain", 2000.0)
    d1.setData(reshape(array((0,100,75,50), uint8), (2,2)))
    d1.addAttribute("how/task", "se.smhi.composite.distance.radar")

    p1.addQualityField(d1)
    
    p2 = _cartesianparam.new()
    p2.quantity = "DBZH"
    p2.setData(reshape(array((255,111,111,0), uint8), (2,2)))
    p2.nodata, p2.undetect, p2.gain, p2.offset = 255.0, 0.0, 0.5, -32.5

    d2 = _ravefield.new()
    d2.addAttribute("what/gain", 2000.0)
    d2.setData(reshape(array((0,0,25,50), uint8), (2,2)))
    d2.addAttribute("how/task", "se.smhi.composite.distance.radar")

    d3 = _ravefield.new()
    d3.addAttribute("what/gain", 2000.0)
    d3.setData(reshape(array((1,1,1,1), uint8), (2,2)))
    d3.addAttribute("how/task", "se.smhi.composite.distance.notused")

    p2.addQualityField(d3)  # Add d3 before d2 so that we know that it performs lookup by name
    p2.addQualityField(d2)

    obj = _acrr.new()
    obj.nodata = -1.0
    obj.undetect = 0.0
    obj.sum(p1, 200.0, 1.6)
    obj.sum(p2, 200.0, 1.6)
    
    result = obj.accumulate(0.0, 2, 1.0)

    self.assertAlmostEquals(1.0, result.getAttribute("what/prodpar"), 4)
    self.assertEquals("ACRR", result.quantity)
    self.assertAlmostEquals(-1.0, result.nodata, 4)
    self.assertAlmostEquals(0.0, result.undetect, 4)
    
    refAcrr = [-1.0, 1.0, 0.5, 0.5]
    refDist = [0.0, 100.0, 50.0, 100.0]
    Acrr = result.getData().flatten()
    Dist = result.getQualityFieldByHowTask("se.smhi.composite.distance.radar").getData().flatten()
    for i in range(len(refAcrr)):
      self.assertAlmostEquals(Acrr[i], refAcrr[i], 2)
      self.assertAlmostEquals(Dist[i], refDist[i], 2)
    