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
import _acrr
import _cartesianparam, _ravefield
import string
from numpy import array, reshape, uint8

NO_DATA = 255.0
UNDETECT = 0.0
ACRR_NO_DATA = -1.0

DEFAULT_DISTANCE_GAIN = 2000.0

class PyAcrrTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass
  
  def calculateAvgDistanceField(self, distance_fields, gain):
    gain_multiplier = gain / 1000.0
    
    sum_dist_field = None
    for dist_field in distance_fields:
      if sum_dist_field is None:
        sum_dist_field = array(dist_field)*gain_multiplier
      else:
        sum_dist_field += array(dist_field)*gain_multiplier
        
    avg_dist_field = sum_dist_field / len(distance_fields)
    avg_dist_field = [max(x, -1.0) for x in avg_dist_field]
    
    return avg_dist_field
  
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
    self.assertAlmostEquals(ACRR_NO_DATA, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEquals(10.0, obj.nodata, 4)
 
  def test_undetect(self):
    obj = _acrr.new()
    self.assertAlmostEquals(UNDETECT, obj.undetect, 4)
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
    p1.setData(reshape(array((NO_DATA,111,0,111), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = NO_DATA, UNDETECT, 0.5, -32.5
 
    d1 = _ravefield.new()
    d1.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
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
    p1.setData(reshape(array((NO_DATA,111,0,111), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = NO_DATA, UNDETECT, 0.5, -32.5
 
    obj = _acrr.new()
    try:
      obj.sum(p1, 200.0, 1.6)
      self.fail("Expected IOError")
    except IOError:
      pass
     
    self.assertEquals(False, obj.isInitialized())
    self.assertEquals(None, obj.getQuantity())
 
  def test_sum_different_quantity(self):
    p1 = _cartesianparam.new()
    p1.quantity = "DBZH"
    p1.setData(reshape(array((NO_DATA,111,0,111), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = NO_DATA, UNDETECT, 0.5, -32.5
 
    d1 = _ravefield.new()
    d1.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d1.setData(reshape(array((0,100,75,50), uint8), (2,2)))
    d1.addAttribute("how/task", "se.smhi.composite.distance.radar")
 
    p1.addQualityField(d1)
  
    p2 = _cartesianparam.new()
    p2.quantity = "TH"
    p2.setData(reshape(array((NO_DATA,111,0,111), uint8), (2,2)))
    p2.nodata, p2.undetect, p2.gain, p2.offset = NO_DATA, UNDETECT, 0.5, -32.5
 
    d2 = _ravefield.new()
    d2.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d2.setData(reshape(array((0,100,75,50), uint8), (2,2)))
    d2.addAttribute("how/task", "se.smhi.composite.distance.radar")
 
    p2.addQualityField(d2)
     
    obj = _acrr.new()
    obj.sum(p1, 200.0, 1.6)
    try:
      obj.sum(p2, 200.0, 1.6)
      self.fail("Expected IOError")
    except IOError:
      pass
     
    self.assertEquals(True, obj.isInitialized())
    self.assertEquals("DBZH", obj.getQuantity())

  def test_accumulate(self):
    
    distance_field1 = (0,100,75,50)
    distance_field2 = (0,0,25,50)
    
    p1 = _cartesianparam.new()
    p1.quantity = "DBZH"
    p1.setData(reshape(array((NO_DATA,111,UNDETECT,111), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = NO_DATA, UNDETECT, 0.5, -32.5
    
    d1 = _ravefield.new()
    d1.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d1.setData(reshape(array(distance_field1, uint8), (2,2)))
    d1.addAttribute("how/task", "se.smhi.composite.distance.radar")

    p1.addQualityField(d1)
    
    p2 = _cartesianparam.new()
    p2.quantity = "DBZH"
    p2.setData(reshape(array((NO_DATA,111,111,UNDETECT), uint8), (2,2)))
    p2.nodata, p2.undetect, p2.gain, p2.offset = NO_DATA, UNDETECT, 0.5, -32.5

    d2 = _ravefield.new()
    d2.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d2.setData(reshape(array(distance_field2, uint8), (2,2)))
    d2.addAttribute("how/task", "se.smhi.composite.distance.radar")

    d3 = _ravefield.new()
    d3.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d3.setData(reshape(array((1,1,1,1), uint8), (2,2)))
    d3.addAttribute("how/task", "se.smhi.composite.distance.notused")

    p2.addQualityField(d3)  # Add d3 before d2 so that we know that it performs lookup by name
    p2.addQualityField(d2)

    obj = _acrr.new()
    obj.nodata = ACRR_NO_DATA
    obj.undetect = UNDETECT
    obj.sum(p1, 200.0, 1.6)
    obj.sum(p2, 200.0, 1.6)
    
    result = obj.accumulate(0.0, 2, 1.0)

    self.assertAlmostEquals(1.0, result.getAttribute("what/prodpar"), 4)
    self.assertEquals("ACRR", result.quantity)
    self.assertAlmostEquals(ACRR_NO_DATA, result.nodata, 4)
    self.assertAlmostEquals(UNDETECT, result.undetect, 4)
    
    refAcrr = [ACRR_NO_DATA, 1.0, 0.5, 0.5]
    refDist = self.calculateAvgDistanceField([distance_field1, distance_field2], DEFAULT_DISTANCE_GAIN)
    
    refDist[0] = ACRR_NO_DATA # shall be no_data since there is 'no_data' in the acrr-fields at this position
    
    Acrr = result.getData().flatten()
    qfield = result.getQualityFieldByHowTask("se.smhi.composite.distance.radar")
    self.assertAlmostEqual(0.0, qfield.getAttribute("what/offset"), 4)
    self.assertAlmostEqual(1000.0, qfield.getAttribute("what/gain"), 4)
    Dist = qfield.getData().flatten()
    for i in range(len(refAcrr)):
      self.assertAlmostEquals(Acrr[i], refAcrr[i], 2)
      self.assertAlmostEquals(Dist[i], refDist[i], 2)
      
  def test_accumulate_distfield_nodata(self):
    
    distance_field1 = (10,20,30,ACRR_NO_DATA)
    distance_field2 = (10,20,30,ACRR_NO_DATA)
    
    p1 = _cartesianparam.new()
    p1.quantity = "DBZH"
    p1.setData(reshape(array((111,111,UNDETECT,NO_DATA), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = NO_DATA, UNDETECT, 0.5, -32.5
    
    d1 = _ravefield.new()
    d1.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d1.setData(reshape(array(distance_field1, uint8), (2,2)))
    d1.addAttribute("how/task", "se.smhi.composite.distance.radar")

    p1.addQualityField(d1)
    
    p2 = _cartesianparam.new()
    p2.quantity = "DBZH"
    p2.setData(reshape(array((111,111,111,NO_DATA), uint8), (2,2)))
    p2.nodata, p2.undetect, p2.gain, p2.offset = NO_DATA, UNDETECT, 0.5, -32.5

    d2 = _ravefield.new()
    d2.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d2.setData(reshape(array(distance_field2, uint8), (2,2)))
    d2.addAttribute("how/task", "se.smhi.composite.distance.radar")

    p2.addQualityField(d2)

    obj = _acrr.new()
    obj.nodata = ACRR_NO_DATA
    obj.undetect = UNDETECT
    obj.sum(p1, 200.0, 1.6)
    obj.sum(p2, 200.0, 1.6)
    
    result = obj.accumulate(0.0, 2, 1.0)

    self.assertAlmostEquals(1.0, result.getAttribute("what/prodpar"), 4)
    self.assertEquals("ACRR", result.quantity)
    self.assertAlmostEquals(ACRR_NO_DATA, result.nodata, 4)
    self.assertAlmostEquals(UNDETECT, result.undetect, 4)
    
    refAcrr = [1.0, 1.0, 0.5, ACRR_NO_DATA]
    refDist = self.calculateAvgDistanceField([distance_field1, distance_field2], DEFAULT_DISTANCE_GAIN)
    Acrr = result.getData().flatten()
    qfield = result.getQualityFieldByHowTask("se.smhi.composite.distance.radar")
    self.assertAlmostEqual(0.0, qfield.getAttribute("what/offset"), 4)
    self.assertAlmostEqual(1000.0, qfield.getAttribute("what/gain"), 4)
    Dist = qfield.getData().flatten()
    for i in range(len(refAcrr)):
      self.assertAlmostEquals(Acrr[i], refAcrr[i], 2)
      self.assertAlmostEquals(Dist[i], refDist[i], 2)
      
  def test_accumulate_tofewfiles(self):
    p1 = _cartesianparam.new()
    p1.quantity = "DBZH"
    p1.setData(reshape(array((NO_DATA,0,0,0), uint8), (2,2)))
    p1.nodata, p1.undetect, p1.gain, p1.offset = NO_DATA, UNDETECT, 0.5, -32.5
     
    d1 = _ravefield.new()
    d1.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d1.setData(reshape(array((0,100,75,50), uint8), (2,2)))
    d1.addAttribute("how/task", "se.smhi.composite.distance.radar")
 
    p1.addQualityField(d1)
     
    p2 = _cartesianparam.new()
    p2.quantity = "DBZH"
    p2.setData(reshape(array((0,NO_DATA,0,0), uint8), (2,2)))
    p2.nodata, p2.undetect, p2.gain, p2.offset = NO_DATA, UNDETECT, 0.5, -32.5
 
    d2 = _ravefield.new()
    d2.addAttribute("what/gain", DEFAULT_DISTANCE_GAIN)
    d2.setData(reshape(array((10,20,25,50), uint8), (2,2)))
    d2.addAttribute("how/task", "se.smhi.composite.distance.radar")
    p2.addQualityField(d2)
 
    obj = _acrr.new()
    obj.nodata = ACRR_NO_DATA
    obj.undetect = UNDETECT
    obj.sum(p1, 200.0, 1.6)
    obj.sum(p2, 200.0, 1.6)
     
    result = obj.accumulate(0.0, 3, 1.0)
 
    self.assertAlmostEquals(1.0, result.getAttribute("what/prodpar"), 4)
    self.assertEquals("ACRR", result.quantity)
    self.assertAlmostEquals(ACRR_NO_DATA, result.nodata, 4)
    self.assertAlmostEquals(UNDETECT, result.undetect, 4)
     
    refAcrr = [ACRR_NO_DATA, ACRR_NO_DATA, ACRR_NO_DATA, ACRR_NO_DATA]
    Acrr = result.getData().flatten()
    for i in range(len(refAcrr)):
      self.assertAlmostEquals(Acrr[i], refAcrr[i], 2)
    
