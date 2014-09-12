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

Tests the polarvolume module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-14
'''
import unittest
import os
import _polarvolume
import _polarscan
import _polarscanparam
import _rave
import _polarnav
import string
import _helpers
import math
import numpy

class PyPolarVolumeTest(unittest.TestCase):
  def setUp(self):
    _helpers.triggerMemoryStatus()

  def tearDown(self):
    pass
    
  def test_new(self):
    obj = _polarvolume.new()
    
    result = string.find(`type(obj)`, "PolarVolumeCore")
    self.assertNotEqual(-1, result) 

  def test_isPolarVolume(self):
    obj = _polarvolume.new()
    scan = _polarscan.new()
    self.assertTrue(_polarvolume.isPolarVolume(obj))
    self.assertFalse(_polarvolume.isPolarVolume(scan))

  def test_attribute_visibility(self):
    attrs = ['longitude', 'latitude', 'height', 'time', 'date', 'source',
             'paramname']
    obj = _polarvolume.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_time(self):
    obj = _polarvolume.new()
    self.assertEquals(None, obj.time)
    obj.time = "200500"
    self.assertEquals("200500", obj.time)
    obj.time = None
    self.assertEquals(None, obj.time)

  def test_time_badValues(self):
    obj = _polarvolume.new()
    values = ["10101", "1010101", "1010ab", "1010x0", "abcdef", 123456]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError, e:
        pass

  def test_date(self):
    obj = _polarvolume.new()
    self.assertEquals(None, obj.date)
    obj.date = "20050101"
    self.assertEquals("20050101", obj.date)
    obj.date = None
    self.assertEquals(None, obj.date)

  def test_date_badValues(self):
    obj = _polarvolume.new()
    values = ["200910101", "2001010", "200a1010", 20091010]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError, e:
        pass

  def test_source(self):
    obj = _polarvolume.new()
    self.assertEquals(None, obj.source)
    obj.source = "ABC:10, ABD:1"
    self.assertEquals("ABC:10, ABD:1", obj.source)
    obj.source = None
    self.assertEquals(None, obj.source)

  def test_longitude(self):
    obj = _polarvolume.new()
    self.assertAlmostEquals(0.0, obj.longitude, 4)
    obj.longitude = 10.0
    self.assertAlmostEquals(10.0, obj.longitude, 4)

  def test_longitude_typeError(self):
    obj = _polarvolume.new()
    self.assertAlmostEquals(0.0, obj.longitude, 4)
    try:
      obj.longitude = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.longitude, 4)

  def test_latitude(self):
    obj = _polarvolume.new()
    self.assertAlmostEquals(0.0, obj.latitude, 4)
    obj.latitude = 10.0
    self.assertAlmostEquals(10.0, obj.latitude, 4)

  def test_latitude_typeError(self):
    obj = _polarvolume.new()
    self.assertAlmostEquals(0.0, obj.latitude, 4)
    try:
      obj.latitude = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.latitude, 4)

  def test_erroneous_member(self):
    obj = _polarvolume.new()
    try:
      obj.thisshouldnotexist = 10
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass

    try:
      v = obj.thisshouldnotexist
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass

  def test_height(self):
    obj = _polarvolume.new()
    self.assertAlmostEquals(0.0, obj.height, 4)
    obj.height = 10.0
    self.assertAlmostEquals(10.0, obj.height, 4)

  def test_height_typeError(self):
    obj = _polarvolume.new()
    self.assertAlmostEquals(0.0, obj.height, 4)
    try:
      obj.height = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.height, 4)
  
  def test_getDistance(self):
    obj = _polarvolume.new()
    obj.longitude = 14.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0

    result = obj.getDistance((14.0 * math.pi/180.0, 61.0 * math.pi/180.0))
    self.assertAlmostEquals(111040.1, result, 1)
    #print "distance = %f"%result
  
  def test_addScan(self):
    obj = _polarvolume.new()
    scan = _polarscan.new()
    obj.addScan(scan)
    #self.assertEquals(1, obj.getNumberOfScans())

  def test_addScan_dateTime(self):
    obj = _polarvolume.new();
    obj.date = "20100101"
    obj.time = "100000"
    scan = _polarscan.new()
    obj.addScan(scan)
    self.assertEquals("20100101", scan.date)
    self.assertEquals("100000", scan.time);

  def test_addScan_navigatorChanged(self):
    obj = _polarvolume.new()
    obj.longitude = 10.0
    scan1 = _polarscan.new()
    scan1.longitude = 5.0

    obj.addScan(scan1)
    self.assertAlmostEquals(10.0, scan1.longitude, 4)

    obj.longitude = 15.0
    self.assertAlmostEquals(15.0, scan1.longitude, 4)

    scan1.longitude = 20.0
    self.assertAlmostEquals(20.0, obj.longitude, 4)

  def test_addScan_refcountIncreaseOnScan(self):
    obj = _polarvolume.new()
    ids = []
    for i in range(50):
      scan = _polarscan.new()
      ids.append(`scan`)
      obj.addScan(scan)
    
    for i in range(obj.getNumberOfScans()):
      scan = obj.getScan(i)
      if `scan` != `obj.getScan(i)`:
        self.fail("Failed to verify scan consistency")


  def test_getNumberOfScans(self):
    obj = _polarvolume.new()
    self.assertEquals(0, obj.getNumberOfScans())
    obj.addScan(_polarscan.new())
    self.assertEquals(1, obj.getNumberOfScans())
    obj.addScan(_polarscan.new())
    self.assertEquals(2, obj.getNumberOfScans())
    
  def test_getScan(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan2 = _polarscan.new()
    
    obj.addScan(scan1)
    obj.addScan(scan2)

    scanresult1 = obj.getScan(0)
    scanresult2 = obj.getScan(1)
    
    self.assertTrue (scan1 == scanresult1)
    self.assertTrue (scan2 == scanresult2)

  def test_removeScan(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan2 = _polarscan.new()
    scan3 = _polarscan.new()
    
    obj.addScan(scan1)
    obj.addScan(scan2)
    obj.addScan(scan3)
    
    obj.removeScan(1)
    self.assertEquals(2, obj.getNumberOfScans())
    self.assertTrue(scan1 == obj.getScan(0))
    self.assertTrue(scan3 == obj.getScan(1))

  def test_getScanWithMaxDistance(self):
    obj = _polarvolume.new()
    s1 = _polarscan.new()
    s1.rscale = 500.0
    s1.elangle = 0.5 * math.pi / 180.0
    s1.longitude = 60.0 * math.pi / 180.0
    s1.latitude = 14.0 * math.pi / 180.0
    s1.height = 100.0
    s1.addAttribute("how/value", "s1")
    p1 = _polarscanparam.new()
    p1.quantity = "DBZH"
    p1.setData(numpy.zeros((100, 120), numpy.uint8))
    s1.addParameter(p1)
    
    s2 = _polarscan.new()
    s2.rscale = 1000.0
    s2.elangle = 0.5 * math.pi / 180.0
    s2.longitude = 60.0 * math.pi / 180.0
    s2.latitude = 14.0 * math.pi / 180.0
    s2.height = 100.0
    s2.addAttribute("how/value", "s2")
    p2 = _polarscanparam.new()
    p2.quantity = "DBZH"
    p2.setData(numpy.zeros((100, 120), numpy.uint8))
    s2.addParameter(p2)
    
    obj.addScan(s1)
    obj.addScan(s2)
    
    result = obj.getScanWithMaxDistance()
    self.assertEquals("s2", result.getAttribute("how/value"))
    
  def test_getScanClosestToElevation_outside(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 0.1 * math.pi / 180.0
    obj.addScan(scan1)
    scan2 = _polarscan.new()
    scan2.elangle = 0.5 * math.pi / 180.0
    obj.addScan(scan2)
    scan3 = _polarscan.new()
    scan3.elangle = 2.0 * math.pi / 180.0
    obj.addScan(scan3)
    
    els = [(0.0, 0.1), (0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.31, 0.5), (1.0, 0.5), (2.0, 2.0)]

    for el in els:
      elevation = el[0]*math.pi / 180.0
      result = obj.getScanClosestToElevation(elevation, 0)
      self.assertAlmostEquals(el[1], result.elangle*180.0/math.pi, 5)

  def test_getScanClosestToElevation_inside(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 0.1 * math.pi / 180.0
    obj.addScan(scan1)
    scan2 = _polarscan.new()
    scan2.elangle = 0.5 * math.pi / 180.0
    obj.addScan(scan2)
    scan3 = _polarscan.new()
    scan3.elangle = 2.0 * math.pi / 180.0
    obj.addScan(scan3)
    
    els = [(0.0, None), (0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.31, 0.5), (1.0, 0.5), (2.0, 2.0), (2.1, None)]

    for el in els:
      elevation = el[0]*math.pi / 180.0
      result = obj.getScanClosestToElevation(elevation, 1)
      if el[1] == None:
        self.assertTrue(result == None)
      else:
        self.assertAlmostEquals(el[1], result.elangle*180.0/math.pi, 5)

  def test_getNearest(self):
    obj = _polarvolume.new()
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    scan1 = _polarscan.new()
    scan1.elangle = 0.1 * math.pi / 180.0
    scan1.rstart = 0.0
    scan1.rscale = 5000.0
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = "DBZH"    
    data = numpy.zeros((100, 120), numpy.uint8)
    param.setData(data)
    scan1.addParameter(param)
    
    scan2 = _polarscan.new()
    scan2.elangle = 1.0 * math.pi / 180.0
    scan2.rstart = 0.0
    scan2.rscale = 5000.0
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = "DBZH"    
    data = numpy.ones((100, 120), numpy.uint8)
    param.setData(data)
    scan2.addParameter(param)
    
    obj.addScan(scan1)
    obj.addScan(scan2)
    
    # Allow outside ranges
    t,v = obj.getNearest((12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0, 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(1.0, v, 4)

    t,v = obj.getNearest((12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0, 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(0.0, v, 4)
    
    # Only allow inside ranges
    t,v = obj.getNearest((12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0, 1)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(1.0, v, 4)

    t,v = obj.getNearest((12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0, 1)
    self.assertEquals(_rave.RaveValueType_NODATA, t)

  def test_getNearestParameterValue(self):
    obj = _polarvolume.new()
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    scan1 = _polarscan.new()
    scan1.elangle = 0.1 * math.pi / 180.0
    scan1.rstart = 0.0
    scan1.rscale = 5000.0
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = "DBZH"
    param.offset = 0.0
    param.gain = 0.0
    data = numpy.zeros((100, 120), numpy.uint8)
    param.setData(data)
    scan1.addParameter(param)
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = "MMM"    
    param.offset = 0.0
    param.gain = 0.0
    data = numpy.ones((100, 120), numpy.uint8)
    param.setData(data)
    scan1.addParameter(param)
    
    scan2 = _polarscan.new()
    scan2.elangle = 1.0 * math.pi / 180.0
    scan2.rstart = 0.0
    scan2.rscale = 5000.0
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = "DBZH"    
    param.offset = 0.0
    param.gain = 0.0
    data = numpy.ones((100, 120), numpy.uint8)
    param.setData(data)
    scan2.addParameter(param)
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = "MMM"    
    param.offset = 0.0
    param.gain = 0.0
    data = numpy.zeros((100, 120), numpy.uint8)
    param.setData(data)
    scan2.addParameter(param)
    
    obj.addScan(scan1)
    obj.addScan(scan2)

    # DBZH
    # Allow outside ranges
    t,v = obj.getNearestParameterValue("DBZH", (12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0, 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(1.0, v, 4)

    t,v = obj.getNearestParameterValue("DBZH", (12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0, 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(0.0, v, 4)
    
    # Only allow inside ranges
    t,v = obj.getNearestParameterValue("DBZH", (12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0, 1)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(1.0, v, 4)

    t,v = obj.getNearestParameterValue("DBZH", (12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0, 1)
    self.assertEquals(_rave.RaveValueType_NODATA, t)

    # MMM
    # Allow outside ranges
    t,v = obj.getNearestParameterValue("MMM", (12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0, 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(0.0, v, 4)

    t,v = obj.getNearestParameterValue("MMM", (12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0, 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(1.0, v, 4)
    
    # Only allow inside ranges
    t,v = obj.getNearestParameterValue("MMM", (12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0, 1)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(0.0, v, 4)

    t,v = obj.getNearestParameterValue("DBZH", (12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0, 1)
    self.assertEquals(_rave.RaveValueType_NODATA, t)

  def test_getNearestConvertedParameterValue(self):
    obj = _polarvolume.new()
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    scan1 = _polarscan.new()
    scan1.elangle = 0.1 * math.pi / 180.0
    scan1.rstart = 0.0
    scan1.rscale = 5000.0
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = "DBZH"
    param.offset = 1.0
    param.gain = 2.0
    data = numpy.ones((100, 120), numpy.uint8)
    param.setData(data)
    scan1.addParameter(param)
    
    scan2 = _polarscan.new()
    scan2.elangle = 1.0 * math.pi / 180.0
    scan2.rstart = 0.0
    scan2.rscale = 5000.0
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = "DBZH"    
    param.offset = 3.0
    param.gain = 4.0
    data = numpy.ones((100, 120), numpy.uint8) + 1
    param.setData(data)
    scan2.addParameter(param)
    
    obj.addScan(scan1)
    obj.addScan(scan2)

    # DBZH
    # Allow outside ranges
    t,v = obj.getNearestConvertedParameterValue("DBZH", (12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0, 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(11.0, v, 4)

    t,v = obj.getNearestConvertedParameterValue("DBZH", (12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0, 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(3.0, v, 4)
    
    # Only allow inside ranges
    t,v = obj.getNearestConvertedParameterValue("DBZH", (12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0, 1)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(11.0, v, 4)

    t,v = obj.getNearestConvertedParameterValue("DBZH", (12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0, 1)
    self.assertEquals(_rave.RaveValueType_NODATA, t)


  def test_paramname(self):
    obj = _polarvolume.new()
    self.assertEquals("DBZH", obj.paramname)
    obj.paramname = "MMM"
    self.assertEquals("MMM", obj.paramname)
    try:
      obj.paramname = None
      self.fail("Expected ValueError")
    except ValueError,e:
      self.assertEquals("MMM", obj.paramname)
    
    
  def test_sortByElevations_ascending(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 2.0
    scan2 = _polarscan.new()
    scan2.elangle = 3.0
    scan3 = _polarscan.new()
    scan3.elangle = 1.0
    
    obj.addScan(scan1)
    obj.addScan(scan2)
    obj.addScan(scan3)
    
    obj.sortByElevations(1)
    
    scanresult1 = obj.getScan(0)
    scanresult2 = obj.getScan(1)
    scanresult3 = obj.getScan(2)
    
    self.assertTrue (scan3 == scanresult1)
    self.assertTrue (scan1 == scanresult2)
    self.assertTrue (scan2 == scanresult3)

  def test_sortByElevations_descending(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 2.0
    scan2 = _polarscan.new()
    scan2.elangle = 3.0
    scan3 = _polarscan.new()
    scan3.elangle = 1.0
    
    obj.addScan(scan1)
    obj.addScan(scan2)
    obj.addScan(scan3)
    
    obj.sortByElevations(0)
    
    scanresult1 = obj.getScan(0)
    scanresult2 = obj.getScan(1)
    scanresult3 = obj.getScan(2)
    
    self.assertTrue (scan2 == scanresult1)
    self.assertTrue (scan1 == scanresult2)
    self.assertTrue (scan3 == scanresult3)

  def test_isAscending(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 0.1
    scan2 = _polarscan.new()
    scan2.elangle = 0.3
    scan3 = _polarscan.new()
    scan3.elangle = 0.5
    obj.addScan(scan1)
    obj.addScan(scan2)
    obj.addScan(scan3)
    
    result = obj.isAscendingScans()
    self.assertEquals(True, result)
    
  def test_isAscending_false(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 0.1
    scan2 = _polarscan.new()
    scan2.elangle = 0.3
    scan3 = _polarscan.new()
    scan3.elangle = 0.5
    obj.addScan(scan1)
    obj.addScan(scan3)
    obj.addScan(scan2)
    
    result = obj.isAscendingScans()
    self.assertEquals(False, result)
    
  def test_isTransformable(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 0.1
    scan2 = _polarscan.new()
    scan2.elangle = 0.3
    scan3 = _polarscan.new()
    scan3.elangle = 0.5
    obj.addScan(scan1)
    obj.addScan(scan2)
    obj.addScan(scan3)

    result = obj.isTransformable()
    self.assertEquals(True, result)
    
  def test_isTransformable_noScans(self):
    obj = _polarvolume.new()
    result = obj.isTransformable()
    self.assertEquals(False, result)

  def test_isTransformable_oneScan(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 0.1
    obj.addScan(scan1)
    result = obj.isTransformable()
    self.assertEquals(True, result)

  def test_isTransformable_descending(self):
    obj = _polarvolume.new()
    scan1 = _polarscan.new()
    scan1.elangle = 0.1
    scan2 = _polarscan.new()
    scan2.elangle = 0.01
    obj.addScan(scan1)
    obj.addScan(scan2)
    result = obj.isTransformable()
    self.assertEquals(False, result)

  def test_addScan_with_noElangleInScan(self):
    vol = _polarvolume.new()
    vol.beamwidth = 2.0*math.pi/180.0
    scan = _polarscan.new()
    self.assertAlmostEquals(1.0*math.pi/180.0, scan.beamwidth, 4)
    vol.addScan(scan)    
    self.assertAlmostEquals(2.0*math.pi/180.0, scan.beamwidth, 4)

  def test_addScan_with_elangleInScan(self):
    vol = _polarvolume.new()
    vol.beamwidth = 2.0*math.pi/180.0
    scan = _polarscan.new()
    scan.beamwidth = 3.0*math.pi/180.0
    vol.addScan(scan)    
    self.assertAlmostEquals(3.0*math.pi/180.0, scan.beamwidth, 4)

  def test_setBeamwidth(self):
    vol = _polarvolume.new()
    vol.beamwidth = 2.0*math.pi/180.0
    scan1 = _polarscan.new()
    scan1.beamwidth = 3.0*math.pi/180.0
    vol.addScan(scan1)
    scan2 = _polarscan.new()
    vol.addScan(scan2)
    self.assertAlmostEquals(3.0*math.pi/180.0, scan1.beamwidth, 4)
    self.assertAlmostEquals(2.0*math.pi/180.0, scan2.beamwidth, 4)
    
    vol.beamwidth = 4.0*math.pi/180.0
    self.assertAlmostEquals(4.0*math.pi/180.0, scan1.beamwidth, 4)
    self.assertAlmostEquals(4.0*math.pi/180.0, scan2.beamwidth, 4)
  
  def test_clone(self):
    vol = _polarvolume.new()
    vol.longitude = 1.0
    vol.latitude = 2.0
    vol.height = 3.0
    vol.time = "200000"
    vol.date= "20110101"
    vol.source = "CMT:123"
    vol.beamwidth = 4.0
    scan1 = _polarscan.new()
    scan1.elangle = 2.0
    vol.addScan(scan1)
        
    cpy = vol.clone()

    # Modify the source volume before we test the clone to verify that they aren't bound to each other
    vol.longitude = 4.0
    vol.latitude = 3.0
    vol.height = 2.0
    vol.time = "210000"
    vol.date= "20120101"
    vol.source = "CMT:124"
    vol.beamwidth = 1.0
    vol.getScan(0).elangle = 3.0
    
    self.assertAlmostEquals(1.0, cpy.longitude, 4)
    self.assertAlmostEquals(2.0, cpy.latitude, 4)
    self.assertAlmostEquals(3.0, cpy.height, 4)
    self.assertEquals("200000", cpy.time)
    self.assertEquals("20110101", cpy.date)
    self.assertAlmostEquals(4.0, cpy.beamwidth, 4)
    self.assertEquals(1, cpy.getNumberOfScans())
    self.assertAlmostEquals(2.0, cpy.getScan(0).elangle, 4)
  
  def test_getConvertedVerticalMaxValue(self):
    import _raveio
    vol = _raveio.open("fixtures/pvol_seang_20090501T120000Z.h5").object
    nrscans = vol.getNumberOfScans()
    lon = 12.879571 * math.pi / 180.0
    lat = 56.356382 * math.pi / 180.0
    
    # First dig out max values scan wise
    svalue = 0.0
    stype = _rave.RaveValueType_NODATA
    
    for i in range(nrscans):
      scan = vol.getScan(i)
      type, value = scan.getNearestConvertedParameterValue("DBZH", (lon, lat))
      if type in [_rave.RaveValueType_DATA, _rave.RaveValueType_UNDETECT]:
        if stype == _rave.RaveValueType_DATA and type == _rave.RaveValueType_DATA:
          if svalue < value:
            svalue = value
        elif stype != _rave.RaveValueType_DATA:
          stype = type
          svalue = value

    # Now test the converted max value method
    type, value = vol.getConvertedVerticalMaxValue("DBZH", (lon, lat))
    self.assertEquals(stype, type)
    self.assertAlmostEquals(svalue, value, 4)

  def test_getDistanceField(self):
    polnav = _polarnav.new()
    polnav.lat0 = 60.0 * math.pi / 180.0
    polnav.lon0 = 12.0 * math.pi / 180.0
    polnav.alt0 = 0.0

    expected = []
    rarr=[]
    for i in range(10):
      rarr.append(polnav.reToDh(100.0 * i, (math.pi / 180.0)*0.5)[0])
    rarr.append(-99999.0)
    rarr.append(-99999.0)
    
    expected.append(rarr)
    rarr=[]
    for i in range(12):
      rarr.append(polnav.reToDh(100.0 * i, (math.pi / 180.0)*1.0)[0])
    expected.append(rarr)
    
    s1 = _polarscan.new()
    s1.longitude = polnav.lon0 # We want same settings as the polar navigator so that we can test result
    s1.latitude = polnav.lat0
    s1.height = polnav.alt0
    s1.rscale = 100.0
    s1.elangle = (math.pi / 180.0)*0.5
    
    p1 = _polarscanparam.new()
    p1.quantity="DBZH"    
    data = numpy.zeros((5, 10), numpy.int8)
    p1.setData(data)
    
    s1.addParameter(p1)

    s2 = _polarscan.new()
    s2.longitude = polnav.lon0 # We want same settings as the polar navigator so that we can test result
    s2.latitude = polnav.lat0
    s2.height = polnav.alt0
    s2.rscale = 100.0
    s2.elangle = (math.pi / 180.0)*1.0
    
    p2 = _polarscanparam.new()
    p2.quantity="DBZH"    
    data = numpy.zeros((5, 12), numpy.int8)
    p2.setData(data)
    
    s2.addParameter(p2)

    obj = _polarvolume.new()
    obj.addScan(s1)
    obj.addScan(s2)

    f = obj.getDistanceField()
    self.assertEquals(12, f.xsize)
    self.assertEquals(2, f.ysize)
    
    for j in range(2):
      for i in range(12):
        self.assertAlmostEquals(expected[j][i], f.getValue(i, j)[1], 4)

  def test_getHeightField(self):
    polnav = _polarnav.new()
    polnav.lat0 = 60.0 * math.pi / 180.0
    polnav.lon0 = 12.0 * math.pi / 180.0
    polnav.alt0 = 0.0

    expected = []
    rarr=[]
    for i in range(10):
      rarr.append(polnav.reToDh(100.0 * i, (math.pi / 180.0)*0.5)[1])
    rarr.append(-99999.0)
    rarr.append(-99999.0)
    
    expected.append(rarr)
    rarr=[]
    for i in range(12):
      rarr.append(polnav.reToDh(100.0 * i, (math.pi / 180.0)*1.0)[1])
    expected.append(rarr)
    
    s1 = _polarscan.new()
    s1.longitude = polnav.lon0 # We want same settings as the polar navigator so that we can test result
    s1.latitude = polnav.lat0
    s1.height = polnav.alt0
    s1.rscale = 100.0
    s1.elangle = (math.pi / 180.0)*0.5
    
    p1 = _polarscanparam.new()
    p1.quantity="DBZH"    
    data = numpy.zeros((5, 10), numpy.int8)
    p1.setData(data)
    
    s1.addParameter(p1)

    s2 = _polarscan.new()
    s2.longitude = polnav.lon0 # We want same settings as the polar navigator so that we can test result
    s2.latitude = polnav.lat0
    s2.height = polnav.alt0
    s2.rscale = 100.0
    s2.elangle = (math.pi / 180.0)*1.0
    
    p2 = _polarscanparam.new()
    p2.quantity="DBZH"    
    data = numpy.zeros((5, 12), numpy.int8)
    p2.setData(data)
    
    s2.addParameter(p2)

    obj = _polarvolume.new()
    obj.addScan(s1)
    obj.addScan(s2)

    f = obj.getHeightField()
    self.assertEquals(12, f.xsize)
    self.assertEquals(2, f.ysize)
    
    for j in range(2):
      for i in range(12):
        self.assertAlmostEquals(expected[j][i], f.getValue(i, j)[1], 2)

  def test_getMaxDistance(self):
    s1 = _polarscan.new()
    s1.longitude = 60.0 * math.pi / 180.0
    s1.latitude = 12.0 * math.pi / 180.0
    s1.height = 0.0
    s1.rscale = 1000.0
    s1.elangle = (math.pi / 180.0)*0.5
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((10, 10), numpy.int8)
    param.setData(data)
    s1.addParameter(param)

    s2 = _polarscan.new()
    s2.longitude = 60.0 * math.pi / 180.0
    s2.latitude = 12.0 * math.pi / 180.0
    s2.height = 0.0
    s2.rscale = 1000.0
    s2.elangle = (math.pi / 180.0)*1.5
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((10, 10), numpy.int8)
    param.setData(data)
    s2.addParameter(param)

    vol = _polarvolume.new()
    vol.addScan(s1)
    vol.addScan(s2)
    
    self.assertAlmostEquals(10999.45, vol.getMaxDistance(), 2)

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()