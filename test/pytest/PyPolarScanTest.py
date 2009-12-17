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

Tests the polarscan module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-14
'''
import unittest
import os
import _polarscan
import _rave
import string
import numpy
import math

class PyPolarScanTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewScan(self):
    obj = _polarscan.new()
    
    isscan = string.find(`type(obj)`, "PolarScanCore")
    self.assertNotEqual(-1, isscan) 

  def test_time(self):
    obj = _polarscan.new()
    self.assertEquals(None, obj.time)
    obj.time = "200500"
    self.assertEquals("200500", obj.time)
    obj.time = None
    self.assertEquals(None, obj.time)

  def test_time_badValues(self):
    obj = _polarscan.new()
    values = ["10101", "1010101", "1010ab", "1010x0", "abcdef", 123456]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError, e:
        pass

  def test_date(self):
    obj = _polarscan.new()
    self.assertEquals(None, obj.date)
    obj.date = "20050101"
    self.assertEquals("20050101", obj.date)
    obj.date = None
    self.assertEquals(None, obj.date)

  def test_date_badValues(self):
    obj = _polarscan.new()
    values = ["200910101", "2001010", "200a1010", 20091010]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError, e:
        pass

  def test_source(self):
    obj = _polarscan.new()
    self.assertEquals(None, obj.source)
    obj.source = "ABC:10, ABD:1"
    self.assertEquals("ABC:10, ABD:1", obj.source)
    obj.source = None
    self.assertEquals(None, obj.source)



  def test_longitude(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.longitude, 4)
    obj.longitude = 1.0
    self.assertAlmostEquals(1.0, obj.longitude, 4)

  def test_latitude(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.latitude, 4)
    obj.latitude = 1.0
    self.assertAlmostEquals(1.0, obj.latitude, 4)

  def test_height(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEquals(1.0, obj.height, 4)

  def test_invalid_attributes(self):
    obj = _polarscan.new()
    try:
      obj.lon = 1.0
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass

  def test_elangle(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.elangle, 4)
    obj.elangle = 10.0
    self.assertAlmostEquals(10.0, obj.elangle, 4)

  def test_elangle_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.elangle, 4)
    try:
      obj.elangle = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.elangle, 4)

  def test_nbins(self):
    obj = _polarscan.new()
    self.assertEquals(0, obj.nbins)
    try:
      obj.nbins = 10
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEquals(0, obj.nbins)

  def test_nbins_withData(self):
    obj = _polarscan.new()
    data = numpy.zeros((4,5), numpy.int8)
    obj.setData(data)
    self.assertEquals(5, obj.nbins)

  def test_rscale(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.rscale, 4)
    obj.rscale = 10.0
    self.assertAlmostEquals(10.0, obj.rscale, 4)

  def test_rscale_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.rscale, 4)
    try:
      obj.rscale = 10
      self.fail("Expected TypeError")
    except TypeError, e:
      pass
    self.assertAlmostEquals(0.0, obj.rscale, 4)

  def test_nrays(self):
    obj = _polarscan.new()
    self.assertEquals(0, obj.nrays)
    try:
      obj.nrays = 10
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEquals(0, obj.nrays)

  def test_nrays_withData(self):
    obj = _polarscan.new()
    data = numpy.zeros((5,4), numpy.int8)
    obj.setData(data)
    self.assertEquals(5, obj.nrays)

  def test_rstart(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.rstart, 4)
    obj.rstart = 10.0
    self.assertAlmostEquals(10.0, obj.rstart, 4)

  def test_rstart_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.rstart, 4)
    try:
      obj.rstart = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.rstart, 4)

  def test_datatype(self):
    obj = _polarscan.new()
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    try:
      obj.datatype = _rave.RaveDataType_INT
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)

  def test_a1gate(self):
    obj = _polarscan.new()
    self.assertEquals(0, obj.a1gate)
    obj.a1gate = 10
    self.assertEquals(10, obj.a1gate)

  def test_a1gate_typeError(self):
    obj = _polarscan.new()
    self.assertEquals(0, obj.a1gate)
    try:
      obj.a1gate = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.a1gate)

  def test_beamwidth(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.beamwidth, 4)
    obj.beamwidth = 10.0
    self.assertAlmostEquals(10.0, obj.beamwidth, 4)

  def test_beamwidth_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.beamwidth, 4)
    try:
      obj.beamwidth = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.beamwidth, 4)

  def test_quantity(self):
    obj = _polarscan.new()
    self.assertEquals(None, obj.quantity)
    obj.quantity = "DBZH"
    self.assertEquals("DBZH", obj.quantity)

  def test_quantity_None(self):
    obj = _polarscan.new()
    obj.quantity = "DBZH"
    obj.quantity = None
    self.assertEquals(None, obj.quantity)

  def test_quantity_typeError(self):
    obj = _polarscan.new()
    self.assertEquals(None, obj.quantity)
    try:
      obj.quantity = 10
      self.fail("Expected ValueError")
    except ValueError,e:
      pass
    self.assertEquals(None, obj.quantity)

  def testScan_gain(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEquals(10.0, obj.gain, 4)

  def testScan_gain_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.gain, 4)

  def testScan_offset(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    obj.offset = 10.0
    self.assertAlmostEquals(10.0, obj.offset, 4)

  def testScan_offset_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    try:
      obj.offset = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.offset, 4)

  def testScan_nodata(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEquals(10.0, obj.nodata, 4)

  def testScan_nodata_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    try:
      obj.nodata = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.nodata, 4)

  def testScan_undetect(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    obj.undetect = 10.0
    self.assertAlmostEquals(10.0, obj.undetect, 4)

  def testScan_undetect_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    try:
      obj.undetect = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.undetect, 4)

  def testScan_getRangeIndex(self):
    obj = _polarscan.new()
    obj.setData(numpy.zeros((1,200), numpy.int8))
    obj.rscale = 1000.0

    # Ranges are tuples (range, expected index)
    ranges = [(499.0, 0),
              (501.0, 0),
              (999.0, 0),
              (1000.0, 1),
              (1999.0, 1),
              (2001.0, 2),
              (199000.0, 199),
              (199999.0, 199),
              (200000.0, -1)]
    
    for rr in ranges:
      result = obj.getRangeIndex(rr[0])
      self.assertEquals(rr[1], result)

  def test_getRangeIndex_rstartSet(self):
    obj = _polarscan.new()
    obj.setData(numpy.zeros((1,200), numpy.int8))
    obj.rscale = 1000.0
    obj.rstart = 2.0

    # Ranges are tuples (range, expected index)
    ranges = [(499.0, -1),
              (501.0, -1),
              (999.0, -1),
              (1000.0, -1),
              (1999.0, -1),
              (2001.0, 0),
              (199000.0, 197),
              (199999.0, 197),
              (200000.0, 198)]
    
    for rr in ranges:
      result = obj.getRangeIndex(rr[0])
      self.assertEquals(rr[1], result)
  
  def testScan_getAzimuthIndex(self):
    obj = _polarscan.new()
    obj.setData(numpy.zeros((400,1),numpy.int8))
    # Azimuths tuple is ordered by an azimuth in degrees and expected index
    azimuths = [(180.0, 200),
                (90.0, 100),
                (0.0, 0),
                (0.1, 0),
                (0.4, 0),
                (359.9, 0),
                (360.4, 0),
                (-0.1,0),
                (-1.0,399)]
    for azv in azimuths:
      result = obj.getAzimuthIndex(azv[0]*math.pi/180.0)
      self.assertEquals(azv[1], result)

  def test_getValue(self):
    obj = _polarscan.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = obj.undetect
    a[2][1] = obj.nodata
    a[4][5] = obj.undetect

    obj.setData(a)

    pts = [((0,0), (_rave.RaveValueType_UNDETECT, 0.0)),
           ((1,0), (_rave.RaveValueType_DATA, 1.0)),
           ((0,1), (_rave.RaveValueType_DATA, 6.0)),
           ((1,2), (_rave.RaveValueType_NODATA, obj.nodata)),
           ((4,4), (_rave.RaveValueType_DATA, 28.0)),
           ((5,4), (_rave.RaveValueType_UNDETECT, obj.undetect)),
           ((5,5), (_rave.RaveValueType_NODATA, obj.nodata))]
    
    for tval in pts:
      result = obj.getValue(tval[0][0], tval[0][1])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def test_getConvertedValue(self):
    obj = _polarscan.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.gain = 0.5
    obj.offset = 10.0
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = obj.undetect
    a[2][1] = obj.nodata
    a[4][5] = obj.undetect

    obj.setData(a)

    pts = [((0,0), (_rave.RaveValueType_UNDETECT, 0.0)),
           ((1,0), (_rave.RaveValueType_DATA, 10.5)),
           ((0,1), (_rave.RaveValueType_DATA, 13.0)),
           ((1,2), (_rave.RaveValueType_NODATA, obj.nodata)),
           ((4,4), (_rave.RaveValueType_DATA, 24.0)),
           ((5,4), (_rave.RaveValueType_UNDETECT, obj.undetect)),
           ((5,5), (_rave.RaveValueType_NODATA, obj.nodata))]
    
    for tval in pts:
      result = obj.getConvertedValue(tval[0][0], tval[0][1])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def testScan_getValueAtAzimuthAndRange(self):
    obj = _polarscan.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.rscale = 1000.0
    a=numpy.arange(3600)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(360,10)).astype(numpy.float64)      
    a[0][0] = obj.undetect
    a[2][1] = obj.nodata
    a[4][5] = obj.undetect

    obj.setData(a)

    pts = [((0.0,0.0), (_rave.RaveValueType_UNDETECT, 0.0)),        #0, 0
           ((10.0,2000.0), (_rave.RaveValueType_DATA, 102.0)),      #10*10 + 2
           ((20.0,9000), (_rave.RaveValueType_DATA, 209.0)),        #20*10 + 9
           ((2.0,1000), (_rave.RaveValueType_NODATA, obj.nodata)),  #10*20 + 9
           ((4.0,5000), (_rave.RaveValueType_UNDETECT, obj.undetect))]
    
    for tval in pts:
      az = tval[0][0]*math.pi/180.0
      ra = tval[0][1]
      result = obj.getValueAtAzimuthAndRange(az,ra)
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def testScan_getValueAtAzimuthAndRange_outsideRange(self):
    obj = _polarscan.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.rscale = 1000.0
    a=numpy.zeros((360,10), numpy.float64)      
    obj.setData(a)    

    t,v = obj.getValueAtAzimuthAndRange(0.0, 15000.0)
    self.assertEquals(_rave.RaveValueType_NODATA, t)
    self.assertAlmostEquals(255.0, v, 4)

  def testScan_getValueAtAzimuthAndRange_undetect(self):
    obj = _polarscan.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.rscale = 1000.0
    a=numpy.zeros((360,10), numpy.float64)      
    obj.setData(a)    

    t,v = obj.getValueAtAzimuthAndRange(0.0, 1000.0)
    self.assertEquals(_rave.RaveValueType_UNDETECT, t)
    self.assertAlmostEquals(0.0, v, 4)

  def testScan_getNearest(self):
    obj = _polarscan.new()
    obj.nodata = 255.0
    obj.longitude = 14.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.undetect = 0.0
    obj.rscale = 1000.0
    a=numpy.zeros((4,9), numpy.float64)      
    a[0][8] = 10.0
    obj.setData(a)
    
    t,v = obj.getNearest((14.0*math.pi/180.0, 60.08*math.pi/180.0))
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(10.0, v, 4)

  def testScan_setData_int8(self):
    obj = _polarscan.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_CHAR, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)


  def testScan_setData_uint8(self):
    obj = _polarscan.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint8),numpy.uint8)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

  def testScan_setData_int16(self):
    obj = _polarscan.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int16),numpy.int16)
    a=numpy.reshape(a,(12,10)).astype(numpy.int16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

  def testScan_setData_uint16(self):
    obj = _polarscan.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint16),numpy.uint16)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()