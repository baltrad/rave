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

Tests the cartesian module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-14
'''
import unittest
import os
import _cartesian
import _rave
import string
import numpy

class PyCartesianTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _cartesian.new()
    
    isscan = string.find(`type(obj)`, "CartesianCore")
    self.assertNotEqual(-1, isscan)
  
  def test_time(self):
    obj = _cartesian.new()
    self.assertEquals(None, obj.time)
    obj.time = "200500"
    self.assertEquals("200500", obj.time)
    obj.time = None
    self.assertEquals(None, obj.time)

  def test_time_badValues(self):
    obj = _cartesian.new()
    values = ["10101", "1010101", "1010ab", "1010x0", "abcdef", 123456]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError, e:
        pass

  def test_date(self):
    obj = _cartesian.new()
    self.assertEquals(None, obj.date)
    obj.date = "20050101"
    self.assertEquals("20050101", obj.date)
    obj.date = None
    self.assertEquals(None, obj.date)

  def test_date_badValues(self):
    obj = _cartesian.new()
    values = ["200910101", "2001010", "200a1010", 20091010]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError, e:
        pass

  def test_source(self):
    obj = _cartesian.new()
    self.assertEquals(None, obj.source)
    obj.source = "ABC:10, ABD:1"
    self.assertEquals("ABC:10, ABD:1", obj.source)
    obj.source = None
    self.assertEquals(None, obj.source)
    
  def test_xsize(self):
    obj = _cartesian.new()
    self.assertEquals(0, obj.xsize)
    try:
      obj.xsize = 10
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEquals(0, obj.xsize)

  def test_xsize_fromArray(self):
    obj = _cartesian.new()
    data = numpy.zeros((11,10), numpy.uint8)
    obj.setData(data)
    self.assertEquals(10, obj.xsize)

  def test_ysize(self):
    obj = _cartesian.new()
    self.assertEquals(0, obj.ysize)
    try:
      obj.ysize = 10
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEquals(0, obj.ysize)

  def test_ysize_fromArray(self):
    obj = _cartesian.new()
    data = numpy.zeros((11,10), numpy.uint8)
    obj.setData(data)
    self.assertEquals(11, obj.ysize)

  def test_xscale(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.xscale, 4)
    obj.xscale = 10.0
    self.assertAlmostEquals(10.0, obj.xscale, 4)

  def test_xscale_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.xscale, 4)
    try:
      obj.xscale = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.xscale, 4)

  def test_yscale(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.yscale, 4)
    obj.yscale = 10.0
    self.assertAlmostEquals(10.0, obj.yscale, 4)

  def test_yscale_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.yscale, 4)
    try:
      obj.yscale = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.yscale, 4)

  def test_datatype(self):
    obj = _cartesian.new()
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    try:
      obj.datatype = _rave.RaveDataType_INT
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)

  def test_datatypes_fromData(self):
    types = [(numpy.int8, _rave.RaveDataType_CHAR),
             (numpy.uint8, _rave.RaveDataType_UCHAR),
             (numpy.int16, _rave.RaveDataType_SHORT),
             (numpy.int32, _rave.RaveDataType_INT),
             (numpy.int64, _rave.RaveDataType_LONG),
             (numpy.float32, _rave.RaveDataType_FLOAT),
             (numpy.float64, _rave.RaveDataType_DOUBLE)]

    obj = _cartesian.new()
    for type in types:
      d = numpy.zeros((10,10), type[0])
      obj.setData(d)
      self.assertEquals(type[1], obj.datatype)

  def Xtest_setValidDataTypes(self):
    dtypes = [_rave.RaveDataType_UNDEFINED, _rave.RaveDataType_CHAR, _rave.RaveDataType_UCHAR,
             _rave.RaveDataType_SHORT, _rave.RaveDataType_INT, _rave.RaveDataType_LONG,
             _rave.RaveDataType_FLOAT, _rave.RaveDataType_DOUBLE]

    obj = _cartesian.new()
    for type in types:
      obj.datatype = type
      self.assertEqual(type, obj.datatype)

  def Xtest_invalidDatatype(self):
    obj = _cartesian.new()
    types = [99,100,-2,30]
    for type in types:
      try:
        obj.datatype = type
        self.fail("Expected ValueError")
      except ValueError, e:
        self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)

    
  def test_quantity(self):
    obj = _cartesian.new()
    self.assertEquals(None, obj.quantity)
    obj.quantity = "DBZH"
    self.assertEquals("DBZH", obj.quantity)

  def test_quantity_typeError(self):
    obj = _cartesian.new()
    try:
      obj.quantity = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(None, obj.quantity)
    
  def test_gain(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEquals(10.0, obj.gain, 4)

  def test_gain_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.gain, 4)

  def test_offset(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    obj.offset = 10.0
    self.assertAlmostEquals(10.0, obj.offset, 4)

  def test_offset_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    try:
      obj.offset = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.offset, 4)

  def test_nodata(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEquals(10.0, obj.nodata, 4)

  def test_nodata_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    try:
      obj.nodata = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.nodata, 4)

  def test_undetect(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    obj.undetect = 10.0
    self.assertAlmostEquals(10.0, obj.undetect, 4)

  def test_undetect_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    try:
      obj.undetect = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.undetect, 4)

  def test_areaextent(self):
    obj = _cartesian.new()
    tt = obj.areaextent
    self.assertEquals(4, len(tt))
    self.assertAlmostEquals(0.0, tt[0], 4)
    self.assertAlmostEquals(0.0, tt[1], 4)
    self.assertAlmostEquals(0.0, tt[2], 4)
    self.assertAlmostEquals(0.0, tt[3], 4)

    obj.areaextent = (10.0, 11.0, 12.0, 13.0)
    tt = obj.areaextent
    self.assertEquals(4, len(tt))
    self.assertAlmostEquals(10.0, tt[0], 4)
    self.assertAlmostEquals(11.0, tt[1], 4)
    self.assertAlmostEquals(12.0, tt[2], 4)
    self.assertAlmostEquals(13.0, tt[3], 4)

  def test_areaextent_badTupleSize(self):
    obj = _cartesian.new()
    try:
      obj.areaextent = (10.0, 20.0, 30.0)
      self.fail("Expected type error")
    except TypeError, e:
      pass

    try:
      obj.areaextent = (10.0, 20.0, 30.0, 40.0, 50.0)
      self.fail("Expected type error")
    except TypeError, e:
      pass

  def test_areaextent_illegalData(self):
    obj = _cartesian.new()
    try:
      obj.areaextent = (10.0, "a", 30.0, 40.0)
      self.fail("Expected type error")
    except TypeError, e:
      pass

  def test_getLocationX(self):
    obj = _cartesian.new();
    obj.areaextent = (100.0, 200.0, 300.0, 400.0)
    obj.xscale = 10.0
    
    xpos = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for x in xpos:
      expected = 100.0 + x*10.0
      result = obj.getLocationX(x)
      self.assertAlmostEqual(expected, result, 4)

  def test_getLocationY(self):
    obj = _cartesian.new();
    obj.areaextent = (100.0, 200.0, 300.0, 400.0)
    obj.yscale = 10.0
    
    ypos = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for y in ypos:
      expected = 400.0 - y*10.0
      result = obj.getLocationY(y)
      self.assertAlmostEqual(expected, result, 4)

  def test_projection(self):
    obj = _cartesian.new()
    proj = _rave.projection("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    obj.projection = proj
    self.assertTrue(proj == obj.projection)

  def test_projection_default(self):
    obj = _cartesian.new()
    self.assertTrue(None == obj.projection)

  def test_projection_setNone(self):
    obj = _cartesian.new()
    proj = _rave.projection("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    obj.projection = proj
    self.assertTrue(proj == obj.projection)
    obj.projection = None
    self.assertTrue(None == obj.projection)

  def test_getValue(self):
    obj = _cartesian.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(12,10)).astype(numpy.float64)    
    a[0][1] = obj.nodata
    a[1][0] = obj.undetect
    obj.setData(a)
    
    pairs = [(0, 0, 0.0, _rave.RaveValueType_UNDETECT),
             (1, 0, obj.nodata, _rave.RaveValueType_NODATA), 
             (0, 1, obj.undetect, _rave.RaveValueType_UNDETECT),
             (2, 0, 2.0, _rave.RaveValueType_DATA),
             (0, 3, 30.0, _rave.RaveValueType_DATA)]

    for cval in pairs:
      result = obj.getValue((cval[0],cval[1]))
      self.assertAlmostEquals(cval[2], result[1], 4)
      self.assertEquals(cval[3], result[0])

  def test_setGetValue(self):
    obj = _cartesian.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(12,10)).astype(numpy.float64)  
    obj.setData(a)
    
    data = [((0,1), 10.0, _rave.RaveValueType_DATA),
            ((1,1), 20.0, _rave.RaveValueType_DATA),
            ((2,2), 30.0, _rave.RaveValueType_DATA),
            ((9,4), 49.0, _rave.RaveValueType_DATA),
            ((8,4), obj.nodata, _rave.RaveValueType_NODATA),
            ((4,8), obj.undetect, _rave.RaveValueType_UNDETECT),]
    
    for v in data:
      obj.setValue(v[0],v[1])
    
    # Verify
    for v in data:
      r = obj.getValue(v[0])
      self.assertAlmostEquals(v[1], r[1], 4)
      self.assertEquals(v[2], r[0])

  def test_setData_int8(self):
    obj = _cartesian.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_CHAR, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)


  def test_setData_uint8(self):
    obj = _cartesian.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint8),numpy.uint8)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_setData_int16(self):
    obj = _cartesian.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int16),numpy.int16)
    a=numpy.reshape(a,(12,10)).astype(numpy.int16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_setData_uint16(self):
    obj = _cartesian.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint16),numpy.uint16)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_getData_int8(self):
    obj = _cartesian.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)    
    
    obj.setData(a)
    obj.setValue((3,2), 5)
    obj.setValue((4,4), 7)
    
    result = obj.getData()
    self.assertEquals(5, result[2][3])
    self.assertEquals(7, result[4][4])
    self.assertEquals("int8", result.dtype.name)

  def test_isTransformable(self):
    proj = _rave.projection("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    data = numpy.zeros((10,10), numpy.float64)

    obj = _cartesian.new()
    obj.xscale = 1000.0
    obj.yscale = 1000.0
    
    obj.projection = proj
    obj.setData(data)

    self.assertEquals(True, obj.isTransformable())
    
  def test_isTransformable_noscale(self):
    proj = _rave.projection("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    data = numpy.zeros((10,10), numpy.float64)

    obj = _cartesian.new()
    obj.xscale = 1000.0
    obj.yscale = 1000.0
    
    obj.projection = proj
    obj.setData(data)

    self.assertEquals(True, obj.isTransformable())
    obj.xscale = 1000.0
    obj.yscale = 0.0
    self.assertEquals(False, obj.isTransformable())
    obj.xscale = 0.0
    obj.yscale = 1000.0
    self.assertEquals(False, obj.isTransformable())

  def test_isTransformable_nodata(self):
    proj = _rave.projection("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")

    obj = _cartesian.new()
    obj.xscale = 1000.0
    obj.yscale = 1000.0
    
    obj.projection = proj

    self.assertEquals(False, obj.isTransformable())

  def test_isTransformable_noproj(self):
    data = numpy.zeros((10,10), numpy.float64)

    obj = _cartesian.new()
    obj.xscale = 1000.0
    obj.yscale = 1000.0
    
    obj.setData(data)

    self.assertEquals(False, obj.isTransformable())
