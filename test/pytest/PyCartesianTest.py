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
import _projection
import _rave
import _area
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
  
  def test_attribute_visibility(self):
    attrs = ['areaextent', 'datatype', 'date', 'gain', 'nodata', 'objectType', 
     'offset', 'product', 'projection', 'quantity', 'source', 'time',
     'undetect', 'xscale', 'xsize', 'yscale', 'ysize']
    obj = _cartesian.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)
  
  def test_init(self):
    obj = _cartesian.new()
    a = _area.new()
    a.xsize = 10
    a.ysize = 10
    a.xscale = 100.0
    a.yscale = 100.0
    a.extent = (1.0, 2.0, 3.0, 4.0)
    a.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")

    obj.init(a, _rave.RaveDataType_SHORT)
    self.assertEquals(10, obj.xsize)
    self.assertEquals(10, obj.ysize)
    self.assertAlmostEquals(100.0, obj.xscale, 4)
    self.assertAlmostEquals(100.0, obj.yscale, 4)
    self.assertAlmostEquals(1.0, obj.areaextent[0], 4)
    self.assertAlmostEquals(2.0, obj.areaextent[1], 4)
    self.assertAlmostEquals(3.0, obj.areaextent[2], 4)
    self.assertAlmostEquals(4.0, obj.areaextent[3], 4)
    self.assertEquals("x", obj.projection.id)
  
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
  
  def test_objectType(self):
    obj = _cartesian.new()
    obj.objectType = _rave.Rave_ObjectType_COMP
    self.assertEquals(_rave.Rave_ObjectType_COMP, obj.objectType)
    obj.objectType = _rave.Rave_ObjectType_IMAGE
    self.assertEquals(_rave.Rave_ObjectType_IMAGE, obj.objectType)
  
  def test_objectType_invalid(self):
    obj = _cartesian.new()
    try:
      obj.objectType = _rave.Rave_ObjectType_CVOL
      fail("Expected ValueError")
    except ValueError, e:
      pass
  
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
    self.assertAlmostEquals(1.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEquals(10.0, obj.gain, 4)

  def test_setGain_zero(self):
    obj = _cartesian.new()
    obj.gain = 0.0
    self.assertAlmostEquals(1.0, obj.gain, 4)


  def test_gain_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEquals(1.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(1.0, obj.gain, 4)

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

  def test_getIndexX(self):
    obj = _cartesian.new();
    obj.areaextent = (100.0, 200.0, 300.0, 400.0)
    obj.xscale = 10.0
    
    xpos = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for x in xpos:
      value = 100.0 + x*10.0
      result = obj.getIndexX(value)
      self.assertEquals(x, result)

  def test_getIndexY(self):
    obj = _cartesian.new();
    obj.areaextent = (100.0, 200.0, 300.0, 400.0)
    obj.yscale = 10.0
    
    ypos = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for y in ypos:
      value = 400.0 - y*10.0
      result = obj.getIndexY(value)
      self.assertEquals(y, result)


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

  def test_getMean(self):
    obj = _cartesian.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    data = numpy.zeros((5,5), numpy.float64)

    for y in range(5):
      for x in range(5):
        data[y][x] = float(x+y*5)
        
    # add some nodata and undetect
    data[0][0] = obj.nodata    # 0
    data[0][3] = obj.nodata    # 3
    data[1][2] = obj.nodata    # 7
    data[1][3] = obj.undetect  # 8
    data[3][2] = obj.undetect  # 17
    data[4][4] = obj.nodata    # 24
    
    obj.setData(data)
    
    # Nodata
    (t,v) = obj.getMean((0,0), 2)
    self.assertEquals(t, _rave.RaveValueType_NODATA)

    # Undetect
    (t,v) = obj.getMean((3,1), 2)
    self.assertEquals(t, _rave.RaveValueType_UNDETECT)
    
    # Left side with one nodata
    expected = data[1][0]
    (t,v) = obj.getMean((0,1), 2) 
    self.assertEquals(t, _rave.RaveValueType_DATA)
    self.assertAlmostEquals(v, expected)

    # Both 1 nodata & 1 undetect
    expected = (data[2][2] + data[2][3])/2
    (t,v) = obj.getMean((3,2), 2) 
    self.assertEquals(t, _rave.RaveValueType_DATA)
    self.assertAlmostEquals(v, expected)
    
    
    
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

  def test_setGetConvertedValue(self):
    obj = _cartesian.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.gain = 2.0
    obj.offset = 1.0
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(12,10)).astype(numpy.float64)  
    obj.setData(a)
    
    obj.setValue((0,1), 10.0)
    obj.setValue((1,1), 20.0)
    obj.setConvertedValue((2,2), 14.5)
    obj.setConvertedValue((3,3), 15.0)

    r = obj.getConvertedValue((0,1))
    self.assertEquals(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEquals(21.0, r[1], 4)

    r = obj.getConvertedValue((1,1))
    self.assertEquals(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEquals(41.0, r[1], 4)
    
    r = obj.getConvertedValue((2,2))
    self.assertEquals(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEquals(14.5, r[1], 4)

    r = obj.getConvertedValue((3,3))
    self.assertEquals(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEquals(15.0, r[1], 4)

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

  def test_isValid_asImage(self):
    obj = _cartesian.new()
    a = _area.new()
    a.xsize = 10
    a.ysize = 10
    a.xscale = 100.0
    a.yscale = 100.0
    a.extent = (1.0, 2.0, 3.0, 4.0)
    a.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    obj.init(a, _rave.RaveDataType_SHORT)
    obj.date = "20100101"
    obj.time = "100000"
    obj.source = "PLC:1234"
    obj.product = _rave.Rave_ProductType_CAPPI
    obj.quantity = "DBZH"
    
    self.assertEquals(True, obj.isValid(_rave.Rave_ObjectType_IMAGE))
    
  def test_isValid_asImage_no_date(self):
    obj = _cartesian.new()
    a = _area.new()
    a.xsize = 10
    a.ysize = 10
    a.xscale = 100.0
    a.yscale = 100.0
    a.extent = (1.0, 2.0, 3.0, 4.0)
    a.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    obj.init(a, _rave.RaveDataType_SHORT)
    obj.time = "100000"
    obj.source = "PLC:1234"
    obj.product = _rave.Rave_ProductType_CAPPI
    obj.quantity = "DBZH"
    
    self.assertEquals(False, obj.isValid(_rave.Rave_ObjectType_IMAGE))

  def test_isValid_asImage_no_quantity(self):
    obj = _cartesian.new()
    a = _area.new()
    a.xsize = 10
    a.ysize = 10
    a.xscale = 100.0
    a.yscale = 100.0
    a.extent = (1.0, 2.0, 3.0, 4.0)
    a.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    obj.init(a, _rave.RaveDataType_SHORT)
    obj.date = "20100101"
    obj.time = "100000"
    obj.source = "PLC:1234"
    obj.product = _rave.Rave_ProductType_CAPPI
    
    self.assertEquals(False, obj.isValid(_rave.Rave_ObjectType_IMAGE))

  def test_isValid_asCvol(self):
    obj = _cartesian.new()
    obj.addAttribute("what/startdate", "20100101")
    obj.addAttribute("what/starttime", "100000")    
    obj.product = _rave.Rave_ProductType_CAPPI
    obj.quantity = "DBZH"
    obj.xscale = 100.0
    obj.yscale = 100.0
    data = numpy.zeros((240,240),numpy.uint8)
    obj.setData(data)
    
    self.assertEquals(True, obj.isValid(_rave.Rave_ObjectType_CVOL))
    