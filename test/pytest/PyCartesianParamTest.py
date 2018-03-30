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

Tests the cartesian parameter module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2012-02-07
'''
import unittest
import os
import _cartesianparam
import _rave
import _ravefield
import string
import numpy

class PyCartesianParamTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _cartesianparam.new()
    
    isparam = str(type(obj)).find("CartesianParamCore")
    self.assertNotEqual(-1, isparam)
  
  def test_attribute_visibility(self):
    attrs = ['xsize', 'ysize', 'quantity', 'gain', 'offset', 'nodata', 
     'undetect', 'datatype']
    obj = _cartesianparam.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)
  
  def test_xsize(self):
    obj = _cartesianparam.new()
    self.assertEqual(0, obj.xsize)
    try:
      obj.xsize = 10
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(0, obj.xsize)

  def test_xsize_fromArray(self):
    obj = _cartesianparam.new()
    data = numpy.zeros((11,10), numpy.uint8)
    obj.setData(data)
    self.assertEqual(10, obj.xsize)

  def test_ysize(self):
    obj = _cartesianparam.new()
    self.assertEqual(0, obj.ysize)
    try:
      obj.ysize = 10
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(0, obj.ysize)

  def test_ysize_fromArray(self):
    obj = _cartesianparam.new()
    data = numpy.zeros((11,10), numpy.uint8)
    obj.setData(data)
    self.assertEqual(11, obj.ysize)

  def test_datatype(self):
    obj = _cartesianparam.new()
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    try:
      obj.datatype = _rave.RaveDataType_INT
      self.fail("Expected AttributeError")
    except AttributeError:
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

    obj = _cartesianparam.new()
    for type in types:
      d = numpy.zeros((10,10), type[0])
      obj.setData(d)
      self.assertEqual(type[1], obj.datatype)
    
  def test_quantity(self):
    obj = _cartesianparam.new()
    self.assertEqual(None, obj.quantity)
    obj.quantity = "DBHH"
    self.assertEqual("DBHH", obj.quantity)

  def test_quantity_typeError(self):
    obj = _cartesianparam.new()
    try:
      obj.quantity = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertEqual(None, obj.quantity)

  def test_gain(self):
    obj = _cartesianparam.new()
    self.assertAlmostEqual(1.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEqual(10.0, obj.gain, 4)

  def test_setGain_zero(self):
    obj = _cartesianparam.new()
    obj.gain = 0.0
    self.assertAlmostEqual(1.0, obj.gain, 4)


  def test_gain_typeError(self):
    obj = _cartesianparam.new()
    self.assertAlmostEqual(1.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(1.0, obj.gain, 4)

  def test_offset(self):
    obj = _cartesianparam.new()
    self.assertAlmostEqual(0.0, obj.offset, 4)
    obj.offset = 10.0
    self.assertAlmostEqual(10.0, obj.offset, 4)

  def test_offset_typeError(self):
    obj = _cartesianparam.new()
    self.assertAlmostEqual(0.0, obj.offset, 4)
    try:
      obj.offset = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.offset, 4)

  def test_nodata(self):
    obj = _cartesianparam.new()
    self.assertAlmostEqual(0.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEqual(10.0, obj.nodata, 4)

  def test_nodata_typeError(self):
    obj = _cartesianparam.new()
    self.assertAlmostEqual(0.0, obj.nodata, 4)
    try:
      obj.nodata = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.nodata, 4)

  def test_undetect(self):
    obj = _cartesianparam.new()
    self.assertAlmostEqual(0.0, obj.undetect, 4)
    obj.undetect = 10.0
    self.assertAlmostEqual(10.0, obj.undetect, 4)

  def test_undetect_typeError(self):
    obj = _cartesianparam.new()
    self.assertAlmostEqual(0.0, obj.undetect, 4)
    try:
      obj.undetect = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.undetect, 4)

  def test_getValue(self):
    obj = _cartesianparam.new()
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
      self.assertAlmostEqual(cval[2], result[1], 4)
      self.assertEqual(cval[3], result[0])

  def test_getMean(self):
    obj = _cartesianparam.new()
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
    self.assertEqual(t, _rave.RaveValueType_NODATA)

    # Undetect
    (t,v) = obj.getMean((3,1), 2)
    self.assertEqual(t, _rave.RaveValueType_UNDETECT)
    
    # Left side with one nodata
    expected = data[1][0]
    (t,v) = obj.getMean((0,1), 2) 
    self.assertEqual(t, _rave.RaveValueType_DATA)
    self.assertAlmostEqual(v, expected)

    # Both 1 nodata & 1 undetect
    expected = (data[2][2] + data[2][3])/2
    (t,v) = obj.getMean((3,2), 2) 
    self.assertEqual(t, _rave.RaveValueType_DATA)
    self.assertAlmostEqual(v, expected)
    
    
  def test_setGetValue(self):
    obj = _cartesianparam.new()
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
      self.assertAlmostEqual(v[1], r[1], 4)
      self.assertEqual(v[2], r[0])

  def test_setGetConvertedValue(self):
    obj = _cartesianparam.new()
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
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(21.0, r[1], 4)

    r = obj.getConvertedValue((1,1))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(41.0, r[1], 4)
    
    r = obj.getConvertedValue((2,2))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(14.5, r[1], 4)

    r = obj.getConvertedValue((3,3))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(15.0, r[1], 4)
    
  def test_setConvertedValue_rawZero(self):
    obj = _cartesianparam.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.gain = 0.4
    obj.offset = -30.0
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(12,10)).astype(numpy.float64)  
    obj.setData(a)
    
    obj.setConvertedValue((0,1), 0.0)

    r = obj.getConvertedValue((0,1))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(0.0, r[1], 4)
    
    r = obj.getValue((0,1))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(75.0, r[1], 4)
    
  def test_setConvertedValue_roundOff(self):
    obj = _cartesianparam.new()
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.gain = 0.4
    obj.offset = -30.0
     
    # test uint8
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint8),numpy.uint8)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint8)  
    obj.setData(a)
     
    obj.setConvertedValue((0,1), -0.5)
     
    r = obj.getValue((0,1))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(74.0, r[1], 4)
     
    #test int8
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)  
    obj.setData(a)
     
    obj.setConvertedValue((0,1), 1.5)
     
    r = obj.getValue((0,1))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(79.0, r[1], 4)
     
    #test uint
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint),numpy.uint)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint)  
    obj.setData(a)
     
    obj.setConvertedValue((0,1), 2.25)
     
    r = obj.getValue((0,1))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(81.0, r[1], 4)
     
    #test int
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int),numpy.int)
    a=numpy.reshape(a,(12,10)).astype(numpy.int)  
    obj.setData(a)
     
    obj.setConvertedValue((0,1), -1.75)
     
    r = obj.getValue((0,1))
    self.assertEqual(_rave.RaveValueType_DATA, r[0])
    self.assertAlmostEqual(71.0, r[1], 4)

  def test_setData_int8(self):
    obj = _cartesianparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_CHAR, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_setData_uint8(self):
    obj = _cartesianparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint8),numpy.uint8)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_setData_int16(self):
    obj = _cartesianparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int16),numpy.int16)
    a=numpy.reshape(a,(12,10)).astype(numpy.int16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_setData_uint16(self):
    obj = _cartesianparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint16),numpy.uint16)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_USHORT, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_setData_uint32(self):
    obj = _cartesianparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint32),numpy.uint32)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint32)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_UINT, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_setData_uint64(self):
    obj = _cartesianparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint64),numpy.uint64)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint64)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_ULONG, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def test_getData_int8(self):
    obj = _cartesianparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)    
    
    obj.setData(a)
    obj.setValue((3,2), 5)
    obj.setValue((4,4), 7)
    
    result = obj.getData()
    self.assertEqual(5, result[2][3])
    self.assertEqual(7, result[4][4])
    self.assertEqual("int8", result.dtype.name)

  def test_isTransformable(self):
    data = numpy.zeros((10,10), numpy.float64)
    obj = _cartesianparam.new()
    obj.setData(data)
    self.assertEqual(True, obj.isTransformable())
    
  def test_qualityfields(self):
    obj = _cartesianparam.new()
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("what/name", "field1")
    field2.addAttribute("what/name", "field2")

    obj.addQualityField(field1)
    obj.addQualityField(field2)
    
    self.assertEqual(2, obj.getNumberOfQualityFields())
    self.assertEqual("field1", obj.getQualityField(0).getAttribute("what/name"))
    obj.removeQualityField(0)
    self.assertEqual(1, obj.getNumberOfQualityFields())
    self.assertEqual("field2", obj.getQualityField(0).getAttribute("what/name"))

  def test_getQualityFieldByHowTask(self):
    param = _cartesianparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)
    
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("how/task", "se.smhi.f1")
    field1.addAttribute("what/value", "f1")
    field2.addAttribute("how/task", "se.smhi.f2")
    field2.addAttribute("what/value", "f2")

    param.addQualityField(field1)
    param.addQualityField(field2)
   
    result = param.getQualityFieldByHowTask("se.smhi.f2")
    self.assertEqual("f2", result.getAttribute("what/value"))

  def test_getQualityFieldByHowTask_notFound(self):
    param = _cartesianparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)

    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("how/task", "se.smhi.f1")
    field1.addAttribute("what/value", "f1")
    field2.addAttribute("how/task", "se.smhi.f2")
    field2.addAttribute("what/value", "f2")

    param.addQualityField(field1)
    param.addQualityField(field2)
    
    try:
      obj.getQualityFieldByHowTask("se.smhi.f3")
      self.fail("Expected NameError")
    except NameError:
      pass
  