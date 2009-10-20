'''
Created on Oct 14, 2009
@author: Anders Henja
'''
import unittest
import os
import _rave
import string
import numpy

class RaveModuleCartesianTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewCartesian(self):
    obj = _rave.cartesian()
    
    isscan = string.find(`type(obj)`, "CartesianCore")
    self.assertNotEqual(-1, isscan)
    
  def testCartesian_xsize(self):
    obj = _rave.cartesian()
    self.assertEquals(0, obj.xsize)
    obj.xsize = 10
    self.assertEquals(10, obj.xsize)

  def testCartesian_xsize_typeError(self):
    obj = _rave.cartesian()
    self.assertEquals(0, obj.xsize)
    try:
      obj.xsize = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.xsize)

  def testCartesian_ysize(self):
    obj = _rave.cartesian()
    self.assertEquals(0, obj.ysize)
    obj.ysize = 10
    self.assertEquals(10, obj.ysize)

  def testCartesian_ysize_typeError(self):
    obj = _rave.cartesian()
    self.assertEquals(0, obj.ysize)
    try:
      obj.ysize = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.ysize)

  def testCartesian_xscale(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.xscale, 4)
    obj.xscale = 10.0
    self.assertAlmostEquals(10.0, obj.xscale, 4)

  def testCartesian_xscale_typeError(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.xscale, 4)
    try:
      obj.xscale = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.xscale, 4)

  def testCartesian_yscale(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.yscale, 4)
    obj.yscale = 10.0
    self.assertAlmostEquals(10.0, obj.yscale, 4)

  def testCartesian_yscale_typeError(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.yscale, 4)
    try:
      obj.yscale = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.yscale, 4)

  def testCartesian_datatype(self):
    obj = _rave.cartesian()
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    obj.datatype = _rave.RaveDataType_INT
    self.assertEqual(_rave.RaveDataType_INT, obj.datatype)

  def testCartesian_setValidDataTypes(self):
    types = [_rave.RaveDataType_UNDEFINED, _rave.RaveDataType_CHAR, _rave.RaveDataType_UCHAR,
             _rave.RaveDataType_SHORT, _rave.RaveDataType_INT, _rave.RaveDataType_LONG,
             _rave.RaveDataType_FLOAT, _rave.RaveDataType_DOUBLE]

    obj = _rave.cartesian()
    for type in types:
      obj.datatype = type
      self.assertEqual(type, obj.datatype)

  def testCartesian_invalidDatatype(self):
    obj = _rave.cartesian()
    types = [99,100,-2,30]
    for type in types:
      try:
        obj.datatype = type
        self.fail("Expected ValueError")
      except ValueError, e:
        self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)

    
  def testCartesian_quantity(self):
    obj = _rave.cartesian()
    self.assertEquals("", obj.quantity)
    obj.quantity = "DBZH"
    self.assertEquals("DBZH", obj.quantity)

  def testCartesian_quantity_typeError(self):
    obj = _rave.cartesian()
    self.assertEquals("", obj.quantity)
    try:
      obj.quantity = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals("", obj.quantity)
    
  def testCartesian_gain(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEquals(10.0, obj.gain, 4)

  def testCartesian_gain_typeError(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.gain, 4)

  def testCartesian_offset(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    obj.offset = 10.0
    self.assertAlmostEquals(10.0, obj.offset, 4)

  def testCartesian_offset_typeError(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    try:
      obj.offset = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.offset, 4)

  def testCartesian_nodata(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEquals(10.0, obj.nodata, 4)

  def testCartesian_nodata_typeError(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    try:
      obj.nodata = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.nodata, 4)

  def testCartesian_undetect(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    obj.undetect = 10.0
    self.assertAlmostEquals(10.0, obj.undetect, 4)

  def testCartesian_undetect_typeError(self):
    obj = _rave.cartesian()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    try:
      obj.undetect = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.undetect, 4)

  def testCartesian_areaextent(self):
    obj = _rave.cartesian()
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

  def testCartesian_areaextent_badTupleSize(self):
    obj = _rave.cartesian()
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

  def testCartesian_areaextent_illegalData(self):
    obj = _rave.cartesian()
    try:
      obj.areaextent = (10.0, "a", 30.0, 40.0)
      self.fail("Expected type error")
    except TypeError, e:
      pass

  def testCartesian_getLocationX(self):
    obj = _rave.cartesian();
    obj.areaextent = (100.0, 200.0, 300.0, 400.0)
    obj.xscale = 10.0
    
    xpos = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for x in xpos:
      expected = 100.0 + x*10.0
      result = obj.getLocationX(x)
      self.assertAlmostEqual(expected, result, 4)

  def testCartesian_getLocationY(self):
    obj = _rave.cartesian();
    obj.areaextent = (100.0, 200.0, 300.0, 400.0)
    obj.yscale = 10.0
    
    ypos = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for y in ypos:
      expected = 400.0 - y*10.0
      result = obj.getLocationY(y)
      self.assertAlmostEqual(expected, result, 4)


  def testCartesian_setData_int8(self):
    obj = _rave.cartesian()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_CHAR, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)


  def testCartesian_setData_uint8(self):
    obj = _rave.cartesian()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint8),numpy.uint8)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def testCartesian_setData_int16(self):
    obj = _rave.cartesian()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int16),numpy.int16)
    a=numpy.reshape(a,(12,10)).astype(numpy.int16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)

  def testCartesian_setData_uint16(self):
    obj = _rave.cartesian()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint16),numpy.uint16)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(12, obj.ysize)
