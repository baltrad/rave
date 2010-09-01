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

Tests the polarscanparam module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2010-01-22
'''
import unittest
import os
import _polarscanparam
import _rave
import _ravefield
import string
import numpy
import math

class PyPolarScanParamTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _polarscanparam.new()
    
    isscan = string.find(`type(obj)`, "PolarScanParamCore")
    self.assertNotEqual(-1, isscan) 

  def test_attribute_visibility(self):
    attrs = ['nbins', 'nrays', 'quantity', 'gain', 'offset', 'nodata',
             'undetect', 'datatype']
    obj = _polarscanparam.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_invalid_attributes(self):
    obj = _polarscanparam.new()
    try:
      obj.lon = 1.0
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass

  def test_nbins(self):
    obj = _polarscanparam.new()
    self.assertEquals(0, obj.nbins)
    try:
      obj.nbins = 10
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEquals(0, obj.nbins)

  def test_nbins_withData(self):
    obj = _polarscanparam.new()
    data = numpy.zeros((4,5), numpy.int8)
    obj.setData(data)
    self.assertEquals(5, obj.nbins)

  def test_nrays(self):
    obj = _polarscanparam.new()
    self.assertEquals(0, obj.nrays)
    try:
      obj.nrays = 10
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEquals(0, obj.nrays)

  def test_nrays_withData(self):
    obj = _polarscanparam.new()
    data = numpy.zeros((5,4), numpy.int8)
    obj.setData(data)
    self.assertEquals(5, obj.nrays)

  def test_datatype(self):
    obj = _polarscanparam.new()
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    try:
      obj.datatype = _rave.RaveDataType_INT
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)

  def test_quantity(self):
    obj = _polarscanparam.new()
    self.assertEquals(None, obj.quantity)
    obj.quantity = "DBZH"
    self.assertEquals("DBZH", obj.quantity)

  def test_quantity_None(self):
    obj = _polarscanparam.new()
    obj.quantity = "DBZH"
    obj.quantity = None
    self.assertEquals(None, obj.quantity)

  def test_quantity_typeError(self):
    obj = _polarscanparam.new()
    self.assertEquals(None, obj.quantity)
    try:
      obj.quantity = 10
      self.fail("Expected ValueError")
    except ValueError,e:
      pass
    self.assertEquals(None, obj.quantity)

  def test_gain(self):
    obj = _polarscanparam.new()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEquals(10.0, obj.gain, 4)

  def test_gain_typeError(self):
    obj = _polarscanparam.new()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.gain, 4)

  def test_offset(self):
    obj = _polarscanparam.new()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    obj.offset = 10.0
    self.assertAlmostEquals(10.0, obj.offset, 4)

  def test_offset_typeError(self):
    obj = _polarscanparam.new()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    try:
      obj.offset = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.offset, 4)

  def test_nodata(self):
    obj = _polarscanparam.new()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEquals(10.0, obj.nodata, 4)

  def test_nodata_typeError(self):
    obj = _polarscanparam.new()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    try:
      obj.nodata = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.nodata, 4)

  def test_undetect(self):
    obj = _polarscanparam.new()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    obj.undetect = 10.0
    self.assertAlmostEquals(10.0, obj.undetect, 4)

  def test_undetect_typeError(self):
    obj = _polarscanparam.new()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    try:
      obj.undetect = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.undetect, 4)

  def test_getValue(self):
    obj = _polarscanparam.new()
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
    obj = _polarscanparam.new()
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

  def test_setValue(self):
    obj = _polarscanparam.new()
    a=numpy.zeros((12,10), numpy.int8)
    obj.setData(a)
    obj.setValue((4,5), 5)
    self.assertAlmostEquals(5.0, obj.getData()[5,4], 4)

  def test_setValue_outOfBounds(self):
    obj = _polarscanparam.new()
    a=numpy.zeros((12,10), numpy.int8)
    obj.setData(a)
    try:
      obj.setValue((15,5), 5)
      self.fail("Expected ValueError")
    except ValueError,e:
      pass

  def test_setData_int8(self):
    obj = _polarscanparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_CHAR, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)


  def test_setData_uint8(self):
    obj = _polarscanparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint8),numpy.uint8)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

  def test_setData_int16(self):
    obj = _polarscanparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int16),numpy.int16)
    a=numpy.reshape(a,(12,10)).astype(numpy.int16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

  def test_setData_uint16(self):
    obj = _polarscanparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint16),numpy.uint16)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

  def test_addAttribute_goodNames(self):
    obj = _polarscanparam.new()
    GOODNAMES = ["how/this", "HOW/this", "HoW/this", "What/that",
                 "wHAT/that", "Where/doobi", "WHERE/DOOBI"]
    for n in GOODNAMES:
      obj.addAttribute(n, "XYZ")
    
  def test_addAttribute_badNames(self):
    obj = _polarscanparam.new()
    BADNAMES = ["/how/this", "/HOW/this", "/how/this/that", "how/this/that"]
    for n in BADNAMES:
      try:
        obj.addAttribute(n, "XYZ")
        self.fail("Expected AttributeError")
      except AttributeError,e:
        pass
  
  def test_attributes(self):
    obj = _polarscanparam.new()
    obj.addAttribute("how/this", "ABC")
    obj.addAttribute("how/that", 1.0)
    obj.addAttribute("what/value", 2)
    obj.addAttribute("where/value", "1.0, 2.0, 3.0")
    
    names = obj.getAttributeNames()
    self.assertEquals(4, len(names))
    self.assertTrue("how/this" in names)
    self.assertTrue("how/that" in names)
    self.assertTrue("what/value" in names)
    self.assertTrue("where/value" in names)
    
    self.assertEquals("ABC", obj.getAttribute("how/this"))
    self.assertAlmostEquals(1.0, obj.getAttribute("how/that"), 4)
    self.assertEquals(2, obj.getAttribute("what/value"))
    self.assertEquals("1.0, 2.0, 3.0", obj.getAttribute("where/value"))

  def test_attributes_nonexisting(self):
    obj = _polarscanparam.new()
    obj.addAttribute("how/this", "ABC")
    obj.addAttribute("how/that", 1.0)
    obj.addAttribute("what/value", 2)
    obj.addAttribute("where/value", "1.0, 2.0, 3.0")
    
    try:
      obj.getAttribute("how/miffo")
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    
  def test_qualityfields(self):
    obj = _polarscanparam.new()
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("what/name", "field1")
    field2.addAttribute("what/name", "field2")

    obj.addQualityField(field1)
    obj.addQualityField(field2)
    
    self.assertEquals(2, obj.getNumberOfQualityFields())
    self.assertEquals("field1", obj.getQualityField(0).getAttribute("what/name"))
    obj.removeQualityField(0)
    self.assertEquals(1, obj.getNumberOfQualityFields())
    self.assertEquals("field2", obj.getQualityField(0).getAttribute("what/name"))    
    
if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()