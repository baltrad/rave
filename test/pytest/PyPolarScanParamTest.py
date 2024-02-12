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
import _ravefield, _ravelegend
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
    
    isscan = str(type(obj)).find("PolarScanParamCore")
    self.assertNotEqual(-1, isscan) 

  def test_attribute_visibility(self):
    attrs = ['nbins', 'nrays', 'quantity', 'gain', 'offset', 'nodata',
             'undetect', 'datatype', 'legend']
    obj = _polarscanparam.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_invalid_attributes(self):
    obj = _polarscanparam.new()
    try:
      obj.lon = 1.0
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_nbins(self):
    obj = _polarscanparam.new()
    self.assertEqual(0, obj.nbins)
    try:
      obj.nbins = 10
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(0, obj.nbins)

  def test_nbins_withData(self):
    obj = _polarscanparam.new()
    data = numpy.zeros((4,5), numpy.int8)
    obj.setData(data)
    self.assertEqual(5, obj.nbins)

  def test_nrays(self):
    obj = _polarscanparam.new()
    self.assertEqual(0, obj.nrays)
    try:
      obj.nrays = 10
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(0, obj.nrays)

  def test_nrays_withData(self):
    obj = _polarscanparam.new()
    data = numpy.zeros((5,4), numpy.int8)
    obj.setData(data)
    self.assertEqual(5, obj.nrays)

  def test_datatype(self):
    obj = _polarscanparam.new()
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    try:
      obj.datatype = _rave.RaveDataType_INT
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)

  def test_quantity(self):
    obj = _polarscanparam.new()
    self.assertEqual(None, obj.quantity)
    obj.quantity = "DBZH"
    self.assertEqual("DBZH", obj.quantity)

  def test_quantity_None(self):
    obj = _polarscanparam.new()
    obj.quantity = "DBZH"
    obj.quantity = None
    self.assertEqual(None, obj.quantity)

  def test_quantity_typeError(self):
    obj = _polarscanparam.new()
    self.assertEqual(None, obj.quantity)
    try:
      obj.quantity = 10
      self.fail("Expected ValueError")
    except ValueError:
      pass
    self.assertEqual(None, obj.quantity)

  def test_gain(self):
    obj = _polarscanparam.new()
    self.assertAlmostEqual(0.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEqual(10.0, obj.gain, 4)

  def test_gain_typeError(self):
    obj = _polarscanparam.new()
    self.assertAlmostEqual(0.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.gain, 4)

  def test_offset(self):
    obj = _polarscanparam.new()
    self.assertAlmostEqual(0.0, obj.offset, 4)
    obj.offset = 10.0
    self.assertAlmostEqual(10.0, obj.offset, 4)

  def test_offset_typeError(self):
    obj = _polarscanparam.new()
    self.assertAlmostEqual(0.0, obj.offset, 4)
    try:
      obj.offset = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.offset, 4)

  def test_nodata(self):
    obj = _polarscanparam.new()
    self.assertAlmostEqual(0.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEqual(10.0, obj.nodata, 4)

  def test_nodata_typeError(self):
    obj = _polarscanparam.new()
    self.assertAlmostEqual(0.0, obj.nodata, 4)
    try:
      obj.nodata = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.nodata, 4)

  def test_undetect(self):
    obj = _polarscanparam.new()
    self.assertAlmostEqual(0.0, obj.undetect, 4)
    obj.undetect = 10.0
    self.assertAlmostEqual(10.0, obj.undetect, 4)

  def test_undetect_typeError(self):
    obj = _polarscanparam.new()
    self.assertAlmostEqual(0.0, obj.undetect, 4)
    try:
      obj.undetect = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.undetect, 4)

  def test_legend(self):
    LEGEND = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]
    obj = _polarscanparam.new()
    legend = _ravelegend.new()
    legend.legend = LEGEND
    obj.legend = legend
    self.assertEqual(LEGEND, obj.legend.legend)

    obj.legend = None
    self.assertEqual(None, obj.legend)

  def test_legend_badvalue(self):
    obj = _polarscanparam.new()
    try:
      obj.legend = [("1","3")]
      self.fail("Expected TypeError")
    except TypeError:
      pass

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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)

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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)

  def test_setValue(self):
    obj = _polarscanparam.new()
    a=numpy.zeros((12,10), numpy.int8)
    obj.setData(a)
    obj.setValue((4,5), 5)
    self.assertAlmostEqual(5.0, obj.getData()[5,4], 4)

  def test_setValue_outOfBounds(self):
    obj = _polarscanparam.new()
    a=numpy.zeros((12,10), numpy.int8)
    obj.setData(a)
    try:
      obj.setValue((15,5), 5)
      self.fail("Expected ValueError")
    except ValueError:
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
    
    self.assertEqual(_rave.RaveDataType_USHORT, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

  def test_setData_uint32(self):
    obj = _polarscanparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint32),numpy.uint32)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint32)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_UINT, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)
    
  def test_setData_uint64(self):
    obj = _polarscanparam.new()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint64),numpy.uint64)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint64)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_ULONG, obj.datatype)
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
    BADNAMES = ["/how/this", "/HOW/this"]
    for n in BADNAMES:
      try:
        obj.addAttribute(n, "XYZ")
        self.fail("Expected AttributeError")
      except AttributeError:
        pass
  
  def test_attributes(self):
    obj = _polarscanparam.new()
    obj.addAttribute("how/this", "ABC")
    obj.addAttribute("how/that", 1.0)
    obj.addAttribute("what/value", 2)
    obj.addAttribute("where/value", "1.0, 2.0, 3.0")
    
    names = obj.getAttributeNames()
    self.assertEqual(4, len(names))
    self.assertTrue("how/this" in names)
    self.assertTrue("how/that" in names)
    self.assertTrue("what/value" in names)
    self.assertTrue("where/value" in names)
    
    self.assertEqual("ABC", obj.getAttribute("how/this"))
    self.assertAlmostEqual(1.0, obj.getAttribute("how/that"), 4)
    self.assertEqual(2, obj.getAttribute("what/value"))
    self.assertEqual("1.0, 2.0, 3.0", obj.getAttribute("where/value"))

  def test_hasAttribute(self):
    obj = _polarscanparam.new()
    obj.addAttribute("how/this", "ABC")
    obj.addAttribute("how/that", 1.0)
    obj.addAttribute("what/value", 2)
    obj.addAttribute("where/value", "1.0, 2.0, 3.0")

    self.assertTrue(obj.hasAttribute("how/this"))
    self.assertTrue(obj.hasAttribute("how/that"))
    self.assertTrue(obj.hasAttribute("what/value"))
    self.assertTrue(obj.hasAttribute("where/value"))
    self.assertFalse(obj.hasAttribute("how/thisandthat"))
    try:
      obj.hasAttribute(None)
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_howSubgroupAttribute(self):
    obj = _polarscanparam.new()

    obj.addAttribute("how/something", 1.0)
    obj.addAttribute("how/grp/something", 2.0)
    try:
      obj.addAttribute("how/grp/else/", 2.0)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

    self.assertAlmostEqual(1.0, obj.getAttribute("how/something"), 2)
    self.assertAlmostEqual(2.0, obj.getAttribute("how/grp/something"), 2)
    self.assertTrue(obj.hasAttribute("how/something"))
    self.assertTrue(obj.hasAttribute("how/grp/something"))

  def test_attributes_nonexisting(self):
    obj = _polarscanparam.new()
    obj.addAttribute("how/this", "ABC")
    obj.addAttribute("how/that", 1.0)
    obj.addAttribute("what/value", 2)
    obj.addAttribute("where/value", "1.0, 2.0, 3.0")
    
    try:
      obj.getAttribute("how/miffo")
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    
  def test_qualityfields(self):
    obj = _polarscanparam.new()
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
  
  def test_toField(self):
    obj = _polarscanparam.new()
    
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint8),numpy.uint8)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint8)    
    
    obj.setData(a)
        
    obj.addAttribute("how/this", "ABC")
    obj.addAttribute("how/that", 1.0)
    obj.addAttribute("what/value", 2)
    
    obj.gain = 2.0
    obj.offset = 3.0
    obj.undetect = 4.0
    obj.nodata = 5.0
    
    result = obj.toField()
    self.assertEqual("ABC", result.getAttribute("how/this"))
    self.assertAlmostEqual(1.0, result.getAttribute("how/that"), 4)
    self.assertEqual(2, result.getAttribute("what/value"))
    self.assertAlmostEqual(2.0, result.getAttribute("what/gain"), 4)
    self.assertAlmostEqual(3.0, result.getAttribute("what/offset"), 4)
    self.assertAlmostEqual(4.0, result.getAttribute("what/undetect"), 4)
    self.assertAlmostEqual(5.0, result.getAttribute("what/nodata"), 4)
    self.assertEqual(_rave.RaveDataType_UCHAR, result.datatype)
    self.assertEqual(10, result.xsize)
    self.assertEqual(12, result.ysize)
  
  def test_fromField(self):
    obj = _ravefield.new()
    obj.setData(numpy.zeros((12,10), numpy.uint8))
    obj.addAttribute("what/gain", 2.0)
    obj.addAttribute("what/offset", 3.0)
    obj.addAttribute("what/nodata", 4.0)
    obj.addAttribute("what/undetect", 5.0)
    obj.addAttribute("what/quantity", "MMH")
    result = _polarscanparam.fromField(obj)
 
    self.assertNotEqual(-1, str(type(result)).find("PolarScanParamCore"))
    self.assertAlmostEqual(2.0, result.gain, 4)
    self.assertAlmostEqual(3.0, result.offset, 4)
    self.assertAlmostEqual(4.0, result.nodata, 4)
    self.assertAlmostEqual(5.0, result.undetect, 4)
    self.assertEqual("MMH", result.quantity)
    self.assertEqual(10, result.nbins)
    self.assertEqual(12, result.nrays)
    self.assertEqual(_rave.RaveDataType_UCHAR, result.datatype)

  def test_shiftData(self):
    obj = _polarscanparam.new()
    f1 = _ravefield.new()
    f2 = _ravefield.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f1.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f2.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.addQualityField(f1)
    obj.addQualityField(f2)
    obj.shiftData(1)
    
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getData()).all())
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getQualityField(0).getData()).all())
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getQualityField(1).getData()).all())

  def test_shiftData_neg(self):
    obj = _polarscanparam.new()
    f1 = _ravefield.new()
    f2 = _ravefield.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f1.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f2.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.addQualityField(f1)
    obj.addQualityField(f2)
    obj.shiftData(-1)
    
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getData()).all())
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getQualityField(0).getData()).all())
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getQualityField(1).getData()).all())

  def test_convertDataDoubleToUchar(self):
    obj = _polarscanparam.new()
    obj.setData(numpy.array([[-32.0,-32.0,-32.0],
                             [ 31.5, 31.5, 31.5],
                             [ 95.5, 95.5, 95.5]]).astype(numpy.float64))      
    self.assertEqual(_rave.RaveDataType_DOUBLE, obj.datatype)

    obj.gain =     0.5
    obj.offset = -32.0
    obj.nodata = 255.0
    obj.undetect = 0.0

    obj.convertDataDoubleToUchar()
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    self.assertEqual([[  0,  0,  0],
                       [127,127,127],
                       [255,255,255]], obj.getData().tolist())

  def test_convertWrongDataType(self):
    obj = _polarscanparam.new()
    obj.setData(numpy.array([[-32.0,-32.0,-32.0],
                             [ 31.5, 31.5, 31.5],
                             [ 95.5, 95.5, 95.5]]).astype(numpy.float32))
    obj.gain =     0.5
    obj.offset = -32.0
    obj.nodata = 255.0
    obj.undetect = 0.0
    try:
        obj.convertDataDoubleToUchar()
    except TypeError:
        pass
  
  def test_clone(self):
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

    obj.addAttribute("how/nisse", 123.0)

    obj.setData(a)

    c = obj.clone()
    self.assertEqual(obj.nodata, c.nodata)
    self.assertEqual(obj.undetect, c.undetect)
    self.assertEqual(obj.gain, c.gain)
    self.assertEqual(obj.offset, c.offset)
    self.assertEqual(obj.nodata, c.nodata)
    self.assertAlmostEqual(123.0, c.getAttribute("how/nisse"), 4)
    
if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()