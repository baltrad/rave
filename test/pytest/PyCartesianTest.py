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
import os, math
import _cartesian
import _cartesianparam
import _projection
import _rave
import _raveio
import _area
import _ravefield
import string
import numpy

def deg2rad(coord):
  return (coord[0]*math.pi/180.0, coord[1]*math.pi/180.0)

class PyCartesianTest(unittest.TestCase):
  CARTESIAN_FIXTURE = "fixtures/cartesian_image.h5"
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _cartesian.new()
    
    iscartesian = str(type(obj)).find("CartesianCore")
    self.assertNotEqual(-1, iscartesian)
  
  def test_isCartesian(self):
    obj = _cartesian.new()
    parm = _cartesianparam.new()
    self.assertTrue(_cartesian.isCartesian(obj))
    self.assertFalse(_cartesian.isCartesian(parm))
  
  def test_attribute_visibility(self):
    attrs = ['areaextent', 'date', 'objectType', 
     'product', 'projection', 'source', 'time',
     'xscale', 'xsize', 'yscale', 'ysize', 'prodname']
    obj = _cartesian.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)
  
  def test_init(self):
    obj = _cartesian.new()
    a = _area.new()
    a.xsize = 10
    a.ysize = 10
    a.xscale = 100.0
    a.yscale = 100.0
    a.extent = (1.0, 2.0, 3.0, 4.0)
    a.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")

    obj.init(a)
    self.assertEqual(10, obj.xsize)
    self.assertEqual(10, obj.ysize)
    self.assertAlmostEqual(100.0, obj.xscale, 4)
    self.assertAlmostEqual(100.0, obj.yscale, 4)
    self.assertAlmostEqual(1.0, obj.areaextent[0], 4)
    self.assertAlmostEqual(2.0, obj.areaextent[1], 4)
    self.assertAlmostEqual(3.0, obj.areaextent[2], 4)
    self.assertAlmostEqual(4.0, obj.areaextent[3], 4)
    self.assertEqual("x", obj.projection.id)
  
  def test_time(self):
    obj = _cartesian.new()
    self.assertEqual(None, obj.time)
    obj.time = "200500"
    self.assertEqual("200500", obj.time)
    obj.time = None
    self.assertEqual(None, obj.time)

  def test_time_badValues(self):
    obj = _cartesian.new()
    values = ["10101", "1010101", "1010ab", "1010x0", "abcdef", 123456]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError:
        pass

  def test_date(self):
    obj = _cartesian.new()
    self.assertEqual(None, obj.date)
    obj.date = "20050101"
    self.assertEqual("20050101", obj.date)
    obj.date = None
    self.assertEqual(None, obj.date)

  def test_date_badValues(self):
    obj = _cartesian.new()
    values = ["200910101", "2001010", "200a1010", 20091010]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError:
        pass

  def test_startdate(self):
    obj = _cartesian.new()
    self.assertEqual(None, obj.startdate)
    obj.date = "20050101"
    self.assertEqual("20050101", obj.startdate)
    obj.startdate = "20060101"
    self.assertEqual("20050101", obj.date)
    self.assertEqual("20060101", obj.startdate)

  def test_starttime(self):
    obj = _cartesian.new()
    self.assertEqual(None, obj.starttime)
    obj.time = "100000"
    self.assertEqual("100000", obj.starttime)
    obj.starttime = "110000"
    self.assertEqual("100000", obj.time)
    self.assertEqual("110000", obj.starttime)

  def test_source(self):
    obj = _cartesian.new()
    self.assertEqual(None, obj.source)
    obj.source = "ABC:10, ABD:1"
    self.assertEqual("ABC:10, ABD:1", obj.source)
    obj.source = None
    self.assertEqual(None, obj.source)

  def test_prodname(self):
    obj = _cartesian.new()
    self.assertEqual(None, obj.prodname)
    obj.prodname = "a product"
    self.assertEqual("a product", obj.prodname)
    obj.prodname = None
    self.assertEqual(None, obj.prodname)

  def test_prodname_typeError(self):
    obj = _cartesian.new()
    try:
      obj.prodname = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertEqual(None, obj.prodname)

  def test_objectType(self):
    obj = _cartesian.new()
    obj.objectType = _rave.Rave_ObjectType_COMP
    self.assertEqual(_rave.Rave_ObjectType_COMP, obj.objectType)
    obj.objectType = _rave.Rave_ObjectType_IMAGE
    self.assertEqual(_rave.Rave_ObjectType_IMAGE, obj.objectType)
  
  def test_objectType_invalid(self):
    obj = _cartesian.new()
    try:
      obj.objectType = _rave.Rave_ObjectType_CVOL
      fail("Expected ValueError")
    except ValueError:
      pass
  
  def test_xsize(self):
    obj = _cartesian.new()
    self.assertEqual(0, obj.xsize)
    try:
      obj.xsize = 10
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(0, obj.xsize)

  def test_ysize(self):
    obj = _cartesian.new()
    self.assertEqual(0, obj.ysize)
    try:
      obj.ysize = 10
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(0, obj.ysize)

  def test_xscale(self):
    obj = _cartesian.new()
    self.assertAlmostEqual(0.0, obj.xscale, 4)
    obj.xscale = 10.0
    self.assertAlmostEqual(10.0, obj.xscale, 4)

  def test_xscale_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEqual(0.0, obj.xscale, 4)
    try:
      obj.xscale = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.xscale, 4)

  def test_yscale(self):
    obj = _cartesian.new()
    self.assertAlmostEqual(0.0, obj.yscale, 4)
    obj.yscale = 10.0
    self.assertAlmostEqual(10.0, obj.yscale, 4)

  def test_yscale_typeError(self):
    obj = _cartesian.new()
    self.assertAlmostEqual(0.0, obj.yscale, 4)
    try:
      obj.yscale = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.yscale, 4)

  def test_areaextent(self):
    obj = _cartesian.new()
    tt = obj.areaextent
    self.assertEqual(4, len(tt))
    self.assertAlmostEqual(0.0, tt[0], 4)
    self.assertAlmostEqual(0.0, tt[1], 4)
    self.assertAlmostEqual(0.0, tt[2], 4)
    self.assertAlmostEqual(0.0, tt[3], 4)

    obj.areaextent = (10.0, 11.0, 12.0, 13.0)
    tt = obj.areaextent
    self.assertEqual(4, len(tt))
    self.assertAlmostEqual(10.0, tt[0], 4)
    self.assertAlmostEqual(11.0, tt[1], 4)
    self.assertAlmostEqual(12.0, tt[2], 4)
    self.assertAlmostEqual(13.0, tt[3], 4)

  def test_areaextent_badTupleSize(self):
    obj = _cartesian.new()
    try:
      obj.areaextent = (10.0, 20.0, 30.0)
      self.fail("Expected type error")
    except TypeError:
      pass

    try:
      obj.areaextent = (10.0, 20.0, 30.0, 40.0, 50.0)
      self.fail("Expected type error")
    except TypeError:
      pass

  def test_areaextent_illegalData(self):
    obj = _cartesian.new()
    try:
      obj.areaextent = (10.0, "a", 30.0, 40.0)
      self.fail("Expected type error")
    except TypeError:
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
      self.assertEqual(x, result)

  def test_getIndexY(self):
    obj = _cartesian.new();
    obj.areaextent = (100.0, 200.0, 300.0, 400.0)
    obj.yscale = 10.0
    
    ypos = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for y in ypos:
      value = 400.0 - y*10.0
      result = obj.getIndexY(value)
      self.assertEqual(y, result)

  def test_getExtremeLonLatBoundaries(self):
    _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84")
    try:
        obj = _raveio.open(self.CARTESIAN_FIXTURE).object
        ul,lr = obj.getExtremeLonLatBoundaries()
        self.assertAlmostEqual(8.73067, ul[0]*180.0/math.pi, 4)
        self.assertAlmostEqual(58.4598, ul[1]*180.0/math.pi, 4)
        self.assertAlmostEqual(16.9439, lr[0]*180.0/math.pi, 4)
        self.assertAlmostEqual(54.1719, lr[1]*180.0/math.pi, 4)
    finally:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84 +datum=WGS84")

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

    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.nodata = 255.0
    param.undetect = 0.0

    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(12,10)).astype(numpy.float64)    
    a[0][1] = param.nodata
    a[1][0] = param.undetect
    
    param.setData(a)
    
    obj.addParameter(param)
    obj.defaultParameter = "DBZH"
    pairs = [(0, 0, 0.0, _rave.RaveValueType_UNDETECT),
             (1, 0, param.nodata, _rave.RaveValueType_NODATA), 
             (0, 1, param.undetect, _rave.RaveValueType_UNDETECT),
             (2, 0, 2.0, _rave.RaveValueType_DATA),
             (0, 3, 30.0, _rave.RaveValueType_DATA)]

    for cval in pairs:
      result = obj.getValue((cval[0],cval[1]))
      self.assertAlmostEqual(cval[2], result[1], 4)
      self.assertEqual(cval[3], result[0])

  def test_getConvertedValueAtLonLat(self):
    obj = _cartesian.new()
    obj.projection = _rave.projection("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544")
    obj.xscale = 100.0
    obj.yscale = 100.0

    xy = obj.projection.fwd(deg2rad((12.8544, 56.3675)))
    obj.areaextent = (xy[0] - 4*100.0, xy[1] - 5*100.0, xy[0] + 6*100.0, xy[1] + 5*100.0)
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.nodata = 255.0
    param.undetect = 0.0

    a=numpy.arange(99)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(11,9)).astype(numpy.float64)    

    param.setData(a)

    obj.addParameter(param)
    obj.defaultParameter = "DBZH"
    
    expected = obj.getConvertedValue((4,5))
    actual = obj.getConvertedValueAtLonLat(deg2rad((12.8544, 56.3675)))
    self.assertEqual(expected[0], actual[0])
    self.assertAlmostEqual(expected[1], actual[1], 4)

  def test_getQualityValueAtLonLat(self):
    obj = _cartesian.new()
    obj.projection = _rave.projection("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544")
    obj.xscale = 100.0
    obj.yscale = 100.0

    xy = obj.projection.fwd(deg2rad((12.8544, 56.3675)))
    obj.areaextent = (xy[0] - 4*100.0, xy[1] - 5*100.0, xy[0] + 6*100.0, xy[1] + 5*100.0)
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.nodata = 255.0
    param.undetect = 0.0

    a=numpy.zeros((11,9))
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(11,9)).astype(numpy.float64)    

    param.setData(a)

    qf = numpy.arange(99)
    qf = numpy.array(qf.astype(numpy.float64),numpy.float64)
    qf = numpy.reshape(qf,(11,9)).astype(numpy.float64)

    qf2 = numpy.arange(99)
    qf2 = numpy.array(qf.astype(numpy.float64),numpy.float64)
    qf2 = numpy.reshape(qf,(11,9)).astype(numpy.float64)
    qf2[5][4]=199.0

    field1 = _ravefield.new()
    field1.addAttribute("how/task", "se.task.1")
    field1.setData(qf)
    param.addQualityField(field1)

    field2 = _ravefield.new()
    field2.addAttribute("how/task", "se.task.2")
    field2.setData(qf2)
    param.addQualityField(field2)

    obj.addParameter(param)
    obj.defaultParameter = "DBZH"
    
    #expected = obj.getConvertedValue((4,5))
    result = obj.getQualityValueAtLonLat(deg2rad((12.8544, 56.3675)), "se.task.1")
    self.assertAlmostEqual(49.0, result, 4)
    result = obj.getQualityValueAtLonLat(deg2rad((12.8544, 56.3675)), "se.task.2")
    self.assertAlmostEqual(199.0, result, 4)

  def test_getConvertedQualityValueAtLonLat(self):
    obj = _cartesian.new()
    obj.projection = _rave.projection("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544")
    obj.xscale = 100.0
    obj.yscale = 100.0

    xy = obj.projection.fwd(deg2rad((12.8544, 56.3675)))
    obj.areaextent = (xy[0] - 4*100.0, xy[1] - 5*100.0, xy[0] + 6*100.0, xy[1] + 5*100.0)
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.nodata = 255.0
    param.undetect = 0.0

    a=numpy.zeros((11,9))
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(11,9)).astype(numpy.float64)    

    param.setData(a)

    qf = numpy.arange(99)
    qf = numpy.array(qf.astype(numpy.float64),numpy.float64)
    qf = numpy.reshape(qf,(11,9)).astype(numpy.float64)

    qf2 = numpy.arange(99)
    qf2 = numpy.array(qf.astype(numpy.float64),numpy.float64)
    qf2 = numpy.reshape(qf,(11,9)).astype(numpy.float64)
    qf2[5][4]=199.0

    field1 = _ravefield.new()
    field1.addAttribute("how/task", "se.task.1")
    field1.addAttribute("what/offset", 10.0)
    field1.addAttribute("what/gain", 2.0)
    field1.setData(qf)
    param.addQualityField(field1)

    field2 = _ravefield.new()
    field2.addAttribute("how/task", "se.task.2")
    field2.addAttribute("what/gain", 3.0)
    field2.setData(qf2)
    param.addQualityField(field2)

    obj.addParameter(param)
    obj.defaultParameter = "DBZH"
    
    #expected = obj.getConvertedValue((4,5))
    result = obj.getConvertedQualityValueAtLonLat(deg2rad((12.8544, 56.3675)), "se.task.1")
    self.assertAlmostEqual(10.0 + 2.0 * 49.0, result, 4)
    result = obj.getConvertedQualityValueAtLonLat(deg2rad((12.8544, 56.3675)), "se.task.2")
    self.assertAlmostEqual(3.0 * 199.0, result, 4)

  def test_getMean(self):
    obj = _cartesian.new()
    
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.nodata = 255.0
    param.undetect = 0.0

    data = numpy.zeros((5,5), numpy.float64)

    for y in range(5):
      for x in range(5):
        data[y][x] = float(x+y*5)
        
    # add some nodata and undetect
    data[0][0] = param.nodata    # 0
    data[0][3] = param.nodata    # 3
    data[1][2] = param.nodata    # 7
    data[1][3] = param.undetect  # 8
    data[3][2] = param.undetect  # 17
    data[4][4] = param.nodata    # 24
    
    param.setData(data)
    
    obj.addParameter(param)
    obj.defaultParameter = "DBZH"
    
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
    obj = _cartesian.new()
    
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.nodata = 255.0
    param.undetect = 0.0
    
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(12,10)).astype(numpy.float64)  
    param.setData(a)
    
    obj.addParameter(param)
    obj.defaultParameter = "DBZH"
    
    data = [((0,1), 10.0, _rave.RaveValueType_DATA),
            ((1,1), 20.0, _rave.RaveValueType_DATA),
            ((2,2), 30.0, _rave.RaveValueType_DATA),
            ((9,4), 49.0, _rave.RaveValueType_DATA),
            ((8,4), param.nodata, _rave.RaveValueType_NODATA),
            ((4,8), param.undetect, _rave.RaveValueType_UNDETECT),]
    
    for v in data:
      obj.setValue(v[0],v[1])
    
    # Verify
    for v in data:
      r = obj.getValue(v[0])
      self.assertAlmostEqual(v[1], r[1], 4)
      self.assertEqual(v[2], r[0])

  def test_setGetConvertedValue(self):
    obj = _cartesian.new()

    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.nodata = 255.0
    param.undetect = 0.0
    param.gain = 2.0
    param.offset = 1.0
    
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(12,10)).astype(numpy.float64)  
    param.setData(a)
    
    obj.addParameter(param)
    obj.defaultParameter = "DBZH"
    
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

  def test_isTransformable(self):
    proj = _rave.projection("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    data = numpy.zeros((10,10), numpy.float64)

    param = _cartesianparam.new()
    param.setData(data)
    param.quantity = "DBZH"
    
    obj = _cartesian.new()
    obj.xscale = 1000.0
    obj.yscale = 1000.0
    
    obj.projection = proj
    obj.addParameter(param);

    self.assertEqual(True, obj.isTransformable())
    
  def test_isTransformable_noscale(self):
    proj = _rave.projection("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    data = numpy.zeros((10,10), numpy.float64)

    param = _cartesianparam.new()
    param.quantity = "DBZH"    
    param.setData(data)
    
    obj = _cartesian.new()
    obj.xscale = 1000.0
    obj.yscale = 1000.0
    
    obj.projection = proj
    obj.addParameter(param)

    self.assertEqual(True, obj.isTransformable())
    obj.xscale = 1000.0
    obj.yscale = 0.0
    self.assertEqual(False, obj.isTransformable())
    obj.xscale = 0.0
    obj.yscale = 1000.0
    self.assertEqual(False, obj.isTransformable())

  def test_isTransformable_nodata(self):
    proj = _rave.projection("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")

    obj = _cartesian.new()
    obj.xscale = 1000.0
    obj.yscale = 1000.0
    
    obj.projection = proj

    self.assertEqual(False, obj.isTransformable())

  def test_isTransformable_noproj(self):
    data = numpy.zeros((10,10), numpy.float64)

    param = _cartesianparam.new()
    param.quantity = "DBZH"    
    param.setData(data)
    
    obj = _cartesian.new()
    obj.xscale = 1000.0
    obj.yscale = 1000.0
    
    obj.addParameter(param)

    self.assertEqual(False, obj.isTransformable())

  def test_attributes_visibility(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.setData(numpy.zeros((10,10), numpy.uint8))
    obj.addParameter(param)
    
    obj.addAttribute("how/something", 1.0)
    self.assertTrue("how/something" in obj.getAttributeNames())
    self.assertTrue("how/something" not in param.getAttributeNames())

    param.addAttribute("how/else", 2.0)
    self.assertTrue("how/else" not in obj.getAttributeNames())
    self.assertTrue("how/else" in param.getAttributeNames())
    
    param.addAttribute("how/something", 2.0)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/something"))
    self.assertAlmostEqual(2.0, param.getAttribute("how/something"))

  def test_howSubgroupAttribute(self):
    obj = _cartesian.new()

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

  def test_product_type(self):
    obj = _cartesian.new()
    producttypes=[_rave.Rave_ProductType_UNDEFINED,
                  _rave.Rave_ProductType_SCAN,
                  _rave.Rave_ProductType_PPI,
                  _rave.Rave_ProductType_CAPPI,
                  _rave.Rave_ProductType_PCAPPI,
                  _rave.Rave_ProductType_ETOP,
                  _rave.Rave_ProductType_MAX,
                  _rave.Rave_ProductType_RR,
                  _rave.Rave_ProductType_VIL,
                  _rave.Rave_ProductType_COMP,
                  _rave.Rave_ProductType_VP,
                  _rave.Rave_ProductType_RHI,
                  _rave.Rave_ProductType_XSEC,
                  _rave.Rave_ProductType_VSP,
                  _rave.Rave_ProductType_HSP,
                  _rave.Rave_ProductType_RAY,
                  _rave.Rave_ProductType_AZIM,
                  _rave.Rave_ProductType_QUAL,
                  _rave.Rave_ProductType_PMAX,
                  _rave.Rave_ProductType_SURF]
    self.assertEqual(obj.product, _rave.Rave_ProductType_UNDEFINED)
    for pt in producttypes:
      obj.product = pt
      self.assertEqual(obj.product, pt)

  def test_isValid_asImage(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.setData(numpy.zeros((10,10), numpy.uint8))
    
    a = _area.new()
    a.xsize = 10
    a.ysize = 10
    a.xscale = 100.0
    a.yscale = 100.0
    a.extent = (1.0, 2.0, 3.0, 4.0)
    a.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    obj.init(a)
    obj.date = "20100101"
    obj.time = "100000"
    obj.source = "PLC:1234"
    obj.product = _rave.Rave_ProductType_CAPPI
    obj.addParameter(param)
    
    self.assertEqual(True, obj.isValid(_rave.Rave_ObjectType_IMAGE))
    
  def test_isValid_asImage_no_date(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.setData(numpy.zeros((10,10), numpy.uint8))
    
    a = _area.new()
    a.xsize = 10
    a.ysize = 10
    a.xscale = 100.0
    a.yscale = 100.0
    a.extent = (1.0, 2.0, 3.0, 4.0)
    a.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    obj.init(a)
    obj.time = "100000"
    obj.source = "PLC:1234"
    obj.product = _rave.Rave_ProductType_CAPPI
    obj.addParameter(param)
    
    self.assertEqual(False, obj.isValid(_rave.Rave_ObjectType_IMAGE))

  def test_addParameter_no_quantity(self):
    obj = _cartesian.new()

    param = _cartesianparam.new()
    param.setData(numpy.zeros((10,10), numpy.uint8))

    try:
      obj.addParameter(param)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_isValid_asCvol(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    
    obj.startdate = "20100101"
    obj.starttime = "100000"
    obj.enddate = "20100101"
    obj.endtime = "100000"
    obj.product = _rave.Rave_ProductType_CAPPI
    obj.xscale = 100.0
    obj.yscale = 100.0
    obj.addParameter(param)
    
    self.assertEqual(True, obj.isValid(_rave.Rave_ObjectType_CVOL))
  
  def test_qualityfields(self):
    obj = _cartesian.new()
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
    obj = _cartesian.new()
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field3 = _ravefield.new()
    field1.addAttribute("how/task", "se.task.1")
    field2.addAttribute("how/task", "se.task.2")
    field3.addAttribute("how/notask", "abc")

    obj.addQualityField(field1)
    obj.addQualityField(field2)
    obj.addQualityField(field3)

    self.assertEqual("se.task.1", obj.getQualityFieldByHowTask("se.task.1").getAttribute("how/task"))
    self.assertEqual("se.task.2", obj.getQualityFieldByHowTask("se.task.2").getAttribute("how/task"))
    self.assertEqual(None, obj.getQualityFieldByHowTask("abc"))

  def test_findQualityFieldByHowTask(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()

    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field3 = _ravefield.new()
    
    field1.addAttribute("how/task", "se.task.1")
    field1.addAttribute("how/some", "should not be found")
    field2.addAttribute("how/task", "se.task.2")
    field3.addAttribute("how/notask", "abc")

    obj.addQualityField(field1)
    obj.addQualityField(field2)
    obj.addQualityField(field3)

    field4 = _ravefield.new()
    field5 = _ravefield.new()
    field4.addAttribute("how/task", "se.task.1")
    field4.addAttribute("how/some", "should be found")
    field5.addAttribute("how/task", "se.task.3")

    param.addQualityField(field4)
    param.addQualityField(field5)    

    param.quantity = "DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    obj.addParameter(param)

    self.assertEqual("should be found", obj.findQualityFieldByHowTask("se.task.1").getAttribute("how/some"))
    self.assertEqual("se.task.2", obj.findQualityFieldByHowTask("se.task.2").getAttribute("how/task"))
    self.assertEqual("se.task.3", obj.findQualityFieldByHowTask("se.task.3").getAttribute("how/task"))
    self.assertEqual(None, obj.findQualityFieldByHowTask("abc"))

  
  def test_addParameter(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity="DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    param2 = _cartesianparam.new()
    param2.quantity="MMH"
    param2.setData(numpy.zeros((240,240),numpy.uint8))
    
    self.assertFalse(obj.hasParameter("DBZH"))
    obj.addParameter(param)
    self.assertTrue(obj.hasParameter("DBZH"))
    obj.addParameter(param2)
    self.assertTrue(obj.hasParameter("DBZH"))
    self.assertTrue(obj.hasParameter("MMH"))

  def test_addParameter_differentSizes(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity="DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    param2 = _cartesianparam.new()
    param2.quantity="MMH"
    param2.setData(numpy.zeros((241,241),numpy.uint8))
    
    obj.addParameter(param)
    try:
      obj.addParameter(param2)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    
    self.assertTrue(obj.hasParameter("DBZH"))
    self.assertFalse(obj.hasParameter("MMH"))
  
  def test_createParameter(self):
    obj = _cartesian.new()
    a = _area.new()
    a.xsize = 10
    a.ysize = 10
    a.xscale = 100.0
    a.yscale = 100.0
    a.extent = (1.0, 2.0, 3.0, 4.0)
    a.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    obj.init(a)
    
    param = obj.createParameter("DBZH", _rave.RaveDataType_UCHAR)
    self.assertEqual(10, param.xsize)
    self.assertEqual(10, param.ysize)
    self.assertEqual("DBZH", param.quantity)

    param = obj.createParameter("MMH", _rave.RaveDataType_UCHAR)
    self.assertEqual(10, param.xsize)
    self.assertEqual(10, param.ysize)
    self.assertEqual("MMH", param.quantity)

    self.assertTrue(obj.hasParameter("DBZH"))
    self.assertTrue(obj.hasParameter("MMH"))
    
  def test_createParameter_notInitialized(self):
    obj = _cartesian.new()

    try:
      obj.createParameter("DBZH",  _rave.RaveDataType_UCHAR)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass    
    self.assertFalse(obj.hasParameter("DBZH"))
    
  def test_getParameter(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity="DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    param2 = _cartesianparam.new()
    param2.quantity="MMH"
    param2.setData(numpy.zeros((240,240),numpy.uint8))
    obj.addParameter(param)
    obj.addParameter(param2)
    
    result = obj.getParameter("DBZH")
    result2 = obj.getParameter("MMH")
    self.assertTrue(param == result)
    self.assertTrue(param2 == result2)

    self.assertTrue(None == obj.getParameter("MMHH"))

  def test_hasParameter(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity="DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    
    self.assertFalse(obj.hasParameter("DBZH"))
    obj.addParameter(param)
    self.assertTrue(obj.hasParameter("DBZH"))
    
  def test_removeParameter(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity="DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    param2 = _cartesianparam.new()
    param2.quantity="MMH"
    param2.setData(numpy.zeros((240,240),numpy.uint8))

    obj.addParameter(param)
    obj.addParameter(param2)
    obj.removeParameter("DBZH")
    
    self.assertTrue(obj.hasParameter("MMH"))
    self.assertFalse(obj.hasParameter("DBZH"))

  def test_getParameterCount(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity="DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    param2 = _cartesianparam.new()
    param2.quantity="MMH"
    param2.setData(numpy.zeros((240,240),numpy.uint8))
    param3 = _cartesianparam.new()
    param3.quantity="MMH"
    param3.setData(numpy.zeros((240,240),numpy.uint8))

    self.assertEqual(0, obj.getParameterCount())
    obj.addParameter(param)
    self.assertEqual(1, obj.getParameterCount())
    obj.addParameter(param2)
    self.assertEqual(2, obj.getParameterCount())
    obj.addParameter(param3)
    self.assertEqual(2, obj.getParameterCount())

  def test_getParameterNames(self):
    obj = _cartesian.new()
    param = _cartesianparam.new()
    param.quantity="DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    param2 = _cartesianparam.new()
    param2.quantity="MMH"
    param2.setData(numpy.zeros((240,240),numpy.uint8))

    result = obj.getParameterNames()
    self.assertEqual(0, len(result))
    
    obj.addParameter(param)
    result = obj.getParameterNames()
    self.assertEqual(1, len(result))
    self.assertTrue("DBZH" in result)

    obj.addParameter(param2)
    result = obj.getParameterNames()
    self.assertEqual(2, len(result))
    self.assertTrue("DBZH" in result)
    self.assertTrue("MMH" in result)
