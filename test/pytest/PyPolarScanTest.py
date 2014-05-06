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
import _polarscanparam
import _rave
import _ravefield
import _polarnav
import string
import numpy
import math

class PyPolarScanTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _polarscan.new()
    
    isscan = string.find(`type(obj)`, "PolarScanCore")
    self.assertNotEqual(-1, isscan)

  def test_isPolarScan(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    self.assertTrue(_polarscan.isPolarScan(obj))
    self.assertFalse(_polarscan.isPolarScan(param))

  def test_attribute_visibility(self):
    attrs = ['elangle', 'nbins', 'rscale', 'nrays', 'rstart', 'a1gate',
             'datatype', 'beamwidth', 'longitude', 'latitude', 'height',
             'time', 'date', 'source', 'defaultparameter']
    obj = _polarscan.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)
  
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
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)
    obj.addParameter(param)
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
    param = _polarscanparam.new()
    param.quantity="DBZH"
    data = numpy.zeros((5,4), numpy.int8)
    param.setData(data)
    obj.addParameter(param)
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
    self.assertAlmostEquals(1.0*math.pi/180.0, obj.beamwidth, 4)
    obj.beamwidth = 10.0*math.pi/180.0
    self.assertAlmostEquals(10.0*math.pi/180.0, obj.beamwidth, 4)

  def test_beamwidth_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEquals(1.0*math.pi/180.0, obj.beamwidth, 4)
    try:
      obj.beamwidth = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(1.0*math.pi/180.0, obj.beamwidth, 4)

  def test_defaultparameter(self):
    obj = _polarscan.new()
    self.assertEquals("DBZH", obj.defaultparameter)
    obj.defaultparameter = "MMM"
    self.assertEquals("MMM", obj.defaultparameter)

  def test_defaultparameter_withParmeters(self):
    obj = _polarscan.new()
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
    param2 = _polarscanparam.new()
    param2.quantity="MMM"
    param2.setData(numpy.ones((3,3), numpy.int8))
    obj.addParameter(param1)
    obj.addParameter(param2)
    
    self.assertAlmostEquals(0.0, obj.getValue(1,1)[1], 4)
    self.assertAlmostEquals(0.0, obj.getValue(2,2)[1], 4)
    obj.defaultparameter = "MMM"
    self.assertAlmostEquals(1.0, obj.getValue(1,1)[1], 4)
    self.assertAlmostEquals(1.0, obj.getValue(2,2)[1], 4)
    obj.defaultparameter = "NONAME"
    self.assertEquals(_rave.RaveValueType_UNDEFINED, obj.getValue(1,1)[0])

  def test_addParameter(self):
    obj = _polarscan.new()
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
    param2 = _polarscanparam.new()
    param2.quantity="MMM"
    param2.setData(numpy.zeros((3,3), numpy.int8))
    obj.addParameter(param1)
    obj.addParameter(param2)

    names = obj.getParameterNames()
    
    self.assertEquals(2, len(names))
    self.assertTrue("DBZH" in names)
    self.assertTrue("MMM" in names)

  def test_addParameter_withChangedDefaultName(self):
    obj = _polarscan.new()
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
    param2 = _polarscanparam.new()
    param2.quantity="MMM"
    param2.setData(numpy.zeros((3,3), numpy.int8))
    obj.addParameter(param1)
    obj.addParameter(param2)
    obj.defaultparameter = "NONAME" #resets to NULL, tested elsewhere
    param3 = _polarscanparam.new()
    param3.quantity="NONAME"
    param3.setData(numpy.ones((3,3), numpy.int8))
    obj.addParameter(param3)

    self.assertAlmostEquals(1.0, obj.getValue(1,1)[1], 4)

  def test_addParameter_conflictingSizes(self):
    obj = _polarscan.new()
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
    param2 = _polarscanparam.new()
    param2.quantity="MMM"
    param2.setData(numpy.zeros((4,4), numpy.int8))
    obj.addParameter(param1)
    
    try:
      obj.addParameter(param2)
      self.fail("Expected AttributeError")
    except AttributeError,e:
      pass

  def test_hasParameter(self):
    obj = _polarscan.new()
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
    param2 = _polarscanparam.new()
    param2.quantity="MMM"
    param2.setData(numpy.zeros((3,3), numpy.int8))
    obj.addParameter(param1)
    obj.addParameter(param2)

    self.assertEquals(True, obj.hasParameter("DBZH"))
    self.assertEquals(True, obj.hasParameter("MMM"))
    self.assertEquals(False, obj.hasParameter("XYZ"))
    self.assertEquals(False, obj.hasParameter("DBZHmm"))

  def test_getParameter(self):
    obj = _polarscan.new()
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
    param2 = _polarscanparam.new()
    param2.quantity="MMM"
    param2.setData(numpy.zeros((3,3), numpy.int8))
    obj.addParameter(param1)
    obj.addParameter(param2)

    result = obj.getParameter("DBZH")
    self.assertEquals("DBZH", result.quantity)
    result = obj.getParameter("MMM")
    self.assertEquals("MMM", result.quantity)
    self.assertEquals(None, obj.getParameter("XYZ"))

  def test_removeParameter(self):
    obj = _polarscan.new()
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
    param2 = _polarscanparam.new()
    param2.quantity="MMM"
    param2.setData(numpy.zeros((3,3), numpy.int8))
    obj.addParameter(param1)
    obj.addParameter(param2)

    result = obj.removeParameter("DBZH")
    self.assertEquals("DBZH", result.quantity)
    
    result = obj.removeParameter("DBZH")
    self.assertEquals(None, result)

    names = obj.getParameterNames()
    self.assertEquals(1, len(names))

  def test_removeAllParameters(self):
    obj = _polarscan.new()
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
    param2 = _polarscanparam.new()
    param2.quantity="MMM"
    param2.setData(numpy.zeros((3,3), numpy.int8))
    obj.addParameter(param1)
    obj.addParameter(param2)

    obj.removeAllParameters()
    names = obj.getParameterNames()
    self.assertEquals(0, len(names))

  def test_getRangeIndex(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((1,200), numpy.int8))
    obj.addParameter(dbzhParam)        
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
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((1,200), numpy.int8))
    obj.addParameter(dbzhParam)    
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
  
  def test_getRange(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((1,200), numpy.int8))
    obj.addParameter(dbzhParam)        
    obj.rscale = 1000.0

    for ri in [0,1,2,3,4,20,40,199]:
      result = obj.getRange(ri)
      self.assertAlmostEquals(ri*1000.0, result)
    self.assertTrue(obj.getRange(200) < 0.0)
    self.assertTrue(obj.getRange(-1) < 0.0)
  
  def test_getAzimuthIndex(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((400,1),numpy.int8))
    obj.addParameter(dbzhParam)
    
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

  def test_setValue(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((10,10), numpy.int8))
    obj.addParameter(dbzhParam)
    obj.setValue((5,5), 10.0)
    
    self.assertAlmostEquals(10.0, obj.getParameter("DBZH").getData()[5,5], 4)

  def test_setValue_otherDefaultParameter(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((10,10), numpy.int8))
    mmhParam = _polarscanparam.new()
    mmhParam.nodata = 255.0
    mmhParam.undetect = 0.0
    mmhParam.quantity = "MMH"
    mmhParam.setData(numpy.zeros((10,10), numpy.int8))

    obj.addParameter(dbzhParam)
    obj.addParameter(mmhParam)
    obj.defaultparameter = "MMH"
    obj.setValue((5,5), 10.0)
    obj.defaultparameter = "DBZH"
    obj.setValue((6,6), 20.0)
    
    self.assertAlmostEquals(10.0, obj.getParameter("MMH").getData()[5,5], 4)
    self.assertAlmostEquals(0.0, obj.getParameter("DBZH").getData()[5,5], 4)
    self.assertAlmostEquals(20.0, obj.getParameter("DBZH").getData()[6,6], 4)
    self.assertAlmostEquals(0.0, obj.getParameter("MMH").getData()[6,6], 4)

  def test_setParameterValue(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((10,10), numpy.int8))
    mmhParam = _polarscanparam.new()
    mmhParam.nodata = 255.0
    mmhParam.undetect = 0.0
    mmhParam.quantity = "MMH"
    mmhParam.setData(numpy.zeros((10,10), numpy.int8))

    obj.addParameter(dbzhParam)
    obj.addParameter(mmhParam)
    obj.setParameterValue("DBZH", (5,5), 10.0)
    obj.setParameterValue("MMH", (6,6), 20.0)
    
    self.assertAlmostEquals(0.0, obj.getParameter("MMH").getData()[5,5], 4)
    self.assertAlmostEquals(10.0, obj.getParameter("DBZH").getData()[5,5], 4)
    self.assertAlmostEquals(20.0, obj.getParameter("MMH").getData()[6,6], 4)
    self.assertAlmostEquals(0.0, obj.getParameter("DBZH").getData()[6,6], 4)

  def test_getValue(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = dbzhParam.undetect
    a[2][1] = dbzhParam.nodata
    a[4][5] = dbzhParam.undetect
    dbzhParam.setData(a)
    obj.addParameter(dbzhParam);
    
    pts = [((0,0), (_rave.RaveValueType_UNDETECT, 0.0)),
           ((1,0), (_rave.RaveValueType_DATA, 1.0)),
           ((0,1), (_rave.RaveValueType_DATA, 6.0)),
           ((1,2), (_rave.RaveValueType_NODATA, dbzhParam.nodata)),
           ((4,4), (_rave.RaveValueType_DATA, 28.0)),
           ((5,4), (_rave.RaveValueType_UNDETECT, dbzhParam.undetect)),
           ((5,5), (_rave.RaveValueType_NODATA, dbzhParam.nodata))]
    
    for tval in pts:
      result = obj.getValue(tval[0][0], tval[0][1])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def test_getParameterValue(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = dbzhParam.undetect
    a[2][1] = dbzhParam.nodata
    a[4][5] = dbzhParam.undetect
    dbzhParam.setData(a)
    obj.addParameter(dbzhParam);
    
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "MMM"
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = dbzhParam.nodata
    a[2][1] = dbzhParam.undetect
    a[4][5] = dbzhParam.nodata
    a[4][4] = 31.0
    dbzhParam.setData(a)
    obj.addParameter(dbzhParam);
    
    # DBZH
    pts = [((0,0), (_rave.RaveValueType_UNDETECT, 0.0)),
           ((1,0), (_rave.RaveValueType_DATA, 1.0)),
           ((0,1), (_rave.RaveValueType_DATA, 6.0)),
           ((1,2), (_rave.RaveValueType_NODATA, dbzhParam.nodata)),
           ((4,4), (_rave.RaveValueType_DATA, 28.0)),
           ((5,4), (_rave.RaveValueType_UNDETECT, dbzhParam.undetect)),
           ((5,5), (_rave.RaveValueType_NODATA, dbzhParam.nodata))]
    
    for tval in pts:
      result = obj.getParameterValue("DBZH", tval[0][0], tval[0][1])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

    #MMM
    pts = [((0,0), (_rave.RaveValueType_NODATA, dbzhParam.nodata)),
           ((1,0), (_rave.RaveValueType_DATA, 1.0)),
           ((0,1), (_rave.RaveValueType_DATA, 6.0)),
           ((1,2), (_rave.RaveValueType_UNDETECT, dbzhParam.undetect)),
           ((4,4), (_rave.RaveValueType_DATA, 31.0)),
           ((5,4), (_rave.RaveValueType_NODATA, dbzhParam.nodata)),
           ((5,5), (_rave.RaveValueType_NODATA, dbzhParam.nodata))]
    
    for tval in pts:
      result = obj.getParameterValue("MMM", tval[0][0], tval[0][1])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def test_getConvertedValue(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.gain = 0.5
    dbzhParam.offset = 10.0
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = dbzhParam.undetect
    a[2][1] = dbzhParam.nodata
    a[4][5] = dbzhParam.undetect
    dbzhParam.setData(a)
    obj.addParameter(dbzhParam);

    pts = [((0,0), (_rave.RaveValueType_UNDETECT, 0.0)),
           ((1,0), (_rave.RaveValueType_DATA, 10.5)),
           ((0,1), (_rave.RaveValueType_DATA, 13.0)),
           ((1,2), (_rave.RaveValueType_NODATA, dbzhParam.nodata)),
           ((4,4), (_rave.RaveValueType_DATA, 24.0)),
           ((5,4), (_rave.RaveValueType_UNDETECT, dbzhParam.undetect)),
           ((5,5), (_rave.RaveValueType_NODATA, dbzhParam.nodata))]
    
    for tval in pts:
      result = obj.getConvertedValue(tval[0][0], tval[0][1])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def test_getConvertedParameterValue(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    param.quantity = "DBZH"
    param.gain = 0.5
    param.offset = 10.0
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = param.undetect
    a[2][1] = param.nodata
    a[4][5] = param.undetect
    param.setData(a)
    obj.addParameter(param);

    param = _polarscanparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    param.quantity = "MMM"
    param.gain = 0.5
    param.offset = 10.0
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = param.nodata
    a[2][1] = param.undetect
    a[0][1] = 10.0
    a[1][0] = 20.0
    a[4][4] = 30.0
    a[4][5] = param.nodata
    param.setData(a)
    obj.addParameter(param);

    # DBZH
    pts = [((0,0), (_rave.RaveValueType_UNDETECT, 0.0)),
           ((1,0), (_rave.RaveValueType_DATA, 10.5)),
           ((0,1), (_rave.RaveValueType_DATA, 13.0)),
           ((1,2), (_rave.RaveValueType_NODATA, param.nodata)),
           ((4,4), (_rave.RaveValueType_DATA, 24.0)),
           ((5,4), (_rave.RaveValueType_UNDETECT, param.undetect)),
           ((5,5), (_rave.RaveValueType_NODATA, param.nodata))]
    
    for tval in pts:
      result = obj.getConvertedParameterValue("DBZH", tval[0][0], tval[0][1])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

    # MMM
    pts = [((0,0), (_rave.RaveValueType_NODATA, param.nodata)),
           ((1,0), (_rave.RaveValueType_DATA, 15.0)),
           ((0,1), (_rave.RaveValueType_DATA, 20.0)),
           ((1,2), (_rave.RaveValueType_UNDETECT, param.undetect)),
           ((4,4), (_rave.RaveValueType_DATA, 25.0)),
           ((5,4), (_rave.RaveValueType_NODATA, param.nodata)),
           ((5,5), (_rave.RaveValueType_NODATA, param.nodata))]
    
    for tval in pts:
      result = obj.getConvertedParameterValue("MMM", tval[0][0], tval[0][1])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def test_getIndexFromAzimuthAndRange(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    param.quantity = "DBZH"
    obj.rscale = 1000.0
    a=numpy.zeros((400,200), numpy.float64)
    param.setData(a)
    obj.addParameter(param)
           
    pts = [((0.0, 0.0), (0, 0)),
           ((0.1, 499.0), (0, 0)),
           ((0.4, 999.0), (0, 0)),
           ((-1.0, 2001.0), (399, 2)),
           ((360.4, 199999.0), (0, 199)),
           ((360.4, 200000.0), None)]
 
    for tval in pts:
      az = tval[0][0]*math.pi/180.0
      ra = tval[0][1]   
      result = obj.getIndexFromAzimuthAndRange(az,ra)
      if tval[1] == None and result == None:
        pass
      elif tval[1] != None and result != None:
        self.assertEquals(2, len(result))
        self.assertEquals(tval[1][0], result[0])
        self.assertEquals(tval[1][1], result[1])
      else:
        self.fail("Unexpected result")

  def test_getValueAtAzimuthAndRange(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    param.quantity = "DBZH"
    obj.rscale = 1000.0
    a=numpy.arange(3600)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(360,10)).astype(numpy.float64)      
    a[0][0] = param.undetect
    a[2][1] = param.nodata
    a[4][5] = param.undetect

    param.setData(a)
    obj.addParameter(param)
    
    pts = [((0.0,0.0), (_rave.RaveValueType_UNDETECT, 0.0)),        #0, 0
           ((10.0,2000.0), (_rave.RaveValueType_DATA, 102.0)),      #10*10 + 2
           ((20.0,9000), (_rave.RaveValueType_DATA, 209.0)),        #20*10 + 9
           ((2.0,1000), (_rave.RaveValueType_NODATA, param.nodata)),  #10*20 + 9
           ((4.0,5000), (_rave.RaveValueType_UNDETECT, param.undetect))]
    
    for tval in pts:
      az = tval[0][0]*math.pi/180.0
      ra = tval[0][1]
      result = obj.getValueAtAzimuthAndRange(az,ra)
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def test_getValueAtAzimuthAndRange_outsideRange(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"    
    obj.rscale = 1000.0
    a=numpy.zeros((360,10), numpy.float64)      
    dbzhParam.setData(a)    
    obj.addParameter(dbzhParam)
    
    t,v = obj.getValueAtAzimuthAndRange(0.0, 15000.0)
    self.assertEquals(_rave.RaveValueType_NODATA, t)
    self.assertAlmostEquals(255.0, v, 4)

  def test_getValueAtAzimuthAndRange_undetect(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"      
    obj.rscale = 1000.0
    a=numpy.zeros((360,10), numpy.float64)      
    dbzhParam.setData(a)    
    obj.addParameter(dbzhParam)

    t,v = obj.getValueAtAzimuthAndRange(0.0, 1000.0)
    self.assertEquals(_rave.RaveValueType_UNDETECT, t)
    self.assertAlmostEquals(0.0, v, 4)
    
  def test_getParameterValueAtAzimuthAndRange(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    param.quantity = "DBZH"
    obj.rscale = 1000.0
    a=numpy.zeros((360,10), numpy.float64)
    a[10][2] = 98
    param.setData(a)
    obj.addParameter(param)

    param = _polarscanparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    param.quantity = "MMM"
    obj.rscale = 1000.0
    a=numpy.zeros((360,10), numpy.float64)
    a[10][2] = 107
    param.setData(a)
    obj.addParameter(param)
    
    result = obj.getParameterValueAtAzimuthAndRange("DBZH", 10.0*math.pi/180.0, 2000.0)
    self.assertEquals(_rave.RaveValueType_DATA, result[0])
    self.assertEquals(98.0, result[1])
    result = obj.getParameterValueAtAzimuthAndRange("MMM", 10.0*math.pi/180.0, 2000.0)
    self.assertEquals(_rave.RaveValueType_DATA, result[0])
    self.assertEquals(107.0, result[1])
  
  def XtestGetConvertedParameterValueAtAzimuthAndRange(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    param.quantity = "DBZH"
    param.gain = 2.0
    param.offset = 3.0
    obj.rscale = 1000.0
    a=numpy.zeros((360,10), numpy.float64)
    a[10][2] = 98
    param.setData(a)
    obj.addParameter(param)

    param = _polarscanparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    param.quantity = "MMM"
    param.gain = 4.0
    param.offset = 5.0
    obj.rscale = 1000.0
    a=numpy.zeros((360,10), numpy.float64)
    a[10][2] = 107
    param.setData(a)
    obj.addParameter(param)
    
    result = obj.getConvertedParameterValueAtAzimuthAndRange("DBZH", 10.0*math.pi/180.0, 2000.0)
    self.assertEquals(_rave.RaveValueType_DATA, result[0])
    self.assertEquals(98.0*2.0 + 3.0, result[1])
    result = obj.getConvertedParameterValueAtAzimuthAndRange("MMM", 10.0*math.pi/180.0, 2000.0)
    self.assertEquals(_rave.RaveValueType_DATA, result[0])
    self.assertEquals(107.0*4.0 + 5.0, result[1])
    
  
  def test_getNearest(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"      
    obj.longitude = 14.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.rscale = 1000.0
    a=numpy.zeros((4,9), numpy.float64)      
    a[0][8] = 10.0
    dbzhParam.setData(a)
    obj.addParameter(dbzhParam)
    t,v = obj.getNearest((14.0*math.pi/180.0, 60.08*math.pi/180.0))
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(10.0, v, 4)

  def test_getNearestParameterValue(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.offset = 0.0
    dbzhParam.gain = 2.0

    mmhParam = _polarscanparam.new()
    mmhParam.nodata = 255.0
    mmhParam.undetect = 0.0
    mmhParam.quantity = "MMM"
    mmhParam.offset = 0.0
    mmhParam.gain = 3.0

    a=numpy.zeros((4,9), numpy.float64)      
    a[0][8] = 10.0
    dbzhParam.setData(a)

    a=numpy.zeros((4,9), numpy.float64)      
    a[0][8] = 20.0
    mmhParam.setData(a)

    obj.addParameter(dbzhParam)
    obj.addParameter(mmhParam)
        
    obj.longitude = 14.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.rscale = 1000.0

    t,v = obj.getNearestParameterValue("DBZH", (14.0*math.pi/180.0, 60.08*math.pi/180.0))
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(10.0, v, 4)

    t,v = obj.getNearestParameterValue("MMM", (14.0*math.pi/180.0, 60.08*math.pi/180.0))
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(20.0, v, 4)

  def test_getNearestConvertedParameterValue(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.offset = 0.0
    dbzhParam.gain = 2.0

    mmhParam = _polarscanparam.new()
    mmhParam.nodata = 255.0
    mmhParam.undetect = 0.0
    mmhParam.quantity = "MMM"
    mmhParam.offset = 0.0
    mmhParam.gain = 3.0

    a=numpy.zeros((4,9), numpy.float64)      
    a[0][8] = 10.0
    dbzhParam.setData(a)

    a=numpy.zeros((4,9), numpy.float64)      
    a[0][8] = 20.0
    mmhParam.setData(a)

    obj.addParameter(dbzhParam)
    obj.addParameter(mmhParam)
        
    obj.longitude = 14.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.rscale = 1000.0

    t,v = obj.getNearestConvertedParameterValue("DBZH", (14.0*math.pi/180.0, 60.08*math.pi/180.0))
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(20.0, v, 4)

    t,v = obj.getNearestConvertedParameterValue("MMM", (14.0*math.pi/180.0, 60.08*math.pi/180.0))
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(60.0, v, 4)

  def test_getNearestIndex(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    obj.longitude = 14.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.rscale = 1000.0
    a=numpy.arange(3600)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(360,10)).astype(numpy.float64)      
    dbzhParam.setData(a)
    obj.addParameter(dbzhParam)

    pts = [((14.0 * math.pi/180.0, 60.08 * math.pi/180.0), (8, 0)),
           ((14.08 * math.pi/180.0, 60.00 * math.pi/180.0), (4, 90)),
           ((17.0 * math.pi/180.0, 60.08 * math.pi/180.0), None)]

    for val in pts:
      result = obj.getNearestIndex(val[0])
      if val[1] == None and result == None:
        pass
      elif val[1] != None and result != None:
        self.assertEquals(len(val[1]), len(result))
        self.assertEquals(val[1][0], result[0])
        self.assertEquals(val[1][1], result[1])
      else:
        if result != None:
          self.fail("Unexpected result: result = %d,%d"%result)
        else:
          self.fail("Unexpected result: result = None")

  def test_qualityfields(self):
    obj = _polarscan.new()
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
  
  def test_add_how_array_attribute_long(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.int32))
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEquals(10, len(result))
    self.assertEquals(0, result[0])
    self.assertEquals(3, result[3])
    self.assertEquals(5, result[5])
    self.assertEquals(9, result[9])

  def test_add_how_array_attribute_double(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.float32))
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEquals(10, len(result))
    self.assertAlmostEquals(0.0, result[0], 2)
    self.assertAlmostEquals(3.0, result[3], 2)
    self.assertAlmostEquals(5.0, result[5], 2)
    self.assertAlmostEquals(9.0, result[9], 2)
  
  def test_hasAttribute(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", 1.0)
    obj.addAttribute("how/something2", "jupp")
    self.assertEquals(True, obj.hasAttribute("how/something"))
    self.assertEquals(True, obj.hasAttribute("how/something2"))
    self.assertEquals(False, obj.hasAttribute("how/something3"))
    
  def test_getQualityFieldByHowTask(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)
    
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("how/task", "se.smhi.f1")
    field1.addAttribute("what/value", "f1")
    field2.addAttribute("how/task", "se.smhi.f2")
    field2.addAttribute("what/value", "f2")

    field3 = _ravefield.new()
    field3.addAttribute("how/task", "se.smhi.f2")
    field3.addAttribute("what/value", "pf2")
    param.addQualityField(field3)
    
    obj.addQualityField(field1)
    obj.addQualityField(field2)
    obj.addParameter(param)
    
    result = obj.getQualityFieldByHowTask("se.smhi.f2")
    self.assertEquals("f2", result.getAttribute("what/value"))

  def test_getQualityFieldByHowTask_notFound(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)

    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("how/task", "se.smhi.f1")
    field1.addAttribute("what/value", "f1")
    field2.addAttribute("how/task", "se.smhi.f2")
    field2.addAttribute("what/value", "f2")

    field3 = _ravefield.new()
    field3.addAttribute("how/task", "se.smhi.f3")
    field3.addAttribute("what/value", "pf2")
    param.addQualityField(field3)

    obj.addQualityField(field1)
    obj.addQualityField(field2)
    
    try:
      obj.getQualityFieldByHowTask("se.smhi.f3")
      self.fail("Expected NameError")
    except NameError, e:
      pass

  def test_findQualityFieldByHowTask(self):
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG);
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)
    
    param2 = _polarscanparam.new()
    param2.quantity="MMH"    
    data = numpy.zeros((4,5), numpy.int8)
    param2.setData(data)
    
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("how/task", "se.smhi.f1")
    field1.addAttribute("what/value", "f1")
    field2.addAttribute("how/task", "se.smhi.f2")
    field2.addAttribute("what/value", "f2")

    field3 = _ravefield.new()
    field3.addAttribute("how/task", "se.smhi.f2")
    field3.addAttribute("what/value", "pf2")
    param.addQualityField(field3)
    field4 = _ravefield.new()
    field4.addAttribute("how/task", "se.smhi.f2")
    field4.addAttribute("what/value", "pf2-mmh")
    param2.addQualityField(field4)
    
    obj.addQualityField(field1)
    obj.addQualityField(field2)
    obj.addParameter(param)
    obj.addParameter(param2)
    
    result = obj.findQualityFieldByHowTask("se.smhi.f2")
    self.assertEquals("pf2", result.getAttribute("what/value"))

    result = obj.findQualityFieldByHowTask("se.smhi.f1")
    self.assertEquals("f1", result.getAttribute("what/value"))

    result = obj.findQualityFieldByHowTask("se.smhi.f2", "MMH")
    self.assertEquals("pf2-mmh", result.getAttribute("what/value"))

  def test_findQualityFieldByHowTask_notFound(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)
    
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("how/task", "se.smhi.f1")
    field1.addAttribute("what/value", "f1")
    field2.addAttribute("how/task", "se.smhi.f2")
    field2.addAttribute("what/value", "f2")

    field3 = _ravefield.new()
    field3.addAttribute("how/task", "se.smhi.f2")
    field3.addAttribute("what/value", "pf2")
    param.addQualityField(field3)
    
    obj.addQualityField(field1)
    obj.addQualityField(field2)
    obj.addParameter(param)
    
    result = obj.findQualityFieldByHowTask("se.smhi.f3")
    self.assertEquals(None, result)
    
  def test_getDistanceField(self):
    polnav = _polarnav.new()
    polnav.lat0 = 60.0 * math.pi / 180.0
    polnav.lon0 = 12.0 * math.pi / 180.0
    polnav.alt0 = 0.0

    expected = []
    for i in range(10):
      expected.append(polnav.reToDh(100.0 * i, (math.pi / 180.0)*0.5)[0])
    
    obj = _polarscan.new()
    obj.longitude = polnav.lon0 # We want same settings as the polar navigator so that we can test result
    obj.latitude = polnav.lat0
    obj.height = polnav.alt0
    
    obj.rscale = 100.0
    obj.elangle = (math.pi / 180.0)*0.5
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((5, 10), numpy.int8)
    param.setData(data)
    obj.addParameter(param)

    f = obj.getDistanceField()
    self.assertEquals(10, f.xsize)
    for i in range(10):
      self.assertAlmostEquals(expected[i], f.getValue(i, 0)[1], 4)

  def test_getHeightField(self):
    polnav = _polarnav.new()
    polnav.lat0 = 60.0 * math.pi / 180.0
    polnav.lon0 = 12.0 * math.pi / 180.0
    polnav.alt0 = 0.0

    expected = []
    for i in range(10):
      expected.append(polnav.reToDh(100.0 * i, (math.pi / 180.0)*0.5)[1])
    
    obj = _polarscan.new()
    obj.longitude = polnav.lon0 # We want same settings as the polar navigator so that we can test result
    obj.latitude = polnav.lat0
    obj.height = polnav.alt0
    
    obj.rscale = 100.0
    obj.elangle = (math.pi / 180.0)*0.5
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((5, 10), numpy.int8)
    param.setData(data)
    obj.addParameter(param)

    f = obj.getHeightField()
    self.assertEquals(10, f.xsize)
    for i in range(10):
      self.assertAlmostEquals(expected[i], f.getValue(i, 0)[1], 4)
    
if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()