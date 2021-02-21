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
        
    self.assertNotEqual(-1, str(type(obj)).find("PolarScanCore"))
    
  def test_isPolarScan(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    self.assertTrue(_polarscan.isPolarScan(obj))
    self.assertFalse(_polarscan.isPolarScan(param))
    
  def test_attribute_visibility(self):
    attrs = ['elangle', 'nbins', 'rscale', 'nrays', 'rstart', 'a1gate',
             'datatype', 'beamwidth', 'beamwH', 'beamwV', 'longitude', 'latitude', 'height',
             'time', 'date', 'source', 'defaultparameter']
    obj = _polarscan.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)
      
  def test_time(self):
    obj = _polarscan.new()
    self.assertEqual(None, obj.time)
    obj.time = "200500"
    self.assertEqual("200500", obj.time)
    obj.time = None
    self.assertEqual(None, obj.time)
    
  def test_time_badValues(self):
    obj = _polarscan.new()
    values = ["10101", "1010101", "1010ab", "1010x0", "abcdef", 123456]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError:
        pass
    
  def test_date(self):
    obj = _polarscan.new()
    self.assertEqual(None, obj.date)
    obj.date = "20050101"
    self.assertEqual("20050101", obj.date)
    obj.date = None
    self.assertEqual(None, obj.date)
    
  def test_date_badValues(self):
    obj = _polarscan.new()
    values = ["200910101", "2001010", "200a1010", 20091010]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError:
        pass
    
  def test_source(self):
    obj = _polarscan.new()
    self.assertEqual(None, obj.source)
    obj.source = "ABC:10, ABD:1"
    self.assertEqual("ABC:10, ABD:1", obj.source)
    obj.source = None
    self.assertEqual(None, obj.source)
  
  def test_longitude(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.longitude, 4)
    obj.longitude = 1.0
    self.assertAlmostEqual(1.0, obj.longitude, 4)
    
  def test_latitude(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.latitude, 4)
    obj.latitude = 1.0
    self.assertAlmostEqual(1.0, obj.latitude, 4)
    
  def test_height(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEqual(1.0, obj.height, 4)
    
  def test_invalid_attributes(self):
    obj = _polarscan.new()
    try:
      obj.lon = 1.0
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    
  def test_elangle(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.elangle, 4)
    obj.elangle = 10.0
    self.assertAlmostEqual(10.0, obj.elangle, 4)
    
  def test_elangle_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.elangle, 4)
    try:
      obj.elangle = 10
      self.fail("Excepted TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.elangle, 4)
    
  def test_nbins(self):
    obj = _polarscan.new()
    self.assertEqual(0, obj.nbins)
    try:
      obj.nbins = 10
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(0, obj.nbins)
    
  def test_nbins_withData(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)
    obj.addParameter(param)
    self.assertEqual(5, obj.nbins)
    
  def test_rscale(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.rscale, 4)
    obj.rscale = 10.0
    self.assertAlmostEqual(10.0, obj.rscale, 4)
    
  def test_rscale_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.rscale, 4)
    try:
      obj.rscale = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.rscale, 4)
    
  def test_nrays(self):
    obj = _polarscan.new()
    self.assertEqual(0, obj.nrays)
    try:
      obj.nrays = 10
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(0, obj.nrays)
    
  def test_nrays_withData(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.quantity="DBZH"
    data = numpy.zeros((5,4), numpy.int8)
    param.setData(data)
    obj.addParameter(param)
    self.assertEqual(5, obj.nrays)
    
  def test_rstart(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.rstart, 4)
    obj.rstart = 10.0
    self.assertAlmostEqual(10.0, obj.rstart, 4)
    
  def test_rstart_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(0.0, obj.rstart, 4)
    try:
      obj.rstart = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(0.0, obj.rstart, 4)
    
  def test_datatype(self):
    obj = _polarscan.new()
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    try:
      obj.datatype = _rave.RaveDataType_INT
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    
  def test_a1gate(self):
    obj = _polarscan.new()
    self.assertEqual(0, obj.a1gate)
    obj.a1gate = 10
    self.assertEqual(10, obj.a1gate)
    
  def test_a1gate_typeError(self):
    obj = _polarscan.new()
    self.assertEqual(0, obj.a1gate)
    try:
      obj.a1gate = 10.0
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertEqual(0, obj.a1gate)
    
  def test_beamwidth(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(1.0*math.pi/180.0, obj.beamwidth, 4)
    obj.beamwidth = 10.0*math.pi/180.0
    self.assertAlmostEqual(10.0*math.pi/180.0, obj.beamwidth, 4)
    
  def test_beamwidth_typeError(self):
    obj = _polarscan.new()
    self.assertAlmostEqual(1.0*math.pi/180.0, obj.beamwidth, 4)
    try:
      obj.beamwidth = 10
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertAlmostEqual(1.0*math.pi/180.0, obj.beamwidth, 4)
    
  def test_use_azimuthal_nav_information(self):
    obj = _polarscan.new()
    self.assertTrue(obj.use_azimuthal_nav_information)
    obj.use_azimuthal_nav_information = False
    self.assertFalse(obj.use_azimuthal_nav_information)
 
  def test_clone(self):
    obj = _polarscan.new()
    obj.elangle = 1.0
    obj.rscale = 1000.0
    obj.rstart = 2.0
    obj.a1gate = 3
    obj.beamwidth = 4.0
    obj.longitude = 5.0
    obj.latitude = 6.0
    obj.height = 7.0
    obj.time = "100000"
    obj.date = "20130101"
    obj.starttime = "110000"
    obj.startdate = "20130102"
    obj.endtime = "120000"
    obj.enddate = "20130103"
    obj.source = "CMT:123"
        
    param1 = _polarscanparam.new()
    param1.quantity="DBZH"
    param1.setData(numpy.zeros((3,3), numpy.int8))
        
    obj.addParameter(param1)
        
    cpy = obj.clone()
    
    obj.elangle = 10.0
    obj.rscale = 1100.0
    obj.rstart = 9.0
    obj.a1gate = 8
    obj.beamwidth = 7.0
    obj.longitude = 6.0
    obj.latitude = 5.0
    obj.height = 4.0
    obj.time = "100001"
    obj.date = "20130201"
    obj.starttime = "110001"
    obj.startdate = "20130202"
    obj.endtime = "120001"
    obj.enddate = "20130203"
    obj.source = "CMT:124"
    
    self.assertAlmostEqual(1.0, cpy.elangle, 4)
    self.assertAlmostEqual(1000.0, cpy.rscale, 4)
    self.assertAlmostEqual(2.0, cpy.rstart, 4)
    self.assertEqual(3, cpy.a1gate)
    self.assertAlmostEqual(4.0, cpy.beamwidth, 4)
    self.assertAlmostEqual(5.0, cpy.longitude, 4)
    self.assertAlmostEqual(6.0, cpy.latitude, 4)
    self.assertAlmostEqual(7.0, cpy.height, 4)
    self.assertEqual("100000", cpy.time)
    self.assertEqual("20130101", cpy.date)
    self.assertEqual("110000", cpy.starttime)
    self.assertEqual("20130102", cpy.startdate)
    self.assertEqual("120000", cpy.endtime)
    self.assertEqual("20130103", cpy.enddate)
    self.assertEqual("CMT:123", cpy.source)
    
  def test_defaultparameter(self):
    obj = _polarscan.new()
    self.assertEqual("DBZH", obj.defaultparameter)
    obj.defaultparameter = "MMM"
    self.assertEqual("MMM", obj.defaultparameter)
    
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
        
    self.assertAlmostEqual(0.0, obj.getValue(1,1)[1], 4)
    self.assertAlmostEqual(0.0, obj.getValue(2,2)[1], 4)
    obj.defaultparameter = "MMM"
    self.assertAlmostEqual(1.0, obj.getValue(1,1)[1], 4)
    self.assertAlmostEqual(1.0, obj.getValue(2,2)[1], 4)
    obj.defaultparameter = "NONAME"
    self.assertEqual(_rave.RaveValueType_UNDEFINED, obj.getValue(1,1)[0])
    
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
        
    self.assertEqual(2, len(names))
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
    
    self.assertAlmostEqual(1.0, obj.getValue(1,1)[1], 4)
    
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
    except AttributeError:
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
    
    self.assertEqual(True, obj.hasParameter("DBZH"))
    self.assertEqual(True, obj.hasParameter("MMM"))
    self.assertEqual(False, obj.hasParameter("XYZ"))
    self.assertEqual(False, obj.hasParameter("DBZHmm"))
    
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
    self.assertEqual("DBZH", result.quantity)
    result = obj.getParameter("MMM")
    self.assertEqual("MMM", result.quantity)
    self.assertEqual(None, obj.getParameter("XYZ"))
    
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
    self.assertEqual("DBZH", result.quantity)
        
    result = obj.removeParameter("DBZH")
    self.assertEqual(None, result)
    
    names = obj.getParameterNames()
    self.assertEqual(1, len(names))
    
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
    self.assertEqual(0, len(names))
    
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
      self.assertEqual(rr[1], result)
    
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
      self.assertEqual(rr[1], result)
      
  def test_getRange(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((1,200), numpy.int8))
    obj.addParameter(dbzhParam)        
    obj.rscale = 1000.0
    
    for ri in [0,1,2,3,4,20,40,199]:
      result = obj.getRange(ri)
      self.assertAlmostEqual(ri*1000.0, result)
    self.assertTrue(obj.getRange(200) < 0.0)
    self.assertTrue(obj.getRange(-1) < 0.0)

  def test_getAzimuthIndex(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
      
    # Azimuths tuple is ordered by an azimuth in degrees and expected index
    azimuths = [(180.0, 180),
                (90.0, 90),
                (0.0, 0),
                (0.1, 0),
                (0.4, 0),
                (359.9, 0),
                (360.4, 0),
                (-0.1, 0),
                (-1.0,359)]
    for azv in azimuths:
      result = obj.getAzimuthIndex(azv[0]*math.pi/180.0)
      self.assertEqual(azv[1], result)

  def test_getAzimuthIndex_2(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((720,1),numpy.int8))
    obj.addParameter(dbzhParam)
      
    # Azimuths tuple is ordered by an azimuth in degrees and expected index
    azimuths = [(180.0, 360),
                (90.0, 180),
                (0.0, 0),
                (0.1, 0),
                (0.4, 1),
                (359.9, 0),
                (360.4, 1),
                (-0.1, 0),
                (-1.0,718)]
    for azv in azimuths:
      result = obj.getAzimuthIndex(azv[0]*math.pi/180.0)
      self.assertEqual(azv[1], result)

  def test_getAzimuthIndex_with_astart(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((450,1),numpy.int8))  # 450 rays => 0.8 degree bandwidth
    obj.addParameter(dbzhParam)

    # Since nr rays = 360 => raywidth will be 1.0 degrees. And as such, astart - halfwidth -> astart + halfwidth will be expected range for each index 
    # Values are (azimuth, astart, expected index)
    #
    astart_test_list = [(0.3, -0.1, 0),
                        (0.69, -0.1, 0),
                        (0.7, -0.1, 1),
                        (0.71, -0.1, 1),
                        (0.71, -0.1, 1),
                        (0.71, 0.1, 0),
                        (0.0, 0.1, 449),
                        (0.1, 0.1, 0),
                        (0.89, 0.1, 0),
                        (0.9, 0.1, 1),
                        (359.9, 0.1, 449),
                        (359.69, 0.1, 449),
                        (359.99, 0.1, 449),
                        ] 

    for (az, astart, index) in astart_test_list:
      #print("Testing %f %f"%(az, astart))
      obj.addAttribute("how/astart", astart) 
      result = obj.getAzimuthIndex(az*math.pi/180.0)
      self.assertEqual(index,result)

  def test_getAzimuthIndex_with_astart_use_azimuthal_nav_information_false(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((400,1),numpy.int8))
    obj.addParameter(dbzhParam)
      
    # Azimuths tuple is ordered by an azimuth in degrees and expected index
    obj.addAttribute("how/astart", 1.0)
    obj.use_azimuthal_nav_information = False
    self.assertAlmostEqual(0, obj.getAzimuthIndex(-0.4*math.pi/180.0))
    obj.use_azimuthal_nav_information = True
    self.assertAlmostEqual(398, obj.getAzimuthIndex(-0.4*math.pi/180.0))


  def test_getAzimuthIndex_with_startazA_stopazA(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) - 0.9
    startazA[0] = 359.1
    stopazA = numpy.arange(0.0, 360.0, 1.0) + 0.1
    #stopazA[359] = 359.1
    #numpy.set_printoptions(precision=3, suppress=True)
    #print(startazA)
    #print(stopazA)
    obj.addAttribute("how/startazA", startazA)
    obj.addAttribute("how/stopazA", stopazA)
    
    test_list = [(359.9, 0),   # All azimuths should be matched inside given array...
                 (359.8, 0),
                 (358.9, 359),
                 (0.09, 0),
                 (0.1, 0),
                 (0.11, 1),
                 (0.91, 1),
                 (0.89, 1),
                 (-0.09, 0), #
                 (-0.1, 0),
                 (-0.11, 0),
                 (-0.4, 0),
                 (-0.89, 0),
                 (-0.9, 359),
                 (-0.91, 359),
                 (359.8, 0),
                 (359.1, 0),
                 (359.09, 359),
                 (358.09, 358)
                 ]
    
    for (az, index) in test_list:
      result = obj.getAzimuthIndex(az*math.pi/180.0)
      self.assertEqual(index,result)

  def test_getAzimuthIndex_with_startazA_stopazA_use_azimuthal_nav_information_false(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) - 0.9
    startazA[0] = 359.1
    stopazA = numpy.arange(0.0, 360.0, 1.0) + 0.1
    #stopazA[359] = 359.1
    #numpy.set_printoptions(precision=3, suppress=True)
    #print(startazA)
    #print(stopazA)
    obj.addAttribute("how/startazA", startazA)
    obj.addAttribute("how/stopazA", stopazA)
    obj.use_azimuthal_nav_information = False

    # Azimuths tuple is ordered by an azimuth in degrees and expected index
    azimuths = [(180.0, 180),
                (90.0, 90),
                (0.0, 0),
                (0.1, 0),
                (0.4, 0),
                (359.9, 0),
                (360.4, 0),
                (-0.1, 0),
                (-1.0,359)]

    for azv in azimuths:
      result = obj.getAzimuthIndex(azv[0]*math.pi/180.0)
      self.assertEqual(azv[1], result)

  def test_getNorthmostIndex_0(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) - 0.4
    startazA[0] = 359.6

    obj.addAttribute("how/startazA", startazA)

    # Index 0, has startaz=-0.4 which is closest to north 
    self.assertEqual(0, obj.getNorthmostIndex())
    
    # Since index 0, no rotation required
    self.assertEqual(0, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_1(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) - 0.9
    startazA[0] = 359.1

    obj.addAttribute("how/startazA", startazA)

    # Index 1, has startaz=0.1 which is closest to north 
    self.assertEqual(1, obj.getNorthmostIndex())
    
    # If getNorthmostIndex() returns a value > 0 && value <= nrays/2, we should rotate counter clockwise and result should be "- getNorthmostIndex()"
    self.assertEqual(-1, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_2(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) - 1.9
    startazA[0] = 358.1
    startazA[1] = 359.1

    obj.addAttribute("how/startazA", startazA)
    
    # Index 2, has startaz=0.1 which is closest to north
    self.assertEqual(2, obj.getNorthmostIndex())

    # If getNorthmostIndex() returns a value > 0 && value <= nrays/2, we should rotate counter clockwise and result should be "- getNorthmostIndex()"
    self.assertEqual(-2, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_3(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) + 0.4
    obj.addAttribute("how/startazA", startazA)

    # Index 0, has startaz=-0.4 which is closest to north 
    self.assertEqual(0, obj.getNorthmostIndex())
    
    # If index = 0, then no rotation
    self.assertEqual(0, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_4(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) + 1.4
    startazA[359]=0.4
    
    obj.addAttribute("how/startazA", startazA)

    # Index 359, has startaz=0.4 which is closest to north 
    self.assertEqual(359, obj.getNorthmostIndex())
    
    # If getNorthmostIndex() returns a value > 0 && value > nrays/2, we should rotate counter clockwise and result should be "nrays - getNorthmostIndex()"
    self.assertEqual(1, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_5(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) + 2.4
    startazA[359]=1.4
    startazA[358]=0.4
    
    obj.addAttribute("how/startazA", startazA)

    # Index 358, has startaz=0.4 which is closest to north 
    self.assertEqual(358, obj.getNorthmostIndex())

    # If getNorthmostIndex() returns a value > 0 && value > nrays/2, we should rotate counter clockwise and result should be "nrays - getNorthmostIndex()"
    self.assertEqual(2, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_6(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((450,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,0.8) + 2.4
    startazA[449]=1.6
    startazA[448]=0.8
    startazA[447]=0.0
    
    obj.addAttribute("how/startazA", startazA)

    # Index 358, has startaz=0.4 which is closest to north 
    self.assertEqual(447, obj.getNorthmostIndex())

    # If getNorthmostIndex() returns a value > 0 && value > nrays/2, we should rotate counter clockwise and result should be "nrays - getNorthmostIndex()"
    self.assertEqual(3, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_astart_0(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    
    obj.addAttribute("how/astart", -0.4999)

    # astart=--0.4999 and ray width ~ 1.0 => index 0 is closest to north 
    self.assertEqual(0, obj.getNorthmostIndex())

    # If getNorthmostIndex() returns a value > 0 && value > nrays/2, we should rotate counter clockwise and result should be "nrays - getNorthmostIndex()"
    self.assertEqual(0, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_astart_1(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    
    obj.addAttribute("how/astart", -0.5)

    # astart=--0.5 and ray width ~ 1.0 => index 1 is closest to north 
    self.assertEqual(1, obj.getNorthmostIndex())

    # If getNorthmostIndex() returns a value > 0 && value > nrays/2, we should rotate counter clockwise and result should be "nrays - getNorthmostIndex()"
    self.assertEqual(-1, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_astart_2(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    
    obj.addAttribute("how/astart", 0.5)

    # astart=0.5 and ray width ~ 1.0 => index 0 is closest to north 
    self.assertEqual(0, obj.getNorthmostIndex())

    # If getNorthmostIndex() returns a value > 0 && value > nrays/2, we should rotate counter clockwise and result should be "nrays - getNorthmostIndex()"
    self.assertEqual(0, obj.getRotationRequiredToNorthmost())

  def test_getNorthmostIndex_astart_3(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    
    obj.addAttribute("how/astart", 0.51)

    # astart=0.51 and ray width ~ 1.0 => index 359 is closest to north 
    self.assertEqual(359, obj.getNorthmostIndex())

    # If getNorthmostIndex() returns a value > 0 && value > nrays/2, we should rotate counter clockwise and result should be "nrays - getNorthmostIndex()"
    self.assertEqual(1, obj.getRotationRequiredToNorthmost())
    
  def test_getNorthmostIndex_startazA_over_astart(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    startazA = numpy.arange(0.0,360.0,1.0) - 0.9
    startazA[0] = 359.1

    obj.addAttribute("how/startazA", startazA)

    obj.addAttribute("how/astart", 0.51)

    # Index 0, has startaz=-0.4 which is closest to north 
    self.assertEqual(1, obj.getNorthmostIndex())
    
    # Since index 0, no rotation required
    self.assertEqual(-1, obj.getRotationRequiredToNorthmost())

  def test_getAzimuth(self):
    obj = _polarscan.new()
    obj.use_azimuthal_nav_information = False
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
     
    # Azimuths tuple is ordered by an index and expected azimuth
    azimuths = [(0, 0.0),
                (1, 1.0),
                (359, 359.0)]
    for azv in azimuths:
      result = obj.getAzimuth(azv[0])
      self.assertAlmostEqual(azv[1], result*180.0/math.pi, 4)
 
  def test_getAzimuth_invalidIndex(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
     
    try:  
      obj.getAzimuth(-1)
      self.fail("Expected ValueError")
    except ValueError:
      pass
     
    try:  
      obj.getAzimuth(360)
      self.fail("Expected ValueError")
    except ValueError:
      pass
 
  def test_getAzimuth_with_astart(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
 
    # Since nr rays = 360 => raywidth will be 1.0 degrees. And as such, astart - halfwidth -> astart + halfwidth will be expected range for each index 
    # Values are (azimuth, astart, expected index)
    #
    # Azimuths tuple is ordered by an index and expected azimuth
    astart_test_list = [(0, 0.5, 0.5),
                        (1, 0.5, 1.5),
                        (359, 0.5, 359.5),
                        (0, -0.5, 359.5),
                        (1, -0.5, 0.5),
                        (359, -0.5, 358.5)]
 
    for (index, astart, az) in astart_test_list:
      obj.addAttribute("how/astart", astart)
      result = obj.getAzimuth(index)
      self.assertAlmostEqual(az, result*180.0/math.pi, 4)
 
  def test_getAzimuth_with_startazA_stopazA(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    #numpy.set_printoptions(suppress=True)
    startazA = numpy.arange(0.0,360.0,1.0) - 0.9
    startazA[0] = 359.1
    #print(startazA)
    stopazA = numpy.arange(0.0,360.0,1.0) + 0.1
    #print(stopazA)
    obj.addAttribute("how/startazA", startazA)
    obj.addAttribute("how/stopazA", stopazA)
 
    # Since nr rays = 360 => raywidth will be 1.0 degrees. And as such, astart - halfwidth -> astart + halfwidth will be expected range for each index 
    # Values are (azimuth, astart, expected index)
    #
    # Azimuths tuple is ordered by an index and expected azimuth
    astart_test_list = [(3, 2.6),
                        (4, 3.6),
                        (5, 4.6),
                        (0, 359.6),
                        (1, 0.6),
                        (359, 358.6)]
 
    for (index, az) in astart_test_list:
      result = obj.getAzimuth(index)
      self.assertAlmostEqual(az, result*180.0/math.pi, 4)
 
  def test_getAzimuth_only_with_startazA(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((360,1),numpy.int8))
    obj.addParameter(dbzhParam)
    #numpy.set_printoptions(suppress=True)
    startazA = numpy.arange(0.0,360.0,1.0) - 0.9
    startazA[0] = 359.1
    obj.addAttribute("how/startazA", startazA)
 
    # Since nr rays = 360 => raywidth will be 1.0 degrees. And as such, astart - halfwidth -> astart + halfwidth will be expected range for each index 
    # Values are (azimuth, astart, expected index)
    #
    # Azimuths tuple is ordered by an index and expected azimuth
    astart_test_list = [(3, 2.6),
                        (4, 3.6),
                        (5, 4.6),
                        (0, 359.6),
                        (1, 0.6),
                        (359, 358.6)]
 
    for (index, az) in astart_test_list:
      result = obj.getAzimuth(index)
      self.assertAlmostEqual(az, result*180.0/math.pi, 4)
 
  def test_setValue(self):
    obj = _polarscan.new()
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.setData(numpy.zeros((10,10), numpy.int8))
    obj.addParameter(dbzhParam)
    obj.setValue((5,5), 10.0)
        
    self.assertAlmostEqual(10.0, obj.getParameter("DBZH").getData()[5,5], 4)
    
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
        
    self.assertAlmostEqual(10.0, obj.getParameter("MMH").getData()[5,5], 4)
    self.assertAlmostEqual(0.0, obj.getParameter("DBZH").getData()[5,5], 4)
    self.assertAlmostEqual(20.0, obj.getParameter("DBZH").getData()[6,6], 4)
    self.assertAlmostEqual(0.0, obj.getParameter("MMH").getData()[6,6], 4)
    
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
        
    self.assertAlmostEqual(0.0, obj.getParameter("MMH").getData()[5,5], 4)
    self.assertAlmostEqual(10.0, obj.getParameter("DBZH").getData()[5,5], 4)
    self.assertAlmostEqual(20.0, obj.getParameter("MMH").getData()[6,6], 4)
    self.assertAlmostEqual(0.0, obj.getParameter("DBZH").getData()[6,6], 4)
    
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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)
    
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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)
    
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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)
    
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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)
    
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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)
    
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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)
    
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
        self.assertEqual(2, len(result))
        self.assertEqual(tval[1][0], result[0])
        self.assertEqual(tval[1][1], result[1])
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
      self.assertEqual(tval[1][0], result[0])
      self.assertAlmostEqual(tval[1][1], result[1], 4)
    
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
    self.assertEqual(_rave.RaveValueType_NODATA, t)
    self.assertAlmostEqual(255.0, v, 4)
    
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
    self.assertEqual(_rave.RaveValueType_UNDETECT, t)
    self.assertAlmostEqual(0.0, v, 4)
        
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
    self.assertEqual(_rave.RaveValueType_DATA, result[0])
    self.assertEqual(98.0, result[1])
    result = obj.getParameterValueAtAzimuthAndRange("MMM", 10.0*math.pi/180.0, 2000.0)
    self.assertEqual(_rave.RaveValueType_DATA, result[0])
    self.assertEqual(107.0, result[1])
      
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
    self.assertEqual(_rave.RaveValueType_DATA, result[0])
    self.assertEqual(98.0*2.0 + 3.0, result[1])
    result = obj.getConvertedParameterValueAtAzimuthAndRange("MMM", 10.0*math.pi/180.0, 2000.0)
    self.assertEqual(_rave.RaveValueType_DATA, result[0])
    self.assertEqual(107.0*4.0 + 5.0, result[1])
        
      
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
    self.assertEqual(_rave.RaveValueType_DATA, t)
    self.assertAlmostEqual(10.0, v, 4)
    
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
    self.assertEqual(_rave.RaveValueType_DATA, t)
    self.assertAlmostEqual(10.0, v, 4)
    
    t,v = obj.getNearestParameterValue("MMM", (14.0*math.pi/180.0, 60.08*math.pi/180.0))
    self.assertEqual(_rave.RaveValueType_DATA, t)
    self.assertAlmostEqual(20.0, v, 4)
    
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
    self.assertEqual(_rave.RaveValueType_DATA, t)
    self.assertAlmostEqual(20.0, v, 4)
    
    t,v = obj.getNearestConvertedParameterValue("MMM", (14.0*math.pi/180.0, 60.08*math.pi/180.0))
    self.assertEqual(_rave.RaveValueType_DATA, t)
    self.assertAlmostEqual(60.0, v, 4)
    
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
        self.assertEqual(len(val[1]), len(result))
        self.assertEqual(val[1][0], result[0])
        self.assertEqual(val[1][1], result[1])
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
        
    self.assertEqual(2, obj.getNumberOfQualityFields())
    self.assertEqual("field1", obj.getQualityField(0).getAttribute("what/name"))
    obj.removeQualityField(0)
    self.assertEqual(1, obj.getNumberOfQualityFields())
    self.assertEqual("field2", obj.getQualityField(0).getAttribute("what/name"))
    
  def test_addOrReplaceQualityField(self):
    obj = _polarscan.new()
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field1.addAttribute("how/task", "field1")
    field2.addAttribute("how/task", "field2")
    
    obj.addQualityField(field1)
    obj.addQualityField(field2)
        
    self.assertEqual(2, obj.getNumberOfQualityFields())
    self.assertEqual("field1", obj.getQualityField(0).getAttribute("what/name"))
    obj.removeQualityField(0)
    self.assertEqual(1, obj.getNumberOfQualityFields())
    self.assertEqual("field2", obj.getQualityField(0).getAttribute("what/name"))
    
  def test_addOrReplaceQualityField(self):
    obj = _polarscan.new()
    field1 = _ravefield.new()
    field2 = _ravefield.new()
    field2_1 = _ravefield.new()
    field1.addAttribute("how/task", "field1")
    field2.addAttribute("how/task", "field2")
    field2_1.addAttribute("how/task", "field2")
    
    obj.addOrReplaceQualityField(field1)
    obj.addOrReplaceQualityField(field2)
    self.assertEqual(2, obj.getNumberOfQualityFields())
    self.assertTrue(field1 == obj.getQualityField(0))
    self.assertTrue(field2 == obj.getQualityField(1))
    
    obj.addOrReplaceQualityField(field2_1)
    self.assertEqual(2, obj.getNumberOfQualityFields())
    self.assertTrue(field1 == obj.getQualityField(0))
    self.assertTrue(field2_1 == obj.getQualityField(1))
        
  def test_add_how_array_attribute_long(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.int32))
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertEqual(0, result[0])
    self.assertEqual(3, result[3])
    self.assertEqual(5, result[5])
    self.assertEqual(9, result[9])

  def test_shift_how_array_attribute_long_right(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.int32))
    obj.shiftAttribute("how/something", 1)
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertEqual(9, result[0])
    self.assertEqual(2, result[3])
    self.assertEqual(4, result[5])
    self.assertEqual(8, result[9])

  def test_shift_how_array_attribute_long_0(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.int32))
    obj.shiftAttribute("how/something", 0)
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertEqual(0, result[0])
    self.assertEqual(3, result[3])
    self.assertEqual(5, result[5])
    self.assertEqual(9, result[9])

  def test_shift_how_array_attribute_long_left(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.int32))
    obj.shiftAttribute("how/something", -1)
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertEqual(1, result[0])
    self.assertEqual(4, result[3])
    self.assertEqual(6, result[5])
    self.assertEqual(0, result[9])

  def test_shift_how_array_attribute_float_right(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.float64))
    obj.shiftAttribute("how/something", 1)
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertAlmostEqual(9, result[0], 4)
    self.assertAlmostEqual(2, result[3], 4)
    self.assertAlmostEqual(4, result[5], 4)
    self.assertAlmostEqual(8, result[9], 4)

  def test_shift_how_array_attribute_float_0(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.float64))
    obj.shiftAttribute("how/something", 0)
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertAlmostEqual(0, result[0], 4)
    self.assertAlmostEqual(3, result[3], 4)
    self.assertAlmostEqual(5, result[5], 4)
    self.assertAlmostEqual(9, result[9], 4)

  def test_shift_how_array_attribute_float_left(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.float64))
    obj.shiftAttribute("how/something", -1)
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertAlmostEqual(1, result[0], 4)
    self.assertAlmostEqual(4, result[3], 4)
    self.assertAlmostEqual(6, result[5], 4)
    self.assertAlmostEqual(0, result[9], 4)

  def test_shiftData(self):
    obj = _polarscan.new()
    p1 = _polarscanparam.new()
    p1.quantity="DBZH"
    p2 = _polarscanparam.new()
    p2.quantity="TH"
    f1 = _ravefield.new()
    f2 = _ravefield.new()
    f3 = _ravefield.new()
    p1.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    p2.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f1.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f2.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f3.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    p1.addQualityField(f1)
    p2.addQualityField(f2)
    obj.addParameter(p1)
    obj.addParameter(p2)
    obj.addQualityField(f3)
    
    obj.shiftData(1)
    
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getParameter("DBZH").getData()).all())
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getParameter("DBZH").getQualityField(0).getData()).all())
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getParameter("TH").getData()).all())
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getParameter("TH").getQualityField(0).getData()).all())
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getQualityField(0).getData()).all())


  def test_shiftData_neg(self):
    obj = _polarscan.new()
    p1 = _polarscanparam.new()
    p1.quantity="DBZH"
    p2 = _polarscanparam.new()
    p2.quantity="TH"
    f1 = _ravefield.new()
    f2 = _ravefield.new()
    f3 = _ravefield.new()
    p1.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    p2.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f1.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f2.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    f3.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    p1.addQualityField(f1)
    p2.addQualityField(f2)
    obj.addParameter(p1)
    obj.addParameter(p2)
    obj.addQualityField(f3)

    obj.shiftData(-1)
    
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getParameter("DBZH").getData()).all())
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getParameter("DBZH").getQualityField(0).getData()).all())
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getParameter("TH").getData()).all())
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getParameter("TH").getQualityField(0).getData()).all())
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getQualityField(0).getData()).all())

  def test_shiftDataAndAttributes(self):
    obj = _polarscan.new()
    p1 = _polarscanparam.new()
    p1.quantity="DBZH"
    p1.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.addParameter(p1)
    
    obj.addAttribute("how/elangles", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/startazA", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/stopazA", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/startazT", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/stopazT", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/startelA", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/stopelA", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/startelT", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/stopelT", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/TXpower", numpy.array([0,1,2,3], numpy.float64))
    
    obj.shiftDataAndAttributes(1)
    
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getParameter("DBZH").getData()).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/elangles")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/startazA")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/stopazA")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/startazT")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/stopazT")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/startelA")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/stopelA")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/startelT")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/stopelT")).all())
    self.assertTrue((numpy.array([3,0,1,2],numpy.uint8)==obj.getAttribute("how/TXpower")).all())

  def test_shiftDataAndAttributes_neg(self):
    obj = _polarscan.new()
    p1 = _polarscanparam.new()
    p1.quantity="DBZH"
    p1.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.addParameter(p1)
    
    obj.addAttribute("how/elangles", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/startazA", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/stopazA", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/startazT", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/stopazT", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/startelA", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/stopelA", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/startelT", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/stopelT", numpy.array([0,1,2,3], numpy.float64))
    obj.addAttribute("how/TXpower", numpy.array([0,1,2,3], numpy.float64))
    
    obj.shiftDataAndAttributes(-1)
    
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getParameter("DBZH").getData()).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/elangles")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/startazA")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/stopazA")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/startazT")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/stopazT")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/startelA")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/stopelA")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/startelT")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/stopelT")).all())
    self.assertTrue((numpy.array([1,2,3,0],numpy.uint8)==obj.getAttribute("how/TXpower")).all())

  def test_add_how_array_attribute_double(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", numpy.arange(10).astype(numpy.float32))
    result = obj.getAttribute("how/something")
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertEqual(10, len(result))
    self.assertAlmostEqual(0.0, result[0], 2)
    self.assertAlmostEqual(3.0, result[3], 2)
    self.assertAlmostEqual(5.0, result[5], 2)
    self.assertAlmostEqual(9.0, result[9], 2)
      
  def test_hasAttribute(self):
    obj = _polarscan.new()
    obj.addAttribute("how/something", 1.0)
    obj.addAttribute("how/something2", "jupp")
    self.assertEqual(True, obj.hasAttribute("how/something"))
    self.assertEqual(True, obj.hasAttribute("how/something2"))
    self.assertEqual(False, obj.hasAttribute("how/something3"))
    try:
      obj.hasAttribute(None)
      self.fail("Expected TypeError")
    except TypeError:
      pass
  
  def test_howSubgroupAttribute(self):
    obj = _polarscan.new()
    param = _polarscanparam.new()
    param.quantity="DBZH"
    data = numpy.zeros((4,5), numpy.int8)
    param.setData(data)
    obj.addParameter(param)
  
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
      
#     obj.date="20200229"
#     obj.time="110000"
#     obj.source="NOD:selul"
#     obj.elangle=1*math.pi/180.0
#     
#     import _raveio
#     rio=_raveio.new()
#     rio.object = obj
#     rio.save("/projects/baltrad/rave-py3/slask.h5")
#     
#     nobj = _raveio.open("/projects/baltrad/rave-py3/slask.h5").object
#     self.assertAlmostEqual(1.0, obj.getAttribute("how/something"), 2)
#     self.assertAlmostEqual(2.0, nobj.getAttribute("how/grp/something"), 2)
  
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
    self.assertEqual("f2", result.getAttribute("what/value"))
    
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
    except NameError:
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
    self.assertEqual("pf2", result.getAttribute("what/value"))
    
    result = obj.findQualityFieldByHowTask("se.smhi.f1")
    self.assertEqual("f1", result.getAttribute("what/value"))
    
    result = obj.findQualityFieldByHowTask("se.smhi.f2", "MMH")
    self.assertEqual("pf2-mmh", result.getAttribute("what/value"))
    
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
    self.assertEqual(None, result)
        
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
    self.assertEqual(10, f.xsize)
    for i in range(10):
      self.assertAlmostEqual(expected[i], f.getValue(i, 0)[1], 4)
    
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
    self.assertEqual(10, f.xsize)
    for i in range(10):
      self.assertAlmostEqual(expected[i], f.getValue(i, 0)[1], 4)
        
  def test_getMaxDistance(self):
    obj = _polarscan.new()
    obj.longitude = 60.0 * math.pi / 180.0
    obj.latitude = 12.0 * math.pi / 180.0
    obj.height = 0.0
    obj.rscale = 1000.0
    obj.elangle = (math.pi / 180.0)*0.5
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((10, 10), numpy.int8)
    param.setData(data)
    obj.addParameter(param)
    self.assertAlmostEqual(10999.45, obj.getMaxDistance(), 2)
        
  def test_getDistance(self):
    obj = _polarscan.new()
    obj.longitude = 12.0 * math.pi / 180.0
    obj.latitude = 60.0 * math.pi / 180.0
    obj.height = 0.0
    obj.rscale = 1000.0
    obj.elangle = (math.pi / 180.0)*0.5
    param = _polarscanparam.new()
    param.quantity="DBZH"    
    data = numpy.zeros((10, 10), numpy.int8)
    param.setData(data)
    obj.addParameter(param)
    self.assertAlmostEqual(222080.29, obj.getDistance((12.0 * math.pi / 180.0, 62.0 * math.pi / 180.0)), 2)

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()