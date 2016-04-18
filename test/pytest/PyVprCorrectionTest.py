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

Tests the vpr correction.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2015-03-23
'''
import unittest
import os
import _rave
import _vprcorrection
import _raveio
import string
import numpy

class PyVprCorrectionTest(unittest.TestCase):
  FIXTURE_VOLUME = "fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"
  FIXTURE_SELUL = "fixtures/selul_pvol_201503300845.h5"
  FIXTURE_NRD_SELUL = "fixtures/vpr-pvol-dbz.lul.20110701T120100Z.h5"
  FIXTURE_NRD_SELUL_TEMP_PROFILE = "fixtures/LUL_H1D_201107011200+001H00M.bpm"
  FIXTURE_NRD_SEKIR = "fixtures/vpr-pvol-dbz.kir.20110701T120100Z.h5"
  FIXTURE_NRD_SEKIR_TEMP_PROFILE = "fixtures/KIR_H1D_201107010100+001H00M.bpm"
  FIXTURE_TEMP_PROFILE = "fixtures/LUL_H1D_201106300000+001H00M.bpm"
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _vprcorrection.new()
    
    self.assertNotEqual(-1, string.find(`type(obj)`, "VprCorrectionCore"))

  def test_attribute_visibility(self):
    attrs = ['minReflectivity', 'heightLimit', 'profileHeight', 'minDistance', 'maxDistance']
    obj = _vprcorrection.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)
  
  def test_attribute_nonexisting(self):
    obj = _vprcorrection.new()
    try:
      obj.nonExistingAttribute = 10
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    
    try:
      x = obj.nonExistingAttribute
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
  
  def test_minReflectivity(self):
    obj = _vprcorrection.new()
    self.assertAlmostEqual(10.0, obj.minReflectivity, 4)
    obj.minReflectivity = 500.0
    self.assertAlmostEqual(500.0, obj.minReflectivity, 4)
    obj.minReflectivity = 600
    self.assertAlmostEqual(600.0, obj.minReflectivity, 4)
  
  def test_heightLimit(self):
    obj = _vprcorrection.new()
    self.assertAlmostEqual(10000.0, obj.heightLimit, 4)
    obj.heightLimit = 500.0
    self.assertAlmostEqual(500.0, obj.heightLimit, 4)
    obj.heightLimit = 600
    self.assertAlmostEqual(600.0, obj.heightLimit, 4)
    
  def test_profileHeight(self):
    obj = _vprcorrection.new()
    self.assertAlmostEqual(100.0, obj.profileHeight, 4)
    obj.profileHeight = 500.0
    self.assertAlmostEqual(500.0, obj.profileHeight, 4)
    obj.profileHeight = 600
    self.assertAlmostEqual(600.0, obj.profileHeight, 4)
    
  def test_minDistance(self):
    obj = _vprcorrection.new()
    self.assertAlmostEqual(1000.0, obj.minDistance, 4)
    obj.minDistance = 500.0
    self.assertAlmostEqual(500.0, obj.minDistance, 4)
    obj.minDistance = 600
    self.assertAlmostEqual(600.0, obj.minDistance, 4)

  def test_maxDistance(self):
    obj = _vprcorrection.new()
    self.assertAlmostEqual(25000.0, obj.maxDistance, 4)
    obj.maxDistance = 500.0
    self.assertAlmostEqual(500.0, obj.maxDistance, 4)
    obj.maxDistance = 600
    self.assertAlmostEqual(600.0, obj.maxDistance, 4)

  def test_plusTemperature(self):
    obj = _vprcorrection.new()
    self.assertAlmostEqual(4.0, obj.plusTemperature, 4)
    obj.plusTemperature = 5.0
    self.assertAlmostEqual(5.0, obj.plusTemperature, 4)

  def test_minusTemperature(self):
    obj = _vprcorrection.new()
    self.assertAlmostEqual(-5.0, obj.minusTemperature, 4)
    obj.minusTemperature = -4.0
    self.assertAlmostEqual(-4.0, obj.minusTemperature, 4)

  def test_dzdh(self):
    obj = _vprcorrection.new()
    self.assertAlmostEqual(-0.003, obj.dzdh, 3)
    obj.dzdh = -2.0
    self.assertAlmostEqual(-2.0, obj.dzdh, 2)

  def test_getNumberOfHeightIntervals(self):
    obj = _vprcorrection.new()
    obj.profileHeight = 100.0
    obj.heightLimit = 1200.0
    self.assertEqual(12, obj.getNumberOfHeightIntervals())    
    obj.heightLimit = 1250.0
    self.assertEqual(12, obj.getNumberOfHeightIntervals())    
    obj.heightLimit = 1300.0
    self.assertEqual(13, obj.getNumberOfHeightIntervals())    

  def test_getHeights(self):
    obj = _vprcorrection.new()
    obj.profileHeight = 100.0
    obj.heightLimit = 1200.0
    self.assertEquals(range(50,1200,100), obj.getHeights())

  def Xtest_getReflectivityArray(self):
    pvol = _raveio.open(self.FIXTURE_SELUL).object
    obj = _vprcorrection.new()
    obj.minReflectivity = 0.0
    obj.minDistance = 1000.0
    obj.maxDistance = 25000.0
    obj.profileHeight = 100.0
    obj.heightLimit = 5000.0
    arr = obj.getReflectivityArray(pvol)
  
  def Xtest_getIdealVpr(self):
    pvol = _raveio.open(self.FIXTURE_SELUL).object
    obj = _vprcorrection.new()
    obj.minReflectivity = 0.0
    obj.minDistance = 1000.0
    obj.maxDistance = 25000.0
    obj.profileHeight = 100.0
    obj.heightLimit = 5000.0
    
    
    htarr = [(50, 10.66), (150, 5.40), (225, 5.20), (200, 5.00), (225, 4.60), (230, 4.10), (260, 3.97), (275, 3.78), (300, 3.5), (325, 2.0), (350, 2.55), (450, 1.15), (550, -1.22), (650, -3.63), (750, -5.16), (850, -7.17), (950, -10.05)]
    #import _rave
    #_rave.setDebugLevel(_rave.Debug_RAVE_DEBUG)
    #heights = obj.getHeights()
    #arr = obj.getIdealVpr(pvol, htarr)
    #for i in range(len(heights)):
    #  print "%f, %f"%(heights[i], arr[i])

  def Xtest_getIdealVpr(self):
    pvol = _raveio.open(self.FIXTURE_SELUL).object
    obj = _vprcorrection.new()
    obj.minReflectivity = 0.0
    obj.minDistance = 1000.0
    obj.maxDistance = 25000.0
    obj.profileHeight = 100.0
    obj.heightLimit = 10000.0
    
    #import _rave
    #_rave.setDebugLevel(_rave.Debug_RAVE_DEBUG)
    htarr = _vprcorrection.readH1D(self.FIXTURE_NRD_SEKIR_TEMP_PROFILE)
    arr = obj.getIdealVpr(pvol, htarr)
    
    #print `arr`
    #for i in range(24):
    #  profilename = "/projects/baltrad/baltrad-vpr/SMHI_VPR/vpr_matlab/example_files/nwp_data/2011/07/01/LUL_H1D_20110701%02d00+001H00M.bpm"%i
    #  print "READING %d"%i
    #  htarr = _vprcorrection.readH1D(profilename)
    #  print "READ %d"%i
    #  #print `htarr`
    #  arr = obj.getIdealVpr(pvol, htarr)
      #heights = obj.getHeights()
    #  print "DONE IDEAL VPR %d"%i
      #for i in range(len(heights)):
      #  print "%f, %f"%(heights[i], arr[i])

  def Xtest_helper_readH1D(self):
    profile = _vprcorrection.readH1D(self.FIXTURE_TEMP_PROFILE)
    print `profile`
  
  def test_helper_lsqFirstOrder(self):
    x=[(0.1,0.1), (1.0, 2.0), (2.0, 3.0), (3.0, 2.0), (4.1,4.1)]
    # Returns constants for ax + b
    a, b = _vprcorrection.lsqFirstOrder(x)
    self.assertAlmostEqual(0.7962, a, 4)
    self.assertAlmostEqual(0.6157, b, 4)
    
#[0.0, 0.0, 0.0, -12.646153587581528, -10.898371050890635, -11.40954762056788, -11.214253617642424, -10.278507864657943, -10.159223824127873, -10.490322289950953, -11.399863017458086, -12.12928340134809, -12.818409219459605, -12.769991218123646, -12.977520024464658, -13.832916024750471, -13.388919940288343, -14.496466398273839, -14.86191514192582, -14.203767168403543, -14.303999766129593, -14.289050860797367, -17.61194011392538, -17.225726681910036, -12.234753833658486, -16.51041142128116, -16.39674776479123, -20.233972457225963, -20.35095223718162, -15.555893320904495, -10.994871511695381, -16.868084910717435, -21.36919905004588, -21.54273115151977, -20.166906328307032, -25.037209228380064, -24.56089377370598, -20.065979233426344, -19.993782234327007, -15.69315047175945, -15.317646840056465, -21.786841982887093, -20.68533319454479, -12.693333075463995, -12.669564959167822, -8.88749968542375, -9.731428269426857, -10.030768933227689, -9.90270240325297, -15.359999781864005
#0.0, -14.399999767559999, -12.399999737759998, -2.799999594719999, -3.1999996006799982, -4.799999624519998, -2.799999594719999, -2.799999594719999, -4.399999618559999, -3.1999996006799982, -1.9999995827999975, -3.5999996066399973, -1.1999995708799993, 0.8000004589199996, -2.39999958876, -2.39999958876, 1.2000004648800022, 0.40000045296000053, 0.40000045296000053, 1.2000004648800022, 1.2000004648800022, 2.000000476800004, 0.8000004589199996, 1.6000004708400013, 3.2000004946800047, 1.2000004648800022, 3.2000004946800047, 4.800000518520001, 3.2000004946800047, 3.2000004946800047, 4.400000512559998, 3.60000050064, 4.000000506600003, 4.400000512559998, 3.2000004946800047, 3.60000050064, 4.000000506600003, 4.000000506600003, 4.800000518520001, 3.2000004946800047, 2.800000488720002, 4.000000506600003, 4.000000506600003, 3.60000050064, 4.400000512559998, 4.800000518520001, 4.800000518520001, 4.800000518520001, 3.2000004946800047, 2.800000488720002
#[50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150, 2250, 2350, 2450, 2550, 2650, 2750, 2850, 2950, 3050, 3150, 3250, 3350, 3450, 3550, 3650, 3750, 3850, 3950, 4050, 4150, 4250, 4350, 4450, 4550, 4650, 4750, 4850, 4950]
    