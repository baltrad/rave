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

Tests the PolarNav module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-21
'''
import unittest
import os
import _polarnav
import string
import numpy
import math

class PyPolarNavTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _polarnav.new()
    
    ispolnav = string.find(`type(obj)`, "PolarNavigatorCore")
    self.assertNotEqual(-1, ispolnav) 
    self.assertAlmostEquals(6356780.0, obj.poleradius, 4)
    self.assertAlmostEquals(6378160.0, obj.equatorradius, 4)
    self.assertAlmostEquals((-3.9e-5)/1000, obj.dndh, 4)

  def test_attribute_visibility(self):
    attrs = ['poleradius', 'equatorradius', 'lon0', 'lat0', 'alt0', 'dndh']
    obj = _polarnav.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_poleradius(self):
    obj = _polarnav.new()
    self.assertAlmostEquals(6356780.0, obj.poleradius, 4)
    obj.poleradius = 10.2
    self.assertAlmostEquals(10.2, obj.poleradius, 4)
  
  def test_equatorradius(self):
    obj = _polarnav.new()
    self.assertAlmostEquals(6378160.0, obj.equatorradius, 4)
    obj.equatorradius = 10.2
    self.assertAlmostEquals(10.2, obj.equatorradius, 4)

  def test_lon0(self):
    obj = _polarnav.new()
    self.assertAlmostEquals(0.0, obj.lon0, 4)
    obj.lon0 = 0.2
    self.assertAlmostEquals(0.2, obj.lon0, 4)

  def test_lat0(self):
    obj = _polarnav.new()
    self.assertAlmostEquals(0.0, obj.lat0, 4)
    obj.lat0 = 0.2
    self.assertAlmostEquals(0.2, obj.lat0, 4)

  def test_alt0(self):
    obj = _polarnav.new()
    self.assertAlmostEquals(0.0, obj.alt0, 4)
    obj.alt0 = 10.2
    self.assertAlmostEquals(10.2, obj.alt0, 4)

  def test_dndh(self):
    obj = _polarnav.new()
    self.assertAlmostEquals((-3.9e-5)/1000.0, obj.dndh, 4)
    obj.dndh = 0.2
    self.assertAlmostEquals(0.2, obj.dndh, 4)

  def test_getDistance(self):
    obj = _polarnav.new()
    obj.lat0 = 60.0 * math.pi/180.0
    obj.lon0 = 14.0 * math.pi/180.0
    
    result = obj.getDistance((61.0*math.pi/180.0, 14.0 * math.pi/180.0))
    
    self.assertAlmostEquals(111040.1, result, 1)

  # Define some clever tests for the APIs... Now they just verify that the returned
  # values are tuples
  def testLlToDa_and_daToLl(self):
    obj = _polarnav.new()
    obj.lat0 = 60.0 * math.pi / 180.0
    obj.lon0 = 14.0 * math.pi / 180.0
    obj.alt0 = 100.0
    
    lat = 61.0 * math.pi / 180.0
    lon = 15.0 * math.pi / 180.0
    
    d,a = obj.llToDa((lat,lon))
    
    nlat,nlon = obj.daToLl(d, a)
    
    self.assertAlmostEquals(lat, nlat, 4)
    self.assertAlmostEquals(lon, nlon, 4)
    
  def test_dhToRe(self):
    obj = _polarnav.new()
    obj.lat0 = 60.0 * math.pi / 180.0
    obj.lon0 = 12.0 * math.pi / 180.0
    obj.alt0 = 0.0

    r,e = obj.dhToRe(50000.0, 1000.0)
    self.assertAlmostEquals(50012.88, r, 2)
    self.assertAlmostEquals(0.976411, e*180.0/math.pi, 4)

  def test_deToRh(self):
    obj = _polarnav.new()
    obj.lat0 = 60.0 * math.pi / 180.0
    obj.lon0 = 12.0 * math.pi / 180.0
    obj.alt0 = 0.0

    r,h = obj.deToRh(50000.0, 0.976411*math.pi/180.0)
    self.assertAlmostEquals(50012.88, r, 2)
    self.assertAlmostEquals(1000.0, h, 2)

  def test_reToDh(self):
    obj = _polarnav.new()
    obj.lat0 = 60.0 * math.pi / 180.0
    obj.lon0 = 12.0 * math.pi / 180.0
    obj.alt0 = 0.0

    d,h = obj.reToDh(50012.88, 0.976411*math.pi/180.0)
    self.assertAlmostEquals(50000.0, d, 2)
    self.assertAlmostEquals(1000.0, h, 2)

  def test_daToLl(self):
    obj = _polarnav.new()
    obj.lat0 = 60.0 * math.pi / 180.0
    obj.lon0 = 12.0 * math.pi / 180.0
    obj.alt0 = 0.0    
    nlat,nlon = obj.daToLl(50000.0, 0.0)
    #print "nlat=%f,nlon=%f"%(nlat*180.0/math.pi, nlon*180.0/math.pi)

  def test_daToLl2(self):
    obj = _polarnav.new()
    obj.lat0 = 60.0 * math.pi / 180.0
    obj.lon0 = 14.0 * math.pi / 180.0
    obj.alt0 = 0.0
    
    lon,lat = obj.daToLl(9000.0, 0.0)
    #print "lon=%f, lat=%f"%(lon*180.0/math.pi, lat*180.0/math.pi)

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()
