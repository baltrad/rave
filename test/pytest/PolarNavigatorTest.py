'''
Created on Oct 21, 2009
@author: Anders Henja
'''
import unittest
import os
import _polarnav
import string
import numpy
import math

class PolarNavigatorTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNew(self):
    obj = _polarnav.new()
    
    ispolnav = string.find(`type(obj)`, "PolarNavigatorCore")
    self.assertNotEqual(-1, ispolnav) 
    self.assertAlmostEquals(6356780.0, obj.poleradius, 4)
    self.assertAlmostEquals(6378160.0, obj.equatorradius, 4)
    self.assertAlmostEquals((-3.9e-5)/1000, obj.dndh, 4)

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
