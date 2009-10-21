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
    
    (nlat,nlon) = obj.daToLl((d, a))
    
    self.assertAlmostEquals(lat, nlat, 4)
    self.assertAlmostEquals(lon, nlon, 4)
    
  