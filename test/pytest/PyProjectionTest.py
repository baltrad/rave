'''
Created on Oct 20, 2009
@author: Anders Henja
'''
import unittest
import os
import _projection
import string
import numpy
import math

def deg2rad(coord):
  return (coord[0]*math.pi/180.0, coord[1]*math.pi/180.0)

def rad2deg(coord):
  return (coord[0]*180.0/math.pi, coord[1]*180.0/math.pi)

class RaveModuleProjectionTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewProjection(self):
    obj = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    
    istransform = string.find(`type(obj)`, "ProjectionCore")
    self.assertNotEqual(-1, istransform) 
    self.assertEqual("x", obj.id)
    self.assertEqual("y", obj.description)
    self.assertEqual("+proj=latlong +ellps=WGS84 +datum=WGS84", obj.definition)

  def testInvalidProjection(self):
    try:
      _projection.new("x", "y", "+proj=unknown + ellps=WGS84")
      self.fail("Expected ValueError")
    except ValueError, e:
      pass
    
  def testProjectFromCartesianToLonLat(self):
    cproj = _projection.new("cartesian", "cartesian", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    llproj = _projection.new("lonlat", "lonlat", "+proj=longlat +ellps=WGS84 +datum=WGS84")
    inx = 60.0 * math.pi / 180.0
    iny = 14.0 * math.pi / 180.0
    x,y = llproj.transform(cproj, (inx,iny))
    
    x,y = cproj.transform(llproj, (x,y))
    
    outx = x * 180.0 / math.pi
    outy = y * 180.0 / math.pi
    
    self.assertAlmostEquals(60.0, outx, 4)
    self.assertAlmostEquals(14.0, outy, 4)

  # Test that we move in the right direction around the center
  def testInv_gnom(self):
    cproj = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544")
    
    ll = rad2deg(cproj.inv((0.0,0.0)))
    self.assertAlmostEquals(12.8544, ll[0], 4)
    self.assertAlmostEquals(56.3675, ll[1], 4)
    
    ll = rad2deg(cproj.inv((0.0, 1000.0)))
    self.assertAlmostEquals(12.8544, ll[0], 4)
    self.assertTrue(ll[1] > 56.3675)
    
    ll = rad2deg(cproj.inv((0.0, -1000.0)))
    self.assertAlmostEquals(12.8544, ll[0], 4)
    self.assertTrue(ll[1] < 56.3675)
    
    ll = rad2deg(cproj.inv((1000.0,0.0)))
    self.assertTrue(ll[0] > 12.8544)
    self.assertAlmostEquals(56.3675, ll[1], 4)

    ll = rad2deg(cproj.inv((-1000.0,0.0)))
    self.assertTrue(ll[0] < 12.8544)
    self.assertAlmostEquals(56.3675, ll[1], 4)

  def testFwd_gnom(self):
    cproj = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544")

    xy = cproj.fwd(deg2rad((12.8544, 56.3675)))
    self.assertAlmostEquals(0.0, xy[0], 4)
    self.assertAlmostEquals(0.0, xy[1], 4)
    
    xy = cproj.fwd(deg2rad((12.8544, 56.5675)))
    self.assertAlmostEquals(0.0, xy[0], 4)
    self.assertTrue(xy[1] > 0.0)

    xy = cproj.fwd(deg2rad((12.8544, 56.1675)))
    self.assertAlmostEquals(0.0, xy[0], 4)
    self.assertTrue(xy[1] < 0.0)

    xy = cproj.fwd(deg2rad((12.8744, 56.3675)))
    self.assertTrue(xy[0] > 0.0)

    xy = cproj.fwd(deg2rad((12.8344, 56.3675)))
    self.assertTrue(xy[0] < 0.0)

if __name__=="__main__":
  unittest.main()
  
    