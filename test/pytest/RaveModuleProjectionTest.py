'''
Created on Oct 20, 2009
@author: Anders Henja
'''
import unittest
import os
import _rave
import string
import numpy
import math

class RaveModuleProjectionTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewProjection(self):
    obj = _rave.projection("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    
    istransform = string.find(`type(obj)`, "ProjectionCore")
    self.assertNotEqual(-1, istransform) 
    self.assertEqual("x", obj.id);
    self.assertEqual("y", obj.description);
    self.assertEqual("+proj=latlong +ellps=WGS84 +datum=WGS84", obj.definition);

  def testInvalidProjection(self):
    try:
      _rave.projection("x", "y", "+proj=unknown + ellps=WGS84")
      self.fail("Expected ValueError")
    except ValueError, e:
      pass
    
  def testProjectFromCartesianToLonLat(self):
    cproj = _rave.projection("cartesian", "cartesian", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    llproj = _rave.projection("lonlat", "lonlat", "+proj=longlat +ellps=WGS84 +datum=WGS84")
    inx = 60.0 * math.pi / 180.0
    iny = 14.0 * math.pi / 180.0
    x,y = llproj.transform(cproj, (inx,iny))
    
    x,y = cproj.transform(llproj, (x,y))
    
    outx = x * 180.0 / math.pi
    outy = y * 180.0 / math.pi
    
    self.assertAlmostEquals(60.0, outx, 4)
    self.assertAlmostEquals(14.0, outy, 4)
      
  # Need some better tests for verifying that projectioning works as expected...

  def transformTo(self, cproj, llproj, coord):
    x,y = cproj.transform(llproj, coord)
    x = x*180.0/math.pi
    y = y*180.0/math.pi
    return (x,y)


if __name__=="__main__":
  unittest.main()
  
    