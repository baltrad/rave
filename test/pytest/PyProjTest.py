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

Tests the _proj module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2021-10-15
'''
import unittest
import os
import _proj
import string
import numpy
import math

def deg2rad(coord):
  return (coord[0]*math.pi/180.0, coord[1]*math.pi/180.0)

def rad2deg(coord):
  return (coord[0]*180.0/math.pi, coord[1]*180.0/math.pi)

class PyProjTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _proj.proj(["+proj=gnom", "+R=6371000.0", "+lat_0=56.3675", "+lon_0=12.8544", "+ellps=WGS84", "+datum=WGS84", "+nadgrids=@null"])
    self.assertTrue(str(type(obj)).find("Proj") >= 0)

  def test_invalid_projection(self):
    try:
      _proj.proj(["+proj=unknown", "+ellps=WGS84"])
      self.fail("Expected ValueError")
    except _proj.error:
      pass

  # Test that we move in the right direction around the center
  def test_invproj(self):
    cproj = _proj.proj(["+proj=gnom", "+R=6371000.0", "+lat_0=56.3675", "+lon_0=12.8544", "+ellps=WGS84", "+datum=WGS84", "+nadgrids=@null"])

    ll = rad2deg(cproj.invproj((0.0,0.0)))
    self.assertAlmostEqual(12.8544, ll[0], 4)
    self.assertAlmostEqual(56.3675, ll[1], 4)
    
    ll = rad2deg(cproj.invproj((0.0, 1000.0)))
    self.assertAlmostEqual(12.8544, ll[0], 4)
    self.assertTrue(ll[1] > 56.3675)
    
    ll = rad2deg(cproj.invproj((0.0, -1000.0)))
    self.assertAlmostEqual(12.8544, ll[0], 4)
    self.assertTrue(ll[1] < 56.3675)
    
    ll = rad2deg(cproj.invproj((1000.0,0.0)))
    self.assertTrue(ll[0] > 12.8544)
    self.assertAlmostEqual(56.3675, ll[1], 4)

    ll = rad2deg(cproj.invproj((-1000.0,0.0)))
    self.assertTrue(ll[0] < 12.8544)
    self.assertAlmostEqual(56.3675, ll[1], 4)

  def test_proj(self):
    cproj = _proj.proj(["+proj=gnom", "+R=6371000.0", "+lat_0=56.3675", "+lon_0=12.8544", "+ellps=WGS84", "+datum=WGS84", "+nadgrids=@null"])

    xy = cproj.proj(deg2rad((12.8544, 56.3675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertAlmostEqual(0.0, xy[1], 4)
    
    xy = cproj.proj(deg2rad((12.8544, 56.5675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertTrue(xy[1] > 0.0)

    xy = cproj.proj(deg2rad((12.8544, 56.1675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertTrue(xy[1] < 0.0)

    xy = cproj.proj(deg2rad((12.8744, 56.3675)))
    self.assertTrue(xy[0] > 0.0)

    xy = cproj.proj(deg2rad((12.8344, 56.3675)))
    self.assertTrue(xy[0] < 0.0)

if __name__=="__main__":
  unittest.main()
  
    