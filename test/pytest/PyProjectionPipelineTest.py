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

Tests the projection pipeline module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2021-10-14
'''
import unittest
import os
import _projection
import _projectionpipeline
import math

def deg2rad(coord):
  return (coord[0]*math.pi/180.0, coord[1]*math.pi/180.0)

def rad2deg(coord):
  return (coord[0]*180.0/math.pi, coord[1]*180.0/math.pi)

class PyProjectionPipelineTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    first = _projection.new("x", "y", "+proj=latlon +ellps=WGS84")
    second = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    
    obj = _projectionpipeline.new(first, second)
    
    self.assertEqual(first.definition, obj.first.definition)
    self.assertEqual(second.definition, obj.second.definition)

  def test_attribute_visibility(self):
    first = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    second = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    obj = _projectionpipeline.new(first, second)
    alist = dir(obj)
    
    attrs = ['first', 'second']
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_fwd_lonlat_to_cartesian(self):
    first = _projection.new("x", "y", "+proj=latlong +ellps=WGS84")
    second = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")

    pipeline = _projectionpipeline.new(first, second)
    
    xy = pipeline.fwd(deg2rad((12.8544, 56.3675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertAlmostEqual(0.0, xy[1], 4)

  def test_inv_lonlat_to_cartesian(self):
    first = _projection.new("x", "y", "+proj=latlong +ellps=WGS84")
    second = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")

    pipeline = _projectionpipeline.new(first, second)
    
    xy = rad2deg(pipeline.inv((0.0, 0.0)))
    self.assertAlmostEqual(12.8544, xy[0], 4)
    self.assertAlmostEqual(56.3675, xy[1], 3)

  def test_createDefaultLonLatPipeline(self):
    other = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")

    pipeline = _projectionpipeline.createDefaultLonLatPipeline(other)
    
    xy = pipeline.fwd(deg2rad((12.8544, 56.3675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertAlmostEqual(0.0, xy[1], 4)

if __name__=="__main__":
  unittest.main()
  
    