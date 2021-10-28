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

Tests the projection module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-20
'''
import unittest
import os
import _projection, _rave
import string
import numpy
import math

def deg2rad(coord):
  return (coord[0]*math.pi/180.0, coord[1]*math.pi/180.0)

def rad2deg(coord):
  return (coord[0]*180.0/math.pi, coord[1]*180.0/math.pi)

class PyProjectionTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    
    istransform = str(type(obj)).find("ProjectionCore")
    self.assertNotEqual(-1, istransform) 
    self.assertEqual("x", obj.id)
    self.assertEqual("y", obj.description)
    self.assertEqual("+proj=latlong +ellps=WGS84 +datum=WGS84", obj.definition)

  def test_attribute_visibility(self):
    attrs = ['id', 'description', 'definition']
    obj = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_invalid_projection(self):
    try:
      _projection.new("x", "y", "+proj=unknown + ellps=WGS84")
      self.fail("Expected ValueError")
    except ValueError:
      pass
    
  def testTransform_cartesianToLonLat(self):
    cproj = _projection.new("cartesian", "cartesian", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    llproj = _projection.new("lonlat", "lonlat", "+proj=longlat +ellps=WGS84 +datum=WGS84")
    inx = 60.0 * math.pi / 180.0
    iny = 14.0 * math.pi / 180.0
    x,y = llproj.transform(cproj, (inx,iny))
    
    x,y = cproj.transform(llproj, (x,y))
    
    outx = x * 180.0 / math.pi
    outy = y * 180.0 / math.pi
    
    self.assertAlmostEqual(60.0, outx, 4)
    self.assertAlmostEqual(14.0, outy, 4)

  def testTransformx_cartesianToLonLat(self):
    cproj = _projection.new("cartesian", "cartesian", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    llproj = _projection.new("lonlat", "lonlat", "+proj=longlat +ellps=WGS84 +datum=WGS84")
    inx = 60.0 * math.pi / 180.0
    iny = 14.0 * math.pi / 180.0
    x,y = llproj.transformx(cproj, (inx,iny))
    
    x,y = cproj.transformx(llproj, (x,y))
    
    outx = x * 180.0 / math.pi
    outy = y * 180.0 / math.pi
    
    self.assertAlmostEqual(60.0, outx, 4)
    self.assertAlmostEqual(14.0, outy, 4)

  # Test that we move in the right direction around the center
  def testInv_gnom(self):
    cproj = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544")
    
    ll = rad2deg(cproj.inv((0.0,0.0)))
    self.assertAlmostEqual(12.8544, ll[0], 4)
    self.assertAlmostEqual(56.3675, ll[1], 4)
    
    ll = rad2deg(cproj.inv((0.0, 1000.0)))
    self.assertAlmostEqual(12.8544, ll[0], 4)
    self.assertTrue(ll[1] > 56.3675)
    
    ll = rad2deg(cproj.inv((0.0, -1000.0)))
    self.assertAlmostEqual(12.8544, ll[0], 4)
    self.assertTrue(ll[1] < 56.3675)
    
    ll = rad2deg(cproj.inv((1000.0,0.0)))
    self.assertTrue(ll[0] > 12.8544)
    self.assertAlmostEqual(56.3675, ll[1], 4)

    ll = rad2deg(cproj.inv((-1000.0,0.0)))
    self.assertTrue(ll[0] < 12.8544)
    self.assertAlmostEqual(56.3675, ll[1], 4)

  def testFwd_gnom(self):
    cproj = _projection.new("gnom","gnom","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544")

    xy = cproj.fwd(deg2rad((12.8544, 56.3675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertAlmostEqual(0.0, xy[1], 4)
    
    xy = cproj.fwd(deg2rad((12.8544, 56.5675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertTrue(xy[1] > 0.0)

    xy = cproj.fwd(deg2rad((12.8544, 56.1675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertTrue(xy[1] < 0.0)

    xy = cproj.fwd(deg2rad((12.8744, 56.3675)))
    self.assertTrue(xy[0] > 0.0)

    xy = cproj.fwd(deg2rad((12.8344, 56.3675)))
    self.assertTrue(xy[0] < 0.0)

  def test_isLatLong_False(self):
    proj = _projection.new("x", "x", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    self.assertFalse(proj.isLatLong())
    llproj = _projection.new("lonlat", "lonlat", "+proj=longlat +ellps=WGS84 +datum=WGS84")

  def test_isLatLong_True(self):
    proj = _projection.new("x", "x", "+proj=lonlat +ellps=WGS84 +datum=WGS84")
    self.assertTrue(proj.isLatLong())

    proj = _projection.new("x", "x", "+proj=latlon +ellps=WGS84 +datum=WGS84")
    self.assertTrue(proj.isLatLong())

    proj = _projection.new("x", "x", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    self.assertTrue(proj.isLatLong())

    proj = _projection.new("x", "x", "+proj=longlat +ellps=WGS84 +datum=WGS84")
    self.assertTrue(proj.isLatLong())
    

  def test_gnom_with_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null");

    xy = proj.fwd(deg2rad((12.8544, 56.3675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    self.assertAlmostEqual(0.0, xy[1], 4)

  def test_gnom_without_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84");
    
    xy = proj.fwd(deg2rad((12.8544, 56.3675)))
    self.assertAlmostEqual(0.0, xy[0], 4)
    if _rave.isLegacyProjEnabled():
      self.assertAlmostEqual(0.0, xy[1], 4)
    else:
      self.assertAlmostEqual(-19759.7461, xy[1], 4)
        

  def test_stere_with_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=stere +R=6370997 +lat_0=90 +lon_0=0 +lat_ts=60 +nadgrids=@null");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(770641.8355, xy[0], 4)
    self.assertAlmostEqual(-3090875.5806, xy[1], 4)

  def test_stere_without_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=stere +R=6370997 +lat_0=90 +lon_0=0 +lat_ts=60");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(770641.8355, xy[0], 4)
    self.assertAlmostEqual(-3090875.5806, xy[1], 4)

  def test_longlat_with_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs +nadgrids=@null");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(14.0, rad2deg(xy)[0], 4)
    self.assertAlmostEqual(60.0, rad2deg(xy)[1], 4)

  def test_longlat_without_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(14.0, rad2deg(xy)[0], 4)
    self.assertAlmostEqual(60.0, rad2deg(xy)[1], 4)
      

  def test_laea_with_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=laea +R=6370997 +lat_0=60 +lon_0=15 +no_defs +nadgrids=@null");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(-55595.1437, xy[0], 4)
    self.assertAlmostEqual(420.1708, xy[1], 4)

  def test_laea_without_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=laea +R=6370997 +lat_0=60 +lon_0=15 +no_defs");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(-55595.1437, xy[0], 4)
    self.assertAlmostEqual(420.1708, xy[1], 4)

  def test_tmerc_with_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=tmerc +ellps=bessel +lon_0=15.808277777778 +x_0=1500000 +nadgrids=@null");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(1399118.9316, xy[0], 4)
    self.assertAlmostEqual(6654754.9388, xy[1], 4)

  def test_tmerc_without_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=tmerc +ellps=bessel +lon_0=15.808277777778 +x_0=1500000");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(1399118.9316, xy[0], 4)
    self.assertAlmostEqual(6654754.9388, xy[1], 4)

  def test_merc_with_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +no_defs +nadgrids=@null");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(1558472.8711, xy[0], 4)
    self.assertAlmostEqual(8399737.8898, xy[1], 4)

  def test_merc_without_nadgrids_null(self):
    proj = _projection.new("x","y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +no_defs");

    xy = proj.fwd(deg2rad((14.0, 60.0)))
    self.assertAlmostEqual(1558472.8711, xy[0], 4)
    self.assertAlmostEqual(8399737.8898, xy[1], 4)

  def test_getProjVersion(self):
    a = _projection.getProjVersion()
    self.assertTrue(len(a) > 0)

  def test_getDefaultLonLatPcsDef(self):
    self.assertEqual("+proj=longlat +ellps=WGS84 +datum=WGS84", _projection.getDefaultLonLatPcsDef())

  def test_setDefaultLonLatPcsDef(self):
    _projection.setDefaultLonLatPcsDef("+proj=longlat +ellps=WGS84")
    try:
        self.assertEqual("+proj=longlat +ellps=WGS84", _projection.getDefaultLonLatPcsDef())
    finally:
        _projection.setDefaultLonLatPcsDef("+proj=longlat +ellps=WGS84 +datum=WGS84") # Reset to not cause problems with other test cases

  def test_createDefaultLonLatProjection(self):
    p = _projection.createDefaultLonLatProjection()
    self.assertEqual("defaultLonLat", p.id)
    self.assertEqual("default lon/lat projection", p.description)
    self.assertEqual("+proj=longlat +ellps=WGS84 +datum=WGS84", p.definition)

  def test_proj_x(self):
    p = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +towgs84=0,0,0")
    lp = _projection.new("x","y", "+proj=longlat +ellps=WGS84 +datum=WGS84");
    
    result = rad2deg(p.transform(lp, (23785.9, -3.24566e+06)))
    self.assertAlmostEqual(14.41989, result[0], 5)
    self.assertAlmostEqual(59.56076, result[1], 5)

      
if __name__=="__main__":
  unittest.main()
  
    