'''
Copyright (C) 2013 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the cartesian composite module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2013-10-10
'''
import unittest
import os
import _cartesian
import _cartesiancomposite
import _rave
import _area
import _projection
import _raveio
import math
import string
import numpy

class PyCartesianCompositeTest(unittest.TestCase):
  FIXTURES=["fixtures/pcappi_gn_seang_20090501120000.h5",
            "fixtures/pcappi_gn_searl_20090501120000.h5",
            "fixtures/pcappi_gn_sease_20090501120000.h5",
            "fixtures/pcappi_gn_sehud_20090501120000.h5",
            "fixtures/pcappi_gn_sekir_20090501120000.h5",
            "fixtures/pcappi_gn_sekkr_20090501120000.h5",
            "fixtures/pcappi_gn_selek_20090501120000.h5",
            "fixtures/pcappi_gn_selul_20090501120000.h5",
            "fixtures/pcappi_gn_seosu_20090501120000.h5",
            "fixtures/pcappi_gn_seovi_20090501120000.h5",
            "fixtures/pcappi_gn_sevar_20090501120000.h5",
            "fixtures/pcappi_gn_sevil_20090501120000.h5"]
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _cartesiancomposite.new()
    self.assertNotEqual(-1, string.find(`type(obj)`, "CartesianCompositeCore"))
    
  def test_attribute_visibility(self):
    attrs = ['date', 'time', 'quantity', 'offset', 'gain']
    obj = _cartesiancomposite.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)
  

  def test_date(self):
    obj = _cartesiancomposite.new()
    self.assertEquals(None, obj.date)
    obj.date = "20120101"
    self.assertEquals("20120101", obj.date)
    obj.date = None
    self.assertEquals(None, obj.date)

  def test_time(self):
    obj = _cartesiancomposite.new()
    self.assertEquals(None, obj.time)
    obj.time = "100000"
    self.assertEquals("100000", obj.time)
    obj.time = None
    self.assertEquals(None, obj.time)
    
  def test_quantity(self):
    obj = _cartesiancomposite.new()
    self.assertEquals("DBZH", obj.quantity)
    obj.quantity = "TH"
    self.assertEquals("TH", obj.quantity)
    try:
      obj.quantity = None
      self.fail("Expected ValueError")
    except ValueError, e:
      pass
    self.assertEquals("TH", obj.quantity)

  def test_offset(self):
    obj = _cartesiancomposite.new()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    obj.offset = 2.5
    self.assertAlmostEquals(2.5, obj.offset, 4)
  
  def test_gain(self):
    obj = _cartesiancomposite.new()
    self.assertAlmostEquals(1.0, obj.gain, 4)
    obj.gain = 2.0
    self.assertAlmostEquals(2.0, obj.gain, 4)
    
  def test_method(self):
    values = [_cartesiancomposite.SelectionMethod_FIRST, 
              _cartesiancomposite.SelectionMethod_MINVALUE, 
              _cartesiancomposite.SelectionMethod_MAXVALUE,
              _cartesiancomposite.SelectionMethod_AVGVALUE,
              _cartesiancomposite.SelectionMethod_NEAREST]
    obj = _cartesiancomposite.new()
    self.assertEquals(_cartesiancomposite.SelectionMethod_FIRST, obj.method)
    for v in values:
      obj.method = v
      self.assertEquals(v, obj.method)
    
    try:
      obj.method = 99
      self.fail("Expected ValueError")
    except ValueError, e:
      pass

  def test_cartesian_objects(self):
    obj = _cartesiancomposite.new()
    self.assertEquals(0, obj.getNumberOfObjects())
    c1 = _cartesian.new()
    c2 = _cartesian.new()
    c3 = _cartesian.new()
    c4 = _cartesian.new()
    obj.add(c1)
    self.assertEquals(1, obj.getNumberOfObjects())
    obj.add(c2)
    obj.add(c3)
    obj.add(c4)
    self.assertEquals(4, obj.getNumberOfObjects())
    rc1 = obj.get(0)
    rc2 = obj.get(1)
    rc3 = obj.get(2)
    rc4 = obj.get(3)
    self.assertTrue(c1 == rc1)
    self.assertTrue(c2 == rc2)
    self.assertTrue(c3 == rc3)
    self.assertTrue(c4 == rc4)
    
  def test_nearest_first(self):
    a = self.create_area()
    
    obj = _cartesiancomposite.new()
    obj.method = _cartesiancomposite.SelectionMethod_FIRST
    for f in self.FIXTURES:
      obj.add(_raveio.open(f).object)
    obj.nodata = 255.0
    obj.undetect = 0.0
    result = obj.nearest(a)
    result.time = "120000"
    result.date = "20090501"
    result.source = "eua_gmaps"
    
    rio = _raveio.new()
    rio.object = result
    rio.save("cart_composite_first.h5")
    
  def create_area(self):
    a = _area.new()
    a.id = "eua_gmaps"
    a.xsize = 800
    a.ysize = 1090
    a.xscale = 6223.0
    a.yscale = 6223.0
    a.extent = (-3117.83526,-6780019.83039,4975312.43200,3215.41216)
    a.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +x_0=1335833 +y_0=-11000715 +a=6378137.0 +b=6378137.0 +no_defs +datum=WGS84")
    return a
  