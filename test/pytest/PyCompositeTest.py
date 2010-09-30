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

Tests the composite module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2010-01-29
'''
import unittest
import os
import _cartesian
import _pycomposite
import _rave
import _area
import _projection
import _raveio
import math
import string
import numpy

class PyCompositeTest(unittest.TestCase):
  SWEDISH_VOLUMES = ["fixtures/pvol_seang_20090501T120000Z.h5",
                     "fixtures/pvol_searl_20090501T120000Z.h5",
                     "fixtures/pvol_sease_20090501T120000Z.h5",
                     "fixtures/pvol_sehud_20090501T120000Z.h5",
                     "fixtures/pvol_sekir_20090501T120000Z.h5",
                     "fixtures/pvol_sekkr_20090501T120000Z.h5",
                     "fixtures/pvol_selek_20090501T120000Z.h5",
                     "fixtures/pvol_selul_20090501T120000Z.h5",
                     "fixtures/pvol_seosu_20090501T120000Z.h5",
                     "fixtures/pvol_seovi_20090501T120000Z.h5",
                     "fixtures/pvol_sevar_20090501T120000Z.h5",
                     "fixtures/pvol_sevil_20090501T120000Z.h5"]


  #SWEDISH_VOLUMES = ["fixtures/pvol_sevar_20090501T120000Z.h5"]
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _pycomposite.new()
    
    isscan = string.find(`type(obj)`, "CompositeCore")
    self.assertNotEqual(-1, isscan)

  def test_attribute_visibility(self):
    attrs = ['height', 'product', 'quantity', 'date', 'time']
    obj = _pycomposite.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_height(self):
    obj = _pycomposite.new()
    self.assertAlmostEquals(1000.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEquals(1.0, obj.height, 4)

  def test_product(self):
    obj = _pycomposite.new()
    self.assertEquals(_rave.Rave_ProductType_PCAPPI, obj.product)

  def test_quantity(self):
    obj = _pycomposite.new()
    self.assertEquals("DBZH", obj.quantity)
    obj.quantity = "MMH"
    self.assertEquals("MMH", obj.quantity)
    

  def test_nearest(self):
    generator = _pycomposite.new()
    
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
    
    generator.quantity = "DBZH"
    generator.product = _rave.Rave_ProductType_PCAPPI
    generator.height = 1000.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    self.assertEquals("DBZH", result.quantity)
    self.assertEquals("120000", result.time)
    self.assertEquals("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEquals(1000.0, prodpar, 4)
    self.assertEquals(_rave.Rave_ProductType_PCAPPI, result.product)
    self.assertEquals(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEquals("nrd2km", result.source);
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite.h5"
    ios.save()

  # To verify ticket 96
  def test_nearest_gmapproj(self):
    generator = _pycomposite.new()

    a = _area.new()
    a.id = "eua_gmaps"
    a.xsize = 800
    a.ysize = 1090
    a.xscale = 6223.0
    a.yscale = 6223.0
    #               llX           llY            urX        urY
    a.extent = (-3117.83526,-6780019.83039,4975312.43200,3215.41216)
    # You can also add  +nadgrids=@null, for usage see PROJ.4 documentation
    a.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +x_0=1335833 +y_0=-11000715 +a=6378137.0 +b=6378137.0 +no_defs +datum=WGS84")

    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
    
    generator.quantity = "DBZH"
    generator.product = _rave.Rave_ProductType_PCAPPI
    generator.height = 1000.0
    result = generator.nearest(a)
    
    result.time = "120000"
    result.date = "20090501"
    result.source = "eua_gmaps"
    
    ios = _raveio.new()
    ios.object = result
    ios.save("swecomposite_gmap.h5")

  def test_nearest_ppi(self):
    generator = _pycomposite.new()
    
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
    
    generator.quantity = "DBZH"
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    self.assertEquals("DBZH", result.quantity)
    self.assertEquals("120000", result.time)
    self.assertEquals("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEquals(0.0, prodpar, 4)
    self.assertEquals(_rave.Rave_ProductType_PPI, result.product)
    self.assertEquals(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEquals("nrd2km", result.source);
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi.h5"
    ios.save()
    
  def test_nearest_ppi_fromscans(self):
    generator = _pycomposite.new()
    
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      scan = rio.object.getScanClosestToElevation(0.0, 0)
      generator.add(scan)
    
    generator.quantity = "DBZH"
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    self.assertEquals("DBZH", result.quantity)
    self.assertEquals("120000", result.time)
    self.assertEquals("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEquals(0.0, prodpar, 4)
    self.assertEquals(_rave.Rave_ProductType_PPI, result.product)
    self.assertEquals(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEquals("nrd2km", result.source);
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi_fromscan.h5"
    ios.save()    