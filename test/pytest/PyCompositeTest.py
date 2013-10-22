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

  QC_VOLUMES = [
                "fixtures/searl_qcvol_20120131T0000Z.h5",
                "fixtures/sease_qcvol_20120131T0000Z.h5",
                "fixtures/sevil_qcvol_20120131T0000Z.h5",
                "fixtures/seang_qcvol_20120131T0000Z.h5"
                ]
  
  MAX_VOLUMES = ["fixtures/prepared_max_fixture_hud.h5",
                 "fixtures/prepared_max_fixture_osu.h5",
                 "fixtures/prepared_max_fixture_ovi.h5"]
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _pycomposite.new()
    
    isscan = string.find(`type(obj)`, "CompositeCore")
    self.assertNotEqual(-1, isscan)

  def test_attribute_visibility(self):
    attrs = ['height', 'range', 'elangle', 'product', 'date', 'time', 'selection_method']
    obj = _pycomposite.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_height(self):
    obj = _pycomposite.new()
    self.assertAlmostEquals(1000.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEquals(1.0, obj.height, 4)
    
  def test_range(self):
    obj = _pycomposite.new()
    self.assertAlmostEquals(500000.0, obj.range, 4)
    obj.range = 1.0
    self.assertAlmostEquals(1.0, obj.range, 4)
 
  def test_elangle(self):
    obj = _pycomposite.new()
    self.assertAlmostEquals(0.0, obj.elangle, 4)
    obj.elangle = 1.0
    self.assertAlmostEquals(1.0, obj.elangle, 4)
    
  def test_product(self):
    obj = _pycomposite.new()
    self.assertEquals(_rave.Rave_ProductType_PCAPPI, obj.product)

  def test_product_valid(self):
    valid_products = [_rave.Rave_ProductType_PCAPPI, 
                      _rave.Rave_ProductType_CAPPI, 
                      _rave.Rave_ProductType_PPI, 
                      _rave.Rave_ProductType_PMAX, 
                      _rave.Rave_ProductType_MAX]
    obj = _pycomposite.new()
    for p in valid_products:
      obj.product = p
      self.assertEquals(p, obj.product)

  def test_product_invalid(self):
    invalid_products = [_rave.Rave_ProductType_SCAN, 
                        _rave.Rave_ProductType_ETOP, 
                        _rave.Rave_ProductType_RR, 
                        _rave.Rave_ProductType_VIL, 
                        _rave.Rave_ProductType_COMP,
                        _rave.Rave_ProductType_VP,
                        _rave.Rave_ProductType_RHI,
                        _rave.Rave_ProductType_XSEC,
                        _rave.Rave_ProductType_VSP,
                        _rave.Rave_ProductType_HSP,
                        _rave.Rave_ProductType_RAY,
                        _rave.Rave_ProductType_AZIM,
                        _rave.Rave_ProductType_QUAL,
                        _rave.Rave_ProductType_HSP]
    obj = _pycomposite.new()
    obj.product = _rave.Rave_ProductType_PCAPPI
    for p in invalid_products:
      obj.product = p
      self.assertEquals(_rave.Rave_ProductType_PCAPPI, obj.product)

  def test_selection_method(self):
    obj = _pycomposite.new()
    self.assertEquals(_pycomposite.SelectionMethod_NEAREST, obj.selection_method)
    obj.selection_method = _pycomposite.SelectionMethod_HEIGHT
    self.assertEquals(_pycomposite.SelectionMethod_HEIGHT, obj.selection_method)

  def test_date(self):
    obj = _pycomposite.new()
    self.assertEquals(None, obj.date)
    obj.date = "20130101"
    self.assertEquals("20130101", obj.date)
    obj.date = None
    self.assertEquals(None, obj.date)

  def test_time(self):
    obj = _pycomposite.new()
    self.assertEquals(None, obj.time)
    obj.time = "101010"
    self.assertEquals("101010", obj.time)
    obj.time = None
    self.assertEquals(None, obj.time)

  def test_selection_method_invalid(self):
    obj = _pycomposite.new()
    try:
      obj.selection_method = 99
      self.fail("Expected ValueError")
    except ValueError, e:
      pass
    self.assertEquals(_pycomposite.SelectionMethod_NEAREST, obj.selection_method)
  
  def test_addParameter(self):
    obj = _pycomposite.new()
    obj.addParameter("DBZH", 2.0, 3.0)
    result = obj.getParameter(0)
    self.assertEquals("DBZH", result[0])
    self.assertAlmostEquals(2.0, result[1], 4)
    self.assertAlmostEquals(3.0, result[2], 4)

  def test_addParameter_duplicate(self):
    obj = _pycomposite.new()
    obj.addParameter("DBZH", 2.0, 3.0)
    obj.addParameter("DBZH", 3.0, 4.0)
    self.assertEquals(1, obj.getParameterCount())
    result = obj.getParameter(0)
    self.assertEquals("DBZH", result[0])
    self.assertAlmostEquals(3.0, result[1], 4)
    self.assertAlmostEquals(4.0, result[2], 4)

  def test_addParameter_multiple(self):
    obj = _pycomposite.new()
    obj.addParameter("DBZH", 2.0, 3.0)
    obj.addParameter("MMH", 3.0, 4.0)
    self.assertEquals(2, obj.getParameterCount())
    result = obj.getParameter(0)
    self.assertEquals("DBZH", result[0])
    self.assertAlmostEquals(2.0, result[1], 4)
    self.assertAlmostEquals(3.0, result[2], 4)
    result = obj.getParameter(1)
    self.assertEquals("MMH", result[0])
    self.assertAlmostEquals(3.0, result[1], 4)
    self.assertAlmostEquals(4.0, result[2], 4)

  def test_hasParameter(self):
    obj = _pycomposite.new()
    self.assertEquals(False, obj.hasParameter("DBZH"))
    obj.addParameter("DBZH", 2.0, 3.0)
    self.assertEquals(True, obj.hasParameter("DBZH"))

  def test_getParameterCount(self):
    obj = _pycomposite.new()
    self.assertEquals(0, obj.getParameterCount())
    obj.addParameter("DBZH", 2.0, 3.0)
    self.assertEquals(1, obj.getParameterCount())
    obj.addParameter("MMH", 1.0, 2.0)
    self.assertEquals(2, obj.getParameterCount())

  def test_rix_nearest(self):
    generator = _pycomposite.new()
    
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    
    for fname in ["fixtures/eesyr_volume.h5", "fixtures/rix_volume.h5"]:
      rio = _raveio.open(fname)
      generator.add(rio.object)
    
    generator.addParameter("DBZH", 1.0, 0.0)
    
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    self.assertEquals("DBZH", result.getParameter("DBZH").quantity)
    self.assertAlmostEquals(1.0, result.getParameter("DBZH").gain, 4)
    self.assertAlmostEquals(0.0, result.getParameter("DBZH").offset, 4)
    self.assertEquals("120000", result.time)
    self.assertEquals("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEquals(0.0, prodpar, 4)
    self.assertEquals(_rave.Rave_ProductType_PPI, result.product)
    self.assertEquals(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEquals("nrd2km", result.source);
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "rixeecomposite.h5"
    ios.save()


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
    
    generator.addParameter("DBZH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_PCAPPI
    generator.height = 1000.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    self.assertEquals("DBZH", result.getParameter("DBZH").quantity)
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

  def test_nearest_multicomposite(self):
    generator = _pycomposite.new()
    
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    
    for fname in self.QC_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
    
    generator.addParameter("DBZH", 1.0, 0.0)
    generator.addParameter("TH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "000000"
    generator.date = "20120131"
    result = generator.nearest(a, ["fi.fmi.ropo.detector.classification", "se.smhi.detector.poo", "se.smhi.composite.distance.radar"])
    
    self.assertEquals("DBZH", result.getParameter("DBZH").quantity)
    self.assertEquals("000000", result.time)
    self.assertEquals("20120131", result.date)
    
    self.assertEquals("TH", result.getParameter("TH").quantity)
    self.assertEquals("000000", result.time)
    self.assertEquals("20120131", result.date)
    
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEquals(0.0, prodpar, 4)
    self.assertEquals(_rave.Rave_ProductType_PPI, result.product)
    self.assertEquals(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEquals("nrd2km", result.source);
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swemulticomposite.h5"
    ios.save()

  def test_nearest_max(self):
    generator = _pycomposite.new()
    
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    
    for fname in self.MAX_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
    
    generator.addParameter("DBZH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_MAX
    generator.height = 0.0
    generator.range = 0.0
    generator.time = "120000"
    generator.date = "20090501"    
    result = generator.nearest(a, ["se.smhi.composite.distance.radar"])
    
    self.assertEquals("DBZH", result.getParameter("DBZH").quantity)
    self.assertEquals("120000", result.time)
    self.assertEquals("20090501", result.date)
    
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEquals(0.0, prodpar, 4)
    self.assertEquals(_rave.Rave_ProductType_MAX, result.product)
    self.assertEquals(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEquals("nrd2km", result.source);
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swemaxcomposite.h5"
    ios.save()


  def test_nearest_pseudomax(self):
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
    
    generator.addParameter("DBZH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_PMAX
    generator.height = 1000.0
    generator.range = 70000.0
    generator.time = "120000"
    generator.date = "20090501"    
    result = generator.nearest(a, ["se.smhi.composite.distance.radar"])
    
    self.assertEquals("DBZH", result.getParameter("DBZH").quantity)
    self.assertEquals("120000", result.time)
    self.assertEquals("20090501", result.date)
    
    prodpar = result.getAttribute("what/prodpar")
    v = prodpar.split(",")
    vh = float(v[0])
    vr = float(v[1])
    self.assertAlmostEquals(1000.0, vh, 4)
    self.assertAlmostEquals(70000.0, vr, 4)
    #self.assertAlmostEquals(70000.0, prodpar, 4)
    self.assertEquals(_rave.Rave_ProductType_PMAX, result.product)
    self.assertEquals(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEquals("nrd2km", result.source);
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swepseudomaxcomposite.h5"
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
    
    generator.addParameter("DBZH", 0.4, -30.0)
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
    
    generator.addParameter("DBZH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    self.assertEquals("DBZH", result.getParameter("DBZH").quantity)
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
    
    generator.addParameter("DBZH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    self.assertEquals("DBZH", result.getParameter("DBZH").quantity)
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

  def test_nearest_ppi_fromscans_byHeight(self):
    generator = _pycomposite.new()
    generator.selection_method = _pycomposite.SelectionMethod_HEIGHT
    
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
    
    generator.addParameter("DBZH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    self.assertEquals("DBZH", result.getParameter("DBZH").quantity)
    self.assertEquals("120000", result.time)
    self.assertEquals("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEquals(0.0, prodpar, 4)
    self.assertEquals(_rave.Rave_ProductType_PPI, result.product)
    self.assertEquals(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEquals("nrd2km", result.source);
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi_fromscan_byheight.h5"
    ios.save()    

  def test_overlapping_objects(self):
    # tests ticket 355, composite selection criterion: nearest
    generator = _pycomposite.new()
    
    a = _area.new()
    a.id = "eua_gmaps"
    a.xsize = 800
    a.ysize = 1090
    a.xscale = 6223.0
    a.yscale = 6223.0
    #               llX           llY            urX        urY
    a.extent = (-3117.83526,-6780019.83039,4975312.43200,3215.41216)
    a.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +x_0=1335833 +y_0=-11000715 +a=6378137.0 +b=6378137.0 +no_defs +datum=WGS84")
    
    rio = _raveio.open("fixtures/pvol_seang_20090501T120000Z.h5")
    scan = rio.object.getScanClosestToElevation(40.0, 0) # Take highest elevation from angelholm
    generator.add(scan)
    
    rio = _raveio.open("fixtures/pvol_sekkr_20090501T120000Z.h5")
    scan = rio.object.getScanClosestToElevation(0.0, 0) # Take lowest elevation from karlskrona
    generator.add(scan)

    generator.addParameter("DBZH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a)
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_gmap_overlapping.h5"
    ios.save()

  def test_nearest_ppi_fromscans_adddistancequality(self):
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
    
    generator.addParameter("DBZH", 1.0, 0.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.nearest(a, ["se.smhi.composite.distance.radar"])
    
    field = result.getParameter("DBZH").getQualityField(0)
    self.assertEquals("se.smhi.composite.distance.radar", field.getAttribute("how/task"))
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi_distancequality.h5"
    ios.save()    

