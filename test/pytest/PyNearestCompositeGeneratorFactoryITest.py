'''
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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

Integration tests for the nearest composite generatory factory to verify that it generates composites
properly.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-03-09
'''
import unittest
import os
import _compositearguments
import _compositegenerator
import _area
import _projection
import _polarscan
import _polarvolume
import _nearestcompositegeneratorfactory
import _raveproperties
import _raveio
import _rave
import _odimsources
import string
import math

class PyNearestCompositeGeneratorFactoryITest(unittest.TestCase):
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

  ODIM_SOURCE_FIXTURE="fixtures/odim_sources_fixture.xml"

  TEMPORARY_FILE="nearest_composite_generator_itest.h5"

  def setUp(self):
    try:
      if os.path.exists(self.TEMPORARY_FILE):
        os.unlink(self.TEMPORARY_FILE)
    except:
      pass

  def tearDown(self):
    try:
      if os.path.exists(self.TEMPORARY_FILE):
        os.unlink(self.TEMPORARY_FILE)
    except:
      pass

  def create_area(self, areaid):
    area = _area.new()
    if areaid == "eua_gmaps":
      area.id = "eua_gmaps"
      area.xsize = 800
      area.ysize = 1090
      area.xscale = 6223.0
      area.yscale = 6223.0
      #               llX           llY            urX        urY
      area.extent = (-3117.83526,-6780019.83039,4975312.43200,3215.41216)
      area.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +x_0=1335833 +y_0=-11000715 +a=6378137.0 +b=6378137.0 +no_defs +datum=WGS84")
    elif areaid == "nrd2km":
      area.id = "nrd2km"
      area.xsize = 848
      area.ysize = 1104
      area.xscale = 2000.0
      area.yscale = 2000.0
      area.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
      area.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")      
    else:
      raise Exception("No such area")
    
    return area

  def test_generate_ppi(self):
    obj = _nearestcompositegeneratorfactory.new()

    args = _compositearguments.new()
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      args.addObject(rio.object)
    args.addParameter("DBZH", 1.0, 0.0)
    args.product = "PPI"
    args.elangle = 0.0
    args.time = "120000"
    args.date = "20090501"
    args.area = self.create_area("nrd2km")

    # arguments are usualy method specific in some way
    args.addArgument("selection_method", "HEIGHT_ABOVE_SEALEVEL")
    args.addArgument("interpolation_method", "NEAREST")

    result = obj.generate(args)

    self.assertEqual(True, result.hasParameter("DBZH"))
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertAlmostEqual(0.0, result.getAttribute("what/prodpar"), 4)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)

    ios = _raveio.new()
    ios.object = result
    ios.filename = "nearest_swecomposite_ppi.h5"
    ios.save()

  def test_generate_ppi_with_all_param_settings(self):
    obj = _nearestcompositegeneratorfactory.new()

    args = _compositearguments.new()
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      args.addObject(rio.object)
    args.addParameter("DBZH", 1.0, 0.0, _rave.RaveDataType_DOUBLE, 255.0, 0.0)
    args.product = "PPI"
    args.elangle = 0.0
    args.time = "120000"
    args.date = "20090501"
    args.area = self.create_area("nrd2km")

    # arguments are usualy method specific in some way
    args.addArgument("selection_method", "HEIGHT_ABOVE_SEALEVEL")
    args.addArgument("interpolation_method", "NEAREST")

    result = obj.generate(args)

    self.assertEqual(True, result.hasParameter("DBZH"))
    param = result.getParameter("DBZH")
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertAlmostEqual(0.0, result.getAttribute("what/prodpar"), 4)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    self.assertAlmostEqual(1.0, param.gain, 4)
    self.assertAlmostEqual(0.0, param.offset, 4)
    self.assertAlmostEqual(255.0, param.nodata, 4)
    self.assertAlmostEqual(0.0, param.undetect, 4)
    self.assertEqual(_rave.RaveDataType_DOUBLE, param.datatype)

  def test_generate_ppi_RATE(self):
    import _rave
    _rave.setTrackObjectCreation(True)
    obj = _nearestcompositegeneratorfactory.new()
    properties = _raveproperties.new()
    properties.set("rave.rate.zr.coefficients", {"sekrn":(210.0, 1.7)})
    properties.sources = _odimsources.load(self.ODIM_SOURCE_FIXTURE)
    args = _compositearguments.new()
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      args.addObject(rio.object)
    args.addParameter("RATE", 1.0, 0.0, _rave.RaveDataType_DOUBLE, 255.0, 0.0)
    args.product = "PPI"
    args.elangle = 0.0
    args.time = "120000"
    args.date = "20090501"
    args.area = self.create_area("nrd2km")

    # arguments are usualy method specific in some way
    args.addArgument("selection_method", "HEIGHT_ABOVE_SEALEVEL")
    args.addArgument("interpolation_method", "NEAREST")

    obj.setProperties(properties)

    result = obj.generate(args)

    self.assertEqual(True, result.hasParameter("RATE"))
    param = result.getParameter("RATE")
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertAlmostEqual(0.0, result.getAttribute("what/prodpar"), 4)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    #print(param.getAttribute("how/product_parameters/json"))  #SHOW RATE ZR coefficients

  def test_generate_pcappi_nearest(self):
    obj = _nearestcompositegeneratorfactory.new()

    args = _compositearguments.new()
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      args.addObject(rio.object)
    args.addParameter("DBZH", 1.0, 0.0)
    args.product = "PCAPPI"
    args.elangle = 0.0
    args.height = 500.0
    args.time = "120000"
    args.date = "20090501"
    args.area = self.create_area("nrd2km")

    # arguments are usualy method specific in some way
    args.addArgument("selection_method", "NEAREST")

    result = obj.generate(args)

    self.assertEqual(True, result.hasParameter("DBZH"))
    self.assertEqual(_rave.Rave_ProductType_PCAPPI, result.product)
    self.assertAlmostEqual(500.0, result.getAttribute("what/prodpar"), 4)
    self.assertEqual("NEAREST", result.getAttribute("how/camethod"))
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)

    ios = _raveio.new()
    ios.object = result
    ios.filename = "nearest_swecomposite_pcappi.h5"
    ios.save()


  def test_generate_pcappi_byheight(self):
    obj = _nearestcompositegeneratorfactory.new()

    args = _compositearguments.new()
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      args.addObject(rio.object)
    args.addParameter("DBZH", 1.0, 0.0)
    args.product = "PCAPPI"
    args.elangle = 0.0
    args.height = 500.0
    args.time = "120000"
    args.date = "20090501"
    args.area = self.create_area("nrd2km")

    # arguments are usualy method specific in some way
    args.addArgument("selection_method", "HEIGHT_ABOVE_SEALEVEL")

    result = obj.generate(args)

    self.assertEqual(True, result.hasParameter("DBZH"))
    self.assertEqual(_rave.Rave_ProductType_PCAPPI, result.product)
    self.assertAlmostEqual(500.0, result.getAttribute("what/prodpar"), 4)
    self.assertEqual("MDE", result.getAttribute("how/camethod"))
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)

    ios = _raveio.new()
    ios.object = result
    ios.filename = "height_swecomposite_pcappi.h5"
    ios.save()

  def test_generate_max(self):
    obj = _nearestcompositegeneratorfactory.new()

    args = _compositearguments.new()
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      args.addObject(rio.object)
    args.addParameter("DBZH", 1.0, 0.0)
    args.product = "MAX"
    args.time = "120000"
    args.date = "20090501"
    args.area = self.create_area("nrd2km")

    # arguments are usualy method specific in some way
    args.addArgument("selection_method", "HEIGHT_ABOVE_SEALEVEL")

    result = obj.generate(args)

    self.assertEqual(True, result.hasParameter("DBZH"))
    self.assertEqual(_rave.Rave_ProductType_MAX, result.product)
    self.assertEqual("MAXIMUM", result.getAttribute("how/camethod"))
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)

    ios = _raveio.new()
    ios.object = result
    ios.filename = "max_swecomposite_pcappi.h5"
    ios.save()
