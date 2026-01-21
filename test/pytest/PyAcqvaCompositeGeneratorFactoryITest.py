'''
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the py acqva composite generatory factory.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2024-12-15
'''
import unittest
import os
import _compositearguments
import _compositegenerator
import _area
import _raveio
import _projection
import _polarscan, _polarvolume, _ravefield
import _acqvacompositegeneratorfactory
import _raveproperties, _odimsources
import string
import math, numpy

class PyAcqvaCompositeGeneratorFactoryITest(unittest.TestCase):
  ODIM_SOURCE_FIXTURE="fixtures/odim_sources_fixture.xml"

  def setUp(self):
    pass

  def tearDown(self):
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
    else:
      raise Exception("No such area")
    
    return area

  def test_generate(self):
    classUnderTest = _acqvacompositegeneratorfactory.new()
    args = _compositearguments.new()
    args.area = self.create_area("eua_gmaps")
    args.product = "PPI"
    args.time = "120000"
    args.date = "20090501"
    args.addParameter("DBZH", 0.1, -30.0)

    seang = _raveio.open("fixtures/pvol_seang_20090501T120000Z.h5").object
    searl = _raveio.open("fixtures/pvol_searl_20090501T120000Z.h5").object

    for v in [seang, searl]:
      for i in range(v.getNumberOfScans()):
        scan = v.getScan(i)
        qdata = numpy.ones(scan.getParameter("DBZH").getData().shape, 'b')
        qfield = _ravefield.new()
        qfield.setData(qdata)
        qfield.addAttribute("how/task", "se.smhi.acqva")
        scan.addQualityField(qfield)
      args.addObject(v)

    args.addQualityFlag("se.smhi.composite.distance.radar")

    result = classUnderTest.generate(args);
    rio = _raveio.new()
    rio.object = result
    rio.save("acqva_factory_test.h5")

    self.assertEqual("QMAXIMUM", result.getAttribute("how/camethod"))


  def Xtest_generate_no_maps(self):
    classUnderTest = _acqvacompositegeneratorfactory.new()
    args = _compositearguments.new()
    args.area = self.create_area("eua_gmaps")
    args.product = "PPI"
    args.time = "120000"
    args.date = "20090501"
    args.addParameter("DBZH", 0.1, -30.0)

    seang = _raveio.open("fixtures/pvol_seang_20090501T120000Z.h5").object
    searl = _raveio.open("fixtures/pvol_searl_20090501T120000Z.h5").object

    for v in [seang, searl]:
      args.addObject(v)

    args.addQualityFlag("se.smhi.composite.distance.radar")

    result = classUnderTest.generate(args);
    rio = _raveio.new()
    rio.object = result
    rio.save("acqva_factory_test_no_maps.h5")

    self.assertEqual("QMAXIMUM", result.getAttribute("how/camethod"))

  def test_generate_with_featuremaps(self):
    classUnderTest = _acqvacompositegeneratorfactory.new()
    args = _compositearguments.new()
    args.area = self.create_area("eua_gmaps")
    args.product = "PPI"
    args.time = "120000"
    args.date = "20090501"
    args.addParameter("DBZH", 0.1, -30.0)

    properties = _raveproperties.new()
    properties.set("rave.acqva.featuremap.dir", "/san1/acqva/featuremaps",)
    properties.set("rave.acqva.featuremap.allow.missing", True)
    properties.set("rave.acqva.featuremap.use_default", True)
    properties.set("rave.acqva.featuremap.use_yearmonth", True)
    properties.sources = _odimsources.load(self.ODIM_SOURCE_FIXTURE)
    classUnderTest.setProperties(properties)

    seang = _raveio.open("fixtures/pvol_seang_20090501T120000Z.h5").object
    searl = _raveio.open("fixtures/pvol_searl_20090501T120000Z.h5").object

    for v in [seang, searl]:
      args.addObject(v)

    args.addQualityFlag("se.smhi.composite.distance.radar")

    result = classUnderTest.generate(args);
    rio = _raveio.new()
    rio.object = result
    rio.save("acqva_factory_test_no_maps.h5")

    self.assertEqual("QMAXIMUM", result.getAttribute("how/camethod"))
