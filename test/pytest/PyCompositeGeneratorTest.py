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

Tests the py composite arguments module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2024-12-13
'''
import unittest
import os
import _compositearguments
import _compositegenerator
import _area
import _projection
import _polarscan
import _polarvolume
import _legacycompositegeneratorfactory
import _acqvacompositegeneratorfactory
import string
import math

class PyCompositeGeneratorTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _compositegenerator.new()
    iscorrect = str(type(obj)).find("CompositeGeneratorCore")
    self.assertNotEqual(-1, iscorrect)

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

  def test_generate_bad_argument(self):
    obj = _compositegenerator.new()
    args = _compositearguments.new()
    try:
      result = obj.generate(None)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_create(self):
    obj = _compositegenerator.create()
    ids = obj.getFactoryIDs()
    self.assertEqual(2, len(ids))
    self.assertTrue("legacy" in ids)
    self.assertTrue("acqva" in ids)

  def test_new_no_factories(self):
    obj = _compositegenerator.new()
    ids = obj.getFactoryIDs()
    self.assertEqual(0, len(ids))

  def test_register(self):
    obj = _compositegenerator.new()
    obj.register("1", _legacycompositegeneratorfactory.new())
    self.assertEqual(1, len(obj.getFactoryIDs()))
    obj.register("2", _acqvacompositegeneratorfactory.new())
    self.assertEqual(2, len(obj.getFactoryIDs()))
    self.assertTrue("1" in obj.getFactoryIDs())
    self.assertTrue("2" in obj.getFactoryIDs())

  def test_uregister(self):
    obj = _compositegenerator.new()
    obj.register("1", _legacycompositegeneratorfactory.new())
    obj.register("2", _acqvacompositegeneratorfactory.new())
    obj.unregister("1")
    self.assertEqual(1, len(obj.getFactoryIDs()))
    self.assertTrue("2" in obj.getFactoryIDs())
    obj.unregister("2")
    self.assertEqual(0, len(obj.getFactoryIDs()))

  def Xtest_generate(self):
    obj = _compositegenerator.create()
    obj.register("legacy", _legacycompositegeneratorfactory.new())

    args = _compositearguments.new()
    args.area = self.create_area("eua_gmaps")
    args.product = "PPI"
    args.elangle = 0.0
    args.time = "120000"
    args.date = "20090501"
    #args.strategy = "acqva"
    args.addParameter("DBZH", 0.1, -30.0)
    #args.quality_flags = ["se.smhi.composite.distance.radar"]

    # arguments are usualy method specific in some way
    args.addArgument("selection_method", "HEIGHT_ABOVE_SEALEVEL")
    args.addArgument("interpolation_method", "NEAREST")

    # args.addObject(_raveio.open(....).object)
    # args.addObject(_raveio.open(....).object)
    # args.addObject(_raveio.open(....).object)

    result = obj.generate(args)

