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
import _raveproperties
import string
import math, numpy

class PyAcqvaCompositeGeneratorFactoryTest(unittest.TestCase):
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

  def test_new(self):
    obj = _acqvacompositegeneratorfactory.new()
    iscorrect = str(type(obj)).find("CompositeGeneratorFactoryCore")
    self.assertNotEqual(-1, iscorrect)

  def test_getName(self):
    obj = _acqvacompositegeneratorfactory.new()
    self.assertEqual("AcqvaCompositeGenerator", obj.getName())

  def test_getDefaultId(self):
    obj = _acqvacompositegeneratorfactory.new()
    self.assertEqual("acqva", obj.getDefaultId())

  def test_canHandle_products(self):
    classUnderTest = _acqvacompositegeneratorfactory.new()
    args = _compositearguments.new()
    args.product = "ACQVA"
    self.assertEqual(True, classUnderTest.canHandle(args))

    for product in ["PPI", "PCAPPI", "CAPPI", "SCAN", "ETOP"]:
      args = _compositearguments.new()
      args.product = product
      self.assertEqual(False, classUnderTest.canHandle(args))

  def test_create(self):
    classUnderTest = _acqvacompositegeneratorfactory.new()
    obj = classUnderTest.create()
    self.assertEqual("AcqvaCompositeGenerator", obj.getName())
    self.assertTrue(classUnderTest != obj)

  def test_properties(self):
    classUnderTest = _acqvacompositegeneratorfactory.new()
    props = _raveproperties.new()
    props.set("t.1", "YES")
    classUnderTest.setProperties(props)
    self.assertEqual("YES", classUnderTest.getProperties().get("t.1"))
    classUnderTest.setProperties(None)
    self.assertEqual(None, classUnderTest.getProperties())
