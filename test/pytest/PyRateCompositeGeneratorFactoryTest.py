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

Tests the py rate composite generatory factory.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-01-31
'''
import unittest
import os
import _compositearguments
import _area
import _projection
import _polarscan
import _polarvolume
import _ratecompositegeneratorfactory
import _raveio
import string
import math

class PyRateCompositeGeneratorFactoryTest(unittest.TestCase):
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

  def test_new(self):
    obj = _ratecompositegeneratorfactory.new()
    iscorrect = str(type(obj)).find("CompositeGeneratorFactoryCore")
    self.assertNotEqual(-1, iscorrect)

  def test_getName(self):
    obj = _ratecompositegeneratorfactory.new()
    self.assertEqual("RateCompositeGenerator", obj.getName())

  def test_getDefaultId(self):
    obj = _ratecompositegeneratorfactory.new()
    self.assertEqual("rate", obj.getDefaultId())

  def Xtest_canHandle_products(self):
    # Rave_ProductType_MAX & NEAREST
    # Rave_ProductType_PMAX & NEAREST
    #
    # Rave_ProductType_MAX
    # Rave_ProductType_PPI
    # Rave_ProductType_PCAPPI
    # Rave_ProductType_CAPPI

    classUnderTest = _legacycompositegeneratorfactory.new()
    for product in ["PPI", "PCAPPI", "CAPPI"]:
      args = _compositearguments.new()
      args.product = product
      self.assertEqual(True, classUnderTest.canHandle(args))

    for product in ["MAX", "PMAX"]:
      args = _compositearguments.new()
      args.product = product
      args.addArgument("interpolation_method", "NEAREST")
      self.assertEqual(True, classUnderTest.canHandle(args))

    for product in ["SCAN", "ETOP", "RHI"]:
      args = _compositearguments.new()
      args.product = product
      self.assertEqual(False, classUnderTest.canHandle(args))

    for product in ["MAX", "PMAX"]:
      args = _compositearguments.new()
      args.product = product
      args.addArgument("interpolation_method", "3D")
      self.assertEqual(False, classUnderTest.canHandle(args))

  def test_generate(self):
    classUnderTest = _ratecompositegeneratorfactory.new()
    args = _compositearguments.new()
    args.product = "PPI"
    args.area = self.create_area("nrd2km")
    args.addParameter("RATE", 0.0, 1.0)
    classUnderTest.generate(args)
