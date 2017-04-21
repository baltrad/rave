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

Tests the bitmap generator module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2016-12-21
'''
import unittest
import os
import _bitmapgenerator
import _raveio
import _cartesian
import _cartesianparam
import _ravefield
import string
import numpy

class PyBitmapGeneratorTest(unittest.TestCase):
  FIXTURE_COMPOSITE_WITH_RADARINDEX = "fixtures/composite_with_radarindex.h5"
  GENERATOR_RADARINDEX_FILENAME = "bitmapgenerator_radarindex_test.h5"
  GENERATOR_SURROUNDING_FILENAME = "bitmapgenerator_surrounding_test.h5"
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _bitmapgenerator.new()
    self.assertNotEqual(-1, string.find(`type(obj)`, "BitmapGeneratorCore"))

  def test_create_surrounding(self):
    obj = _bitmapgenerator.new()
    cartesian = _raveio.open(self.FIXTURE_COMPOSITE_WITH_RADARINDEX).object.getImage(0)
    result = obj.create_surrounding(cartesian.getParameter("DBZH"))
    cartesian.getParameter("DBZH").addQualityField(result)
    dbzh = cartesian.getParameter("DBZH").getData()
    bitmap = result.getData()
    dbzh = numpy.where(numpy.equal(dbzh, 255.0), 0.0, dbzh).astype(numpy.uint8)
    d = numpy.where(numpy.equal(bitmap, 1.0), 255.0, dbzh).astype(numpy.uint8)
    np = _cartesianparam.new()
    np.setData(d)
    np.quantity="BRDR"
    cartesian.addParameter(np)
    rio = _raveio.new()
    rio.object = cartesian
    rio.save(self.GENERATOR_SURROUNDING_FILENAME)
    
  def test_create_intersect(self):
    obj = _bitmapgenerator.new()
    cartesian = _raveio.open(self.FIXTURE_COMPOSITE_WITH_RADARINDEX).object.getImage(0)
    result = obj.create_intersect(cartesian.getParameter("DBZH"), "se.smhi.composite.index.radar")
    dbzh = cartesian.getParameter("DBZH")
    bitmap = result.getData()
    d = numpy.where(numpy.equal(bitmap, 1.0), 255.0, dbzh.getData()).astype(numpy.uint8)
    np = _cartesianparam.new()
    np.setData(d)
    np.quantity="BRDR"
    cartesian.addParameter(np)
    cartesian.getParameter("DBZH").addQualityField(result)
    rio = _raveio.new()
    rio.object = cartesian
    rio.save(self.GENERATOR_RADARINDEX_FILENAME)
    
    