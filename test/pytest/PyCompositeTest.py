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
    
    result = generator.nearest(a, 1000.0)
    
    result.quantity = "DBZH"
    result.time = "120000"
    result.date = "20090501"
    result.product = _rave.Rave_ProductType_COMP
    result.source = "nrd2km"
    result.objectType = _rave.Rave_ObjectType_IMAGE
    data = result.getData()
    
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite.h5"
    ios.save()
