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

Tests the py radar definition module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2010-08-31
'''
import unittest
import os
import _radardef
import _projection
import string
import math

class PyRadarDefinitionTest(unittest.TestCase):
  
  def setUp(self):
    pass

  def tearDown(self):
    pass
  
  def test_new(self):
    obj = _radardef.new()
    
    isradardef = string.find(`type(obj)`, "RadarDefinitionCore")
    self.assertNotEqual(-1, isradardef)

  def test_attribute_visibility(self):
    attrs = ['id', 'description', 'longitude', 'latitude', 'height',
             'elangles', 'nrays', 'nbins', 'scale', 
             'beamwidth', 'wavelength', 'projection']
    radar = _radardef.new()
    alist = dir(radar)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_id(self):
    obj = _radardef.new()

    self.assertEquals(None, obj.id)
    obj.id = "something"
    self.assertEquals("something", obj.id)
    obj.id = None
    self.assertEquals(None, obj.id)

  def test_id_typeError(self):
    obj = _radardef.new()

    try:
      obj.id = 1.2
      self.fail("Expected TypeError")
    except TypeError, e:
      pass
    self.assertEquals(None, obj.id)
    
  def test_description(self):
    obj = _radardef.new()
    self.assertEquals(None, obj.description)
    obj.description = "abc"
    self.assertEquals("abc", obj.description)
  
  def test_longitude(self):
    obj = _radardef.new()
    self.assertAlmostEquals(0.0, obj.longitude, 4)
    obj.longitude = 2.0
    self.assertAlmostEquals(2.0, obj.longitude, 4)

  def test_latitude(self):
    obj = _radardef.new()
    self.assertAlmostEquals(0.0, obj.latitude, 4)
    obj.latitude = 2.0
    self.assertAlmostEquals(2.0, obj.latitude, 4)

  def test_height(self):
    obj = _radardef.new()
    self.assertAlmostEquals(0.0, obj.height, 4)
    obj.height = 2.0
    self.assertAlmostEquals(2.0, obj.height, 4)

  def test_scale(self):
    obj = _radardef.new()
    self.assertAlmostEquals(0.0, obj.scale, 4)
    obj.scale = 2.0
    self.assertAlmostEquals(2.0, obj.scale, 4)

  def test_beamwidth(self):
    obj = _radardef.new()
    self.assertAlmostEquals(0.0, obj.beamwidth, 4)
    obj.beamwidth = 2.0
    self.assertAlmostEquals(2.0, obj.beamwidth, 4)

  def test_wavelength(self):
    obj = _radardef.new()
    self.assertAlmostEquals(0.0, obj.wavelength, 4)
    obj.wavelength = 2.0
    self.assertAlmostEquals(2.0, obj.wavelength, 4)

  def test_nbins(self):
    obj = _radardef.new()
    self.assertEquals(0, obj.nbins)
    obj.nbins = 11
    self.assertEquals(11, obj.nbins)

  def test_nrays(self):
    obj = _radardef.new()
    self.assertEquals(0, obj.nrays)
    obj.nrays = 8
    self.assertEquals(8, obj.nrays)

  def test_elangles(self):
    obj = _radardef.new()
    self.assertTrue(type([]) == type(obj.elangles))
    self.assertEquals(0, len(obj.elangles))
    obj.elangles = [1.0, 2.0, 3.0]
    self.assertEquals(3, len(obj.elangles))
    self.assertAlmostEquals(1.0, obj.elangles[0], 4)
    self.assertAlmostEquals(2.0, obj.elangles[1], 4)
    self.assertAlmostEquals(3.0, obj.elangles[2], 4)
    
  def test_projection(self):
    obj = _radardef.new()
    self.assertEquals(None, obj.projection)
    
    obj.projection = _projection.new("x", "y", "+proj=latlong +ellps=WGS84 +datum=WGS84")
    
    self.assertEquals("x", obj.projection.id)
    
    obj.projection = None
    
    self.assertEquals(None, obj.projection)
    
  def test_projection_typeError(self):
    obj = _radardef.new()

    try:
      obj.projection = "+proj=latlong +ellps=WGS84 +datum=WGS84"
      self.fail("Expected TypeError")
    except TypeError, e:
      pass
    
    self.assertEquals(None, obj.projection)
    