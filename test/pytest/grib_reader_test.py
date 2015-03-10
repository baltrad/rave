'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.

'''
import unittest, os
from grib import grib_reader, grib_reader_factory
import numpy

##
# Tests the grib reading functionality. Requires that pygrib exists
#
class grib_reader_test(unittest.TestCase):
  FIXTURE="fixtures/model_file_with_level_0_850_and_925.grb"
  
  def setUp(self):
    self.classUnderTest = grib_reader.pygrib_grib_reader(self.FIXTURE)

  def tearDown(self):
    self.classUnderTest = None

  def test_get_field(self):
    field = self.classUnderTest.get_field(grib_reader.grib_reader.TEMPERATURE)
    self.assertTrue(field is not None)
    self.assertEquals("20150303", field.getAttribute("what/date"))
    self.assertEquals("000000", field.getAttribute("what/time"))
    self.assertEquals(grib_reader.grib_reader.TEMPERATURE, field.getAttribute("what/name"))
    self.assertEquals("K", field.getAttribute("what/units"))
    self.assertEquals("regular_ll", field.getAttribute("what/gridType"))
    self.assertEquals(850, field.getAttribute("what/level"))
    self.assertAlmostEquals(9999.0, field.getAttribute("what/nodata"), 4)
    self.assertAlmostEquals(256.8679, numpy.min(field.getData()), 4)
    self.assertAlmostEquals(272.7273, numpy.max(field.getData()), 4)
    self.assertEquals(179, field.ysize)
    self.assertEquals(428, field.xsize)
    
  def test_get_field_with_level(self):
    field = self.classUnderTest.get_field(grib_reader.grib_reader.TEMPERATURE, 925)
    self.assertTrue(field is not None)
    self.assertEquals("20150303", field.getAttribute("what/date"))
    self.assertEquals("000000", field.getAttribute("what/time"))
    self.assertEquals(grib_reader.grib_reader.TEMPERATURE, field.getAttribute("what/name"))
    self.assertEquals("K", field.getAttribute("what/units"))
    self.assertEquals("regular_ll", field.getAttribute("what/gridType"))
    self.assertEquals(925, field.getAttribute("what/level"))
    self.assertAlmostEquals(9999.0, field.getAttribute("what/nodata"), 4)
    self.assertAlmostEquals(256.7686, numpy.min(field.getData()), 4)
    self.assertAlmostEquals(275.5498, numpy.max(field.getData()), 4)
    self.assertEquals(179, field.ysize)
    self.assertEquals(428, field.xsize)
  
  def test_get_field_with_non_existing_name(self):
    try:
      self.classUnderTest.get_field("No such name")
      self.fail("Expected IOError")
    except IOError, e:
      pass

  def test_get_field_with_non_existing_level(self):
    try:
      self.classUnderTest.get_field(grib_reader.grib_reader.TEMPERATURE, 999)
      self.fail("Expected IOError")
    except IOError, e:
      pass
  
  def test_iterator(self):
    DEFS=[("Skin temperature",0),
          ("High cloud cover",0),
          ("Total column ozone",0),
          ("Surface pressure",0),
          ("2 metre temperature",0),
          ("Snowfall",0),
          ("10 metre wind gust since previous post-processing",0),
          ("Medium cloud cover",0),
          ("Large-scale precipitation",0),
          ("Convective available potential energy",0),
          ("10 metre U wind component",0),
          ("Mean sea level pressure",0),
          ("Total cloud cover",0),
          ("Convective precipitation",0),
          ("Total column water vapour",0),
          ("Low cloud cover",0),
          ("Total precipitation",0),
          ("10 metre V wind component",0),
          ("2 metre dewpoint temperature",0),
          ("Specific humidity",850),
          ("Vertical velocity",850),
          ("Geopotential Height",850),
          ("Divergence",850),
          ("Temperature",850),
          ("V component of wind",850),
          ("U component of wind",850),
          ("Relative humidity",850),
          ("Temperature",925),
          ("V component of wind",925),
          ("U component of wind",925),
          ("Relative humidity",925),
          ("Vertical velocity",925),
          ("Divergence",925),
          ("Specific humidity",925),
          ("Geopotential Height",925)]

    didx = 0
    for c in self.classUnderTest.iterator():
      self.assertEquals(DEFS[didx][0], c.getAttribute("what/name"))
      self.assertEquals(DEFS[didx][1], c.getAttribute("what/level"))
      didx = didx + 1
    