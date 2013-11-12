'''
Copyright (C) 2009- Swedish Meteorological and Hydrological Institute, SMHI,

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

Test case for the rave_wmo_flatfile parser

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2013-11-07
'''
import unittest, os, datetime
from rave_wmo_flatfile import rave_wmo_flatfile

class rave_wmo_flatfile_test(unittest.TestCase):
  FIXTURE="fixtures/flatfile.txt"
  
  def setUp(self):
    self.classUnderTest = rave_wmo_flatfile()

  def tearDown(self):
    self.classUnderTest = None
    
  def test_gps_to_decimal(self):
    self.assertAlmostEquals(10 + 20/60.0 + 30/3600.0, self.classUnderTest._gps_to_decimal(10,20,30))
    self.assertAlmostEquals(10 + 5/60.0 + 3600/3600.0, self.classUnderTest._gps_to_decimal(10,5,3600))

  def test_parse_row(self):
    IDS=["CountryArea", "CountryCode", "IndexNbr", "IndexSubNbr", "StationName", "Latitude", "Longitude"]
    
    rowstr = "SWE\t123\t12345\t1\tEn station\t10 20 30N\t30 20 10W"
    
    result = self.classUnderTest._parse_row(IDS, rowstr)
    self.assertEquals("SWE", result.country)
    self.assertEquals("123", result.countrycode)
    self.assertEquals("12345", result.stationnumber)
    self.assertEquals("1", result.stationsubnumber)
    self.assertEquals("En station", result.stationname)
    self.assertAlmostEquals(10+20/60.0+30/3600.0, result.latitude, 4)
    self.assertAlmostEquals(-(30+20/60.0+10/3600.0), result.longitude, 4)
    
  def test_parse(self):
    result = self.classUnderTest.parse(self.FIXTURE)
    self.assertEquals(8, len(result))
    self.assertEquals("02013", result[0].stationnumber)#67 43 37N  17 28 16E
    self.assertAlmostEquals(67+43/60.0+37/3600.0, result[0].latitude, 4)
    self.assertAlmostEquals(17+28/60.0+16/3600.0, result[0].longitude, 4)
    self.assertEquals("10015", result[7].stationnumber)#67 43 37N  17 28 16E
    self.assertAlmostEquals(54+10/60.0+35/3600.0, result[7].latitude, 4)
    self.assertAlmostEquals(07+53/60.0+35/3600.0, result[7].longitude, 4)
    