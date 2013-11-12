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

Test case for the rave_fm12 synop parsing

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2013-11-08
'''

import unittest, os, datetime
from rave_fm12 import fm12_base_info, fm12_obs, fm12_parser

class rave_fm12_parser_test(unittest.TestCase):
  #FIXTURE="fixtures/flatfile.txt"

  FIXTURE_1="fixtures/A_SMNO01ENMI300600RRC_C_ENMI_20131030061900.txt"
  FIXTURE_2="fixtures/A_SMSN86ESWI310600_C_ESWI_20131031061530.txt"
  FIXTURE_3="fixtures/A_SMUR11UKMS300600RRA_C_UKMS_20131030061900.txt"
  FIXTURE_4="fixtures/A_SMUR11UKMS300600RRB_C_UKMS_20131030061900.txt"
  
  def setUp(self):
    self.classUnderTest = fm12_parser()

  def tearDown(self):
    self.classUnderTest = None
    
  def test_parse_1(self):
    #AAXX 30061
    # 01026 16/// /2004 10032 20013 39705 49843 52023 6//// 333 20026 91112=
    result = self.classUnderTest.parse(self.FIXTURE_1)
    
    self.assertEquals(1, len(result))
    
    self.assertEquals("01026", result[0].station)
    self.assertEquals(fm12_obs.SYNOP, result[0].type)
    self.assertEquals("20131030", result[0].date)
    self.assertEquals("061900", result[0].time)
    self.assertEquals(None, result[0].visibility)
    self.assertEquals(None, result[0].cloudbase)
    self.assertEquals(None, result[0].cloudcover)
    self.assertEquals(200, result[0].winddirection)
    self.assertEquals(4, result[0].windspeed)
    self.assertAlmostEquals(3.2, result[0].temperature, 4)
    self.assertAlmostEquals(1.3, result[0].dewpoint, 4)
    self.assertAlmostEquals(970.5, result[0].pressure, 4)
    self.assertAlmostEquals(984.3, result[0].sea_lvl_pressure, 4)
    self.assertAlmostEquals(2.3, result[0].pressure_change, 4)
    self.assertEquals(fm12_obs.DELAYED, result[0].updated)
    
  def test_parse_2(self):
    result = self.classUnderTest.parse(self.FIXTURE_2)

    self.assertEquals(41, len(result))
    
    r1 = self.get_station_from_list(result, "02321")
    self.assertEquals("02321", r1.station)
    self.assertEquals(fm12_obs.SYNOP, r1.type)
    self.assertEquals("20131031", r1.date)
    self.assertEquals("061530", r1.time)
    self.assertAlmostEquals(51.0, r1.visibility,4)
    self.assertEquals(2, r1.cloudbase)
    self.assertEquals(None, r1.cloudcover)
    self.assertEquals(0, r1.winddirection)
    self.assertEquals(0, r1.windspeed)
    self.assertAlmostEquals(-3.3, r1.temperature, 4)
    self.assertAlmostEquals(-4.2, r1.dewpoint, 4)
    self.assertAlmostEquals(978.7, r1.pressure, 4)
    self.assertAlmostEquals(1010.2, r1.sea_lvl_pressure, 4)
    self.assertAlmostEquals(-1.6, r1.pressure_change, 4)
    self.assertAlmostEquals(0.0, r1.liquid_precipitation, 4)
    self.assertEquals(12, r1.accumulation_period)
    self.assertAlmostEquals(-2.3, r1.max_24hr_temperature, 4)
    self.assertAlmostEquals(-7.1, r1.min_24hr_temperature, 4)
    self.assertEquals(fm12_obs.ORIGINAL, r1.updated)
    
    r2 = self.get_station_from_list(result, "02462") # Got a NIL
    self.assertEquals(None, r2)
    
  def get_station_from_list(self, slist, stationnumber):
    for s in slist:
      if s.station == stationnumber:
        return s
    return None