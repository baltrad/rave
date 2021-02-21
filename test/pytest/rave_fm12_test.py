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
  FIXTURE_5="fixtures/SNDN54_EKMI_201511260800.txt"

  def setUp(self):
    self.classUnderTest = fm12_parser()

  def tearDown(self):
    self.classUnderTest = None
    
  def test_parse_1(self):
    #AAXX 30061
    # 01026 16/// /2004 10032 20013 39705 49843 52023 6//// 333 20026 91112=
    result = self.classUnderTest.parse(self.FIXTURE_1)
    
    self.assertEqual(1, len(result))
    
    self.assertEqual("01026", result[0].station)
    self.assertEqual(fm12_obs.SYNOP, result[0].type)
    self.assertEqual("20131030", result[0].date)
    self.assertEqual("061900", result[0].time)
    self.assertEqual(None, result[0].visibility)
    self.assertEqual(None, result[0].cloudbase)
    self.assertEqual(None, result[0].cloudcover)
    self.assertEqual(200, result[0].winddirection)
    self.assertEqual(4, result[0].windspeed)
    self.assertAlmostEqual(3.2, result[0].temperature, 4)
    self.assertAlmostEqual(1.3, result[0].dewpoint, 4)
    self.assertAlmostEqual(770.5, result[0].pressure, 4)
    self.assertAlmostEqual(984.3, result[0].sea_lvl_pressure, 4)
    self.assertAlmostEqual(2.3, result[0].pressure_change, 4)
    self.assertEqual(fm12_obs.DELAYED, result[0].updated)
    
  def test_parse_2(self):
    result = self.classUnderTest.parse(self.FIXTURE_2)

    self.assertEqual(41, len(result))
    
    r1 = self.get_station_from_list(result, "02321")
    self.assertEqual("02321", r1.station)
    self.assertEqual(fm12_obs.SYNOP, r1.type)
    self.assertEqual("20131031", r1.date)
    self.assertEqual("061530", r1.time)
    self.assertAlmostEqual(51.0, r1.visibility,4)
    self.assertEqual(2, r1.cloudbase)
    self.assertEqual(None, r1.cloudcover)
    self.assertEqual(0, r1.winddirection)
    self.assertEqual(0, r1.windspeed)
    self.assertAlmostEqual(-3.3, r1.temperature, 4)
    self.assertAlmostEqual(-4.2, r1.dewpoint, 4)
    self.assertAlmostEqual(978.7, r1.pressure, 4)
    self.assertAlmostEqual(1080.2, r1.sea_lvl_pressure, 4)
    self.assertAlmostEqual(-1.6, r1.pressure_change, 4)
    self.assertAlmostEqual(0.0, r1.liquid_precipitation, 4)
    self.assertEqual(12, r1.accumulation_period)
    self.assertAlmostEqual(-2.3, r1.max_24hr_temperature, 4)
    self.assertAlmostEqual(-7.1, r1.min_24hr_temperature, 4)
    self.assertEqual(fm12_obs.ORIGINAL, r1.updated)
    
    r2 = self.get_station_from_list(result, "02462") # Got a NIL
    self.assertEqual(None, r2)

  def test_parse_3(self):
    result = self.classUnderTest.parse(self.FIXTURE_3)

    self.assertEqual(1, len(result))
    
    r1 = self.get_station_from_list(result, "33466")
    self.assertEqual("33466", r1.station)
    self.assertEqual(fm12_obs.SYNOP, r1.type)
    self.assertEqual("20131030", r1.date)
    self.assertEqual("061900", r1.time)
    self.assertAlmostEqual(8.0, r1.visibility,4)
    self.assertEqual(6, r1.cloudbase)
    self.assertEqual(4, r1.cloudcover)
    self.assertEqual(0, r1.winddirection)
    self.assertEqual(0, r1.windspeed)
    self.assertAlmostEqual(9.5, r1.temperature, 4)
    self.assertAlmostEqual(8.5, r1.dewpoint, 4)
    self.assertAlmostEqual(999.9, r1.pressure, 4)
    self.assertAlmostEqual(1018.5, r1.sea_lvl_pressure, 4)
    self.assertAlmostEqual(1.2, r1.pressure_change, 4)
    self.assertAlmostEqual(0.0, r1.liquid_precipitation, 4)
    self.assertEqual(0, r1.accumulation_period)
    self.assertEqual(None, r1.max_24hr_temperature)
    self.assertAlmostEqual(8.5, r1.min_24hr_temperature, 4)
    self.assertEqual(fm12_obs.DELAYED, r1.updated)
    
    r2 = self.get_station_from_list(result, "02462") # Got a NIL
    self.assertEqual(None, r2)

  def test_parse_5(self):
    result = self.classUnderTest.parse(self.FIXTURE_5)

    self.assertEqual(58, len(result))
    
    # 3 different sections so we probably should check that each section has been handled
    
    r1 = self.get_station_from_list(result, "06019")
    self.assertEqual("06019", r1.station)
    self.assertEqual(fm12_obs.SYNOP, r1.type)
    self.assertEqual("20151126", r1.date)
    self.assertEqual("080000", r1.time)
    self.assertEqual(None, r1.visibility)
    self.assertEqual(None, r1.cloudbase)
    self.assertEqual(None, r1.cloudcover)
    self.assertEqual(260, r1.winddirection)
    self.assertEqual(1, r1.windspeed)
    self.assertAlmostEqual(2.8, r1.temperature, 4)
    self.assertAlmostEqual(-0.5, r1.dewpoint, 4)
    self.assertAlmostEqual(0.0, r1.pressure, 4)
    self.assertAlmostEqual(0.0, r1.sea_lvl_pressure, 4)
    self.assertAlmostEqual(0.0, r1.pressure_change, 4)
    self.assertAlmostEqual(0.0, r1.liquid_precipitation, 4)
    self.assertEqual(0, r1.accumulation_period)
    self.assertEqual(None, r1.max_24hr_temperature)
    self.assertAlmostEqual(3.6, r1.min_24hr_temperature, 4)
    self.assertEqual(fm12_obs.ORIGINAL, r1.updated)
    
    r2 = self.get_station_from_list(result, "06165") # Got a NIL
    self.assertEqual(None, r2)

    r3 = self.get_station_from_list(result, "26144")
    self.assertEqual("26144", r3.station)
    self.assertEqual(fm12_obs.SYNOP, r3.type)
    self.assertEqual("20151126", r3.date)
    self.assertEqual("080000", r3.time)
    self.assertAlmostEqual(24.0, r3.visibility, 4)
    self.assertEqual(None, r3.cloudbase)
    self.assertEqual(None, r3.cloudcover)
    self.assertEqual(180, r3.winddirection)
    self.assertEqual(4, r3.windspeed)
    self.assertAlmostEqual(-3.1, r3.temperature, 4)
    self.assertAlmostEqual(-6.4, r3.dewpoint, 4)
    self.assertAlmostEqual(1004.9, r3.pressure, 4)
    self.assertAlmostEqual(1013.8, r3.sea_lvl_pressure, 4)
    self.assertAlmostEqual(1.0, r3.pressure_change, 4)
    self.assertAlmostEqual(0.0, r3.liquid_precipitation, 4)
    self.assertEqual(0, r3.accumulation_period)
    self.assertAlmostEqual(-1.2, r3.max_24hr_temperature, 4)
    self.assertAlmostEqual(-3.9, r3.min_24hr_temperature, 4)
    self.assertEqual(fm12_obs.ORIGINAL, r3.updated)

    
  def get_station_from_list(self, slist, stationnumber):
    for s in slist:
      if s.station == stationnumber:
        return s
    return None