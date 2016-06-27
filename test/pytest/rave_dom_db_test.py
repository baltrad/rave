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
import unittest, os, datetime
from rave_dom import wmo_station, observation
import rave_dom_db
import _rave

import contextlib
from sqlalchemy import engine, event, exc as sqlexc, sql
from sqlalchemy.orm import mapper, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError
from gadjust.gra import gra_coefficient
from gadjust.grapoint import grapoint

Base = declarative_base()

#from sqlalchemy.orm import Base

DB_URL = os.environ.get("RAVE_TESTDB_URI", "")

##
# Simple helper for verifying that data access is behaving as expected
#
class testdb(object):
  def __init__(self):
    self._engine = engine.create_engine(DB_URL)
  
  def query(self, str):
    result = []
    with self.get_connection() as conn:
      cursor = conn.execute(str)
      for row in cursor:
        keys = row.keys()
        v = {}
        for k in keys:
          v[k] = row[k]
        #print row.keys()
        result.append(v)
    return result
  
  def add(self, obj):
    Session = sessionmaker(bind=self._engine)
    session = Session()
    session.add(obj)
    session.commit()
    session = None
  
  def get_connection(self):
    return contextlib.closing(self._engine.connect())

class rave_dom_db_test(unittest.TestCase):
  # I am using class under test for creating and dropping the tables.
  
  def setUp(self):
    self.classUnderTest = rave_dom_db.create_db(DB_URL)
    self.classUnderTest.drop()
    self.classUnderTest.create() 

  def tearDown(self):
    #self.classUnderTest.drop()
    pass

  def test_add(self):
    #station, country, type, date, time, longitude, latitude
    obs = observation("12345", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    self.classUnderTest.add(obs)
    
    result = testdb().query("SELECT * FROM rave_observation where station = '12345'")
    self.assertEquals(1, len(result))
    self.assertEquals("12345", result[0]["station"])
    self.assertEquals("SWEDEN", result[0]["country"])
    self.assertEquals("2010-10-10", result[0]["date"].strftime("%Y-%m-%d"))
    self.assertEquals("11:30:00", result[0]["time"].strftime("%H:%M:%S"))
    self.assertAlmostEquals(60.123, result[0]["latitude"], 4)
    self.assertAlmostEquals(13.031, result[0]["longitude"], 4)


  def test_add_2(self):
    #station, country, type, date, time, longitude, latitude
    obs = observation("12345", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    obs.visibility = 1.0
    obs.windtype = observation.WIND_TYPE_ESTIMATED
    obs.cloudcover = 2
    obs.winddirection = 3
    obs.temperature = 4.0
    obs.dewpoint = 5.0
    obs.relativehumidity = 6.0
    obs.pressure = 7.0
    obs.sea_lvl_pressure = 8.0
    obs.pressure_change = 9.0
    obs.liquid_precipitation = 10.0
    obs.accumulation_period = 11
    
    self.classUnderTest.add(obs)
    
    result = testdb().query("SELECT * FROM rave_observation where station = '12345'")
    self.assertEquals(1, len(result))
    self.assertEquals("12345", result[0]["station"])
    self.assertEquals("SWEDEN", result[0]["country"])
    self.assertEquals("2010-10-10", result[0]["date"].strftime("%Y-%m-%d"))
    self.assertEquals("11:30:00", result[0]["time"].strftime("%H:%M:%S"))
    self.assertAlmostEquals(60.123, result[0]["latitude"], 4)
    self.assertAlmostEquals(13.031, result[0]["longitude"], 4)
    
    self.assertAlmostEquals(1.0, result[0]["visibility"], 4)
    self.assertEquals(0, result[0]["windtype"])
    self.assertEquals(2, result[0]["cloudcover"])
    self.assertEquals(3, result[0]["winddirection"])
    self.assertAlmostEquals(4.0, result[0]["temperature"], 4)
    self.assertAlmostEquals(5.0, result[0]["dewpoint"], 4)
    self.assertAlmostEquals(6.0, result[0]["relativehumidity"], 4)
    self.assertAlmostEquals(7.0, result[0]["pressure"], 4)
    self.assertAlmostEquals(8.0, result[0]["sea_lvl_pressure"], 4)
    self.assertAlmostEquals(9.0, result[0]["pressure_change"], 4)
    self.assertAlmostEquals(10.0, result[0]["liquid_precipitation"], 4)
    self.assertEquals(11, result[0]["accumulation_period"])

  def test_get_observation(self):
    # I am using the dom defined observation instead of trying to define some generic dict variant
    # for populating a db table
    obs = observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    obs.visibility = 1.0
    obs.windtype = observation.WIND_TYPE_ESTIMATED
    obs.cloudcover = 2
    obs.winddirection = 3
    obs.temperature = 4.0
    obs.dewpoint = 5.0
    obs.relativehumidity = 6.0
    obs.pressure = 7.0
    obs.sea_lvl_pressure = 8.0
    obs.pressure_change = 9.0
    obs.liquid_precipitation = 10.0
    obs.accumulation_period = 11
    obs2 = observation("54322", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    
    testdb().add(obs)
    testdb().add(obs2)
    
    # Test
    with self.classUnderTest.get_session() as s:
      result = s.query(observation).filter_by(station="54321").all()[0]
    #result = self.classUnderTest.get_session().query(observation).filter_by(station="54321").all()[0]
    self.assertEquals("54321", result.station)
    self.assertEquals("SWEDEN", result.country)
    self.assertEquals("2010-10-10", result.date.strftime("%Y-%m-%d"))
    self.assertEquals("11:30:00", result.time.strftime("%H:%M:%S"))
    self.assertAlmostEquals(60.123, result.latitude, 4)
    self.assertAlmostEquals(13.031, result.longitude, 4)

  def test_get_observations_in_bbox(self):
    db = testdb()
    db.add(observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123))
    db.add(observation("54322", "SWEDEN", 0, "20101010", "113000", 14.031, 61.123))  #X
    db.add(observation("54323", "SWEDEN", 0, "20101010", "113000", 13.531, 60.523))  #X
    db.add(observation("54324", "SWEDEN", 0, "20101010", "113000", 16.031, 62.123))
    db.add(observation("54325", "SWEDEN", 0, "20101010", "113000", 13.131, 60.223))
    db.add(observation("54326", "SWEDEN", 0, "20101010", "113000", 14.531, 61.523))
    db.add(observation("54325", "SWEDEN", 0, "20101010", "114500", 13.131, 60.223))
    
    # Test
    result = self.classUnderTest.get_observations_in_bbox(13.0,61.5,15.0,60.5)

    # Verify result
    self.assertEquals(2, len(result))
    self.assertEquals("54322", result[0].station)
    self.assertEquals("54323", result[1].station)

  def test_get_observations_in_bbox_2(self):
    # Tests that if we don't specify a dateinterval, we always get the latest
    db = testdb()
    db.add(observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123))
    db.add(observation("54322", "SWEDEN", 0, "20101010", "113000", 14.031, 61.123))  #X
    db.add(observation("54323", "SWEDEN", 0, "20101010", "113000", 13.531, 60.523))  #X
    db.add(observation("54324", "SWEDEN", 0, "20101010", "113000", 16.031, 62.123))
    db.add(observation("54325", "SWEDEN", 0, "20101010", "113000", 13.131, 60.223))  #X
    db.add(observation("54326", "SWEDEN", 0, "20101010", "113000", 14.531, 61.523))
    db.add(observation("54325", "SWEDEN", 0, "20101010", "114500", 13.131, 60.223))  #X
    
    # Test
    result = self.classUnderTest.get_observations_in_bbox(13.0,61.5,15.0,60.2)

    # Verify result
    self.assertEquals(4, len(result))
    self.assertEquals("54322", result[0].station)
    self.assertEquals("54323", result[1].station)   
    self.assertEquals("54325", result[2].station)   
    self.assertEquals("54325", result[3].station)   

  def test_get_observations_in_bbox_3(self):
    # Tests that date interval is working
    db = testdb()
    db.add(observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "114500", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "120000", 13.031, 60.123))
    db.add(observation("54322", "SWEDEN", 0, "20101010", "113000", 14.031, 61.123)) #X
    db.add(observation("54323", "SWEDEN", 0, "20101010", "114500", 13.531, 60.523)) #X
    db.add(observation("54324", "SWEDEN", 0, "20101010", "120000", 16.031, 62.123))
    db.add(observation("54324", "SWEDEN", 0, "20101010", "120015", 16.031, 62.123))
    db.add(observation("54325", "SWEDEN", 0, "20101010", "113000", 13.131, 60.223)) #X
    db.add(observation("54326", "SWEDEN", 0, "20101010", "113000", 14.531, 61.523))
    
    # Test
    result = self.classUnderTest.get_observations_in_bbox(13.0,61.5,15.0,60.2,
                                                          datetime.datetime(2010,10,10,11,30),
                                                          datetime.datetime(2010,10,10,11,44))

    # Verify result
    self.assertEquals(2, len(result))
    self.assertEquals("54322", result[0].station)
    self.assertEquals("2010-10-10", result[0].date.strftime("%Y-%m-%d"))
    self.assertEquals("11:30:00", result[0].time.strftime("%H:%M:%S"))
    self.assertEquals("54325", result[1].station)   
    self.assertEquals("2010-10-10", result[1].date.strftime("%Y-%m-%d"))
    self.assertEquals("11:30:00", result[1].time.strftime("%H:%M:%S"))

  def test_get_observations_in_bbox_4(self):
    # Another test to verify that date interval is working with mixed dates
    db = testdb()
    db.add(observation("54321", "SWEDEN", 0, "20101010", "180000", 13.031, 60.123))
    db.add(observation("54322", "SWEDEN", 0, "20101011", "060000", 13.031, 60.123))
    
    # Test
    result = self.classUnderTest.get_observations_in_bbox(13.0,61.5,15.0,60.1,
                                                          datetime.datetime(2010,10,10,18,00))#,
                                                          #datetime.datetime(2010,10,11,06,00))
    result2 = self.classUnderTest.get_observations_in_bbox(13.0,61.5,15.0,60.1,
                                                           datetime.datetime(2010,10,10,18,00),
                                                           datetime.datetime(2010,10,11,06,00))

    # Verify result
    self.assertEquals(2, len(result))
    self.assertEquals(2, len(result2))
 
  def test_get_observations_in_bbox_5(self):
    # Another test to verify that date interval is working with mixed dates
    db = testdb()
    db.add(observation("54321", "SWEDEN", 0, "20101010", "180000", 13.031, 60.123,accumulation_period=0))
    db.add(observation("54322", "SWEDEN", 0, "20101011", "060000", 13.031, 60.123,accumulation_period=0))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "180001", 13.031, 60.123,accumulation_period=12))
    db.add(observation("54322", "SWEDEN", 0, "20101011", "060001", 13.031, 60.123,accumulation_period=12))
    
    # Test
    result = self.classUnderTest.get_observations_in_bbox(13.0,61.5,15.0,60.1,
                                                          datetime.datetime(2010,10,10,18,00),
                                                          datetime.datetime(2010,10,11,06,00))

    # Verify result
    self.assertEquals(3, len(result))
    
  def test_get_observations_in_interval(self):
    db = testdb()
    db.add(observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "114500", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "120000", 13.031, 60.123))
    db.add(observation("54322", "SWEDEN", 0, "20101010", "113000", 14.031, 61.123)) #X
    db.add(observation("54322", "SWEDEN", 0, "20101010", "114500", 13.531, 60.523)) #X
    db.add(observation("54323", "SWEDEN", 0, "20101010", "120000", 16.031, 62.123))
    db.add(observation("54324", "SWEDEN", 0, "20101010", "120030", 16.031, 62.123))
    
    # Test
    result = self.classUnderTest.get_observations_in_interval(datetime.datetime(2010,10,10,11,30),
                                                              datetime.datetime(2010,10,10,11,45))
    self.assertEquals(4, len(result))
    self.assertEquals("54321", result[0].station)
    self.assertEquals("11:30:00", result[0].time.strftime("%H:%M:%S"))
    self.assertEquals("54321", result[1].station)
    self.assertEquals("11:45:00", result[1].time.strftime("%H:%M:%S"))
    self.assertEquals("54322", result[2].station)
    self.assertEquals("11:30:00", result[2].time.strftime("%H:%M:%S"))
    self.assertEquals("54322", result[3].station)
    self.assertEquals("11:45:00", result[3].time.strftime("%H:%M:%S"))

  def test_get_observations_in_interval_2(self):
    db = testdb()
    db.add(observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "114500", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "120000", 13.031, 60.123))
    db.add(observation("54322", "SWEDEN", 0, "20101010", "113000", 14.031, 61.123)) #X
    db.add(observation("54322", "SWEDEN", 0, "20101010", "114500", 13.531, 60.523)) #X
    db.add(observation("54323", "SWEDEN", 0, "20101010", "120000", 16.031, 62.123))
    db.add(observation("54324", "SWEDEN", 0, "20101010", "120030", 16.031, 62.123))
    
    # Test
    result = self.classUnderTest.get_observations_in_interval(datetime.datetime(2010,10,10,11,30),
                                                              datetime.datetime(2010,10,10,11,45),
                                                              ["54322","54323"])
    self.assertEquals(2, len(result))
    self.assertEquals("54322", result[0].station)
    self.assertEquals("11:30:00", result[0].time.strftime("%H:%M:%S"))
    self.assertEquals("54322", result[1].station)
    self.assertEquals("11:45:00", result[1].time.strftime("%H:%M:%S"))

  def test_get_observations_in_interval_3(self):
    db = testdb()
    db.add(observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "114500", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "120000", 13.031, 60.123))
    db.add(observation("54322", "SWEDEN", 0, "20101010", "113000", 14.031, 61.123)) #X
    db.add(observation("54322", "SWEDEN", 0, "20101010", "114500", 13.531, 60.523)) #X
    db.add(observation("54323", "SWEDEN", 0, "20101010", "120000", 16.031, 62.123))
    db.add(observation("54324", "SWEDEN", 0, "20101010", "120030", 16.031, 62.123))
    
    # Test
    result = self.classUnderTest.get_observations_in_interval(datetime.datetime(2010,10,10,11,30),
                                                              datetime.datetime(2010,10,10,12,00),
                                                              ["54322","54323"])
    self.assertEquals(3, len(result))
    self.assertEquals("54322", result[0].station)
    self.assertEquals("11:30:00", result[0].time.strftime("%H:%M:%S"))
    self.assertEquals("54322", result[1].station)
    self.assertEquals("11:45:00", result[1].time.strftime("%H:%M:%S"))
    self.assertEquals("54323", result[2].station)
    self.assertEquals("12:00:00", result[2].time.strftime("%H:%M:%S"))
 
  def test_get_observations_in_interval_4(self):
    db = testdb()
    db.add(observation("54320", "SWEDEN", 0, "20101010", "060000", 13.031, 60.123))
    db.add(observation("54321", "SWEDEN", 0, "20101010", "180000", 13.031, 60.123))
    db.add(observation("54322", "SWEDEN", 0, "20101011", "060000", 13.031, 60.123))
    db.add(observation("54323", "SWEDEN", 0, "20101011", "180000", 13.031, 60.123))
    
    # Test
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(4, len(result))
    self.assertEquals("54320", result[0].station)
    self.assertEquals("54321", result[1].station)
    self.assertEquals("54322", result[2].station)
    self.assertEquals("54323", result[3].station)

    # Test
    result = self.classUnderTest.get_observations_in_interval(datetime.datetime(2010,10,10,18,00), None)
    self.assertEquals(3, len(result))
    self.assertEquals("54321", result[0].station)
    self.assertEquals("54322", result[1].station)
    self.assertEquals("54323", result[2].station)

    result = self.classUnderTest.get_observations_in_interval(datetime.datetime(2010,10,10,18,00),
                                                              datetime.datetime(2010,10,11,06,00),
                                                              ["54322"])
    self.assertEquals(1, len(result))
    self.assertEquals("54322", result[0].station)
    
  def test_delete_observations_in_interval_with_age_minlimit(self):
    db = testdb()
    db.add(observation("64320", "SWEDEN", 0, "20160610", "060000", 13.031, 60.123))
    db.add(observation("64321", "SWEDEN", 0, "20160610", "180000", 13.031, 60.123))
    db.add(observation("64322", "SWEDEN", 0, "20160611", "060000", 13.031, 60.123))
    db.add(observation("64323", "SWEDEN", 0, "20160611", "180000", 13.031, 60.123))

    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(4, len(result))
    self.assertEquals("64320", result[0].station)
    self.assertEquals("64321", result[1].station)
    self.assertEquals("64322", result[2].station)
    self.assertEquals("64323", result[3].station)
    
    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(datetime.datetime(2016,06,11,00,00), None)
    self.assertEquals(2, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(2, len(result))
    self.assertEquals("64320", result[0].station)
    self.assertEquals("64321", result[1].station)
    
    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(datetime.datetime(2016,06,10,18,00), None)
    self.assertEquals(1, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(1, len(result))
    self.assertEquals("64320", result[0].station)

    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(datetime.datetime(2010,01,01,00,00), None)
    self.assertEquals(1, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(0, len(result))
    
  def test_delete_observations_in_interval_with_age_maxlimit(self):
    db = testdb()
    db.add(observation("64320", "SWEDEN", 0, "20160531", "060000", 13.031, 60.123))
    db.add(observation("64321", "SWEDEN", 0, "20160531", "180000", 13.031, 60.123))
    db.add(observation("64322", "SWEDEN", 0, "20160601", "060000", 13.031, 60.123))
    db.add(observation("64323", "SWEDEN", 0, "20160601", "180000", 13.031, 60.123))

    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(4, len(result))
    self.assertEquals("64320", result[0].station)
    self.assertEquals("64321", result[1].station)
    self.assertEquals("64322", result[2].station)
    self.assertEquals("64323", result[3].station)
    
    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(None, datetime.datetime(2016,06,01,00,00))
    self.assertEquals(2, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(2, len(result))
    self.assertEquals("64322", result[0].station)
    self.assertEquals("64323", result[1].station)
    
    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(None, datetime.datetime(2016,06,01,06,00))
    self.assertEquals(1, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(1, len(result))
    self.assertEquals("64323", result[0].station)

    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(None, datetime.datetime(2020,01,01,00,00))
    self.assertEquals(1, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(0, len(result))
    
  def test_delete_observations_in_interval_with_intervallimit(self):
    db = testdb()
    db.add(observation("64309", "SWEDEN", 0, "20160520", "050000", 13.031, 60.123))
    db.add(observation("64310", "SWEDEN", 0, "20160520", "060000", 13.031, 60.123))
    db.add(observation("64311", "SWEDEN", 0, "20160520", "061000", 13.031, 60.123))
    db.add(observation("64312", "SWEDEN", 0, "20160520", "061005", 13.031, 60.123))
    db.add(observation("64313", "SWEDEN", 0, "20160520", "063100", 13.031, 60.123))

    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(5, len(result))
    self.assertEquals("64309", result[0].station)
    self.assertEquals("64310", result[1].station)
    self.assertEquals("64311", result[2].station)
    self.assertEquals("64312", result[3].station)
    self.assertEquals("64313", result[4].station)
    
    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(datetime.datetime(2016,05,20,06,00), 
                                                                                     datetime.datetime(2016,05,20,06,10))
    self.assertEquals(2, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(3, len(result))
    self.assertEquals("64309", result[0].station)
    self.assertEquals("64312", result[1].station)
    self.assertEquals("64313", result[2].station)
    
    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(datetime.datetime(2016,05,20,05,05), 
                                                                                     datetime.datetime(2016,05,20,06,30))
    self.assertEquals(1, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(2, len(result))
    self.assertEquals("64309", result[0].station)
    self.assertEquals("64313", result[1].station)
    
    # Function under test
    no_of_deleted_observations = self.classUnderTest.delete_observations_in_interval(datetime.datetime(2016,05,20,00,00), 
                                                                                     datetime.datetime(2016,05,20,07,00))
    self.assertEquals(2, no_of_deleted_observations)
    
    result = self.classUnderTest.get_observations_in_interval(None, None)
    self.assertEquals(0, len(result))

  def test_add_duplicate_observation(self):
    db = testdb()
    obs = observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    obs.temperature = 10.0
    db.add(obs)
    obs = observation("54322", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    obs.temperature = 11.0
    db.add(obs)

    # Test
    nobs = observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    nobs.temperature = 11.0
    try:
      self.classUnderTest.add(nobs)
      self.fail("Expected IntegrityError")
    except IntegrityError, e:
      pass

  def test_merge_observation(self):
    db = testdb()
    obs = observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    obs.temperature = 10.0
    db.add(obs)

    # Test
    nobs = observation("54321", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    nobs.temperature = 11.0
    nobs2 = observation("54322", "SWEDEN", 0, "20101010", "113000", 13.031, 60.123)
    nobs2.temperature = 12.0

    self.classUnderTest.merge([nobs,nobs2])
    result = testdb().query("SELECT * FROM rave_observation order by station")
    self.assertEquals(2, len(result))
    self.assertEquals("54321", result[0]["station"])
    self.assertAlmostEquals(11.0, result[0]["temperature"], 4)
    self.assertEquals("54322", result[1]["station"])
    self.assertAlmostEquals(12.0, result[1]["temperature"], 4)

  def test_add_station(self):
    #__init__(self, country, countrycode, stationnbr, subnbr, stationname, longitude, latitude):
    station = wmo_station("SWEDEN", "0123", "12345", "0", "Pelle", 10.10, 20.20)
    
    self.classUnderTest.add(station)
    
    result = testdb().query("SELECT * FROM rave_wmo_station where stationnumber = '12345'")
    self.assertEquals(1, len(result))
    self.assertEquals("12345", result[0]["stationnumber"])
    self.assertEquals("0", result[0]["stationsubnumber"])
    self.assertEquals("Pelle", result[0]["stationname"])
    self.assertAlmostEquals(10.10, result[0]["longitude"], 4)
    self.assertAlmostEquals(20.20, result[0]["latitude"], 4)

  def test_add_substation(self):
    #__init__(self, country, countrycode, stationnbr, subnbr, stationname, longitude, latitude):
    station1 = wmo_station("SWEDEN", "0123", "12345", "0", "Pelle", 10.10, 20.20)
    station2 = wmo_station("SWEDEN", "0123", "12345", "1", "Pelle", 10.10, 20.20)
    
    self.classUnderTest.add(station1)
    self.classUnderTest.add(station2)
    
    result = testdb().query("SELECT * FROM rave_wmo_station where stationnumber = '12345' order by stationsubnumber")
    self.assertEquals(2, len(result))
    self.assertEquals("12345", result[0]["stationnumber"])
    self.assertEquals("0", result[0]["stationsubnumber"])
    self.assertEquals("12345", result[1]["stationnumber"])
    self.assertEquals("1", result[1]["stationsubnumber"])

  def test_get_station(self):
    #__init__(self, country, countrycode, stationnbr, subnbr, stationname, longitude, latitude):
    station1 = wmo_station("SWEDEN", "0123", "12345", "0", "Pelle", 10.10, 20.20)
    station2 = wmo_station("SWEDEN", "0123", "12346", "0", "Nisse", 30.10, 40.20)
    
    testdb().add(station1)
    testdb().add(station2)
    
    result = self.classUnderTest.get_station("12346")
    self.assertEquals("12346", result.stationnumber)
    self.assertEquals("Nisse", result.stationname)
    self.assertAlmostEquals(30.10, result.longitude, 4)
    self.assertAlmostEquals(40.20, result.latitude, 4)
    
  def test_get_gra_coefficient(self):
    #def __init__(self, area, date, time, significant, points, loss, r, r_significant, corr_coeff, a, b, c, mean, stddev):
    gc1 = gra_coefficient("A1", "20140301", "100000", "True", 10, 5, 1.0, "True", 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    gc2 = gra_coefficient("A1", "20140301", "103000", "True", 11, 6, 8.0, "True", 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
    
    testdb().add(gc1)
    testdb().add(gc2)
    
    result = self.classUnderTest.get_gra_coefficient(datetime.datetime(2014,3,1,10,30,0))
    self.assertEquals("20140301", result.date.strftime("%Y%m%d"))
    self.assertEquals("103000", result.time.strftime("%H%M%S"))
    self.assertEquals("True", result.significant)
    self.assertEquals(11, result.points)
    self.assertEquals(6, result.loss)
    self.assertAlmostEquals(8.0, result.r, 4)
    self.assertEquals("True", result.r_significant)
    self.assertAlmostEquals(9.0, result.corr_coeff, 4)
    self.assertAlmostEquals(10.0, result.a, 4)
    self.assertAlmostEquals(11.0, result.b, 4)
    self.assertAlmostEquals(12.0, result.c, 4)
    self.assertAlmostEquals(13.0, result.mean, 4)
    self.assertAlmostEquals(14.0, result.stddev, 4)
    
    result = self.classUnderTest.get_gra_coefficient(datetime.datetime(2014,3,1,10,00,0))
    self.assertEquals("20140301", result.date.strftime("%Y%m%d"))
    self.assertEquals("100000", result.time.strftime("%H%M%S"))
    self.assertEquals("True", result.significant)
    self.assertEquals(10, result.points)
    self.assertEquals(5, result.loss)
    self.assertAlmostEquals(1.0, result.r, 4)
    self.assertEquals("True", result.r_significant)
    self.assertAlmostEquals(2.0, result.corr_coeff, 4)
    self.assertAlmostEquals(3.0, result.a, 4)
    self.assertAlmostEquals(4.0, result.b, 4)
    self.assertAlmostEquals(5.0, result.c, 4)
    self.assertAlmostEquals(6.0, result.mean, 4)
    self.assertAlmostEquals(7.0, result.stddev, 4)
  
  def test_get_newest_gra_coefficient(self):
    gc1 = gra_coefficient("A1", "20140301", "100000", "True", 10, 5, 1.0, "True", 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    gc2 = gra_coefficient("A1", "20140301", "103000", "True", 11, 6, 8.0, "True", 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
    gc3 = gra_coefficient("A1", "20140301", "110000", "True", 12, 7, 9.0, "True", 10.0, 11.0, 12.0, 13.0, 14.0, 15.0)
    
    testdb().add(gc1)
    testdb().add(gc2)
    testdb().add(gc3)
    
    result = self.classUnderTest.get_newest_gra_coefficient(datetime.datetime(2014,3,1,10,30,0))
    self.assertEquals("20140301", result.date.strftime("%Y%m%d"))
    self.assertEquals("110000", result.time.strftime("%H%M%S"))
    
    result = self.classUnderTest.get_newest_gra_coefficient(datetime.datetime(2014,3,1,10,00,0))
    self.assertEquals("20140301", result.date.strftime("%Y%m%d"))
    self.assertEquals("110000", result.time.strftime("%H%M%S"))

  def test_get_newest_gra_coefficient_in_range(self):
    gc1 = gra_coefficient("A1", "20140301", "100000", "True", 10, 5, 1.0, "True", 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    gc2 = gra_coefficient("A1", "20140301", "103000", "True", 11, 6, 8.0, "True", 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
    gc3 = gra_coefficient("A1", "20140301", "110000", "True", 12, 7, 9.0, "True", 10.0, 11.0, 12.0, 13.0, 14.0, 15.0)
    gc4 = gra_coefficient("A1", "20140301", "113000", "True", 13, 8,10.0, "True", 12.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    
    testdb().add(gc1)
    testdb().add(gc2)
    testdb().add(gc3)
    testdb().add(gc4)
    
    result = self.classUnderTest.get_newest_gra_coefficient(datetime.datetime(2014,3,1,10,30,0), datetime.datetime(2014,3,1,11,00,0))
    self.assertEquals("20140301", result.date.strftime("%Y%m%d"))
    self.assertEquals("110000", result.time.strftime("%H%M%S"))
    
    result = self.classUnderTest.get_newest_gra_coefficient(datetime.datetime(2014,3,1,11,00,0), datetime.datetime(2014,3,1,11,30,0))
    self.assertEquals("20140301", result.date.strftime("%Y%m%d"))
    self.assertEquals("113000", result.time.strftime("%H%M%S"))
  
  def test_get_grapoints(self):
    gp1 = grapoint(_rave.RaveValueType_DATA, 1.0, 234, 10.0, 60.0, "20140416", "110000", 20.0, 2)
    gp2 = grapoint(_rave.RaveValueType_DATA, 2.0, 235, 11.0, 61.0, "20140416", "010000", 21.0, 3)
    gp3 = grapoint(_rave.RaveValueType_DATA, 3.0, 236, 12.0, 62.0, "20140415", "110000", 22.0, 4)
    gp4 = grapoint(_rave.RaveValueType_DATA, 4.0, 237, 13.0, 63.0, "20140415", "010000", 23.0, 5)
    gp5 = grapoint(_rave.RaveValueType_DATA, 5.0, 238, 14.0, 64.0, "20140414", "110000", 24.0, 6)
    gp6 = grapoint(_rave.RaveValueType_DATA, 6.0, 239, 15.0, 65.0, "20140414", "010000", 25.0, 7)
    
    testdb().add(gp1)
    testdb().add(gp2)
    testdb().add(gp3)
    testdb().add(gp4)
    testdb().add(gp5)
    testdb().add(gp6)
    
    result = self.classUnderTest.get_grapoints(datetime.datetime(2014,4,15,0,1,0))
    self.assertEquals(4, len(result))
    self.assertEquals("20140415", result[0].date.strftime("%Y%m%d"))
    self.assertEquals("010000", result[0].time.strftime("%H%M%S"))
    self.assertAlmostEquals(4.0, result[0].radarvalue, 4)
    self.assertAlmostEquals(237.0, result[0].radardistance, 4)
    self.assertAlmostEquals(13.0, result[0].longitude, 4)
    self.assertAlmostEquals(63.0, result[0].latitude, 4)
    self.assertAlmostEquals(23.0, result[0].observation, 4)
    self.assertEquals(5, result[0].accumulation_period)

  def test_get_grapoints_startend(self):
    gp1 = grapoint(_rave.RaveValueType_DATA, 1.0, 234, 10.0, 60.0, "20140416", "110000", 20.0, 2)
    gp2 = grapoint(_rave.RaveValueType_DATA, 2.0, 235, 11.0, 61.0, "20140416", "010000", 21.0, 3)
    gp3 = grapoint(_rave.RaveValueType_DATA, 3.0, 236, 12.0, 62.0, "20140415", "110000", 22.0, 4)
    gp4 = grapoint(_rave.RaveValueType_DATA, 4.0, 237, 13.0, 63.0, "20140415", "010000", 23.0, 5)
    gp5 = grapoint(_rave.RaveValueType_DATA, 5.0, 238, 14.0, 64.0, "20140414", "110000", 24.0, 6)
    gp6 = grapoint(_rave.RaveValueType_DATA, 6.0, 239, 15.0, 65.0, "20140414", "010000", 25.0, 7)
    
    testdb().add(gp1)
    testdb().add(gp2)
    testdb().add(gp3)
    testdb().add(gp4)
    testdb().add(gp5)
    testdb().add(gp6)
    
    result = self.classUnderTest.get_grapoints(datetime.datetime(2014,4,14,11,0,0), datetime.datetime(2014,4,16,1,0,0))
    self.assertEquals(4, len(result))
    self.assertEquals("20140414", result[0].date.strftime("%Y%m%d"))
    self.assertEquals("110000", result[0].time.strftime("%H%M%S"))
    self.assertEquals("20140415", result[1].date.strftime("%Y%m%d"))
    self.assertEquals("010000", result[1].time.strftime("%H%M%S"))
    self.assertEquals("20140415", result[2].date.strftime("%Y%m%d"))
    self.assertEquals("110000", result[2].time.strftime("%H%M%S"))
    self.assertEquals("20140416", result[3].date.strftime("%Y%m%d"))
    self.assertEquals("010000", result[3].time.strftime("%H%M%S"))

  def test_delete_grapoints(self):
    gp1 = grapoint(_rave.RaveValueType_DATA, 1.0, 234, 10.0, 60.0, "20140416", "110000", 20.0, 2)
    gp2 = grapoint(_rave.RaveValueType_DATA, 2.0, 235, 11.0, 61.0, "20140416", "010000", 21.0, 3)
    gp3 = grapoint(_rave.RaveValueType_DATA, 3.0, 236, 12.0, 62.0, "20140415", "110000", 22.0, 4)
    gp4 = grapoint(_rave.RaveValueType_DATA, 4.0, 237, 13.0, 63.0, "20140415", "010000", 23.0, 5)
    gp5 = grapoint(_rave.RaveValueType_DATA, 5.0, 238, 14.0, 64.0, "20140414", "110000", 24.0, 6)
    gp6 = grapoint(_rave.RaveValueType_DATA, 6.0, 239, 15.0, 65.0, "20140414", "010000", 25.0, 7)
    
    testdb().add(gp1)
    testdb().add(gp2)
    testdb().add(gp3)
    testdb().add(gp4)
    testdb().add(gp5)
    testdb().add(gp6)

    # Verify all data    
    result = self.classUnderTest.get_grapoints(datetime.datetime(2014,4,14,0,1,0))
    self.assertEquals(6, len(result))
    self.assertEquals("20140414", result[0].date.strftime("%Y%m%d"))
    self.assertEquals("010000", result[0].time.strftime("%H%M%S"))

    # Execute delete
    self.classUnderTest.delete_grapoints(datetime.datetime(2014,4,15,0,1,0))
    result = self.classUnderTest.get_grapoints(datetime.datetime(2014,4,14,0,1,0))
    self.assertEquals(4, len(result))
    self.assertEquals("20140415", result[0].date.strftime("%Y%m%d"))
    self.assertEquals("010000", result[0].time.strftime("%H%M%S"))

  def test_delete_grapoints_startend(self):
    gp1 = grapoint(_rave.RaveValueType_DATA, 1.0, 234, 10.0, 60.0, "20140416", "110000", 20.0, 2)
    gp2 = grapoint(_rave.RaveValueType_DATA, 2.0, 235, 11.0, 61.0, "20140416", "010000", 21.0, 3)
    gp3 = grapoint(_rave.RaveValueType_DATA, 3.0, 236, 12.0, 62.0, "20140415", "110000", 22.0, 4)
    gp4 = grapoint(_rave.RaveValueType_DATA, 4.0, 237, 13.0, 63.0, "20140415", "010000", 23.0, 5)
    gp5 = grapoint(_rave.RaveValueType_DATA, 5.0, 238, 14.0, 64.0, "20140414", "110000", 24.0, 6)
    gp6 = grapoint(_rave.RaveValueType_DATA, 6.0, 239, 15.0, 65.0, "20140414", "010000", 25.0, 7)
    
    testdb().add(gp1)
    testdb().add(gp2)
    testdb().add(gp3)
    testdb().add(gp4)
    testdb().add(gp5)
    testdb().add(gp6)

    # Execute delete
    self.classUnderTest.delete_grapoints(datetime.datetime(2014,4,14,11,0,0), datetime.datetime(2014,4,16,1,0,0))
    result = self.classUnderTest.get_grapoints(datetime.datetime(2014,4,14,0,1,0))
    self.assertEquals(2, len(result))
    self.assertEquals("20140414", result[0].date.strftime("%Y%m%d"))
    self.assertEquals("010000", result[0].time.strftime("%H%M%S"))
    self.assertEquals("20140416", result[1].date.strftime("%Y%m%d"))
    self.assertEquals("110000", result[1].time.strftime("%H%M%S"))
    
  def test_delete_stations(self):
    station1 = wmo_station("SWEDEN", "0123", "12345", "0", "Pelle", 10.10, 20.20)
    station2 = wmo_station("SWEDEN", "0123", "12346", "0", "Nisse", 30.10, 40.20)
    
    testdb().add(station1)
    testdb().add(station2)
    
    result = self.classUnderTest.get_stations_in_bbox(0.0, 100.0, 50.0, 0.0)
    self.assertEquals(2, len(result))
    
    # Execute delete
    deleted_stations = self.classUnderTest.delete_all_stations()
    self.assertEquals(2, deleted_stations)
    
    # check that all (both) stations are no longer in db
    result = self.classUnderTest.get_stations_in_bbox(0.0, 100.0, 50.0, 0.0)
    self.assertEquals(0, len(result))
    