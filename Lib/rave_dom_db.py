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
##
# The database - domain model setup
#

## 
# @file
# @author Anders Henja, SMHI
# @date 2013-11-06
from __future__ import absolute_import

import contextlib
import jprops
from sqlalchemy import engine, event, exc as sqlexc, sql
from sqlalchemy.orm import mapper, sessionmaker

from sqlalchemy import (
    Column,
    ForeignKey,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    UniqueConstraint,
)

from sqlalchemy.types import (
    BigInteger,
    Boolean,
    Date,
    Float,
    Integer,
    LargeBinary,
    Text,
    Time,
)
from sqlalchemy import asc,desc

from rave_dom import wmo_station, observation
from gadjust.gra import gra_coefficient
from rave_defines import BDB_CONFIG_FILE

def psql_set_extra_float_digits(dbapi_con, con_record):
    cursor = dbapi_con.cursor()
    cursor.execute("SET extra_float_digits=2")
    dbapi_con.commit()

# We use sqlalchemy for creating the tables. If we need to upgrade
# later on, migrate the code to sqlalchemy-migrate.
meta = MetaData()
    
rave_wmo_station=Table("rave_wmo_station", meta,
                       Column("stationnumber", Text, nullable=False),
                       Column("stationsubnumber", Text, nullable=False),
                       Column("stationname", Text, nullable=True),
                       Column("country", Text, nullable=True),
                       Column("countrycode", Text, nullable=True),
                       Column("longitude", Float, nullable=True),
                       Column("latitude", Float, nullable=True),
                       PrimaryKeyConstraint('stationnumber','stationsubnumber',name='pk_wmo_station'),
)

# Even though we are using the wmo station setup for station number we don't want to
# set a foreign constraint on it since we don't know if the stations location will be moved.
#
rave_observation=Table("rave_observation", meta,
                       Column("station", Text, nullable=True),
                       Column("country", Text, nullable=True),
                       Column("type", Integer, nullable=True),
                       Column("date", Date, nullable=False),
                       Column("time", Time, nullable=False),
                       Column("longitude", Float, nullable=False),
                       Column("latitude", Float, nullable=False),
                       Column("visibility", Float, nullable=True),
                       Column("windtype", Integer, nullable=True),
                       Column("cloudcover", Integer, nullable=True),
                       Column("winddirection", Float, nullable=True),
                       Column("windspeed", Float, nullable=True),
                       Column("temperature", Float, nullable=True),
                       Column("dewpoint", Float, nullable=True),
                       Column("relativehumidity", Float, nullable=True),
                       Column("pressure", Float, nullable=True),
                       Column("sea_lvl_pressure", Float, nullable=True),
                       Column("pressure_change", Float, nullable=True),
                       Column("liquid_precipitation", Float, nullable=True),
                       Column("accumulation_period", Integer, nullable=True),
                       PrimaryKeyConstraint("station", "date", "time"))

rave_gra_coefficient=Table("rave_gra_coefficient", meta,
                           Column("area", Text, nullable=False),
                           Column("date", Date, nullable=False),
                           Column("time", Time, nullable=False),
                           Column("significant", Text, nullable=False),
                           Column("points", Integer, nullable=False),
                           Column("loss", Integer, nullable=False),
                           Column("r", Float, nullable=False),
                           Column("r_significant", Text, nullable=False),
                           Column("corr_coeff", Float, nullable=False),
                           Column("a", Float, nullable=False),
                           Column("b", Float, nullable=False),
                           Column("c", Float, nullable=False),
                           Column("mean", Float, nullable=False),
                           Column("stddev", Float, nullable=False),
                           PrimaryKeyConstraint("area","date","time"))
                       
mapper(wmo_station, rave_wmo_station)
mapper(observation, rave_observation)
mapper(gra_coefficient, rave_gra_coefficient)

##
# Class for connecting with the database
class rave_db(object):
  def __init__(self, engine_or_url):
    if isinstance(engine_or_url, basestring):
      self._engine = engine.create_engine(engine_or_url, echo=False)
    else:
      self._engine = engine_or_url
    
    if self._engine.name == "postgresql":
      event.listen(self._engine, "connect", psql_set_extra_float_digits)
  
    meta.bind = self._engine
    
  def get_connection(self):
    """get a context managed connection to the database
    """
    return contextlib.closing(self._engine.connect())

  ##
  # Creates the tables if they don't exist
  def create(self):
    meta.create_all()
    
  ##
  # Drops the database tables if they exist
  def drop(self):
    meta.drop_all()
    
  ##
  # Adds an object to the associated table
  #
  def add(self, obj):
    Session = sessionmaker(bind=self._engine)
    session = Session()
    xlist = obj
    if not isinstance(obj, list):
      xlist = [obj]
    for x in xlist:
      session.add(x)
    session.commit()
    session = None    

  def merge(self, obj):
    Session = sessionmaker(bind=self._engine)
    session = Session()
    xlist = obj
    if not isinstance(obj, list):
      xlist = [obj]
    for x in xlist:
      nx = session.merge(x)
      session.add(nx)
    session.commit()
    session = None    
    

  def get_session(self):
    Session = sessionmaker(bind=self._engine)
    session = Session()
    return contextlib.closing(session)
  
  def get_observations_in_bbox(self, ullon, ullat, lrlon, lrlat, startdt=None, enddt=None):
    with self.get_session() as s:
      q = s.query(observation).filter(observation.longitude>=ullon) \
                                 .filter(observation.longitude<=lrlon) \
                                 .filter(observation.latitude<=ullat) \
                                 .filter(observation.latitude>=lrlat)
      if startdt != None:
        q = q.filter(observation.time>=startdt.time()).filter(observation.date>=startdt.date())
      if enddt != None:
        q = q.filter(observation.time<=enddt.time()).filter(observation.date<=enddt.date())
      
      return q.order_by(asc(observation.station)) \
              .order_by(desc(observation.date)) \
              .order_by(desc(observation.time)) \
              .distinct(observation.station).all()
  
  def get_observations_in_interval(self, startdt, enddt, stations=[]):
    with self.get_session() as s:
      q = s.query(observation)
      if startdt != None:
        q = q.filter(observation.time>=startdt.time()).filter(observation.date>=startdt.date())

      if enddt != None:
        q = q.filter(observation.time<=enddt.time()).filter(observation.date<=enddt.date())

      if stations != None and len(stations) > 0:
        q = q.filter(observation.station.in_(stations))
      
      return q.order_by(asc(observation.station)) \
              .order_by(asc(observation.date)) \
              .order_by(asc(observation.time)).all()
  
  def get_station(self, stationid):
    with self.get_session() as s:
      return s.query(wmo_station).filter(wmo_station.stationnumber==stationid).first()
  
  def get_gra_coefficient(self, dt):
    with self.get_session() as s:
      q = s.query(gra_coefficient).filter(gra_coefficient.date>=dt.date()).filter(gra_coefficient.time>=dt.time())
      q = q.order_by(asc(gra_coefficient.date)).order_by(asc(gra_coefficient.time))
      return q.first()
  
##
# Creates a rave db instance
# If create_schema = True (default) then the tables will be created
#
def create_db(url, create_schema=True):
  db = rave_db(url)
  if create_schema:
    db.create()
  return db

##
# Creates a rave_db instance.
def create_db_from_conf(configfile=BDB_CONFIG_FILE, create_schema=True):
    properties = {}
    try:
      with open(configfile) as fp:
        properties = jprops.load_properties(fp)
    except Exception,e:
      print e.__str__()

    propname = "rave.db.uri"
    if not properties.has_key(propname):
      propname = "baltrad.bdb.server.backend.sqla.uri"

    return create_db(properties[propname], create_schema)
