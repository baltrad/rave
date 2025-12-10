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

try:
    from baltradutils import jprops
except:
    import jprops

import datetime
from sqlalchemy import engine, event
from sqlalchemy.orm import mapper, sessionmaker
import rave_pgf_logger

import migrate.versioning.api
import migrate.versioning.repository
from migrate import exceptions as migrateexc


from sqlalchemy import (
    Column,
    MetaData,
    PrimaryKeyConstraint,
    Table,
)

from sqlalchemy.types import Date, Float, Integer, Text, Time, DateTime
from sqlalchemy import asc, desc
from sqlalchemy.exc import OperationalError

from rave_dom import wmo_station, observation, melting_layer
from gadjust.gra import gra_coefficient
from gadjust.grapoint import grapoint
from rave_defines import BDB_CONFIG_FILE

import os
from sqlalchemy.exc import OperationalError
import psycopg2

logger = rave_pgf_logger.create_logger()

MIGRATION_REPO_PATH = os.path.join(os.path.dirname(__file__), "ravemigrate")


def psql_set_extra_float_digits(dbapi_con, con_record):
    cursor = dbapi_con.cursor()
    cursor.execute("SET extra_float_digits=2")
    dbapi_con.commit()


# def psql_checkout(dbapi_conn, connection_rec, connection_proxy):
#    logger.info("CHECKOUT")

# def psql_checkin(dbapi_conn, connection_rec):
#    logger.info("CHECKIN")

# def psql_reset(dbapi_conn, connection_rec):
#    logger.info("RESETING")

# We use sqlalchemy for creating the tables. If we need to upgrade
# later on, migrate the code to sqlalchemy-migrate.
meta = MetaData()

rave_wmo_station = Table(
    "rave_wmo_station",
    meta,
    Column("stationnumber", Text, nullable=False),
    Column("stationsubnumber", Text, nullable=False),
    Column("stationname", Text, nullable=True),
    Column("country", Text, nullable=True),
    Column("countrycode", Text, nullable=True),
    Column("longitude", Float, nullable=True),
    Column("latitude", Float, nullable=True),
    PrimaryKeyConstraint('stationnumber', 'stationsubnumber', name='pk_wmo_station'),
)

# Even though we are using the wmo station setup for station number we don't want to
# set a foreign constraint on it since we don't know if the stations location will be moved.
#
rave_observation = Table(
    "rave_observation",
    meta,
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
    Column("valid_fields_bitmask", Integer, nullable=True),
    PrimaryKeyConstraint("station", "date", "time", "accumulation_period"),
)

# The coefficients used for gra
rave_gra_coefficient = Table(
    "rave_gra_coefficient",
    meta,
    Column("identifier", Text, nullable=False),
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
    PrimaryKeyConstraint("identifier", "area", "date", "time"),
)

rave_grapoint = Table(
    "rave_grapoint",
    meta,
    Column("date", Date, nullable=False),
    Column("time", Time, nullable=False),
    Column("radarvaluetype", Integer, nullable=False),
    Column("radarvalue", Float, nullable=False),
    Column("radardistance", Float, nullable=False),
    Column("longitude", Float, nullable=False),
    Column("latitude", Float, nullable=False),
    Column("observation", Float, nullable=False),
    Column("accumulation_period", Integer, nullable=False),
    Column("gr", Float, nullable=False),
    PrimaryKeyConstraint("date", "time", "longitude", "latitude"),
)

rave_melting_layer = Table(
    "rave_melting_layer",
    meta,
    Column("nod", Text, nullable=False),
    Column("datetime", DateTime, nullable=False),
    Column("top", Float, nullable=True),
    Column("bottom", Float, nullable=True),
    PrimaryKeyConstraint("datetime", "nod"),
)

mapper(wmo_station, rave_wmo_station)
mapper(observation, rave_observation)
mapper(melting_layer, rave_melting_layer)

mapper(gra_coefficient, rave_gra_coefficient)
mapper(grapoint, rave_grapoint)

dburipool = {}


##
# Class for connecting with the database
class rave_db(object):
    def __init__(self, engine_or_url):
        if isinstance(engine_or_url, str):
            self._engine = engine.create_engine(engine_or_url, echo=False)
        else:
            self._engine = engine_or_url

        if self._engine.name == "postgresql":
            event.listen(self._engine, "connect", psql_set_extra_float_digits)
            event.listen(self._engine, 'invalidate', self.psql_invalidate)

        meta.bind = self._engine
        self.Session = sessionmaker(bind=self._engine)

    def get_connection(self):
        """get a context managed connection to the database"""
        return contextlib.closing(self._engine.connect())

    def psql_invalidate(self, dbapi_conn, connection_rec, exception):
        if exception != None and isinstance(exception, OperationalError):
            if "server closed the connection unexpectedly" in exception.message:
                logger.warning(
                    "Got invalidation message indicating that there has been connection problems. Recreating pool."
                )
                self._engine.dispose()
        elif exception != None and isinstance(exception, psycopg2.OperationalError):
            logger.warning("psycopg2,OperationalError will be tested")
            if "server closed the connection unexpectedly" in exception.message:
                logger.warning(
                    "Got invalidation message indicating that there has been connection problems. Recreating pool."
                )
                self._engine.dispose()

    ##
    # Creates the tables if they don't exist
    def create(self):
        repo = migrate.versioning.repository.Repository(MIGRATION_REPO_PATH)

        # try setting up version control for the databases created before
        # we started using sqlalchemy-migrate
        try:
            migrate.versioning.api.version_control(self._engine, repo, version=0)
        except migrateexc.DatabaseAlreadyControlledError:
            pass

        migrate.versioning.api.upgrade(self._engine, repo)

    ##
    # Drops the database tables if they exist
    def drop(self):
        repo = migrate.versioning.repository.Repository(MIGRATION_REPO_PATH)
        try:
            migrate.versioning.api.downgrade(self._engine, repo, 0)
            migrate.versioning.api.drop_version_control(self._engine, repo)
        except migrateexc.DatabaseNotControlledError:
            pass

    ##
    # Adds an object to the associated table
    #
    def add(self, obj):
        session = self.Session()
        xlist = obj
        if not isinstance(obj, list):
            xlist = [obj]
        try:
            for x in xlist:
                session.add(x)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
        session = None

    def merge(self, obj):
        session = self.Session()
        xlist = obj
        if not isinstance(obj, list):
            xlist = [obj]
        try:
            for x in xlist:
                nx = session.merge(x)
                session.add(nx)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
        session = None

    def get_session(self):
        session = self.Session()
        return contextlib.closing(session)

    def get_observations_in_bbox(self, ullon, ullat, lrlon, lrlat, startdt=None, enddt=None):
        with self.get_session() as s:
            q = (
                s.query(observation)
                .filter(observation.longitude >= ullon)
                .filter(observation.longitude <= lrlon)
                .filter(observation.latitude <= ullat)
                .filter(observation.latitude >= lrlat)
            )
            if startdt != None:
                q = q.filter(observation.date + observation.time >= startdt)
            if enddt != None:
                q = q.filter(observation.date + observation.time <= enddt)

            return (
                q.order_by(asc(observation.station))
                .order_by(desc(observation.date))
                .order_by(desc(observation.time))
                .all()
            )

    def get_observations_in_interval(self, startdt, enddt, stations=[]):
        with self.get_session() as s:
            q = s.query(observation)
            if startdt != None:
                q = q.filter(observation.date + observation.time >= startdt)

            if enddt != None:
                q = q.filter(observation.date + observation.time <= enddt)

            if stations != None and len(stations) > 0:
                q = q.filter(observation.station.in_(stations))

            return (
                q.order_by(asc(observation.station))
                .order_by(asc(observation.date))
                .order_by(asc(observation.time))
                .all()
            )

    def delete_observations_in_interval(self, startdt, enddt):
        with self.get_session() as s:
            q = s.query(observation)
            if startdt != None:
                q = q.filter(observation.date + observation.time >= startdt)

            if enddt != None:
                q = q.filter(observation.date + observation.time <= enddt)

            no_of_observations = q.delete(synchronize_session=False)
            s.commit()

            return no_of_observations

    def get_station(self, stationid):
        with self.get_session() as s:
            return s.query(wmo_station).filter(wmo_station.stationnumber == stationid).first()

    def get_stations_in_bbox(self, ullon, ullat, lrlon, lrlat):
        with self.get_session() as s:
            q = (
                s.query(wmo_station)
                .filter(wmo_station.longitude >= ullon)
                .filter(wmo_station.longitude <= lrlon)
                .filter(wmo_station.latitude <= ullat)
                .filter(wmo_station.latitude >= lrlat)
            )

            return q.all()

    def delete_all_stations(self):
        with self.get_session() as s:
            q = s.query(wmo_station)
            no_of_stations = q.delete()
            s.commit()
            return no_of_stations

    def get_gra_coefficient(self, dt, identifier=None):
        with self.get_session() as s:
            q = s.query(gra_coefficient).filter(gra_coefficient.date + gra_coefficient.time >= dt)
            if identifier:
                q = q.filter(gra_coefficient.identifier == identifier)
            else:
                q = q.filter(gra_coefficient.identifier == '')
            q = q.order_by(asc(gra_coefficient.date)).order_by(asc(gra_coefficient.time))
            return q.first()

    ## Return the most recent gra coefficient since dt
    # @param dt: From time when to search for coefficients
    def get_newest_gra_coefficient(self, dt, identifier=None, dtmax=None):
        with self.get_session() as s:
            q = s.query(gra_coefficient).filter(gra_coefficient.date + gra_coefficient.time >= dt)
            if dtmax != None:
                q = q.filter(gra_coefficient.date + gra_coefficient.time <= dtmax)
            if identifier:
                q = q.filter(gra_coefficient.identifier == identifier)
            else:
                q = q.filter(gra_coefficient.identifier == '')

            q = q.order_by(desc(gra_coefficient.date + gra_coefficient.time))
            return q.first()

    def get_grapoints(self, dt, edt=None):
        with self.get_session() as s:
            q = s.query(grapoint).filter(grapoint.date + grapoint.time >= dt)
            if edt is not None:
                q = q.filter(grapoint.date + grapoint.time <= edt)
            q = q.order_by(asc(grapoint.date)).order_by(asc(grapoint.time))
            return q.all()

    def delete_grapoints(self, dt, edt=None):
        with self.get_session() as s:
            q = s.query(grapoint).filter(grapoint.date + grapoint.time <= dt)
            if edt is not None:
                # If edt is specified, we want to delete within specified range
                q = s.query(grapoint).filter(grapoint.date + grapoint.time >= dt)
                q = q.filter(grapoint.date + grapoint.time <= edt)

            pts = q.delete(synchronize_session=False)
            s.commit()
            return pts

    def purge_grapoints(self):
        with self.get_session() as s:
            q = s.query(grapoint)
            pts = q.delete()
            s.commit()
            return pts

    def get_latest_melting_layer(self, nod, hours=None, ct=datetime.datetime.utcnow()):
        with self.get_session() as s:
            q = s.query(melting_layer).filter(melting_layer.nod == nod)
            if hours is not None:
                q = q.filter(melting_layer.datetime > ct - datetime.timedelta(hours=hours)).filter(
                    melting_layer.datetime <= ct
                )
            q = q.order_by(desc(melting_layer.datetime)).limit(1)
            return q.first()

    def remove_old_melting_layers(self, ct=None):
        with self.get_session() as s:
            if ct is None:
                ct = datetime.datetime.utcnow() - datetime.timedelta(hours=168)
            q = s.query(melting_layer).filter(melting_layer.datetime < ct)
            pts = q.delete()
            s.commit()
            return pts


##
# Creates a rave db instance. This instance will be remembered for the same url which means
# that the same db-instance will be used for the same url.
# If create_schema = True (default) then the tables will be created
#
def create_db(url, create_schema=True):
    if url not in dburipool:
        db = rave_db(url)
        if create_schema:
            db.create()
        dburipool[url] = db
    return dburipool[url]


##
# Creates a rave_db instance.
def create_db_from_conf(configfile=BDB_CONFIG_FILE, create_schema=True):
    properties = {}
    try:
        with open(configfile) as fp:
            properties = jprops.load_properties(fp)
    except Exception as e:
        print(e.__str__())

    propname = "rave.db.uri"
    if not propname in properties:
        propname = "baltrad.bdb.server.backend.sqla.uri"

    return create_db(properties[propname], create_schema)
