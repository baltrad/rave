from sqlalchemy import (
    Column,
    MetaData,
    PrimaryKeyConstraint,
    Table
)

from sqlalchemy.types import (
    Date,
    Float,
    Integer,
    Text,
    Time
)

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
                       PrimaryKeyConstraint("station", "date", "time", "type"))

# The coefficients used for gra
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
    
#                           self.radarvaluetype = rt
#    self.radarvalue = rv
#    self.radardistance = rd
#    self.longitude = longitude
#    self.latitude = latitude
#    self.date = date
#    self.time = time
#    self.observation = liquid_precipitation
#    self.accumulation_period = accumulation_period
#    self.gr = -1
#    if self.radarvaluetype == _rave.RaveValueType_DATA and self.radarvalue >= 0.1:
#      self.gr = 10 * log10(self.observation / self.radarvalue)
rave_grapoint=Table("rave_grapoint", meta,
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
                    PrimaryKeyConstraint("date","time","longitude","latitude"))


def upgrade(migrate_engine):
  meta.bind = migrate_engine
  meta.create_all()

def downgrade(migrate_engine):
  meta.bind = migrate_engine
  meta.drop_all()
