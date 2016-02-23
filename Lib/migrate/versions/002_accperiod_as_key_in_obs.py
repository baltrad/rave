from sqlalchemy import Table, Column, MetaData, Integer
from migrate import constraint

def upgrade(migrate_engine):
  meta = MetaData(bind=migrate_engine)
  rave_observation = Table('rave_observation', meta, autoload=True)
  bitmask_column = Column("valid_fields_bitmask", Integer, nullable=True)
  bitmask_column.create(rave_observation)
    
  old_key = constraint.PrimaryKeyConstraint("station", "date", "time", table=rave_observation)
  new_key = constraint.PrimaryKeyConstraint("station", "date", "time", "accumulation_period", table=rave_observation)
  old_key.drop()  
  new_key.create()


def downgrade(migrate_engine):
  meta = MetaData(bind=migrate_engine)
  rave_observation = Table('rave_observation', meta, autoload=True)
  rave_observation.c.valid_fields_bitmask.drop()

  old_key = constraint.PrimaryKeyConstraint("station", "date", "time", table=rave_observation)
  new_key = constraint.PrimaryKeyConstraint("station", "date", "time", "accumulation_period", table=rave_observation)
  new_key.drop()
  old_key.create()