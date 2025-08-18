from sqlalchemy import Table, Column, MetaData, Integer
from migrate.changeset import constraint


def upgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)
    rave_observation = Table('rave_observation', meta, autoload=True)
    bitmask_column = Column("valid_fields_bitmask", Integer, nullable=True)
    bitmask_column.create(rave_observation)

    old_key = constraint.PrimaryKeyConstraint("station", "date", "time", "type", table=rave_observation)
    old_key.drop()

    for primary_key in rave_observation.primary_key:
        primary_key.primary_key = False

    new_key = constraint.PrimaryKeyConstraint("station", "date", "time", "accumulation_period", table=rave_observation)
    new_key.create()


def downgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)
    rave_observation = Table('rave_observation', meta, autoload=True)
    rave_observation.c.valid_fields_bitmask.drop()

    new_key = constraint.PrimaryKeyConstraint("station", "date", "time", "accumulation_period", table=rave_observation)
    new_key.drop()

    for primary_key in rave_observation.primary_key:
        primary_key.primary_key = False

    old_key = constraint.PrimaryKeyConstraint("station", "date", "time", "type", table=rave_observation)
    old_key.create()
