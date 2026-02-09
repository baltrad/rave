from sqlalchemy import Table, Column, MetaData, Integer, Text
from migrate.changeset import constraint
import contextlib


def upgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)
    rave_grapoint = Table('rave_grapoint', meta, autoload=True)
    identifier_column = Column("identifier", Text, nullable=True)
    identifier_column.create(rave_grapoint)

    old_key = constraint.PrimaryKeyConstraint("date", "time", "longitude", "latitude", table=rave_grapoint)
    old_key.drop()

    for primary_key in rave_grapoint.primary_key:
        primary_key.primary_key = False

    with contextlib.closing(migrate_engine.connect()) as conn:
        conn.execute("update rave_grapoint set identifier = ''")

    rave_grapoint.c.identifier.alter(nullable=False)

    new_key = constraint.PrimaryKeyConstraint("identifier", "date", "time", "longitude", "latitude", table=rave_grapoint)
    new_key.create()


def downgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)
    rave_grapoint = Table('rave_grapoint', meta, autoload=True)

    old_key = constraint.PrimaryKeyConstraint("identifier", "date", "time", "longitude", "latitude", table=rave_grapoint)
    old_key.drop()

    rave_grapoint.c.identifier.drop()
    for primary_key in rave_grapoint.primary_key:
        primary_key.primary_key = False

    new_key = constraint.PrimaryKeyConstraint("date", "time", "longitude", "latitude", table=rave_grapoint)
    new_key.create()
