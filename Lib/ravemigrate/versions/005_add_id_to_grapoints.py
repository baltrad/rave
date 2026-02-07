from sqlalchemy import Table, Column, MetaData, Integer, Text
from migrate.changeset import constraint
import contextlib


def upgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)
    rave_grapoint = Table('rave_grapoint', meta, autoload=True)
    identifier_column = Column("identifier", Text, nullable=True)
    identifier_column.create(rave_grapoint)

def downgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)
    rave_grapoint = Table('rave_grapoint', meta, autoload=True)
    rave_grapoint.c.identifier.drop()
