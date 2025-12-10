from sqlalchemy import Table, Column, MetaData, Integer, Text
from migrate.changeset import constraint
import contextlib


def upgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)
    rave_gra_coefficient = Table('rave_gra_coefficient', meta, autoload=True)
    identifier_column = Column("identifier", Text, nullable=False)
    identifier_column.create(rave_gra_coefficient)

    old_key = constraint.PrimaryKeyConstraint("area", "date", "time", table=rave_gra_coefficient)
    old_key.drop()

    for primary_key in rave_gra_coefficient.primary_key:
        primary_key.primary_key = False

    new_key = constraint.PrimaryKeyConstraint("identifier", "area", "date", "time", table=rave_gra_coefficient)
    new_key.create()

    with contextlib.closing(migrate_engine.connect()) as conn:
        conn.execute("update rave_gra_coefficient set area = '' where area = 'n/a'")

def downgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)
    rave_gra_coefficient = Table('rave_gra_coefficient', meta, autoload=True)
    rave_gra_coefficient.c.identifier.drop()

    new_key = constraint.PrimaryKeyConstraint("identifier", "area", "date", "time", table=rave_gra_coefficient)
    new_key.drop()

    for primary_key in rave_gra_coefficient.primary_key:
        primary_key.primary_key = False

    old_key = constraint.PrimaryKeyConstraint("area", "date", "time", table=rave_gra_coefficient)
    old_key.create()
