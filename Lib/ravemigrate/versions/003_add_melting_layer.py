from migrate.changeset import constraint
from sqlalchemy import (
    Column,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    sql,
)

from sqlalchemy.types import (
    DateTime,
    Float,
    Text,
)


def upgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)

    rave_melting_layer = Table(
        "rave_melting_layer",
        meta,
        Column("nod", Text, nullable=False),
        Column("datetime", DateTime, nullable=False),
        Column("top", Float, nullable=True),
        Column("bottom", Float, nullable=True),
        PrimaryKeyConstraint("datetime", "nod"),
    )

    rave_melting_layer.create()


def downgrade(migrate_engine):
    meta = MetaData(bind=migrate_engine)

    rave_melting_layer = Table("rave_melting_layer", meta, autoload=True)

    rave_melting_layer.drop()
