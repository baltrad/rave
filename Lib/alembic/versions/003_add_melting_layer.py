from alembic import op
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

revision = '003'
down_revision = '002'

def upgrade():
    op.create_table(
        "rave_melting_layer",
        Column("nod", Text, nullable=False),
        Column("datetime", DateTime, nullable=False),
        Column("top", Float, nullable=True),
        Column("bottom", Float, nullable=True),
        PrimaryKeyConstraint("datetime", "nod")
    )


def downgrade():
    op.drop_table("rave_melting_layer")
