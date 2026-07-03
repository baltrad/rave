from sqlalchemy import Table, Column, MetaData, Integer, Text
from sqlalchemy import inspect
from alembic import op
import contextlib

revision = '004'
down_revision = '003'

def upgrade():
    op.add_column("rave_gra_coefficient", Column("identifier", Text, nullable=True))

    key_name=inspect(op.get_context().connection.engine).get_pk_constraint('rave_gra_coefficient')['name']
    op.drop_constraint(key_name, 'rave_gra_coefficient')

    op.execute("update rave_gra_coefficient set area = '' where area = 'n/a'")
    op.execute("update rave_gra_coefficient set identifier = ''")

    op.alter_column('rave_gra_coefficient', 'identifier', nullable=False)

    op.create_primary_key('rave_gra_coefficient_pkey', 'rave_gra_coefficient', ["identifier", "area", "date", "time"])

def downgrade():
    key_name=inspect(op.get_context().connection.engine).get_pk_constraint('rave_gra_coefficient')['name']
    op.drop_constraint(key_name, 'rave_gra_coefficient')

    op.drop_column("rave_gra_coefficient", "identifier")

    op.create_primary_key('rave_gra_coefficient_pkey', 'rave_gra_coefficient', ["area", "date", "time"])
