from sqlalchemy import Table, Column, MetaData, Integer
from sqlalchemy import inspect
from alembic import op

revision = '002'
down_revision = '001'

def upgrade():
    engine = op.get_context().connection.engine

    op.add_column("rave_observation", Column("valid_fields_bitmask", Integer, nullable=True))

    key_name=inspect(engine).get_pk_constraint('rave_observation')['name']
    op.drop_constraint(key_name, 'rave_observation')

    op.create_primary_key('rave_observation_pkey', 'rave_observation', ["station", "date", "time", "accumulation_period"])

def downgrade():
    key_name=inspect(op.get_context().connection.engine).get_pk_constraint('rave_observation')['name']
    op.drop_constraint(key_name, 'rave_observation')

    op.drop_column("rave_observation", "valid_fields_bitmask")

    op.create_primary_key('rave_observation_pkey', 'rave_observation', ["station", "date", "time", "accumulation_period"])  # We really  can't remove the accumulation period since acc-periods always is needed
