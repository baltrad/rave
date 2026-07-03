from sqlalchemy import Column, Text
from sqlalchemy import inspect
from alembic import op

revision = '005'
down_revision = '004'

def upgrade():
    op.add_column("rave_grapoint", Column("identifier", Text, nullable=True))
    key_name = inspect(op.get_context().connection.engine).get_pk_constraint('rave_grapoint')['name']
    op.drop_constraint(key_name, 'rave_grapoint')

    op.execute("update rave_grapoint set identifier = ''")

    op.alter_column('rave_grapoint', 'identifier', nullable=False)

    op.create_primary_key('rave_grapoint_pkey', 'rave_grapoint', ["identifier", "date", "time", "longitude", "latitude"])

def downgrade():
    key_name = inspect(op.get_context().connection.engine).get_pk_constraint('rave_grapoint')['name']

    op.drop_constraint(key_name, 'rave_grapoint')

    op.drop_column("rave_grapoint", "identifier")

    op.create_primary_key('rave_grapoint_pkey', 'rave_grapoint', ["date", "time", "longitude", "latitude"])
