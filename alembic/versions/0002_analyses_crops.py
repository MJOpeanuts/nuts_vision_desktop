"""Central tables: analyses (was log_jobs) and crops (was ics_cropped)

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.rename_table("log_jobs", "analyses")
    op.rename_table("ics_cropped", "crops")
    op.execute("UPDATE analyses SET status = 'processing' WHERE status = 'running'")
    with op.batch_alter_table("analyses") as batch:
        batch.add_column(sa.Column("source_path", sa.Text))
        batch.add_column(sa.Column("result_image_path", sa.Text))
        batch.add_column(sa.Column("metadata_path", sa.Text))
        batch.add_column(sa.Column("input_copy_path", sa.Text))
        batch.alter_column("status", server_default="processing")
        batch.create_check_constraint(
            "ck_analyses_status", "status IN ('processing','completed','error')")


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.drop_constraint("ck_analyses_status", type_="check")
        for c in ("source_path", "result_image_path", "metadata_path", "input_copy_path"):
            batch.drop_column(c)
    op.rename_table("crops", "ics_cropped")
    op.rename_table("analyses", "log_jobs")
