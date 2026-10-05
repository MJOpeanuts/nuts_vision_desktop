"""Initial SQLite schema for nuts_vision

Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_now = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "images_input",
        sa.Column("image_id", sa.Integer, primary_key=True),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("upload_at", sa.DateTime, server_default=_now),
        sa.Column("format", sa.String(10)),
    )
    op.create_table(
        "log_jobs",
        sa.Column("job_id", sa.Integer, primary_key=True),
        sa.Column("image_id", sa.Integer,
                  sa.ForeignKey("images_input.image_id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_name", sa.String(255)),
        sa.Column("job_folder_path", sa.Text),
        sa.Column("started_at", sa.DateTime, server_default=_now),
        sa.Column("ended_at", sa.DateTime),
        sa.Column("model", sa.String(255)),
        sa.Column("model_version", sa.String(255)),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("error_message", sa.Text),
    )
    op.create_index("idx_log_jobs_image_id", "log_jobs", ["image_id"])
    op.create_table(
        "detections",
        sa.Column("detection_id", sa.Integer, primary_key=True),
        sa.Column("job_id", sa.Integer,
                  sa.ForeignKey("log_jobs.job_id", ondelete="CASCADE"), nullable=False),
        sa.Column("class_name", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("bbox_x1", sa.Float, nullable=False),
        sa.Column("bbox_y1", sa.Float, nullable=False),
        sa.Column("bbox_x2", sa.Float, nullable=False),
        sa.Column("bbox_y2", sa.Float, nullable=False),
    )
    op.create_index("idx_detections_job_id", "detections", ["job_id"])
    op.create_table(
        "ics_cropped",
        sa.Column("cropped_id", sa.Integer, primary_key=True),
        sa.Column("job_id", sa.Integer,
                  sa.ForeignKey("log_jobs.job_id", ondelete="CASCADE"), nullable=False),
        sa.Column("detection_id", sa.Integer,
                  sa.ForeignKey("detections.detection_id", ondelete="CASCADE"), nullable=False),
        sa.Column("cropped_file_path", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=_now),
    )
    op.create_index("idx_ics_cropped_job_id", "ics_cropped", ["job_id"])
    op.create_index("idx_ics_cropped_detection_id", "ics_cropped", ["detection_id"])

    # PCBA Photo Booth tables (UUIDs stored as text, JSON as JSON)
    op.create_table(
        "log_pcba_pb_import",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
        sa.Column("user_id", sa.String(36)),
        sa.Column("pcba_id", sa.String(36)),
        sa.Column("org_id", sa.String(36)),
        sa.Column("image_storage_path", sa.Text),
        sa.Column("detection_config", sa.JSON),
        sa.Column("total_detections", sa.Integer),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.CheckConstraint(
            "status IN ('pending','processing','completed','error','cancelled')",
            name="ck_lppi_status"),
    )
    op.create_index("idx_lppi_status", "log_pcba_pb_import", ["status"])
    op.create_index("idx_lppi_created_at", "log_pcba_pb_import", ["created_at"])
    op.create_table(
        "log_pcba_pb_row_import",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("log_pcba_pb_import_id", sa.String(36),
                  sa.ForeignKey("log_pcba_pb_import.id", ondelete="CASCADE"), nullable=False),
        sa.Column("row_number", sa.Integer),
        sa.Column("detection_type", sa.Text),
        sa.Column("ic_subtype", sa.Text),
        sa.Column("detection_confidence", sa.Float),
        sa.Column("ic_confidence", sa.Float),
        sa.Column("bounding_box", sa.JSON),
        sa.Column("cropped_image_path", sa.Text),
        sa.Column("processing_status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=_now),
        sa.CheckConstraint(
            "processing_status IN ('pending','processed','error','cancelled')",
            name="ck_lppri_status"),
    )
    op.create_index("idx_lppri_import_id", "log_pcba_pb_row_import", ["log_pcba_pb_import_id"])


def downgrade() -> None:
    for t in ("log_pcba_pb_row_import", "log_pcba_pb_import", "ics_cropped",
              "detections", "log_jobs", "images_input"):
        op.drop_table(t)
