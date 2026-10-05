"""Tests for the local SQLite storage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from database import DatabaseManager  # noqa: E402


def test_schema_persists_and_is_not_reset(tmp_path):
    path = tmp_path / "db" / "nuts_vision.sqlite3"
    db = DatabaseManager(path)
    image_id = db.log_image_upload("a.jpg", "/tmp/a.jpg", "jpg")
    job_id = db.start_job(image_id, "m.pt", job_name="a", job_folder_path="/tmp/job")
    det_id = db.log_detection(job_id, "IC", 0.9, [1, 2, 3, 4])
    db.log_cropped_component(job_id, det_id, "/tmp/c.jpg")
    db.end_job(job_id)

    db2 = DatabaseManager(path)  # restart: upgrade only, data kept
    jobs = db2.get_all_jobs()
    assert len(jobs) == 1 and jobs[0]["status"] == "completed"
    stats = db2.get_job_statistics(job_id)
    assert stats["total_detections"] == 1 and stats["total_crops"] == 1
    assert db2.get_detection_statistics()["component_counts"] == {"IC": 1}
    with db2.get_connection() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_pcba_roundtrip(tmp_path):
    db = DatabaseManager(tmp_path / "x.sqlite3")
    imp = db.create_pcba_import("board.jpg", {"conf": 0.3}, 1, "completed")
    db.log_pcba_row_import(imp, 1, "IC", 0.8, {"x": 1, "y": 2})
    assert db.get_all_pcba_imports()[0]["detection_config"] == {"conf": 0.3}
    assert db.get_pcba_import_rows(imp)[0]["bounding_box"] == {"x": 1, "y": 2}
    assert db.get_pcba_statistics()["component_counts"] == {"IC": 1}
