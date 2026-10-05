#!/usr/bin/env python3
"""
Database utilities for nuts_vision.

Local SQLite storage (SQLAlchemy 2) with an Alembic-managed schema.
Public methods keep the signatures and return formats of the former
PostgreSQL implementation.

Every operation uses its own short transaction and its own connection, so no
transaction stays open during inference or image writing and nothing is shared
between threads.
"""

import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

try:
    from .config import get_paths
except ImportError:  # imported as a top-level module (src/ on sys.path)
    from config import get_paths

BUSY_TIMEOUT_MS = 5000
_ALEMBIC_DIR = Path(__file__).resolve().parent.parent / "alembic"
_init_lock = threading.Lock()


def _set_sqlite_pragmas(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    try:
        cur.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass  # e.g. filesystem not supporting WAL: keep default journal
    cur.close()


def _rows(result) -> List[Dict[str, Any]]:
    return [dict(r) for r in result.mappings().all()]


class DatabaseManager:
    """Manages the local SQLite database for nuts_vision."""

    def __init__(self, db_path: "str | Path | None" = None, auto_init: bool = True):
        """
        Args:
            db_path: SQLite file (default: <data dir>/database/nuts_vision.sqlite3)
            auto_init: create folders/database and upgrade the schema
        """
        self.db_path = Path(db_path) if db_path else get_paths().database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"timeout": BUSY_TIMEOUT_MS / 1000},
        )
        event.listen(self.engine, "connect", _set_sqlite_pragmas)
        if auto_init:
            self.init_schema()

    def init_schema(self) -> None:
        """Create or upgrade the schema (never resets existing data)."""
        from alembic import command
        from alembic.config import Config

        cfg = Config()
        cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
        cfg.set_main_option("sqlalchemy.url", self.engine.url.render_as_string(hide_password=False).replace("%", "%%"))
        with _init_lock:  # safe against Streamlit reruns / threads
            with self.engine.begin() as conn:
                cfg.attributes["connection"] = conn
                command.upgrade(cfg, "head")

    @contextmanager
    def get_connection(self):
        """Short transaction: commits on success, rolls back on error."""
        with self.engine.begin() as conn:
            yield conn

    # ------------------------------------------------------------------
    # Images / jobs / detections
    # ------------------------------------------------------------------

    def log_image_upload(self, file_name: str, file_path: str, format: str = None) -> int:
        """Log an uploaded image. Returns image_id."""
        with self.get_connection() as conn:
            res = conn.execute(
                text("INSERT INTO images_input (file_name, file_path, format) "
                     "VALUES (:n, :p, :f)"),
                {"n": file_name, "p": file_path, "f": format},
            )
            return int(res.lastrowid)

    def start_job(
        self,
        image_id: int,
        model: str,
        job_name: str = None,
        job_folder_path: str = None,
        model_version: str = None,
        source_path: str = None,
    ) -> int:
        """Start a detection job (status 'processing'). Returns job_id."""
        with self.get_connection() as conn:
            res = conn.execute(
                text("INSERT INTO analyses (image_id, model, model_version, job_name, "
                     "job_folder_path, source_path, status) "
                     "VALUES (:i, :m, :mv, :jn, :jp, :sp, 'processing')"),
                {"i": image_id, "m": model, "mv": model_version,
                 "jn": job_name, "jp": job_folder_path, "sp": source_path},
            )
            return int(res.lastrowid)

    def set_job_files(self, job_id: int, job_name: str = None, job_folder_path: str = None,
                      input_copy_path: str = None, result_image_path: str = None,
                      metadata_path: str = None):
        """Record the files written for an analysis (images stay on disk)."""
        with self.get_connection() as conn:
            conn.execute(
                text("UPDATE analyses SET job_name = :n, job_folder_path = :f, "
                     "input_copy_path = :i, result_image_path = :r, "
                     "metadata_path = :m WHERE job_id = :j"),
                {"n": job_name, "f": job_folder_path, "i": input_copy_path, "r": result_image_path, "m": metadata_path, "j": job_id},
            )

    def end_job(self, job_id: int, status: str = "completed", error_message: str = None):
        """Mark a job as ended ('completed' or 'error')."""
        if status not in ("completed", "error"):
            raise ValueError("status must be 'completed' or 'error'")
        with self.get_connection() as conn:
            conn.execute(
                text("UPDATE analyses SET ended_at = CURRENT_TIMESTAMP, status = :s, "
                     "error_message = :e WHERE job_id = :j"),
                {"s": status, "e": error_message, "j": job_id},
            )

    def log_detection(self, job_id: int, class_name: str, confidence: float,
                      bbox: List[float]) -> int:
        """Log a detection (bbox = [x1, y1, x2, y2]). Returns detection_id."""
        with self.get_connection() as conn:
            res = conn.execute(
                text("INSERT INTO detections (job_id, class_name, confidence, "
                     "bbox_x1, bbox_y1, bbox_x2, bbox_y2) "
                     "VALUES (:j, :c, :conf, :x1, :y1, :x2, :y2)"),
                {"j": job_id, "c": class_name, "conf": float(confidence),
                 "x1": float(bbox[0]), "y1": float(bbox[1]),
                 "x2": float(bbox[2]), "y2": float(bbox[3])},
            )
            return int(res.lastrowid)

    def log_cropped_component(self, job_id: int, detection_id: int,
                              cropped_file_path: str) -> int:
        """Log a cropped component image. Returns cropped_id."""
        with self.get_connection() as conn:
            res = conn.execute(
                text("INSERT INTO crops (job_id, detection_id, cropped_file_path) "
                     "VALUES (:j, :d, :p)"),
                {"j": job_id, "d": detection_id, "p": cropped_file_path},
            )
            return int(res.lastrowid)

    def get_job_statistics(self, job_id: int) -> Dict[str, Any]:
        """Statistics for a specific job."""
        with self.get_connection() as conn:
            res = conn.execute(
                text("""
                    SELECT j.*, i.file_name, i.file_path,
                        COUNT(DISTINCT d.detection_id) AS total_detections,
                        COUNT(DISTINCT ic.cropped_id) AS total_crops
                    FROM analyses j
                    JOIN images_input i ON j.image_id = i.image_id
                    LEFT JOIN detections d ON j.job_id = d.job_id
                    LEFT JOIN crops ic ON j.job_id = ic.job_id
                    WHERE j.job_id = :j
                    GROUP BY j.job_id, i.file_name, i.file_path
                """),
                {"j": job_id},
            )
            rows = _rows(res)
            return rows[0] if rows else {}

    def test_connection(self) -> bool:
        """Return True if the database is reachable."""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False

    def get_all_images(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            return _rows(conn.execute(
                text("SELECT * FROM images_input ORDER BY upload_at DESC, image_id DESC "
                     "LIMIT :l"), {"l": limit}))

    def get_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """All jobs (most recent first) with image information."""
        with self.get_connection() as conn:
            return _rows(conn.execute(
                text("""
                    SELECT j.*, i.file_name, i.file_path, i.format,
                        COUNT(DISTINCT d.detection_id) AS detection_count
                    FROM analyses j
                    JOIN images_input i ON j.image_id = i.image_id
                    LEFT JOIN detections d ON j.job_id = d.job_id
                    GROUP BY j.job_id, i.file_name, i.file_path, i.format
                    ORDER BY j.started_at DESC, j.job_id DESC
                    LIMIT :l
                """), {"l": limit}))

    def get_all_detections(self, job_id: int = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            if job_id:
                res = conn.execute(
                    text("SELECT * FROM detections WHERE job_id = :j "
                         "ORDER BY detection_id DESC LIMIT :l"),
                    {"j": job_id, "l": limit})
            else:
                res = conn.execute(
                    text("SELECT * FROM detections ORDER BY detection_id DESC LIMIT :l"),
                    {"l": limit})
            return _rows(res)

    def get_all_cropped_components(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            return _rows(conn.execute(
                text("""
                    SELECT ic.*, d.class_name, j.job_name
                    FROM crops ic
                    JOIN detections d ON ic.detection_id = d.detection_id
                    JOIN analyses j ON ic.job_id = j.job_id
                    ORDER BY ic.created_at DESC, ic.cropped_id DESC LIMIT :l
                """), {"l": limit}))

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Overall detection statistics."""
        with self.get_connection() as conn:
            stats = _rows(conn.execute(text("""
                SELECT
                    COUNT(DISTINCT i.image_id) AS total_images,
                    COUNT(DISTINCT j.job_id) AS total_jobs,
                    COUNT(DISTINCT d.detection_id) AS total_detections
                FROM images_input i
                LEFT JOIN analyses j ON i.image_id = j.image_id
                LEFT JOIN detections d ON j.job_id = d.job_id
            """)))
            counts = conn.execute(text(
                "SELECT class_name, COUNT(*) AS count FROM detections "
                "GROUP BY class_name ORDER BY count DESC")).mappings().all()
        result = stats[0] if stats else {}
        result["component_counts"] = {r["class_name"]: r["count"] for r in counts}
        return result

    # ------------------------------------------------------------------
    # PCBA Photo Booth logging (log_pcba_pb_import / log_pcba_pb_row_import)
    # ------------------------------------------------------------------

    def create_pcba_import(
        self,
        image_storage_path: str,
        detection_config: Optional[dict] = None,
        total_detections: int = 0,
        status: str = "pending",
        user_id: Optional[str] = None,
        pcba_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> str:
        """Create a PCBA Photo Booth import session. Returns its UUID (str)."""
        import json
        import_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            conn.execute(
                text("""
                    INSERT INTO log_pcba_pb_import
                        (id, image_storage_path, detection_config, total_detections,
                         status, user_id, pcba_id, org_id)
                    VALUES (:id, :p, :cfg, :t, :s, :u, :pc, :o)
                """),
                {"id": import_id, "p": image_storage_path,
                 "cfg": json.dumps(detection_config) if detection_config else None,
                 "t": total_detections, "s": status,
                 "u": user_id, "pc": pcba_id, "o": org_id},
            )
        return import_id

    def update_pcba_import_status(self, import_id: str, status: str,
                                  total_detections: Optional[int] = None) -> None:
        with self.get_connection() as conn:
            if total_detections is not None:
                conn.execute(
                    text("UPDATE log_pcba_pb_import SET status = :s, total_detections = :t "
                         "WHERE id = :i"),
                    {"s": status, "t": total_detections, "i": import_id})
            else:
                conn.execute(
                    text("UPDATE log_pcba_pb_import SET status = :s WHERE id = :i"),
                    {"s": status, "i": import_id})

    def log_pcba_row_import(
        self,
        import_id: str,
        row_number: int,
        detection_type: str,
        detection_confidence: float,
        bounding_box: dict,
        ic_subtype: Optional[str] = None,
        ic_confidence: Optional[float] = None,
        cropped_image_path: Optional[str] = None,
        processing_status: str = "pending",
    ) -> str:
        """Log one detected row of a PCBA import. Returns its UUID (str)."""
        import json
        row_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            conn.execute(
                text("""
                    INSERT INTO log_pcba_pb_row_import
                        (id, log_pcba_pb_import_id, row_number, detection_type,
                         ic_subtype, detection_confidence, ic_confidence,
                         bounding_box, cropped_image_path, processing_status)
                    VALUES (:id, :imp, :rn, :dt, :ics, :dc, :icc, :bb, :cp, :ps)
                """),
                {"id": row_id, "imp": import_id, "rn": row_number, "dt": detection_type,
                 "ics": ic_subtype, "dc": detection_confidence, "icc": ic_confidence,
                 "bb": json.dumps(bounding_box), "cp": cropped_image_path,
                 "ps": processing_status},
            )
        return row_id

    @staticmethod
    def _decode_json(rows: List[Dict[str, Any]], *keys: str) -> List[Dict[str, Any]]:
        import json
        for row in rows:
            for k in keys:
                if isinstance(row.get(k), str):
                    try:
                        row[k] = json.loads(row[k])
                    except ValueError:
                        pass
        return rows

    def get_all_pcba_imports(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = _rows(conn.execute(
                text("""
                    SELECT p.*, COUNT(r.id) AS row_count
                    FROM log_pcba_pb_import p
                    LEFT JOIN log_pcba_pb_row_import r ON r.log_pcba_pb_import_id = p.id
                    GROUP BY p.id
                    ORDER BY p.created_at DESC, p.rowid DESC
                    LIMIT :l
                """), {"l": limit}))
        return self._decode_json(rows, "detection_config")

    def get_pcba_import_rows(self, import_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = _rows(conn.execute(
                text("SELECT * FROM log_pcba_pb_row_import "
                     "WHERE log_pcba_pb_import_id = :i ORDER BY row_number LIMIT :l"),
                {"i": import_id, "l": limit}))
        return self._decode_json(rows, "bounding_box")

    def get_all_pcba_rows(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = _rows(conn.execute(
                text("SELECT * FROM log_pcba_pb_row_import "
                     "ORDER BY created_at DESC, rowid DESC LIMIT :l"), {"l": limit}))
        return self._decode_json(rows, "bounding_box")

    def get_pcba_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics from PCBA Photo Booth tables."""
        with self.get_connection() as conn:
            stats = _rows(conn.execute(text("""
                SELECT
                    COUNT(DISTINCT p.id) AS total_imports,
                    COALESCE(SUM(p.total_detections), 0) AS total_detections,
                    COUNT(DISTINCT CASE WHEN p.status = 'completed' THEN p.id END) AS completed,
                    COUNT(DISTINCT CASE WHEN p.status = 'error' THEN p.id END) AS errors
                FROM log_pcba_pb_import p
            """)))
            counts = conn.execute(text(
                "SELECT detection_type, COUNT(*) AS count FROM log_pcba_pb_row_import "
                "GROUP BY detection_type ORDER BY count DESC")).mappings().all()
        result = stats[0] if stats else {}
        result["component_counts"] = {r["detection_type"]: r["count"] for r in counts}
        return result


def get_db_manager_from_env() -> DatabaseManager:
    """
    Create a DatabaseManager on the local SQLite database.

    The location comes from the paths configuration (src/config.py):
    NUTS_VISION_DATA_DIR overrides the default
    %LOCALAPPDATA%\\DataPeanuts\\NutsVision\\database\\nuts_vision.sqlite3.
    """
    paths = get_paths()
    paths.ensure_dirs()
    return DatabaseManager(paths.database_path)
