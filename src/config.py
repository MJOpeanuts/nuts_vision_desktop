#!/usr/bin/env python3
"""
Local paths configuration for nuts_vision.

Technical data (SQLite database, logs) and user data (images and analysis
results) are kept outside the repository / installation folder.

Environment overrides:
    NUTS_VISION_DATA_DIR    technical folder (database/, logs/)
    NUTS_VISION_OUTPUT_DIR  folder receiving images and analysis results
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_VENDOR = "DataPeanuts"
APP_NAME = "NutsVision"


def _windows_known_folder(folder_guid: str) -> "Path | None":
    """Resolve a Windows Known Folder (works even if relocated)."""
    try:
        import ctypes
        from ctypes import wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8),
            ]

        import uuid
        u = uuid.UUID(folder_guid)
        guid = _GUID(u.time_low, u.time_mid, u.time_hi_version,
                     (ctypes.c_ubyte * 8).from_buffer_copy(u.bytes[8:]))
        path_ptr = ctypes.c_wchar_p()
        res = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(path_ptr))
        if res != 0:
            return None
        path = Path(path_ptr.value)
        ctypes.windll.ole32.CoTaskMemFree(path_ptr)
        return path
    except Exception:
        return None


# FOLDERID_Pictures
_PICTURES_GUID = "33E28130-4E1E-4676-835A-98395C3BC3BB"


def default_data_dir() -> Path:
    """Technical folder: %LOCALAPPDATA%\\DataPeanuts\\NutsVision on Windows."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_VENDOR / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_VENDOR / APP_NAME


def default_output_dir() -> Path:
    """User images folder: the real Windows Pictures folder + NutsVision."""
    pictures = _windows_known_folder(_PICTURES_GUID) if sys.platform == "win32" else None
    if pictures is None:
        pictures = Path.home() / "Pictures"
    return pictures / APP_NAME


@dataclass(frozen=True)
class AppPaths:
    data_dir: Path
    output_dir: Path

    @property
    def database_dir(self) -> Path:
        return self.data_dir / "database"

    @property
    def database_path(self) -> Path:
        return self.database_dir / "nuts_vision.sqlite3"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def jobs_dir(self) -> Path:
        """Analysis results (one sub-folder per analysis)."""
        return self.output_dir / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.output_dir / "imports"

    def ensure_dirs(self) -> None:
        for d in (self.database_dir, self.logs_dir, self.jobs_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_paths(output_dir: "str | os.PathLike | None" = None) -> AppPaths:
    """Build the paths configuration (env overrides, then defaults)."""
    data = os.environ.get("NUTS_VISION_DATA_DIR")
    out = output_dir or os.environ.get("NUTS_VISION_OUTPUT_DIR")
    return AppPaths(
        data_dir=Path(data).expanduser() if data else default_data_dir(),
        output_dir=Path(out).expanduser() if out else default_output_dir(),
    )


def open_folder(path: "str | os.PathLike") -> bool:
    """Open a folder in the system file explorer."""
    import subprocess
    p = Path(path)
    if not p.exists():
        return False
    try:
        if sys.platform == "win32":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
        return True
    except Exception:
        return False
