# nuts_vision

Automated electronic component detector for circuit boards — upload a photo of a PCB, detect components with YOLOv8 using dual-model inference (`smd_comp` + `ic_detect`), crop each one individually, and browse the results through a Streamlit web interface.

## What the application does

1. **Upload** one or more PCB photos via the web interface.
2. **Detect** — the `smd_comp` YOLOv8 model identifies 13 component types on the board (Button, Capacitor, Connector, Diode, Electrolytic Capacitor, IC, Inductor, Led, Pads, Pins, Resistor, Switch, Transistor).
3. **IC sub-classification** — the `ic_detect` model classifies ICs by pin layout (`four_side`, `two_side`, `without_side`).
4. **Crop** — each detected component is saved as a separate image.
5. **Browse** — every analysis is stored as a *job* (its own folder) that you can review in the **Job Viewer** page.
6. **Log** *(optional)* — all results are stored in a local SQLite database for later querying and statistics.

### Detected component classes (13)

Button, Capacitor, Connector, Diode, Electrolytic Capacitor, IC, Inductor, Led, Pads, Pins, Resistor, Switch, Transistor

---

## Project structure

```
nuts_vision/
├── app.py                  # Streamlit web interface (main entry point)
├── smd_comp.pt             # Trained YOLOv8 model (component detection, 13 classes)
├── smd_comp.onnx           # ONNX export (deployment: Raspberry Pi, web, etc.)
├── ic_detect_best.onnx     # IC sub-classification model (four_side, two_side, without_side)
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── docker-compose.yml      # Legacy (PostgreSQL), unused
├── start_web.sh / .bat     # Convenience launchers
├── check_dependencies.py   # Dependency checker
├── example.py              # Python usage examples
├── README.md               # This file
├── src/
│   ├── pipeline.py         # Full detect + crop pipeline
│   ├── detect.py           # Component detector + DualModelDetector (YOLOv8 wrapper)
│   ├── crop.py             # Component cropper
│   ├── visualize.py        # Visualization utilities
│   └── database.py         # SQLite logging
└── database/
    └── init.sql            # Database schema
```

### Auto-generated and dataset directories

The following directories are **not tracked by git** (they are listed in `.gitignore`) and can be safely deleted at any time:

| Directory | Origin | Can be deleted? |
|-----------|--------|----------------|
| `outputs/` | Created automatically when running the CLI scripts (`detect.py`, `crop.py`, `visualize.py`, `example.py`) to store results | ✅ Yes — recreated on next run |
| `runs/` | Created automatically by YOLO/Ultralytics during training or inference | ✅ Yes — recreated by YOLO |
| `train/` | YOLO training dataset (images + labels) | ✅ Yes — only needed to retrain the model |
| `valid/` | YOLO validation dataset (images + labels) | ✅ Yes — only needed to retrain the model |
| `test/` | YOLO test dataset (images + labels) | ✅ Yes — only needed to retrain the model |

> **Note:** The web app (`app.py`) does not use `train/`, `valid/`, `test/`, or `runs/`. It stores its results in the `jobs/` folder instead.

### Job output structure

Each processed image produces a job folder:

```
jobs/
  <image_name>_<YYYYMMDD>_<HHMMSS>/
    input.<ext>     — original photo
    result.jpg      — annotated photo with bounding boxes
    crops/          — one cropped image per detected IC
    metadata.json   — detection data (class, confidence, bbox, crop filename)
```

---

## Installation

**Requirements:** Python 3.8+.

```bash
# Install Python dependencies
pip install -r requirements.txt
```

---

## Running the web interface

```bash
# Quickest way (handles venv + .env automatically)
bash start_web.sh          # Linux / macOS
start_web.bat              # Windows

# Or launch directly
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

### Web interface pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview and quick statistics |
| 📤 Upload & Process | Upload PCB images and run the detection pipeline |
| 🔍 Job Viewer | Browse per-job results: input photo, annotated result, crops, metadata |
| 🗄️ Database Viewer | Browse the local SQLite database tables |
| 📊 Statistics | IC counts and job history charts (requires DB) |
| ℹ️ About | Version and environment info |

---

## Running from the command line

```bash
# Process a single image
python src/pipeline.py --model smd_comp.pt --image path/to/board.jpg

# Process a whole directory
python src/pipeline.py --model smd_comp.pt --image-dir path/to/images/

# With database logging
python src/pipeline.py --model smd_comp.pt --image path/to/board.jpg --use-database
```

---

## Local storage (SQLite)

Nuts Vision is self-contained: no Docker, PostgreSQL or network configuration is needed.
On first launch the folders and the SQLite database are created automatically (schema managed by Alembic,
upgraded at each start, never reset).

| Data | Default location (Windows) |
|------|----------------------------|
| Database & logs | `%LOCALAPPDATA%\DataPeanuts\NutsVision\` (`database\nuts_vision.sqlite3`, `logs\`) |
| Images & results | `<Pictures folder>\NutsVision\` (`jobs\`, `imports\`) |

Override with `NUTS_VISION_DATA_DIR` / `NUTS_VISION_OUTPUT_DIR` (see `.env.example`); the results folder can also
be changed from the sidebar **Storage** panel. An **Open analysis folder** button in the Job Viewer opens an analysis in the file explorer.
The history persists across restarts.

> `docker-compose.yml` and `database/*.sql` are legacy PostgreSQL files, kept untouched and no longer used by the app.
> No data is migrated from the old PostgreSQL storage.

---

## YOLO models

The detection system uses a **dual-model** approach:

1. **`smd_comp.pt` / `smd_comp.onnx`** — YOLOv8 model trained to detect **13 component types** on PCB images (Button, Capacitor, Connector, Diode, Electrolytic Capacitor, IC, Inductor, Led, Pads, Pins, Resistor, Switch, Transistor)
2. **`ic_detect_best.onnx`** — YOLOv8 model for IC sub-classification by pin layout (`four_side`, `two_side`, `without_side`)

Both models are forced in the web interface — no manual model selection is needed.

---

## License

CC BY 4.0
