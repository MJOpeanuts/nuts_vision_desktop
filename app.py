#!/usr/bin/env python3
"""
Streamlit Web Interface for nuts_vision
Provides a graphical interface for:
- PCB image upload and component detection
- PCBA Photo Booth (dual-model detection pipeline)
- Per-job viewer (input photo, result photo, cropped components, metadata)
- Database viewer
- Statistics
"""

import io
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import modules
try:
    from config import get_paths, open_folder
    from database import get_db_manager_from_env
    from pipeline import ComponentAnalysisPipeline
    DB_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    DB_AVAILABLE = False

try:
    from detect import DualModelDetector
    DUAL_DETECTOR_AVAILABLE = True
except ImportError:
    DUAL_DETECTOR_AVAILABLE = False

# All 13 component classes for smd_comp
COMP_DETECT_CLASSES = [
    'Button', 'Capacitor', 'Connector', 'Diode',
    'Electrolytic Capacitor', 'IC', 'Inductor', 'Led',
    'Pads', 'Pins', 'Resistor', 'Switch', 'Transistor'
]


def _detect_model_formats(comp_base: str = "smd_comp", ic_base: str = "ic_detect_best",
                           root: Path = None) -> list:
    """Return list of available model format labels (e.g. ['ONNX (.onnx)', 'PT (.pt)'])
    based on whether the *comp* model file exists in each format.
    The ic_detect model is optional; it does not gate format availability."""
    root = root or Path(".")
    formats = []
    if (root / f"{comp_base}.onnx").exists():
        formats.append("ONNX (.onnx)")
    if (root / f"{comp_base}.pt").exists():
        formats.append("PT (.pt)")
    return formats


# Page configuration
st.set_page_config(
    page_title="nuts_vision - IC Detector",
    page_icon="\U0001f527",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .dataframe { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# Local storage: output folder is selectable (sidebar), data folder is technical
if DB_AVAILABLE:
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = str(get_paths().output_dir)
    APP_PATHS = get_paths(st.session_state.output_dir)
    JOBS_DIR = APP_PATHS.jobs_dir
    try:
        APP_PATHS.ensure_dirs()
    except OSError as e:
        st.error(f"Cannot create the data folders: {e}")


@st.cache_resource
def _get_db():
    """One DatabaseManager per server process (schema init is rerun-safe)."""
    return get_db_manager_from_env()


# Initialize session state
if DB_AVAILABLE and not st.session_state.get("db_connected", False):
    try:
        st.session_state.db = _get_db()
        st.session_state.db_connected = st.session_state.db.test_connection()
        if st.session_state.db_connected and "db_error" in st.session_state:
            del st.session_state.db_error
    except Exception as e:
        st.session_state.db_connected = False
        st.session_state.db_error = str(e)

# Sidebar
st.sidebar.markdown("## \U0001f527 nuts_vision")
st.sidebar.markdown("**Electronic Component Analyzer**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["\U0001f3e0 Home", "\U0001f4e4 Upload & Process", "\U0001f4f7 PCBA Photo Booth",
     "\U0001f50d Job Viewer", "\U0001f5c4\ufe0f Database Viewer",
     "\U0001f4ca Statistics", "\u2139\ufe0f About"]
)

st.sidebar.markdown("---")
if DB_AVAILABLE:
    if st.session_state.get("db_connected", False):
        st.sidebar.success("\u2705 Local database ready")
    else:
        st.sidebar.warning("\u26a0\ufe0f Local database unavailable")
        if "db_error" in st.session_state:
            st.sidebar.text(f"Error: {st.session_state.db_error}")
        if st.sidebar.button("\U0001f504 Retry Connection"):
            st.session_state.db_connected = False
            if "db_error" in st.session_state:
                del st.session_state.db_error
            st.rerun()
else:
    st.sidebar.warning("\u26a0\ufe0f Database Module Not Available")

if DB_AVAILABLE:
    with st.sidebar.expander("\U0001f4c2 Storage"):
        new_out = st.text_input("Images & results folder", value=st.session_state.output_dir)
        if new_out and new_out != st.session_state.output_dir:
            st.session_state.output_dir = new_out
            st.rerun()
        st.caption(f"Database: {APP_PATHS.database_path}")
        if st.button("Open results folder"):
            if not open_folder(JOBS_DIR):
                st.warning(f"Cannot open: {JOBS_DIR}")


# ========== HOME PAGE ==========
if page == "\U0001f3e0 Home":
    st.markdown('<div class="main-header">\U0001f527 nuts_vision</div>', unsafe_allow_html=True)
    st.markdown("### IC Detection System")
    st.markdown(r"""
    Welcome to **nuts_vision** — an automated system for detecting electronic components on circuit boards.

    #### 🎯 How it works:
    1. **Upload** a photo of an electronic circuit board
    2. **Detect** — YOLOv8 identifies up to 13 component types on the board using `smd_comp`
    3. **IC sub-classification** — a dedicated `ic_detect` model classifies ICs by pin layout (`four_side`, `two_side`, `without_side`)
    4. **Crop** — each detected component is saved as a separate image
    5. **Browse** — all results are organized in a per-job folder

    #### 📋 Detected components (13 classes):
    """)
    _class_cols = st.columns(4)
    for idx, cls in enumerate(COMP_DETECT_CLASSES):
        with _class_cols[idx % 4]:
            st.markdown(f"\u2713 {cls}")

    st.markdown("---")
    st.markdown("""
    #### \U0001f4c1 Output structure (per job):
    ```
    <output folder>/jobs/
      <image_name>_<date>_<time>/
        input.<ext>    — original photo
        result.jpg     — annotated photo with bounding boxes
        crops/         — one cropped image per detected component
        metadata.json  — detection data
    ```
    """)

    if st.session_state.get("db_connected", False):
        try:
            stats = st.session_state.db.get_detection_statistics()
            st.markdown("---")
            st.markdown("### \U0001f4ca Quick Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", stats.get("total_images", 0))
            with col2:
                st.metric("Total Jobs", stats.get("total_jobs", 0))
            with col3:
                st.metric("Detections", stats.get("total_detections", 0))
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")


# ========== UPLOAD & PROCESS PAGE ==========
elif page == "\U0001f4e4 Upload & Process":
    st.markdown('<div class="main-header">\U0001f4e4 Upload & Process Images</div>', unsafe_allow_html=True)

    # Detect available model formats
    _up_root = Path(__file__).parent.resolve()
    _up_formats = _detect_model_formats(comp_base="smd_comp", root=_up_root)

    if not _up_formats:
        st.error("""
        \u274c **No model found!**
        Make sure `smd_comp.pt` or `smd_comp.onnx` is present in the project root directory.
        """)
        model_path = None
    else:
        st.markdown("### Model Configuration")
        col1, col2 = st.columns([3, 1])
        with col1:
            if len(_up_formats) > 1:
                _up_fmt = st.radio(
                    "Model format",
                    options=_up_formats,
                    index=0,
                    horizontal=True,
                    help="Choose between PyTorch (.pt) and ONNX (.onnx) model format",
                    key="up_model_format",
                )
            else:
                _up_fmt = _up_formats[0]
                st.text_input("Model format", value=_up_fmt, disabled=True, key="up_model_format_display")

            if "ONNX" in _up_fmt:
                model_path = str(_up_root / "smd_comp.onnx")
            else:
                model_path = str(_up_root / "smd_comp.pt")
            st.text_input("Model", value=Path(model_path).name, disabled=True,
                           help="smd_comp model — selected for component detection")
        with col2:
            conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9,
                                        value=0.25, step=0.05)

    if model_path is None:
        st.stop()

    st.markdown("### Class Filter")
    selected_classes = st.multiselect(
        "Classes to detect (leave empty = detect all)",
        options=COMP_DETECT_CLASSES,
        default=[],
        help="Select which component types to include in the results. Empty means all 13 classes."
    )

    st.markdown("### Upload Images")
    uploaded_files = st.file_uploader("Choose PCB images", type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True)

    st.markdown("### Processing Options")
    use_database = st.checkbox("Log to Database", value=True)

    if st.button("\U0001f680 Start Processing", type="primary",
                 disabled=not uploaded_files or not model_path):
        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
        else:
            upload_dir = APP_PATHS.uploads_dir
            upload_dir.mkdir(parents=True, exist_ok=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("Initializing pipeline...")
                pipeline = ComponentAnalysisPipeline(
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                    use_database=use_database and st.session_state.get("db_connected", False)
                )
                total_files = len(uploaded_files)
                results_summary = []
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name} ({idx+1}/{total_files})...")
                    progress_bar.progress(idx / total_files)
                    file_path = upload_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        result = pipeline.process_image(str(file_path.resolve()), jobs_base_dir=str(JOBS_DIR))
                        # Apply optional class filter on the metadata
                        dets = result["metadata"]["total_detections"]
                        if selected_classes:
                            filtered = [d for d in result["metadata"]["detections"]
                                        if d["class_name"] in selected_classes]
                            dets = len(filtered)
                        results_summary.append({
                            "file": uploaded_file.name,
                            "status": "\u2705 Success",
                            "job_folder": result["job_folder"],
                            "detections": dets
                        })
                    except Exception as e:
                        results_summary.append({
                            "file": uploaded_file.name,
                            "status": f"\u274c Error: {str(e)}",
                            "job_folder": "",
                            "detections": 0
                        })
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                st.success("\u2705 Processing completed!")
                st.markdown("### Results Summary")
                st.dataframe(pd.DataFrame(results_summary), width="stretch")
                st.info("\U0001f4c1 View detailed results in the **Job Viewer** page.")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if uploaded_files:
        st.markdown("---")
        st.markdown("### \U0001f4f7 Uploaded Images Preview")
        cols = st.columns(min(3, len(uploaded_files)))
        for idx, uploaded_file in enumerate(uploaded_files[:6]):
            with cols[idx % 3]:
                _preview = Image.open(uploaded_file)
                _preview = ImageOps.exif_transpose(_preview)
                st.image(_preview, caption=uploaded_file.name, width="stretch")



# ========== PCBA PHOTO BOOTH PAGE ==========
elif page == "\U0001f4f7 PCBA Photo Booth":
    st.markdown('<div class="main-header">\U0001f4f7 PCBA Photo Booth</div>', unsafe_allow_html=True)
    st.markdown(
        "Dual-model PCBA inspection pipeline: upload → infer → validate → crop."
    )

    if not DUAL_DETECTOR_AVAILABLE:
        st.error("DualModelDetector could not be imported. Check that `src/detect.py` is present.")
        st.stop()

    # ------------------------------------------------------------------
    # Helper: draw bounding boxes with semi-transparent filled zones
    # ------------------------------------------------------------------
    DETECTION_PALETTE = {
        'Button': '#FF5733', 'Capacitor': '#33FFF5', 'Connector': '#A833FF',
        'Diode': '#FF8C00', 'Electrolytic Capacitor': '#1E90FF',
        'IC': '#FF33A8', 'Inductor': '#32CD32', 'Led': '#33FF57',
        'Pads': '#FFD700', 'Pins': '#00CED1', 'Resistor': '#FF6347',
        'Switch': '#7CFC00', 'Transistor': '#DC143C',
    }

    def _hex_to_rgba(hex_color: str, alpha: int = 60):
        """Convert '#RRGGBB' to an (R, G, B, A) tuple."""
        h = hex_color.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)

    def _draw_boxes(pil_img: Image.Image, detections: list) -> Image.Image:
        img = pil_img.copy().convert("RGBA")
        # Transparent overlay for filled zones
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(img)

        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cls = det.get('class_name', '?')
            color_hex = DETECTION_PALETTE.get(cls, '#FFFFFF')
            fill_rgba = _hex_to_rgba(color_hex, alpha=90)

            # Semi-transparent filled zone
            overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_rgba)

            # Solid outline (thicker for visibility)
            draw.rectangle([x1, y1, x2, y2], outline=color_hex, width=4)

            # Label
            label = cls
            if det.get('ic_subtype'):
                label += f" [{det['ic_subtype']}]"
            conf = det.get('confidence', 0)
            label += f" {conf:.2f}"

            # Label background for readability
            text_y = max(0, y1 - 16)
            text_w = draw.textlength(label) if hasattr(draw, 'textlength') else len(label) * 7
            draw.rectangle(
                [x1, text_y, x1 + int(text_w) + 6, text_y + 14],
                fill=color_hex,
            )
            draw.text((x1 + 2, text_y + 1), label, fill="white")

        # Composite overlay onto image
        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    # ------------------------------------------------------------------
    # STEP 1 — Upload
    # ------------------------------------------------------------------
    st.markdown("## Step 1 — Upload Image")
    uploaded = st.file_uploader(
        "Drop a PCB image here",
        type=["jpg", "jpeg", "png"],
        key="pb_upload"
    )

    if uploaded:
        # Only reset detections when a genuinely new file is uploaded,
        # not on every Streamlit rerun (which would wipe Step 4 state).
        current_file_id = uploaded.file_id
        if current_file_id != st.session_state.get("pb_upload_file_id"):
            st.session_state["pb_upload_file_id"] = current_file_id
            st.session_state["pb_image_bytes"] = uploaded.read()
            # Sanitize: keep only the basename, strip any path separators
            st.session_state["pb_image_name"] = Path(uploaded.name).name
            st.session_state.pop("pb_detections", None)  # reset previous run

    if "pb_image_bytes" not in st.session_state:
        st.info("Upload a PCB image to start.")
        st.stop()

    img_bytes = st.session_state["pb_image_bytes"]
    img_name  = st.session_state["pb_image_name"]
    pil_image = Image.open(io.BytesIO(img_bytes))
    # Apply EXIF orientation so the displayed image matches what the
    # detection pipeline sees (OpenCV ignores EXIF tags).
    pil_image = ImageOps.exif_transpose(pil_image)

    st.image(pil_image, caption=img_name, width="stretch")

    # Job name input
    default_job_name = Path(img_name).stem
    job_name_input = st.text_input(
        "Job name",
        value=default_job_name,
        help="Give this job a meaningful name. It will be used for the output folder.",
        key="pb_job_name",
    )

    # ------------------------------------------------------------------
    # STEP 2 — Model configuration & inference
    # ------------------------------------------------------------------
    st.markdown("## Step 2 — Detection Configuration")

    _project_root = Path(__file__).parent.resolve()

    # Detect available model formats
    _pb_formats = _detect_model_formats(comp_base="smd_comp", root=_project_root)

    if not _pb_formats:
        st.error("⚠️ No smd_comp model files found in the project root.")
        st.stop()

    if len(_pb_formats) > 1:
        _pb_fmt = st.radio(
            "Model format",
            options=_pb_formats,
            index=0,
            horizontal=True,
            help="Choose between ONNX and PyTorch (.pt) model format",
            key="pb_model_format",
        )
    else:
        _pb_fmt = _pb_formats[0]
        st.info(f"Model format: **{_pb_fmt}** (only format available)")

    if "ONNX" in _pb_fmt:
        comp_model_name = "smd_comp.onnx"
        ic_model_name = "ic_detect_best.onnx"
    else:
        comp_model_name = "smd_comp.pt"
        ic_model_name = "ic_detect_best.pt"

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.text_input("smd_comp model", value=comp_model_name, disabled=True,
                       help="smd_comp model — selected for component detection",
                       key="pb_comp_model")
        comp_conf = st.slider("smd_comp confidence", 0.1, 0.9, 0.25, 0.05, key="pb_comp_conf")
    with col_m2:
        st.text_input("ic_detect model", value=ic_model_name, disabled=True,
                       help="ic_detect_best model — selected for IC sub-classification",
                       key="pb_ic_model")
        ic_conf = st.slider("ic_detect confidence", 0.1, 0.9, 0.25, 0.05, key="pb_ic_conf")

    # Check model files exist
    _comp_model_path = _project_root / comp_model_name
    _ic_model_path = _project_root / ic_model_name
    if not _comp_model_path.exists():
        st.warning(f"⚠️ smd_comp model not found: {comp_model_name}")
    if not _ic_model_path.exists():
        st.warning(f"⚠️ ic_detect model not found: {ic_model_name}")

    st.markdown("### Classes to detect")
    selected_classes = st.multiselect(
        "Select component types (empty = all 13)",
        options=COMP_DETECT_CLASSES,
        default=COMP_DETECT_CLASSES,
        key="pb_class_filter"
    )

    st.markdown("### Image Preprocessing")
    col_pp1, col_pp2 = st.columns(2)
    with col_pp1:
        pb_apply_clahe = st.checkbox(
            "CLAHE contrast enhancement",
            value=True,
            key="pb_apply_clahe",
            help="Adaptive histogram equalization — improves contrast in dark peripheral zones (vignetting).",
        )
    with col_pp2:
        pb_apply_sharpen = st.checkbox(
            "Sharpening",
            value=False,
            key="pb_apply_sharpen",
            help="Mild unsharp-mask kernel — recovers edge detail lost to optical blur at the frame edges.",
        )

    run_inference = st.button("\U0001f50d Run Detection", type="primary",
                              disabled=not _comp_model_path.exists())

    if run_inference:
        import uuid as _uuid
        # Use only server-side paths resolved from the project root
        comp_model_resolved = _comp_model_path.resolve()
        ic_model_resolved = _ic_model_path.resolve() if _ic_model_path.exists() else None

        if not comp_model_resolved.exists():
            st.error(f"smd_comp model not found: {comp_model_name}")
        else:
            # Save image to a temp file using a UUID filename (never user-controlled)
            tmp_dir = Path("/tmp/pb_uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            # Preserve the original extension but use a random name to avoid collisions
            _orig_suffix = Path(img_name).suffix or ".jpg"
            tmp_path = tmp_dir / f"{_uuid.uuid4().hex}{_orig_suffix}"
            tmp_path.write_bytes(img_bytes)

            ic_path_arg = str(ic_model_resolved) if ic_model_resolved else None
            if not ic_model_resolved:
                st.warning(f"ic_detect model not found: {ic_model_name} — running single-model mode.")

            with st.spinner("Running dual-model inference…"):
                try:
                    detector = DualModelDetector(
                        comp_model_path=str(comp_model_resolved),
                        ic_model_path=ic_path_arg,
                        comp_conf=comp_conf,
                        ic_conf=ic_conf,
                    )
                    detections = detector.detect(
                        str(tmp_path),
                        class_filter=selected_classes if selected_classes else None,
                        apply_clahe=pb_apply_clahe,
                        apply_sharpen=pb_apply_sharpen,
                    )
                    st.session_state["pb_detections"] = detections
                    st.session_state["pb_detection_config"] = {
                        "comp_model": comp_model_name,
                        "ic_model": ic_model_name if ic_model_resolved else None,
                        "comp_conf": comp_conf,
                        "ic_conf": ic_conf,
                        "class_filter": selected_classes,
                        "iou_threshold": DualModelDetector.IOU_THRESHOLD,
                    }
                    st.success(f"\u2705 Detected {len(detections)} components.")
                except Exception as exc:
                    import traceback
                    st.error(f"Inference error: {exc}")
                    st.code(traceback.format_exc())

    # ------------------------------------------------------------------
    # STEP 3 — Interactive annotated preview
    # ------------------------------------------------------------------
    if "pb_detections" in st.session_state:
        raw_detections: list = st.session_state["pb_detections"]

        # For ICs, keep only those confirmed by ic_detect
        detections = [
            d for d in raw_detections
            if d.get('class_name', '').upper() != 'IC' or d.get('ic_confirmed', False)
        ]

        st.markdown("## Step 3 — Annotated Preview")
        st.markdown(
            "Toggle component classes on/off to show or hide them on the image."
        )

        # Identify which classes are present in detections
        detected_types = sorted({d.get('class_name', '?') for d in detections})

        # Per-class toggle checkboxes — two-column layout: toggles | image
        col_toggles, col_image = st.columns([1, 4])
        with col_toggles:
            st.markdown("**Show / Hide**")
            visible_classes = set()
            for cls in detected_types:
                count = sum(1 for d in detections if d.get('class_name') == cls)
                is_visible = st.checkbox(
                    f"{cls} ({count})",
                    value=True,
                    key=f"pb_cls_toggle_{cls}",
                    help=f"Show/hide all {cls} detections",
                )
                if is_visible:
                    visible_classes.add(cls)

        # Filter detections to only visible classes
        visible_detections = [
            d for d in detections
            if d.get('class_name', '?') in visible_classes
        ]

        with col_image:
            annotated = _draw_boxes(pil_image, visible_detections)
            st.image(annotated, caption="Detected components", width="stretch")

            # Color legend (only for visible classes)
            visible_types = sorted(visible_classes & set(detected_types))
            if visible_types:
                legend_html = " ".join(
                    f'<span style="display:inline-block;margin:2px 6px;padding:2px 8px;'
                    f'background:{DETECTION_PALETTE.get(t, "#888")};color:#fff;'
                    f'border-radius:4px;font-size:0.85rem;">{t}</span>'
                    for t in visible_types
                )
                st.markdown(legend_html, unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # STEP 4 — User validation (editable detection list)
        # ------------------------------------------------------------------
        st.markdown("## Step 4 — Validate & Edit Detections")
        st.markdown(
            "Review the detections below. Uncheck rows to exclude them, "
            "or edit the component type directly."
        )

        # Build an editable dataframe
        rows = []
        for i, d in enumerate(detections):
            rows.append({
                "keep":             True,
                "#":                i,
                "type":             d.get("class_name", ""),
                "ic_subtype":       d.get("ic_subtype") or "",
                "confidence":       round(d.get("confidence", 0), 4),
                "ic_confirmed":     d.get("ic_confirmed", False),
                "x1": round(d["bbox"][0], 1), "y1": round(d["bbox"][1], 1),
                "x2": round(d["bbox"][2], 1), "y2": round(d["bbox"][3], 1),
            })

        if not rows:
            st.info("No detections to display.")
            st.stop()

        df_edit = pd.DataFrame(rows)
        edited_df = st.data_editor(
            df_edit,
            column_config={
                "keep": st.column_config.CheckboxColumn("Keep", default=True),
                "type": st.column_config.SelectboxColumn(
                    "Type", options=COMP_DETECT_CLASSES, required=True
                ),
                "ic_subtype": st.column_config.SelectboxColumn(
                    "IC sub-type", options=["", "four_side", "two_side", "without_side"]
                ),
            },
            disabled=["#", "confidence", "ic_confirmed", "x1", "y1", "x2", "y2"],
            width="stretch",
            num_rows="fixed",
            key="pb_edit_df"
        )

        kept_rows = edited_df[edited_df["keep"]]
        st.caption(f"{len(kept_rows)} / {len(detections)} detections kept.")

        # ------------------------------------------------------------------
        # Confirm & generate crops
        # ------------------------------------------------------------------
        if st.button("\u2702\ufe0f Confirm & Generate Crops", type="primary",
                     disabled=len(kept_rows) == 0):
            import cv2
            # Use the EXIF-corrected PIL image (already transposed above)
            # to produce a BGR numpy array consistent with detection results.
            corrected_rgb = pil_image.convert("RGB")
            cv_img = cv2.cvtColor(np.array(corrected_rgb), cv2.COLOR_RGB2BGR)

            # ---- Create job folder ----
            now = datetime.now()
            sanitized_job_name = "".join(
                c if (c.isalnum() or c in "._-") else "_"
                for c in (job_name_input or "photobooth").strip()
            ).strip("._-") or "photobooth"
            job_folder_name = f"{sanitized_job_name}_{now.strftime('%Y%m%d_%H%M%S')}"
            jobs_base = JOBS_DIR.resolve()
            job_dir = jobs_base / job_folder_name
            # Guard against path traversal
            if not job_dir.resolve().is_relative_to(jobs_base):
                st.error("Invalid job name.")
                st.stop()
            crops_dir = job_dir / "crops"
            job_dir.mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(exist_ok=True)

            # ---- Save EXIF-corrected input photo ----
            orig_suffix = Path(img_name).suffix or ".jpg"
            input_path = job_dir / f"input{orig_suffix}"
            corrected_rgb.save(str(input_path), quality=95)

            # ---- Save annotated photo ----
            annotated_rgb = annotated.convert("RGB")
            result_path = job_dir / "result.jpg"
            annotated_rgb.save(str(result_path), quality=95)

            # ---- Generate and save crops ----
            saved_rows = []
            col_imgs = st.columns(4)
            col_idx = 0

            for _, row in kept_rows.iterrows():
                i = int(row["#"])
                d = detections[i]
                bbox = d["bbox"]
                x1, y1, x2, y2 = (max(0, int(v)) for v in bbox[:4])
                # Apply small padding
                pad = 10
                h_img, w_img = cv_img.shape[:2]
                x1c = max(0, x1 - pad)
                y1c = max(0, y1 - pad)
                x2c = min(w_img, x2 + pad)
                y2c = min(h_img, y2 + pad)
                crop = cv_img[y1c:y2c, x1c:x2c]
                cls = row["type"]
                crop_name = f"{i:03d}_{cls}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                saved_rows.append({
                    "row_number":           i,
                    "detection_type":       cls,
                    "ic_subtype":           row["ic_subtype"] or None,
                    "detection_confidence": float(row["confidence"]),
                    "ic_confidence":        d.get("ic_confidence"),
                    "bounding_box": {
                        "x": x1, "y": y1,
                        "width": x2 - x1, "height": y2 - y1
                    },
                    "cropped_image_path":   str(crop_path),
                    "crop_file":            crop_name,
                    "processing_status":    "pending",
                })

                # Display crop thumbnail
                with col_imgs[col_idx % 4]:
                    st.image(
                        Image.open(io.BytesIO(cv2.imencode(".jpg", crop)[1].tobytes())),
                        caption=f"{cls} ({row['ic_subtype'] or '—'})",
                        width="stretch"
                    )
                col_idx += 1

            # ---- Save metadata.json ----
            config = st.session_state.get("pb_detection_config", {})
            metadata = {
                "job_name": job_folder_name,
                "input_file": img_name,
                "date": now.isoformat(),
                "model": config.get("comp_model", ""),
                "ic_model": config.get("ic_model"),
                "detection_config": config,
                "total_detections": len(saved_rows),
                "detections": [
                    {
                        "index":       r["row_number"],
                        "class_name":  r["detection_type"],
                        "confidence":  r["detection_confidence"],
                        "ic_subtype":  r["ic_subtype"],
                        "bbox": [
                            r["bounding_box"]["x"],
                            r["bounding_box"]["y"],
                            r["bounding_box"]["x"] + r["bounding_box"]["width"],
                            r["bounding_box"]["y"] + r["bounding_box"]["height"],
                        ],
                        "crop_file":   r["crop_file"],
                    }
                    for r in saved_rows
                ],
            }
            metadata_path = job_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            st.success(
                f"\u2705 {len(saved_rows)} crops generated.\n\n"
                f"\U0001f4c1 Job saved to `{job_dir}`"
            )

            # Log to database if available
            if st.session_state.get("db_connected", False):
                try:
                    db = st.session_state.db
                    import_id = db.create_pcba_import(
                        image_storage_path=img_name,
                        detection_config=config,
                        total_detections=len(saved_rows),
                        status="completed",
                    )

                    for row_data in saved_rows:
                        db.log_pcba_row_import(
                            import_id=import_id,
                            row_number=row_data["row_number"],
                            detection_type=row_data["detection_type"],
                            detection_confidence=row_data["detection_confidence"],
                            bounding_box=row_data["bounding_box"],
                            ic_subtype=row_data["ic_subtype"],
                            ic_confidence=row_data["ic_confidence"],
                            cropped_image_path=row_data["cropped_image_path"],
                            processing_status=row_data["processing_status"],
                        )

                    st.info(f"\U0001f4be Session logged to database (import id: {import_id})")
                except Exception as exc:
                    st.warning(f"Database logging failed: {exc}")


# ========== JOB VIEWER PAGE ==========
elif page == "\U0001f50d Job Viewer":
    st.markdown('<div class="main-header">\U0001f50d Job Viewer</div>', unsafe_allow_html=True)
    st.markdown("Browse results: input photo, annotated result, cropped components, and metadata.")

    jobs_base = JOBS_DIR
    job_folders = sorted(
        [d for d in jobs_base.iterdir() if d.is_dir() and (d / "metadata.json").exists()],
        key=lambda d: d.stat().st_mtime, reverse=True
    ) if jobs_base.exists() else []

    db_jobs = []
    if st.session_state.get("db_connected", False):
        try:
            db_jobs = st.session_state.db.get_all_jobs()
        except Exception:
            pass

    if not job_folders and not db_jobs:
        st.info("No jobs found yet. Upload and process a PCB image first!")
    else:
        fs_job_names = {d.name for d in job_folders}
        job_options = [d.name for d in job_folders]
        job_folder_map = {d.name: d for d in job_folders}
        for job in db_jobs:
            jname = job.get("job_name") or f"job_{job['job_id']}"
            if jname not in fs_job_names:
                job_options.append(f"{jname} (folder missing)")

        if not job_options:
            st.info("No jobs found.")
        else:
            selected_label = st.selectbox("Choose a job to inspect", job_options)
            selected_name = selected_label.replace(" (folder missing)", "")
            job_dir = job_folder_map.get(selected_name)

            if job_dir and job_dir.exists():
                with open(job_dir / "metadata.json") as f:
                    metadata = json.load(f)

                st.markdown("---")
                st.markdown("### \U0001f4cb Job Information")
                st.caption(f"\U0001f4c1 {job_dir}")
                if st.button("\U0001f4c2 Open analysis folder", key="open_job_dir"):
                    if not open_folder(job_dir):
                        st.warning(f"Cannot open: {job_dir}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Job Name", metadata.get("job_name", "—"))
                with col2:
                    st.metric("Total Detections", metadata.get("total_detections", 0))
                with col3:
                    date_str = metadata.get("date", "")
                    try:
                        dt = datetime.fromisoformat(date_str)
                        st.metric("Date", dt.strftime("%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        st.metric("Date", date_str)
                st.text(f"Model: {metadata.get('model', '—')}")
                st.text(f"Folder: {job_dir}")

                st.markdown("---")
                st.markdown("### \U0001f4f8 Input Photo")
                input_photos = list(job_dir.glob("input.*"))
                if input_photos:
                    try:
                        st.image(Image.open(input_photos[0]), caption="Input", width="stretch")
                    except Exception as e:
                        st.error(f"Could not display input photo: {e}")
                else:
                    st.warning("Input photo not found.")

                st.markdown("---")
                st.markdown("### \U0001f3af Result Photo (Annotated)")
                result_photo = job_dir / "result.jpg"
                if result_photo.exists():
                    try:
                        st.image(Image.open(result_photo), caption="Detected components", width="stretch")
                    except Exception as e:
                        st.error(f"Could not display result photo: {e}")
                else:
                    st.warning("Result photo not found.")

                crops_dir = job_dir / "crops"
                crop_files = sorted(crops_dir.glob("*.jpg")) if crops_dir.exists() else []
                detections = metadata.get("detections", [])

                if crop_files:
                    st.markdown("---")
                    st.markdown(f"### \u2702\ufe0f Cropped Components ({len(crop_files)} total)")
                    det_by_file = {d.get("crop_file"): d for d in detections if d.get("crop_file")}
                    cols_per_row = 4
                    rows = [crop_files[i:i+cols_per_row] for i in range(0, len(crop_files), cols_per_row)]
                    for row in rows:
                        cols = st.columns(cols_per_row)
                        for col, crop_file in zip(cols, row):
                            with col:
                                try:
                                    det = det_by_file.get(crop_file.name, {})
                                    caption = det.get("class_name", crop_file.stem)
                                    if "confidence" in det:
                                        caption += f" ({det['confidence']:.2f})"
                                    st.image(Image.open(crop_file), caption=caption, width="stretch")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                else:
                    st.info("No cropped components found.")

                st.markdown("---")
                st.markdown("### \U0001f4c4 Metadata")
                if detections:
                    df_det = pd.DataFrame([
                        {
                            "index": d["index"], "class": d["class_name"],
                            "confidence": d["confidence"],
                            "x1": round(d["bbox"][0], 1), "y1": round(d["bbox"][1], 1),
                            "x2": round(d["bbox"][2], 1), "y2": round(d["bbox"][3], 1),
                            "crop_file": d.get("crop_file", "—")
                        } for d in detections
                    ])
                    st.dataframe(df_det, width="stretch")
                else:
                    st.info("No detections in this job.")

                with st.expander("\U0001f4c4 Raw metadata.json"):
                    st.json(metadata)
            else:
                st.warning(f"Job folder not found on disk for: {selected_name}")


# ========== DATABASE VIEWER PAGE ==========
elif page == "\U0001f5c4\ufe0f Database Viewer":
    st.markdown('<div class="main-header">\U0001f5c4\ufe0f Database Viewer</div>', unsafe_allow_html=True)

    if not st.session_state.get("db_connected", False):
        st.error("\u274c **Local database unavailable.** Restart the application; "
                 "see the logs folder if the problem persists.")
    else:
        table_view = st.selectbox("Select Table",
            ["\U0001f4f8 Images Input", "\U0001f504 Jobs Log",
             "\U0001f3af Detections", "\u2702\ufe0f Cropped Components",
             "\U0001f4f7 PCBA Imports", "\U0001f4cb PCBA Detection Rows"])

        if st.button("\U0001f504 Refresh"):
            st.rerun()

        try:
            if table_view == "\U0001f4f8 Images Input":
                data = st.session_state.db.get_all_images()
                if data:
                    df = pd.DataFrame(data)
                    if "upload_at" in df.columns:
                        df["upload_at"] = pd.to_datetime(df["upload_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(df, width="stretch", height=400)
                    st.caption(f"Total records: {len(df)}")
                else:
                    st.info("No images in database yet.")

            elif table_view == "\U0001f504 Jobs Log":
                data = st.session_state.db.get_all_jobs()
                if data:
                    df = pd.DataFrame(data)
                    for col in ["started_at", "ended_at"]:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(df, width="stretch", height=400)
                    st.caption(f"Total records: {len(df)}")
                    if len(df) > 0:
                        st.markdown("---")
                        job_id = st.selectbox("Select Job ID", df["job_id"].tolist())
                        if job_id:
                            stats = st.session_state.db.get_job_statistics(job_id)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Detections", stats.get("total_detections", 0))
                            with col2:
                                st.metric("Components Cropped", stats.get("total_crops", 0))
                else:
                    st.info("No jobs in database yet.")

            elif table_view == "\U0001f3af Detections":
                all_jobs = st.session_state.db.get_all_jobs()
                job_options = ["All Jobs"] + [f"Job {j['job_id']} - {j['file_name']}" for j in all_jobs]
                selected_job = st.selectbox("Filter by Job", job_options)
                job_id = None
                if selected_job != "All Jobs":
                    job_id = int(selected_job.split()[1])
                data = st.session_state.db.get_all_detections(job_id=job_id)
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, width="stretch", height=400)
                    st.caption(f"Total records: {len(df)}")
                    if "class_name" in df.columns:
                        st.markdown("---")
                        st.bar_chart(df["class_name"].value_counts())
                else:
                    st.info("No detections in database yet.")

            elif table_view == "\u2702\ufe0f Cropped Components":
                data = st.session_state.db.get_all_cropped_components()
                if data:
                    df = pd.DataFrame(data)
                    if "created_at" in df.columns:
                        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(df, width="stretch", height=400)
                    st.caption(f"Total records: {len(df)}")
                else:
                    st.info("No cropped components in database yet.")

            elif table_view == "\U0001f4f7 PCBA Imports":
                data = st.session_state.db.get_all_pcba_imports()
                if data:
                    df = pd.DataFrame(data)
                    if "created_at" in df.columns:
                        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    # Show detection_config as truncated string for readability
                    if "detection_config" in df.columns:
                        df["detection_config"] = df["detection_config"].apply(
                            lambda v: json.dumps(v, ensure_ascii=False)[:80] + "…" if v else "—"
                        )
                    st.dataframe(df, width="stretch", height=400)
                    st.caption(f"Total records: {len(df)}")

                    if len(df) > 0:
                        st.markdown("---")
                        import_ids = df["id"].tolist()
                        selected_import = st.selectbox("Inspect import session", import_ids)
                        if selected_import:
                            rows = st.session_state.db.get_pcba_import_rows(str(selected_import))
                            if rows:
                                st.dataframe(pd.DataFrame(rows), width="stretch", height=400)
                            else:
                                st.info("No detection rows for this import session.")
                else:
                    st.info("No PCBA imports in database yet.")

            elif table_view == "\U0001f4cb PCBA Detection Rows":
                imports = st.session_state.db.get_all_pcba_imports()
                import_options = ["All Imports"] + [
                    f"{imp['id']} — {imp.get('status', '')} ({imp.get('row_count', 0)} rows)"
                    for imp in imports
                ]
                selected = st.selectbox("Filter by import session", import_options)
                if selected == "All Imports":
                    data = st.session_state.db.get_all_pcba_rows()
                else:
                    imp_id = selected.split(" — ")[0]
                    data = st.session_state.db.get_pcba_import_rows(imp_id)
                if data:
                    df = pd.DataFrame(data)
                    if "created_at" in df.columns:
                        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(df, width="stretch", height=400)
                    st.caption(f"Total records: {len(df)}")
                    if "detection_type" in df.columns:
                        st.markdown("---")
                        st.bar_chart(df["detection_type"].value_counts())
                else:
                    st.info("No PCBA detection rows in database yet.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ========== STATISTICS PAGE ==========
elif page == "\U0001f4ca Statistics":
    st.markdown('<div class="main-header">\U0001f4ca Statistics & Analytics</div>', unsafe_allow_html=True)

    if not st.session_state.get("db_connected", False):
        st.warning("Local database unavailable. Statistics require database access.")
    else:
        try:
            stats = st.session_state.db.get_detection_statistics()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", stats.get("total_images", 0))
            with col2:
                st.metric("Total Jobs", stats.get("total_jobs", 0))
            with col3:
                st.metric("Total Detections", stats.get("total_detections", 0))

            st.markdown("---")
            component_counts = stats.get("component_counts", {})
            if component_counts:
                df_c = pd.DataFrame(list(component_counts.items()),
                                     columns=["Component", "Count"]).sort_values("Count", ascending=False)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(df_c.set_index("Component"))
                with col2:
                    st.dataframe(df_c, width="stretch")
            else:
                st.info("No component data available yet.")

            st.markdown("---")
            recent_jobs = st.session_state.db.get_all_jobs(limit=10)
            if recent_jobs:
                st.markdown("### Recent Jobs")
                df_j = pd.DataFrame(recent_jobs)
                for col in ["started_at", "ended_at"]:
                    if col in df_j.columns:
                        df_j[col] = pd.to_datetime(df_j[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
                show_cols = [c for c in ["job_id", "job_name", "file_name", "started_at", "detection_count"]
                             if c in df_j.columns]
                st.dataframe(df_j[show_cols], width="stretch")
            else:
                st.info("No jobs recorded yet.")

            # --- PCBA Photo Booth statistics ---
            st.markdown("---")
            st.markdown("### \U0001f4f7 PCBA Photo Booth")
            try:
                pcba_stats = st.session_state.db.get_pcba_statistics()
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    st.metric("Imports", pcba_stats.get("total_imports", 0))
                with pcol2:
                    st.metric("Detections", pcba_stats.get("total_detections", 0))
                with pcol3:
                    st.metric("Completed", pcba_stats.get("completed", 0))
                with pcol4:
                    st.metric("Errors", pcba_stats.get("errors", 0))

                pcba_counts = pcba_stats.get("component_counts", {})
                if pcba_counts:
                    df_pc = pd.DataFrame(
                        list(pcba_counts.items()),
                        columns=["Component", "Count"]
                    ).sort_values("Count", ascending=False)
                    col_left, col_right = st.columns([2, 1])
                    with col_left:
                        st.bar_chart(df_pc.set_index("Component"))
                    with col_right:
                        st.dataframe(df_pc, width="stretch")
                else:
                    st.info("No PCBA detection data yet.")
            except Exception:
                st.info("PCBA logging tables not available yet.")

        except Exception as e:
            st.error(f"Error: {str(e)}")


# ========== ABOUT PAGE ==========
elif page == "\u2139\ufe0f About":
    st.markdown('<div class="main-header">\u2139\ufe0f About nuts_vision</div>', unsafe_allow_html=True)
    st.markdown(r"""
    ### 🔧 Electronic Component Detection System

    **nuts_vision** analyses photos of electronic circuit boards, detects up to **13 component types**,
    and produces cropped images of each detected component.

    #### 🎯 Dual-Model Detection:
    - **`smd_comp`**: detects 13 classes — Button, Capacitor, Connector, Diode, Electrolytic Capacitor, IC, Inductor, Led, Pads, Pins, Resistor, Switch, Transistor
    - **`ic_detect_best`**: classifies ICs by pin layout — `four_side` (4 rows of pins), `two_side` (2 rows), `without_side` (BGA/QFN)
    - Results are cross-referenced via IoU matching for enhanced IC detection accuracy

    #### 🛠️ Key Technologies:
    - **YOLOv8**: component detection (ONNX & PyTorch models)
    - **SQLite** (local): analysis history
    - **Streamlit**: web interface
    - **OpenCV**: image processing & cropping

    #### 📁 Job Folder Structure:
    ```
    <output folder>/jobs/
      <image_name>_<YYYYMMDD>_<HHMMSS>/
        input.<ext>    — original photo
        result.jpg     — annotated photo
        crops/         — one image per component
        metadata.json  — detection data
    ```

    **Version**: 2.2.0 | **License**: CC BY 4.0
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Database Status:**")
        if st.session_state.get("db_connected", False):
            st.success("\u2705 Local SQLite database")
            st.text(f"File: {APP_PATHS.database_path}")
            st.text(f"Results: {JOBS_DIR}")
        else:
            st.error("\u274c Not Connected")
    with col2:
        st.markdown("**Environment:**")
        st.text(f"Python: {sys.version.split()[0]}")
        st.text(f"Streamlit: {st.__version__}")
        _about_models = {
            "smd_comp.pt": Path("smd_comp.pt").exists(),
            "smd_comp.onnx": Path("smd_comp.onnx").exists(),
            "ic_detect_best.onnx": Path("ic_detect_best.onnx").exists(),
            "ic_detect_best.pt": Path("ic_detect_best.pt").exists(),
        }
        for model_name, found in _about_models.items():
            st.text(f"{model_name}: {'✅ Found' if found else '❌ Not found'}")
