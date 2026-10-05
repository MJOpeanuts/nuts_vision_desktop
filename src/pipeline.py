#!/usr/bin/env python3
"""
Complete Pipeline for Electronic Component Detection
Combines detection and cropping into a single workflow.
Each job is stored in a dedicated folder containing the input photo,
the annotated result photo, all cropped components, and a JSON metadata file.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import cv2

# Import our modules
from detect import ComponentDetector, load_image_with_exif
from crop import ComponentCropper

# Import database module if available
try:
    from database import DatabaseManager, get_db_manager_from_env
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: Database module not available. Install sqlalchemy and alembic to enable database logging.")


class ComponentAnalysisPipeline:
    """Complete pipeline for component detection and cropping."""
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        padding: int = 10,
        use_database: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
            padding: Padding for cropped components
            use_database: Whether to log to database
        """
        self.detector = ComponentDetector(model_path, conf_threshold)
        self.cropper = ComponentCropper(padding)
        self.use_database = use_database and DB_AVAILABLE
        self.model_path = model_path
        
        if self.use_database:
            try:
                self.db = get_db_manager_from_env()
                if not self.db.test_connection():
                    print("Warning: Local database unavailable. Continuing without database logging.")
                    self.use_database = False
            except Exception as e:
                print(f"Warning: Could not initialize database: {e}")
                self.use_database = False

    def process_image(
        self,
        image_path: str,
        jobs_base_dir: str = "jobs"
    ) -> dict:
        """
        Process a single image: detect components, crop them, save results.
        
        The output is stored in a job folder named:
            {jobs_base_dir}/{input_stem}_{YYYYMMDD}_{HHMMSS}/
        
        The folder contains:
            input{ext}      — copy of the original photo
            result.jpg      — annotated photo with bounding boxes
            crops/          — one cropped image per detected component
            metadata.json   — detection data and job info

        Args:
            image_path: Path to the input PCB image
            jobs_base_dir: Base directory where job folders are created

        Returns:
            Dictionary with job_folder, job_name, detections, crop_paths
        """
        img_path = Path(image_path)
        now = datetime.now()
        job_name = f"{img_path.stem}_{now.strftime('%Y%m%d_%H%M%S')}"
        job_dir = Path(jobs_base_dir) / job_name
        crops_dir = job_dir / "crops"
        job_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"JOB: {job_name}")
        print(f"Output folder: {job_dir}")
        print("="*60)

        # --- Copy input photo (EXIF-corrected so it matches what the viewer shows) ---
        input_copy = job_dir / f"input{img_path.suffix}"

        # --- Detection ---
        print("\n[STEP 1/2] Detecting components...")
        image = load_image_with_exif(str(img_path))

        # Save the EXIF-corrected input copy for consistency with the viewer
        cv2.imwrite(str(input_copy), image)

        detections = self.detector.detect_components(
            str(img_path),
            save_visualization=False,
            image=image,
        )
        print(f"  Detected {len(detections)} components")

        # --- Save annotated result image ---
        results = self.detector.model(image, conf=self.detector.conf_threshold, verbose=False)
        annotated = results[0].plot()
        result_path = job_dir / "result.jpg"
        cv2.imwrite(str(result_path), annotated)
        print(f"  Saved result image: {result_path}")

        # --- Crop all detected components ---
        print("\n[STEP 2/2] Cropping components...")
        crop_paths = []
        for i, detection in enumerate(detections):
            cropped = self.cropper.crop_component(image, detection['bbox'])
            crop_filename = f"{i:03d}_{detection['class_name']}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), cropped)
            crop_paths.append(str(crop_path))
        print(f"  Saved {len(crop_paths)} cropped components to {crops_dir}")

        # --- Save metadata JSON ---
        metadata = {
            "job_name": job_name,
            "input_file": str(img_path.resolve()),
            "date": now.isoformat(),
            "model": str(self.model_path),
            "total_detections": len(detections),
            "detections": [
                {
                    "index": i,
                    "class_name": d["class_name"],
                    "confidence": round(d["confidence"], 4),
                    "bbox": d["bbox"],
                    "crop_file": Path(crop_paths[i]).name if i < len(crop_paths) else None
                }
                for i, d in enumerate(detections)
            ]
        }
        metadata_path = job_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")

        # --- Database logging ---
        if self.use_database:
            try:
                file_fmt = img_path.suffix.lstrip(".")
                image_id = self.db.log_image_upload(img_path.name, str(img_path.resolve()), file_fmt)
                job_id = self.db.start_job(
                    image_id, self.model_path,
                    job_name=job_name,
                    job_folder_path=str(job_dir.resolve())
                )
                detection_ids = {}
                for i, d in enumerate(detections):
                    det_id = self.db.log_detection(job_id, d["class_name"], d["confidence"], d["bbox"])
                    detection_ids[i] = det_id
                for i, crop_path in enumerate(crop_paths):
                    if i in detection_ids:
                        self.db.log_cropped_component(job_id, detection_ids[i], str(Path(crop_path).resolve()))
                self.db.end_job(job_id)
            except Exception as e:
                print(f"Warning: Database logging failed: {e}")

        print(f"\n✅ Job complete: {job_dir}")
        return {
            "job_name": job_name,
            "job_folder": str(job_dir),
            "input_photo": str(input_copy),
            "result_photo": str(result_path),
            "crop_photos": crop_paths,
            "metadata": metadata
        }

    def run_pipeline(
        self,
        image_path: str = None,
        image_dir: str = None,
        output_base_dir: str = "jobs",
        **kwargs
    ):
        """
        Run the pipeline on one or more images (kept for backward-compatibility).

        Args:
            image_path: Path to single image (optional)
            image_dir: Directory of images (optional)
            output_base_dir: Base directory for job folders
        """
        images_to_process = []
        if image_path:
            images_to_process = [image_path]
        elif image_dir:
            image_dir = Path(image_dir)
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                images_to_process.extend(image_dir.glob(f"*{ext}"))
                images_to_process.extend(image_dir.glob(f"*{ext.upper()}"))
            images_to_process = [str(p) for p in images_to_process]

        results = []
        for img in images_to_process:
            try:
                result = self.process_image(str(img), jobs_base_dir=output_base_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img}: {e}")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Component detection pipeline — detects and crops electronic components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --model smd_comp.pt --image board.jpg
  python pipeline.py --model smd_comp.pt --image-dir images/
  python pipeline.py --model smd_comp.pt --image board.jpg --conf 0.5
        """
    )
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument("--image", type=str, help="Path to single image to process")
    parser.add_argument("--image-dir", type=str, help="Directory of images to process")
    parser.add_argument("--output-dir", type=str, default="jobs", help="Base directory for job folders (default: jobs)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--padding", type=int, default=10, help="Padding around crops in pixels (default: 10)")
    parser.add_argument("--use-database", action="store_true", help="Enable logging to the local SQLite database")

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    pipeline = ComponentAnalysisPipeline(
        model_path=args.model,
        conf_threshold=args.conf,
        padding=args.padding,
        use_database=args.use_database
    )

    pipeline.run_pipeline(
        image_path=args.image,
        image_dir=args.image_dir,
        output_base_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
