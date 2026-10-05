#!/usr/bin/env python3
"""
Component Detection Script with Image Preprocessing
Detects electronic components in circuit board images using YOLO and preprocessing.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Optional
from PIL import Image, ImageOps

try:
    from .image_limits import open_image, InvalidImageError
except ImportError:  # imported as a top-level module (src/ on sys.path)
    from image_limits import open_image, InvalidImageError
import json


def load_image_with_exif(image_path: str) -> np.ndarray:
    """
    Load an image with EXIF orientation correction.

    PIL/Streamlit auto-rotate images according to EXIF orientation tags,
    but OpenCV's ``cv2.imread`` ignores them.  This helper opens the file
    with PIL, applies ``ImageOps.exif_transpose()`` so the pixel data
    matches the visual orientation, then converts to a BGR numpy array
    suitable for OpenCV processing.

    Args:
        image_path: Path to the image file.

    Returns:
        BGR numpy array with correct orientation.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    try:
        pil_img = open_image(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        # Ensure 3-channel RGB
        pil_img = pil_img.convert("RGB")
        # Convert to BGR numpy array for OpenCV
        rgb_array = np.array(pil_img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except InvalidImageError:
        raise  # clear user-facing message (too large / invalid image)
    except Exception as e:
        raise ValueError(f"Could not load image: {image_path} — {e}")


class ComponentDetector:
    """Detector for electronic components on circuit boards."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize the component detector.
        
        Args:
            model_path: Path to the trained YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def preprocess_image(
        self, 
        image: np.ndarray,
        apply_blur: bool = True,
        blur_kernel: Tuple[int, int] = (5, 5),
        detect_edges: bool = False,
        apply_clahe: bool = False,
        apply_sharpen: bool = False,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess image with optional blur, CLAHE contrast enhancement,
        sharpening, and edge detection.

        Args:
            image: Input image (BGR format)
            apply_blur: Whether to apply Gaussian blur
            blur_kernel: Kernel size for Gaussian blur
            detect_edges: Whether to detect edges
            apply_clahe: Whether to apply CLAHE contrast enhancement
            apply_sharpen: Whether to apply mild sharpening
            clahe_clip: CLAHE clip limit (default 2.0)
            clahe_grid: CLAHE tile grid size (default (8, 8))

        Returns:
            Tuple of (preprocessed_image, edge_map)
        """
        preprocessed = image.copy()

        # Apply Gaussian blur to reduce noise
        if apply_blur:
            preprocessed = cv2.GaussianBlur(preprocessed, blur_kernel, 0)

        # CLAHE on the L channel in LAB colour space — improves local
        # contrast and compensates for vignetting / uneven illumination
        if apply_clahe:
            preprocessed = self._apply_clahe(preprocessed, clahe_clip, clahe_grid)

        # Mild sharpening to recover edge detail in peripheral zones
        if apply_sharpen:
            preprocessed = self._apply_sharpen(preprocessed)

        # Detect edges (optional, for visualization)
        edge_map = None
        if detect_edges:
            gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
            edge_map = cv2.Canny(gray, 50, 150)

        return preprocessed, edge_map

    # ------------------------------------------------------------------
    # Peripheral-detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Apply CLAHE on the L channel of a LAB image."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge([l_chan, a_chan, b_chan])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_sharpen(image: np.ndarray) -> np.ndarray:
        """Apply a mild unsharp-mask style sharpening kernel."""
        kernel = np.array(
            [[0, -1,  0],
             [-1,  5, -1],
             [0, -1,  0]], dtype=np.float32
        )
        return cv2.filter2D(image, -1, kernel)
    
    def detect_components(
        self, 
        image_path: str,
        preprocess: bool = True,
        save_visualization: bool = True,
        output_dir: str = "outputs/results",
        apply_clahe: bool = False,
        apply_sharpen: bool = False,
        image: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Detect components in an image.
        
        Args:
            image_path: Path to input image
            preprocess: Whether to preprocess the image
            save_visualization: Whether to save annotated image
            output_dir: Directory to save results
            apply_clahe: Whether to apply CLAHE contrast enhancement
            apply_sharpen: Whether to apply mild sharpening
            image: Optional pre-loaded BGR image (skips file I/O if provided)
            
        Returns:
            List of detection dictionaries
        """
        # Use pre-loaded image or load from file with EXIF correction
        if image is None:
            image = load_image_with_exif(str(image_path))
        
        original_image = image.copy()
        
        # Preprocess if requested
        if preprocess:
            image, edge_map = self.preprocess_image(
                image,
                apply_clahe=apply_clahe,
                apply_sharpen=apply_sharpen,
            )
        
        # Run detection
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        # Parse detections
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'bbox_center': box.xywh[0].cpu().numpy().tolist()  # [x_center, y_center, width, height]
                }
                detections.append(detection)
        
        # Save visualization if requested
        if save_visualization:
            output_path = Path(output_dir) / f"{Path(image_path).stem}_detected.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Annotate image
            annotated = results[0].plot()
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved visualization to: {output_path}")
        
        return detections
    
    def batch_detect(
        self,
        image_dir: str,
        output_dir: str = "outputs/results",
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ) -> dict:
        """
        Detect components in multiple images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            extensions: List of valid image extensions
            
        Returns:
            Dictionary mapping image paths to detections
        """
        image_dir = Path(image_dir)
        all_detections = {}
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_path in image_files:
            print(f"\nProcessing: {image_path.name}")
            try:
                detections = self.detect_components(
                    str(image_path),
                    output_dir=output_dir
                )
                all_detections[str(image_path)] = detections
                print(f"  Detected {len(detections)} components")
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
        
        # Save results to JSON
        results_file = Path(output_dir) / "detections.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        print(f"\nSaved detection results to: {results_file}")
        
        return all_detections


class DualModelDetector:
    """
    Dual-model detector that combines smd_comp and ic_detect_best
    for enhanced IC detection with visual sub-type classification.

    Cross-referencing rules (IoU threshold = 0.5):
    - IC in comp_detect + match in ic_detect  → confirmed IC, enriched with sub-type
    - IC in ic_detect without match           → added as IC (missed by comp_detect)
    - IC in comp_detect without ic_detect match → kept, flagged as "unconfirmed"
    - Non-IC classes from comp_detect         → kept as-is (after optional class filter)
    """

    IOU_THRESHOLD = 0.5

    COMP_DETECT_CLASSES = [
        'Button', 'Capacitor', 'Connector', 'Diode',
        'Electrolytic Capacitor', 'IC', 'Inductor', 'Led',
        'Pads', 'Pins', 'Resistor', 'Switch', 'Transistor'
    ]
    IC_SUBTYPES = ['four_side', 'two_side', 'without_side']

    def __init__(
        self,
        comp_model_path: str,
        ic_model_path: Optional[str] = None,
        comp_conf: float = 0.25,
        ic_conf: float = 0.25
    ):
        """
        Args:
            comp_model_path: Path to smd_comp model (.onnx or .pt)
            ic_model_path:   Path to ic_detect_best model (.onnx or .pt), optional
            comp_conf:       Confidence threshold for comp_detect
            ic_conf:         Confidence threshold for ic_detect
        """
        self.comp_detector = ComponentDetector(comp_model_path, comp_conf)
        self.ic_detector = ComponentDetector(ic_model_path, ic_conf) if ic_model_path else None

    # ------------------------------------------------------------------
    # IoU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image_path: str,
        class_filter: Optional[List[str]] = None,
        apply_clahe: bool = False,
        apply_sharpen: bool = False,
    ) -> List[dict]:
        """
        Run dual-model inference and return a unified detection list.

        Each detection dict contains:
            class_name        (str)  – component type from COMP_DETECT_CLASSES
            ic_subtype        (str|None)  – 'four_side', 'two_side', 'without_side' or None
            confidence        (float) – best confidence from comp_detect (or ic_detect if new)
            ic_confidence     (float|None) – ic_detect confidence when applicable
            ic_confirmed      (bool)  – True if IC was confirmed by ic_detect match
            bbox              (list)  – [x1, y1, x2, y2]
            bbox_center       (list)  – [cx, cy, w, h]
            class_id          (int)

        Args:
            image_path:    Path to the PCB image
            class_filter:  Optional list of class names to keep (e.g. ['IC', 'capacitor'])
                           If None or empty, all 13 classes are returned.
            apply_clahe:   Whether to apply CLAHE contrast enhancement
            apply_sharpen: Whether to apply mild sharpening
        """
        # --- 1. comp_detect (always run) ---
        comp_dets = self.comp_detector.detect_components(
            image_path, save_visualization=False,
            apply_clahe=apply_clahe, apply_sharpen=apply_sharpen,
        )

        # --- 2. ic_detect (optional) ---
        ic_dets: List[dict] = []
        if self.ic_detector is not None:
            ic_dets = self.ic_detector.detect_components(
                image_path, save_visualization=False,
                apply_clahe=apply_clahe, apply_sharpen=apply_sharpen,
            )

        # --- 3. Cross-reference ICs ---
        unified = self._cross_reference(comp_dets, ic_dets)

        # --- 4. Apply class filter ---
        if class_filter:
            filter_set = {c.lower() for c in class_filter}
            unified = [d for d in unified if d['class_name'].lower() in filter_set]

        return unified

    # ------------------------------------------------------------------
    # Internal cross-referencing logic
    # ------------------------------------------------------------------

    def _cross_reference(
        self,
        comp_dets: List[dict],
        ic_dets: List[dict]
    ) -> List[dict]:
        """Merge comp_detect and ic_detect detections."""
        matched_ic_indices: set = set()  # indices into ic_dets already consumed

        result: List[dict] = []

        for cd in comp_dets:
            entry = {
                'class_id':      cd['class_id'],
                'class_name':    cd['class_name'],
                'confidence':    cd['confidence'],
                'ic_confidence': None,
                'ic_subtype':    None,
                'ic_confirmed':  False,
                'bbox':          cd['bbox'],
                'bbox_center':   cd['bbox_center'],
            }

            if cd['class_name'].upper() == 'IC' and ic_dets:
                best_iou = 0.0
                best_idx = -1
                for idx, icd in enumerate(ic_dets):
                    iou = self._compute_iou(cd['bbox'], icd['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou >= self.IOU_THRESHOLD:
                    matched_ic = ic_dets[best_idx]
                    matched_ic_indices.add(best_idx)
                    entry['ic_subtype']    = matched_ic['class_name']
                    entry['ic_confidence'] = matched_ic['confidence']
                    entry['ic_confirmed']  = True
                    # Keep the best confidence from either model
                    entry['confidence']    = max(cd['confidence'], matched_ic['confidence'])
                else:
                    # IC detected by comp_detect but not confirmed by ic_detect
                    entry['ic_confirmed'] = False

            result.append(entry)

        # --- ICs found by ic_detect but missed by comp_detect ---
        for idx, icd in enumerate(ic_dets):
            if idx not in matched_ic_indices:
                bbox = icd['bbox']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                w  = bbox[2] - bbox[0]
                h  = bbox[3] - bbox[1]
                result.append({
                    'class_id':      None,   # not assigned by comp_detect
                    'class_name':    'IC',
                    'confidence':    icd['confidence'],
                    'ic_confidence': icd['confidence'],
                    'ic_subtype':    icd['class_name'],
                    'ic_confirmed':  True,
                    'bbox':          bbox,
                    'bbox_center':   [cx, cy, w, h],
                })

        return result


def main():
    parser = argparse.ArgumentParser(description="Detect electronic components in images")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image to process"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Directory of images to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable image preprocessing"
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")
    
    # Initialize detector
    detector = ComponentDetector(args.model, conf_threshold=args.conf)
    
    # Process images
    if args.image:
        detections = detector.detect_components(
            args.image,
            preprocess=not args.no_preprocess,
            output_dir=args.output_dir
        )
        print(f"\nDetected {len(detections)} components:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    else:
        detector.batch_detect(args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
