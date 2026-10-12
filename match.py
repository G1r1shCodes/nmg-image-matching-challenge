"""
match.py — NMG Image Matching Challenge
CLI tool to compare two stone slab images and determine if they show the same physical piece.

Usage:
    python match.py --image-a data/images/img_001.jpg --image-b data/images/img_002.jpg --out output/
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# ── Default Hyperparameters ───────────────────────────────────────────
RESIZE_LONG_SIDE   = 800
RATIO_THRESHOLD    = 0.75
RANSAC_REPROJ      = 5.0
SAME_THRESHOLD     = 40
SIMILAR_THRESHOLD  = 15
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_GRID_SIZE    = (8, 8)


def preprocess(image_path: Path, long_side: int = RESIZE_LONG_SIDE, apply_clahe: bool = True):
    """Loads image with error handling, resizes, converts to grayscale and enhances with CLAHE."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not decode image: {image_path}")
        
    h, w = img_bgr.shape[:2]
    scale = long_side / max(h, w)
    img_resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        processed = clahe.apply(gray)
    else:
        processed = gray
        
    return img_resized, processed


def create_detector(name: str):
    name = name.upper()
    if name == 'SIFT':
        return cv2.SIFT_create(), cv2.NORM_L2
    elif name == 'ORB':
        return cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING
    elif name == 'AKAZE':
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    elif name == 'BRISK':
        return cv2.BRISK_create(), cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported detector: {name}. Choose SIFT, ORB, AKAZE, or BRISK.")


def compute_confidence(inliers: int, inlier_ratio: float, has_valid_homography: bool) -> tuple[float, str]:
    """
    Computes a 0–100% confidence score based on verified RANSAC inliers and inlier ratio.
    Returns (confidence_score, band).
    """
    if not has_valid_homography or inliers < 4:
        return 0.0, 'DIFFERENT'
        
    if inliers >= SAME_THRESHOLD:
        # Scale between 70.0% and 100.0%
        bonus = min(30.0, 30.0 * ((inliers - SAME_THRESHOLD) / 60.0))
        ratio_weight = min(1.0, inlier_ratio / 0.5)
        conf = 70.0 + (bonus * 0.7) + (30.0 * ratio_weight * 0.3)
        return min(100.0, round(conf, 1)), 'SAME'
    elif inliers >= SIMILAR_THRESHOLD:
        # Scale between 35.0% and 69.9%
        ratio_norm = (inliers - SIMILAR_THRESHOLD) / (SAME_THRESHOLD - SIMILAR_THRESHOLD)
        conf = 35.0 + (34.0 * ratio_norm)
        return round(conf, 1), 'SIMILAR'
    else:
        # Scale between 0.0% and 34.9%
        conf = 34.0 * (inliers / SIMILAR_THRESHOLD)
        return max(0.0, round(conf, 1)), 'DIFFERENT'


def match_pair(img_a_path: Path, img_b_path: Path, detector_name: str = 'SIFT', out_dir: Path = None):
    t0 = time.perf_counter()
    
    # 1. Preprocess
    img_a, proc_a = preprocess(img_a_path)
    img_b, proc_b = preprocess(img_b_path)
    
    # 2. Extract features
    det, norm = create_detector(detector_name)
    kps_a, des_a = det.detectAndCompute(proc_a, None)
    kps_b, des_b = det.detectAndCompute(proc_b, None)
    
    n_kps_a = len(kps_a) if kps_a else 0
    n_kps_b = len(kps_b) if kps_b else 0
    
    if des_a is None or des_b is None or n_kps_a < 4 or n_kps_b < 4:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            'keypoints_a': n_kps_a, 'keypoints_b': n_kps_b,
            'total_matches': 0, 'good_matches': 0,
            'inliers': 0, 'inlier_ratio': 0.0,
            'confidence': 0.0, 'band': 'DIFFERENT',
            'time_ms': round(elapsed_ms, 1)
        }
        
    # 3. Match with Lowe's ratio test (k=2)
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw_matches = bf.knnMatch(des_a, des_b, k=2)
    total_matches = len(raw_matches)
    
    good_matches = [m for m, n in raw_matches if m.distance < RATIO_THRESHOLD * n.distance]
    n_good = len(good_matches)
    
    # 4. Geometric verification (RANSAC)
    inliers = 0
    inlier_ratio = 0.0
    has_valid_h = False
    inlier_match_objs = []
    
    if n_good >= 4:
        pts_a = np.float32([kps_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_b = np.float32([kps_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, RANSAC_REPROJ)
        
        if mask is not None and H is not None:
            # Determinant sanity check (reject degenerate zero/negative mappings)
            det_h = np.linalg.det(H[:2, :2])
            if det_h > 1e-4:
                has_valid_h = True
                mask_flat = mask.ravel()
                inliers = int(mask_flat.sum())
                inlier_ratio = round(inliers / n_good, 3)
                inlier_match_objs = [good_matches[i] for i, v in enumerate(mask_flat) if v == 1]
                
    # 5. Confidence score and decision band
    confidence, band = compute_confidence(inliers, inlier_ratio, has_valid_h)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    # 6. Save visualization if requested
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Uniformly subsample inliers for clean visualization across slab
        max_draw = 50
        if len(inlier_match_objs) > max_draw:
            idx = np.linspace(0, len(inlier_match_objs) - 1, max_draw, dtype=int)
            draw_matches = [inlier_match_objs[i] for i in idx]
        else:
            draw_matches = inlier_match_objs
            
        viz = cv2.drawMatches(
            img_a, kps_a, img_b, kps_b, draw_matches, None,
            matchColor=(0, 255, 0), singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        out_filename = f"{img_a_path.stem}__{img_b_path.stem}.jpg"
        out_path = out_dir / out_filename
        cv2.imwrite(str(out_path), viz)
        
    return {
        'keypoints_a': n_kps_a,
        'keypoints_b': n_kps_b,
        'total_matches': total_matches,
        'good_matches': n_good,
        'inliers': inliers,
        'inlier_ratio': inlier_ratio,
        'confidence': confidence,
        'band': band,
        'time_ms': round(elapsed_ms, 1),
        'saved_vis': str(out_dir / f"{img_a_path.stem}__{img_b_path.stem}.jpg") if out_dir else None
    }


def main():
    parser = argparse.ArgumentParser(description="NMG Image Matching Challenge — Pair Matcher")
    parser.add_argument('--image-a', required=True, type=str, help="Path to reference image A")
    parser.add_argument('--image-b', required=True, type=str, help="Path to query image B")
    parser.add_argument('--out', default='output', type=str, help="Output directory for inlier visualization")
    parser.add_argument('--detector', default='SIFT', type=str, choices=['SIFT', 'ORB', 'AKAZE', 'BRISK'], help="Feature detector")
    
    args = parser.parse_args()
    
    img_a_path = Path(args.image_a)
    img_b_path = Path(args.image_b)
    out_dir = Path(args.out)
    
    try:
        res = match_pair(img_a_path, img_b_path, detector_name=args.detector, out_dir=out_dir)
        
        print("\n" + "=" * 65)
        print(f"IMAGE MATCHING RESULTS ({args.detector})")
        print("=" * 65)
        print(f"Image A          : {img_a_path.name}")
        print(f"Image B          : {img_b_path.name}")
        print(f"Keypoints A / B  : {res['keypoints_a']} / {res['keypoints_b']}")
        print(f"Total Matches    : {res['total_matches']}")
        print(f"Good Matches     : {res['good_matches']} (Ratio Test 0.75)")
        print(f"RANSAC Inliers   : {res['inliers']}")
        print(f"Inlier Ratio     : {res['inlier_ratio']:.1%}")
        print(f"Confidence Score : {res['confidence']}%")
        print(f"Decision Band    : {res['band']}")
        print(f"Processing Time  : {res['time_ms']:.1f} ms")
        if res.get('saved_vis'):
            print(f"Visualization    : {res['saved_vis']}")
        print("=" * 65 + "\n")
        
    except Exception as e:
        print(f"Error matching images: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
