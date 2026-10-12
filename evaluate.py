"""
evaluate.py — NMG Image Matching Challenge
Batch evaluation CLI script to test pairs and generate results.csv and summary statistics.

Usage:
    python evaluate.py --data . --out results.csv
    python evaluate.py --data data --pairs data/pairs.csv --out results.csv
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

from match import match_pair


def get_piece_code(path: Path) -> str:
    parts = path.stem.split('__')
    return parts[1].split('_')[0] if len(parts) > 1 else 'UNKNOWN'


def get_condition(path: Path) -> str:
    stem = path.stem
    for cond in ['rotation_scale', 'viewpoint', 'lighting', 'occlusion']:
        if cond in stem:
            return cond
    return 'unknown'


def load_evaluation_pairs(data_dir: Path, pairs_csv: Path = None) -> list[dict]:
    """
    Loads pairs from pairs_csv if provided, or discovers all pairs between reference/ and queries/.
    """
    pairs_list = []
    
    if pairs_csv is not None and pairs_csv.exists():
        df_pairs = pd.read_csv(pairs_csv)
        for _, row in df_pairs.iterrows():
            img_a = data_dir / row['image_a'] if not Path(row['image_a']).is_absolute() else Path(row['image_a'])
            img_b = data_dir / row['image_b'] if not Path(row['image_b']).is_absolute() else Path(row['image_b'])
            pairs_list.append({
                'image_a_path': img_a,
                'image_b_path': img_b,
                'image_a_name': img_a.name,
                'image_b_name': img_b.name,
                'label'       : str(row.get('label', 'unknown')),
                'pair_type'   : str(row.get('pair_type', 'unknown'))
            })
    else:
        # Fallback to standard reference and query directories
        ref_dir = data_dir / 'reference'
        qry_dir = data_dir / 'queries'
        
        if not ref_dir.exists() or not qry_dir.exists():
            # Check current working directory
            ref_dir = Path('reference')
            qry_dir = Path('queries')
            
        refs = sorted(ref_dir.glob('*.jpg'))
        qrys = sorted(qry_dir.glob('*.jpg'))
        
        for r in refs:
            for q in qrys:
                code_r = get_piece_code(r)
                code_q = get_piece_code(q)
                is_same = (code_r == code_q)
                
                label = 'same' if is_same else 'different'
                cond = get_condition(q)
                pair_type = cond if is_same else ('lookalike' if code_r[:3] == code_q[:3] else 'different')
                
                pairs_list.append({
                    'image_a_path': r,
                    'image_b_path': q,
                    'image_a_name': r.name,
                    'image_b_name': q.name,
                    'label'       : label,
                    'pair_type'   : pair_type
                })
                
    return pairs_list


def main():
    parser = argparse.ArgumentParser(description="NMG Image Matching Challenge — Batch Evaluator")
    parser.add_argument('--data', default='.', type=str, help="Base data directory")
    parser.add_argument('--pairs', default=None, type=str, help="Optional path to pairs.csv file")
    parser.add_argument('--out', default='results.csv', type=str, help="Output CSV path")
    parser.add_argument('--detector', default='SIFT', type=str, choices=['SIFT', 'ORB', 'AKAZE', 'BRISK'], help="Detector algorithm")
    parser.add_argument('--save-vis', action='store_true', help="Save inlier visualizations to output/ folder")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    pairs_csv = Path(args.pairs) if args.pairs else None
    out_path = Path(args.out)
    out_vis_dir = Path('output') if args.save_vis else None
    
    pairs = load_evaluation_pairs(data_dir, pairs_csv)
    if not pairs:
        print("No evaluation pairs found!", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nRunning batch evaluation on {len(pairs)} pairs using {args.detector}...")
    t_start = time.perf_counter()
    
    results = []
    for i, pair in enumerate(pairs, 1):
        try:
            match_res = match_pair(
                pair['image_a_path'],
                pair['image_b_path'],
                detector_name=args.detector,
                out_dir=out_vis_dir
            )
            
            results.append({
                'image_a'      : pair['image_a_name'],
                'image_b'      : pair['image_b_name'],
                'label'        : pair['label'],
                'pair_type'    : pair['pair_type'],
                'keypoints_a'  : match_res['keypoints_a'],
                'keypoints_b'  : match_res['keypoints_b'],
                'total_matches': match_res['total_matches'],
                'good_matches' : match_res['good_matches'],
                'inliers'      : match_res['inliers'],
                'inlier_ratio' : match_res['inlier_ratio'],
                'confidence'   : match_res['confidence'],
                'band'         : match_res['band'],
                'time_ms'      : match_res['time_ms']
            })
            
            if i % 25 == 0 or i == len(pairs):
                print(f"  Processed {i}/{len(pairs)} pairs...")
                
        except Exception as e:
            print(f"Error on pair {pair['image_a_name']} vs {pair['image_b_name']}: {e}", file=sys.stderr)
            
    total_time = time.perf_counter() - t_start
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved evaluation results to {out_path} ({len(df)} rows, completed in {total_time:.2f}s)")
    
    # Check if ground truth labels are known
    known_labels = df[df['label'].isin(['same', 'different', 'SAME', 'DIFFERENT'])]
    if not known_labels.empty:
        print("\n" + "=" * 80)
        print("EVALUATION PERFORMANCE SUMMARY")
        print("=" * 80)
        
        same_pairs = df[df['label'].str.lower() == 'same']
        diff_pairs = df[df['label'].str.lower() == 'different']
        lookalikes = df[df['pair_type'].str.lower() == 'lookalike']
        
        tp = (same_pairs['band'] == 'SAME').sum()
        fp = (diff_pairs['band'] == 'SAME').sum()
        fp_lookalike = (lookalikes['band'] == 'SAME').sum() if not lookalikes.empty else 0
        
        accuracy = ((same_pairs['band'] == 'SAME').sum() + (diff_pairs['band'] != 'SAME').sum()) / len(df) * 100
        
        print(f"Total Pairs Evaluated   : {len(df)}")
        print(f"SAME Recall             : {tp}/{len(same_pairs)} ({tp/len(same_pairs)*100:.1f}%)" if len(same_pairs) else "SAME Recall: N/A")
        print(f"Total False Positives   : {fp}/{len(diff_pairs)} (Goal: 0)")
        print(f"Lookalikes Called SAME  : {fp_lookalike}/{len(lookalikes)} (Goal: 0)")
        print(f"Overall Accuracy        : {accuracy:.1f}%")
        print(f"Min Inliers on SAME     : {same_pairs['inliers'].min() if len(same_pairs) else 'N/A'}")
        print(f"Max Inliers on NON-SAME : {diff_pairs['inliers'].max() if len(diff_pairs) else 'N/A'}")
        print(f"Average Time per Pair   : {df['time_ms'].mean():.1f} ms")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
