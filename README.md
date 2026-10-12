# NMG Image Matching Challenge — Solution

A computer vision tool built for **NMG Labs** to decide whether two stone slab photographs show the **same physical object**, or merely a visually similar lookalike from the same quarry block.

---

## Quickstart (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run batch evaluation on all 100 image pairs (writes results.csv)
python evaluate.py --out results.csv

# 3. Compare any two individual images (saves visualization in output/)
python match.py --image-a reference/img_036__B01A_carrara_reference.jpg --image-b queries/img_021__B01A_rotation_scale_same.jpg --out output/
```

---

## Benchmark Results (100 Image Pairs)

Evaluated across all 5 reference slabs and 20 query images (10 same, 10 lookalike, 80 different):

| Algorithm | SAME Recall | False Positives on Lookalikes | Accuracy | Weakest SAME (Inliers) | Strongest NON-SAME (Inliers) | Margin | Avg Time/Pair |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT (Default)** | **10 / 10 (100%)** | **0 / 10 (0%)** | **100.0%** | **126** | **8** | **+118 inliers** | 132 ms |
| **ORB** | 10 / 10 (100%) | 0 / 10 (0%) | 100.0% | 107 | 8 | +99 inliers | **39 ms** ⚡ |
| **AKAZE** | 10 / 10 (100%) | 0 / 10 (0%) | 100.0% | 72 | 6 | +66 inliers | 107 ms |
| **BRISK** | 9 / 10 (90%) | 0 / 10 (0%) | 99.0% | 31 | 10 | +21 inliers | 144 ms |

---

## Completed Bonus Deliverables (PDF Part 2 · B)

1. **Partial Occlusion Breaking Point Analysis**:
   - Synthetically stress-tested occlusion from 0% to 95% (`output/occlusion_stress_test.png`).
   - The pipeline operates reliably up to **85% occlusion** (100 verified inliers) before dropping below the 40-inlier threshold at $\ge 90\%$.
2. **Production Feature Caching Speedup**:
   - Pre-computing catalog reference descriptors in memory once achieves a **76.3% runtime reduction** (drops from 13.16s down to 3.11s for 100 pairs — **31 ms per comparison**).
3. **SIFT vs. ORB Head-to-Head Comparison**:
   - SIFT provides the highest safety margin (+118 inliers); ORB delivers 4x faster mobile-grade inference (39 ms).

---

## Repository Structure

```
Intern_Assignment_Dataset/
├── match.py                         # CLI tool to compare any two images
├── evaluate.py                      # CLI tool to batch evaluate pairs and write results.csv
├── image_matching_challenge.ipynb   # Step-by-step interactive Jupyter notebook
├── REPORT.md                        # Technical brief (methodology, formulas, failure cases, bonuses)
├── requirements.txt                 # Python package dependencies
├── results.csv                      # Full evaluation output table on the 100 pairs
├── reference/                       # 5 catalog reference images
├── queries/                         # 20 query images (rotation, viewpoint, lighting, occlusion)
└── output/                          # Inlier visualization images, benchmark tables, and stress plots
```

---

## Methodology Highlights

1. **Preprocessing & Enhancement**: Normalizes longest dimension to 800px. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization, clip limit 2.0, tile grid 8×8) to reveal subtle veins in shadows without amplifying noise.
2. **Feature Extraction**: SIFT generates scale-invariant keypoint descriptors across octave spaces.
3. **Lowe's Ratio Test (0.75)**: Filters ambiguous matches on repetitive marble backgrounds.
4. **RANSAC Geometric Verification (5.0px)**: Computes a planar homography and counts inliers agreeing with rigid surface transformation. Sibling lookalikes fail this check, keeping lookalike inliers to random noise ($\le 8$).
5. **Three Decision Bands**:
   - **`SAME`** ($\ge 40$ inliers, Confidence $70\% - 100\%$): Exact physical piece verified.
   - **`SIMILAR`** ($15 - 39$ inliers, Confidence $35\% - 69\%$): Quarry block lookalike candidate.
   - **`DIFFERENT`** ($< 15$ inliers, Confidence $0\% - 34\%$): Different stone block.
