# NMG Image Matching Challenge — Technical Report
**Author:** Candidate Submission  
**Focus:** Fine-Grained Identification of Natural Stone Slabs  

---

## 1. Executive Summary
Natural stone slabs sliced sequentially from the same quarry block share near-identical base coloration and macro-veining structures ("the family resemblance"). However, for architectural provenance, inventory tracking, and quality assurance, our system must determine whether two photographs show the **exact same physical piece** ("the fingerprint"), even under severe real-world distortions.

Using a classical Computer Vision pipeline combining **CLAHE contrast enhancement**, **SIFT feature extraction**, **Lowe's ratio testing ($0.75$)**, and **RANSAC geometric homography verification ($5.0\text{px}$)**, our system achieves:
- **100% Recall on SAME physical pieces ($10/10$)**
- **0% False Positive Rate on Lookalikes ($0/10$)**
- **0% False Positive Rate on Different Stones ($0/80$)**
- **Overall Accuracy: 100.0%**
- **Average Inference Latency: $\approx 132 \text{ ms}$ per pair (optimizable to $31 \text{ ms}$ via caching)**

---

## 2. Algorithm Selection & Pipeline Design

```
[Raw Image A & B]
       │
       ▼
[1. Preprocessing] ──► Resize (long-edge = 800px) + CLAHE (clip=2.0, grid=8x8)
       │
       ▼
[2. Feature Extractor] ──► SIFT (Scale-Invariant Feature Transform, L2 Norm)
       │
       ▼
[3. Feature Matching] ──► Brute-Force Matcher + Lowe's Ratio Test (k=2, ratio=0.75)
       │
       ▼
[4. Geometric Check] ──► RANSAC Homography (5.0px reprojection threshold)
       │
       ▼
[5. Scoring & Bands] ──► Confidence Score (0–100%) → SAME / SIMILAR / DIFFERENT
```

### Why SIFT (Scale-Invariant Feature Transform)?
- Natural stone has intricate crystal fracturing and vein intersections. SIFT builds a scale-space Gaussian pyramid with orientation gradient histograms, making keypoints invariant to arbitrary rotations and zooming.
- Compared to ORB, SIFT generates richer descriptors ($128$-dimensional floating-point vectors vs. $256$-bit binary strings), producing an order of magnitude more verified inliers on textured granite ($>3,000$ inliers).

### Why CLAHE Preprocessing?
- Slabs stored in warehouses suffer from localized spotlight glare or dim ambient shadows. Standard global histogram equalization washes out subtle marble veins.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** operates on local $8 \times 8$ tiles, equalizing contrast while suppressing noise amplification, ensuring subtle veining remains visible to the detector.

### Why Lowe's Ratio Test ($0.75$)?
- Plain marble fields contain visually repetitive white/beige patches. Nearest-neighbor matching alone produces ambiguous false positives.
- By demanding that the best match distance $d_1$ is at least $25\%$ shorter than the second-best match $d_2$ ($d_1 / d_2 < 0.75$), ambiguous matches are eliminated before RANSAC.

### Why RANSAC ($5.0\text{px}$)?
- Two photos of the same rigid planar slab are related by a 2D projective transformation (Homography $H$).
- RANSAC randomly samples 4 candidate correspondences, fits $H$, and projects all points. If a projected point lands within a $5.0$-pixel tolerance radius, it is classified as an **inlier**.
- Inliers must agree on the rigid geometric transformation. Sibling lookalike slabs from the same block have slightly shifted 3D vein structures that fail this rigid planar check, keeping lookalike inliers down to random noise levels ($<16$).

---

## 3. Confidence Score Formula & Decision Thresholds

### Mathematical Formulation
A confidence score must be grounded in geometric evidence and strictly output $0\%$ if verification fails:

$$\text{Inlier Ratio} = \frac{\text{Inliers}}{\text{Good Matches}}$$

$$\text{Confidence (\%)} = \begin{cases} 
0.0\% & \text{if inliers } < 4 \text{ or } \det(H_{2\times 2}) \le 0 \\
\min\left(100.0, \, 70.0 + 30.0 \times \left[0.7 \cdot \frac{\text{Inliers} - 40}{60} + 0.3 \cdot \min\left(1.0, \frac{\text{Inlier Ratio}}{0.5}\right)\right]\right) & \text{if inliers } \ge 40 \\
35.0 + 34.0 \times \left(\frac{\text{Inliers} - 15}{25}\right) & \text{if } 15 \le \text{inliers} < 40 \\
34.0 \times \left(\frac{\text{Inliers}}{15}\right) & \text{if inliers } < 15
\end{cases}$$

### The Three Decision Bands:
1. **SAME ($\ge 70.0\%$ Confidence, $\ge 40$ Inliers):** Overwhelming geometric consensus proving the photos show the same physical object.
2. **SIMILAR ($35.0\% - 69.9\%$ Confidence, $15 - 39$ Inliers):** Statistically significant feature similarity, but insufficient geometric evidence (quarry twin / lookalike candidates).
3. **DIFFERENT ($0.0\% - 34.9\%$ Confidence, $< 15$ Inliers):** No coherent geometric alignment.

---

## 4. Benchmark Results on Provided Dataset (100 Pairs)

### Detector Comparison Table:
| Algorithm | SAME Recall | False Positives | Accuracy | Weakest SAME (Inliers) | Strongest NON-SAME (Inliers) | Margin | Avg Time/Pair |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT (Selected)** | **10 / 10 (100%)** | **0 / 90 (0%)** | **100.0%** | **126** | **16** | **+110** | 155 ms |
| **ORB** | 10 / 10 (100%) | 0 / 90 (0%) | 100.0% | 107 | 8 | +99 | **39 ms** |
| **AKAZE** | 10 / 10 (100%) | 0 / 90 (0%) | 100.0% | 72 | 6 | +66 | 107 ms |
| **BRISK** | 9 / 10 (90%) | 0 / 90 (0%) | 99.0% | 31 | 10 | +21 | 144 ms |

### SIFT Performance Breakdown by Condition:
| Condition | Evaluated Pairs | Correct Matches | Average Inliers |
| :--- | :---: | :---: | :---: |
| **Rotation & Scale** | 3 | 3 / 3 (100%) | **1,566** |
| **Viewpoint Angle** | 3 | 3 / 3 (100%) | **1,870** |
| **Partial Occlusion** | 2 | 2 / 2 (100%) | **454** |
| **Lighting / Glare** | 2 | 2 / 2 (100%) | **162** |
| **Lookalike Sibling** | 10 | 0 / 10 False Positives | **6.2** (Max: 16) |
| **Different Quarry Block**| 80 | 0 / 80 False Positives | **1.8** (Max: 8) |

---

## 5. Challenges Faced & Failure Case Analysis

### Failure Case 1: BRISK Under Severe Spotlight Glare (`img_039__B02A` vs `img_008__B02A`)
- **Observed Behavior:** On Statuario white marble under the `lighting` condition, BRISK yielded only **31 inliers**, falling below the uniform $40$-inlier threshold and getting classified as `SIMILAR`.
- **Root Cause:** Statuario marble consists of a stark white background with faint, delicate grey veins. The query image contains an intense specular spotlight reflection in the center. BRISK relies on the AGAST corner detector with fixed intensity difference comparisons around a circular sampling ring. The blinding glare flattened local intensity gradients to zero, preventing corner responses.
- **Resolution:** SIFT constructs Difference-of-Gaussians (DoG) responses across multiple octaves and successfully extracted **126 inliers** on the exact same pair. Furthermore, adding CLAHE improved BRISK from its original baseline of 28 inliers to 31 inliers.

### Failure Case 2: Repetitive Texture Drift on Fine-Grained Granite (`B05A` vs `B05B`)
- **Observed Behavior:** On fine-grained grey granite, raw nearest-neighbor matching generated over **1,200 initial matches** between two different slabs from the same block.
- **Root Cause:** Grey granite exhibits high spatial frequency texture (thousands of black and white crystal speckles per square inch) that look mathematically identical at a local patch level.
- **Resolution:** Lowe's ratio test discarded 85% of ambiguous speckle matches. Subsequent RANSAC homography estimation discarded remaining false correspondences, yielding only **5 geometric inliers**—far below the $40$-inlier threshold and ensuring zero false positives.

---

## 6. Completed Bonus Tasks (From PDF Part 2 · B)

### Bonus 1: Partial Occlusion Breaking Point Analysis
* **Question from Brief:** *"How much of the piece can be hidden before matching fails?"*
* **Experimental Findings:** We synthetically masked an increasing percentage of the slab from $0\%$ to $95\%$ with an opaque barrier (`output/occlusion_stress_test.png`).
  * At $50\%$ occlusion: **699 inliers** (Overwhelming match).
  * At $70\%$ occlusion: **322 inliers** (Solid match).
  * At $85\%$ occlusion: **100 inliers** (Still well above the 40-inlier threshold).
  * At $90\%$ occlusion: **31 inliers** (Drops below 40 into `SIMILAR`).
* **Conclusion:** The pipeline is exceptionally resilient, operating reliably up to **$85\%$ occlusion** (only $15\%$ of the stone visible).

### Bonus 2: Production Speedup via Reference Feature Caching
* **Question from Brief:** *"Execution time and at least one optimization, such as caching features, with before and after times."*
* **Implementation:** In a production catalog system, reference slabs are static. We pre-extract and cache the 5 reference SIFT descriptors once in memory, rather than re-computing them on every query comparison.
* **Benchmark Results (100 Comparisons):**
  * **Baseline Runtime (No Caching):** $13.16\text{ seconds}$ ($131.6\text{ ms/pair}$)
  * **Optimized Runtime (With Caching):** $3.11\text{ seconds}$ ($31.1\text{ ms/pair}$)
  * **Performance Gain:** **$76.3\%$ Speedup** ($4.23\times$ faster throughput).

### Bonus 3: SIFT vs. ORB Head-to-Head Benchmark
* Both evaluated on the exact same 100 pairs:
  * **SIFT:** Superior robustness on white marbles with subtle veins (126 inliers on hardest pair; +110 margin; 155 ms/pair).
  * **ORB:** Exceptional throughput for real-time mobile apps (39 ms/pair; 100% accuracy; +99 margin).

---

## 7. Recommended Future Improvements
1. **Adaptive Multi-Scale Pyramid:** Processing an image pyramid at $600\text{px}$, $800\text{px}$, and $1200\text{px}$ would capture both macro-vein layouts and micro-crystal fissures under extreme zoom.
2. **Homography Area & Aspect Sanity Checks:** Discarding transformations whose projected quadrilaterals have self-intersections, concave angles, or aspect ratios $>5:1$ protects against accidental collinear match alignments.
3. **Two-Stage Cascade Architecture:** Run cached ORB first for instantaneous 31ms screening. If inliers land in the borderline 20–45 range, trigger SIFT for high-precision verification.
