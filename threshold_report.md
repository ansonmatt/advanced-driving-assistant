# ADAS Confidence Threshold Evaluation Report

**Model Path:** `models/ADAS_Trained.pt`  
**Model Classes:** `['Bus', 'Car', 'Fixed Obstacle', 'Pothole', 'Three Wheeler', 'Tractor', 'Truck', 'Two Wheeler', 'person']` (excluding 'text')  
**Evaluation Date:** 2026-06-22 22:52:27  
**Total Sampled Frames Processed:** 2344 (sampled every 5th frame) across 8 videos.  

## Executive Summary & Recommendation

Based on the evaluation results, we analyzed thresholds from **0.15** to **0.35** to find the optimal balance for a live demo on Indian roads:
- **0.15 to 0.20**: High sensitivity but likely introduces substantial false positives (noise), especially in busy traffic scenes.
- **0.25 to 0.30**: Appears to be the 'sweet spot' with stable detections and moderate zero-detection frames.
- **0.35**: Might be too conservative for this model, leading to missed detections (higher zero-detection frames, especially on highways or potholes).

*(See the detailed recommendations section at the bottom for final selection guidance)*

## Overall Performance Summary

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| **0.15** | 0.481 | 1448 / 2344 | 61.8% | 1127 |
| **0.20** | 0.403 | 1544 / 2344 | 65.9% | 945 |
| **0.25** | 0.349 | 1631 / 2344 | 69.6% | 818 |
| **0.30** | 0.311 | 1693 / 2344 | 72.2% | 730 |
| **0.35** | 0.273 | 1768 / 2344 | 75.4% | 641 |


## Per-Video Breakdown

### Video: `highway1.mp4`

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| 0.15 | 0.558 | 138 / 260 | 53.1% | 145 |
| 0.20 | 0.458 | 158 / 260 | 60.8% | 119 |
| 0.25 | 0.392 | 168 / 260 | 64.6% | 102 |
| 0.30 | 0.354 | 177 / 260 | 68.1% | 92 |
| 0.35 | 0.331 | 182 / 260 | 70.0% | 86 |

### Video: `pothole1.mp4`

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| 0.15 | 0.729 | 102 / 339 | 30.1% | 247 |
| 0.20 | 0.652 | 120 / 339 | 35.4% | 221 |
| 0.25 | 0.608 | 134 / 339 | 39.5% | 206 |
| 0.30 | 0.581 | 143 / 339 | 42.2% | 197 |
| 0.35 | 0.540 | 157 / 339 | 46.3% | 183 |

### Video: `test1.mp4`

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| 0.15 | 0.038 | 207 / 212 | 97.6% | 8 |
| 0.20 | 0.028 | 207 / 212 | 97.6% | 6 |
| 0.25 | 0.024 | 208 / 212 | 98.1% | 5 |
| 0.30 | 0.005 | 211 / 212 | 99.5% | 1 |
| 0.35 | 0.000 | 212 / 212 | 100.0% | 0 |

### Video: `test2.mp4`

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| 0.15 | 0.104 | 467 / 518 | 90.2% | 54 |
| 0.20 | 0.083 | 476 / 518 | 91.9% | 43 |
| 0.25 | 0.069 | 483 / 518 | 93.2% | 36 |
| 0.30 | 0.060 | 488 / 518 | 94.2% | 31 |
| 0.35 | 0.037 | 499 / 518 | 96.3% | 19 |

### Video: `test3.mp4`

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| 0.15 | 1.373 | 21 / 83 | 25.3% | 114 |
| 0.20 | 1.169 | 23 / 83 | 27.7% | 97 |
| 0.25 | 1.048 | 29 / 83 | 34.9% | 87 |
| 0.30 | 0.928 | 30 / 83 | 36.1% | 77 |
| 0.35 | 0.807 | 35 / 83 | 42.2% | 67 |

### Video: `test4.mp4`

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| 0.15 | 0.781 | 176 / 374 | 47.1% | 292 |
| 0.20 | 0.644 | 197 / 374 | 52.7% | 241 |
| 0.25 | 0.553 | 214 / 374 | 57.2% | 207 |
| 0.30 | 0.476 | 232 / 374 | 62.0% | 178 |
| 0.35 | 0.401 | 254 / 374 | 67.9% | 150 |

### Video: `traffic1.mp4`

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| 0.15 | 0.677 | 117 / 263 | 44.5% | 178 |
| 0.20 | 0.559 | 131 / 263 | 49.8% | 147 |
| 0.25 | 0.468 | 149 / 263 | 56.7% | 123 |
| 0.30 | 0.418 | 160 / 263 | 60.8% | 110 |
| 0.35 | 0.380 | 169 / 263 | 64.3% | 100 |

### Video: `traffic2.mp4`

| Threshold | Avg Detections/Frame | Zero-Detection Frames | Zero-Detection % | Total Detections |
|-----------|----------------------|-----------------------|------------------|------------------|
| 0.15 | 0.302 | 220 / 295 | 74.6% | 89 |
| 0.20 | 0.241 | 232 / 295 | 78.6% | 71 |
| 0.25 | 0.176 | 246 / 295 | 83.4% | 52 |
| 0.30 | 0.149 | 252 / 295 | 85.4% | 44 |
| 0.35 | 0.122 | 260 / 295 | 88.1% | 36 |

## Detailed Analysis & Recommendations

### 1. Sensitivity vs. Reliability
- At **0.15**, the average detection rate is highest, which ensures that small objects (like potholes or distant persons/two-wheelers) are caught. However, this is prone to bounding box flicker and false positives on texture (like pavement patches, shadow lines, etc.).
- At **0.35** (the default in `adas_engine.py`), the zero-detection frame count rises significantly. On Indian roads, where obstacles are dense, having a high percentage of zero-detection frames suggests the system is under-detecting active traffic/hazards.

### 2. Live Demo Selection
- **Recommendation:** We recommend setting the threshold to **0.25** or **0.30** for the live demo.
- If the demo is on a **highway** (clearer lane markings, higher speeds, fewer obstacles), a **0.30** threshold is ideal to keep the HUD clean and focus on clear, confident vehicle targets.
- If the demo is on a **city road / pothole route** (dense two-wheelers, three-wheelers, pedestrians, potholes), a **0.20 or 0.25** threshold is preferred so the system doesn't miss potholes and smaller traffic actors that have slightly lower confidence scores due to partial occlusions.
