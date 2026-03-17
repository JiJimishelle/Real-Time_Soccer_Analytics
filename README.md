# Real-Time Soccer Analytics on the Edge

### A Lightweight Framework and Data Standardization (SAF-SVA)

A lightweight computer vision framework for **real-time soccer player tracking, team labeling, and ball possession estimation**, designed to run on **consumer-grade hardware**.

This repository implements a modernized pipeline inspired by classical soccer video analysis systems and enhanced with modern deep learning techniques.


# Overview

Soccer video analysis has progressed significantly with advances in **computer vision and artificial intelligence**, enabling automatic extraction of tactical and performance information from broadcast footage.

Early research such as **Liu et al. (2009)** introduced classical pipelines for:

* Player detection
* Player labeling
* Player tracking

using handcrafted visual features.

More recent approaches rely on **deep learning detectors and multi-object tracking algorithms**, such as **YOLO** and **ByteTrack**, which achieve strong benchmark performance.

However, current soccer analytics research still suffers from:

* inconsistent task definitions
* different evaluation protocols
* incompatible data formats

These issues make **direct comparison and reproducibility difficult**.


# Proposed System

This project introduces a **lightweight real-time soccer analytics framework** capable of running on **edge devices**.

The system automatically performs:

* Player detection
* Ball detection
* Team color labeling
* Multi-object tracking
* Ball possession estimation

The framework is optimized for **real-time performance** while maintaining high accuracy.


# System Architecture

Pipeline overview:

```
Video Input
     │
     ▼
YOLOv8 Detection
(Player + Ball)
     │
     ▼
Adaptive K-Means
Team Color Clustering
     │
     ▼
Norfair Tracking
(Temporal Inertia + Hold Rule)
     │
     ▼
Ball Possession Estimation
     │
     ▼
SAF-SVA Output Format
(JSON)
```


# Key Components

## Player & Ball Detection

* Model: **YOLOv8**
* Detects:

  * players
  * ball
* Ball detector:

  * initialized from **COCO pretrained weights**
  * fine-tuned with **~1000 manually annotated soccer-ball samples**


## Team Labeling

Team identity is determined using **adaptive K-Means clustering** based on dominant uniform colors.

This approach enables **unsupervised team separation** without manual annotation.


## Multi-Object Tracking

Tracking is implemented using **Norfair**, enhanced with:

* **Temporal inertia** to stabilize player identity
* **Hold-rule mechanism** to prevent short-term ID switches
* robust player trajectory generation


## Ball Possession Estimation

Ball possession is determined by analyzing:

* spatial proximity between player and ball
* temporal continuity of possession

This produces **stable possession timelines** across the match.


# Data Standardization: SAF-SVA

This work introduces a lightweight data schema:

**Soccer Analytics Format for Spatial-Visual Annotation (SAF-SVA)**

SAF-SVA is a **JSON-based format** designed to standardize:

* player tracking data
* team labeling
* ball position
* ball possession

Example structure:

```json
{
  "frame": 1250,
  "players": [
    {"id": 4, "team": "A", "bbox": [x, y, w, h]},
    {"id": 7, "team": "B", "bbox": [x, y, w, h]}
  ],
  "ball": {"x": 502, "y": 311},
  "possession": "Team A"
}
```

This format enables:

* reproducible experiments
* standardized evaluation
* fair benchmarking across studies


# Performance

Evaluation was conducted on **four full-match broadcast videos**.

Hardware:

* **MacBook Air (M3)**

Results:

| Task             | Precision | Recall |
| ---------------- | --------- | ------ |
| Player Detection | 96.3%     | 93.8%  |
| Ball Detection   | 96.1%     | 92.4%  |

System Performance:

* **25–30 FPS real-time processing**

These results demonstrate the framework's suitability for **edge deployment**.

---

# Key Features

* Real-time soccer analytics
* Lightweight architecture
* Runs on consumer hardware
* Automatic team labeling
* Ball possession estimation
* Standardised data format (SAF-SVA)

---

# Keywords

Soccer analytics
Player tracking
Lightweight framework
Ball possession estimation
Data standardization
SAF-SVA

