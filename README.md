# Out-of-Distribution Anomaly Detection for Robust Autonomous Driving Object Detectors
> COMP 4610SEF Data Science Project | Hong Kong Metropolitan University | Group 15

## Overview
This project builds an end-to-end OOD (Out-of-Distribution) warning pipeline that acts as an independent safety monitor on top of 2D object detectors for autonomous driving.

Instead of designing a new detector from scratch, we add a parallel anomaly scoring module based on **Mahalanobis distance on backbone features**. The module evaluates whether the current driving scene falls outside the training distribution and triggers hierarchical warnings before the detector fails.

## Core Research Objectives
1. Integrate a Mahalanobis-based OOD module with a CNN-Transformer hybrid object detector, and verify reliable warnings for both autonomous agents and human drivers across three corner-case categories.
2. Investigate how targeted data augmentation affects detector generalization and OOD warning reliability across different corner-case types.

## Key Novel Contributions
1. **Dual-Pathway End-to-End Warning Pipeline**
   Separate threshold logic for autonomous driving agents (high recall, safety-first) and human drivers (low false alarm, usability-first), with per-category FPR/FNR evaluation.

2. **Hybrid Backbone + OOD Empirical Study**
   Systematic comparison of pure CNN (Faster R-CNN) and CNN-Transformer hybrid (Deformable DETR) backbones paired with Mahalanobis OOD scoring, to verify whether global contextual features improve anomaly robustness.

3. **Fine-Grained Corner Case Evaluation**
   Performance analysis across three corner-case categories: adverse weather, occluded targets, and novel obstacles.

## Project Structure
