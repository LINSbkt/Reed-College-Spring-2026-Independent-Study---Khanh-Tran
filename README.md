## Segmentation-Guided Spatial Grounding for Mitotic Figure VQA
Independent Study Project | Reed College | Spring 2025

Advisor: Anna Ritz   
Student: Tran Bao Khanh 
## Overview
This project investigates whether spatial grounding — via bounding boxes or SAM segmentation masks — improves Vision-Language Model (VLM) performance on mitotic figure identification in H&E breast cancer histopathology.

Mitotic figures are cells caught in active division. Their count per unit area is a standard component of tumour grading for breast carcinoma, but manual counting is laborious and subject to ~20% inter-observer disagreement. This study uses **Gemini 3.1 Flash** as the VLM and **SAM ViT-H** as the segmentation backbone, evaluated on the **MIDOG++ Domain 1a** dataset.

## Research Question
> Does providing a VLM with explicit spatial information (bounding boxes or segmentation masks) improve its ability to distinguish mitotic figures from visually similar hard negatives (apoptotic cells, hyperchromatic nuclei, pyknotic nuclei) in H&E histopathology?

## Dataset

**MIDOG++ Domain 1a — Human Breast Carcinoma**
- 20 whole-slide image regions downloaded from the public MIDOG++ dataset
- Image size: 7,215 × 5,412 pixels, RGBA, ~148 MB each
- **388 total annotations: 141 mitotic figures, 247 hard negatives**
- Ground truth provided as bounding boxes with binary category labels
- All images digitized at 40× magnification (0.25 µm/pixel)

## Repository Structure

## Experimental Design

## Results

## Acknowledgements

- Dr. Anna Ritz (Reed College) — supervision and guidance
- MIDOG++ dataset — Aubreville et al., Scientific Data 2023
- SAM — Kirillov et al., ICCV 2023
- Gemini 3.1 Flash — Google DeepMind
