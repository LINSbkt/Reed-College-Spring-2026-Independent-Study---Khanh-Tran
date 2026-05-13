# Spatial Grounding for Mitotic Figure Identification with Vision-Language Models

Investigating whether bounding boxes or SAM segmentation masks improve 
VLM performance on mitotic figure detection in H&E breast cancer histopathology.

Dataset: MIDOG++ Domain 1a (Human Breast Carcinoma)  
Model: Gemini 3.1 Flash Lite (Google DeepMind) via API  
Segmentation: Segment Anything Model 

---

## Background

Mitotic figures (cells caught in active division) are a key component of 
histological grading in breast cancer. Distinguishing them from visually similar 
hard negatives (apoptotic cells, hyperchromatic and pyknotic nuclei) is 
difficult even for trained pathologists, with ~20% inter-observer disagreement.

This project asks: can a general-purpose Vision-Language Model identify mitotic 
figures from H&E image crops, and does providing spatial grounding (a bounding 
box or SAM segmentation mask) improve its performance?

---

## Dataset

MIDOG++ (Aubreville et al., Scientific Data 2023)  
- Domain 1a: 20 whole-slide image regions, human breast carcinoma  
- 388 annotated cells: 141 mitotic figures, 247 hard negatives  
- Images: 7,215 × 5,412 px TIFF, ~148 MB each, digitized at 40×  
- Ground truth: bounding boxes with binary category labels

---

## Experimental Design

Each annotation was presented to the VLM as a 256×256 pixel crop under 
three spatial grounding conditions:

| Condition | Description |
|---|---|
| Raw Crop | No spatial grounding |
| Bounding Box | Red bounding box drawn around target cell |
| Mask | Semi-transparent SAM segmentation mask overlay |

Four experiments varied prompting strategy and crop positioning:

| Experiment | Crops | Prompts | Confidence score |
|---|---|---|---|
| 1 | Centered | Basic | No |
| 2 | Centered | Improved (domain knowledge) | No |
| 3 | Off-center | Improved | Yes |
| 4 | Centered | Improved | Yes |

---

## Setup

Python 3.12+
Set your Gemini API key as an environment variable
Download SAM

---

## Running the Experiments

Run in order as each experiment builds on patch files and results from previous ones.

Explore the dataset
jupyter notebook notebooks/data_exploration.ipynb

Download patches: patches, patches_exp3

Experiment 1: basic prompts, centered crops
python notebooks/experiment_1.py

Experiment 2: improved prompts with domain knowledge
python notebooks/experiment_2.py

Experiment 3: off-center crops + confidence score
python notebooks/experiment_3.py

Experiment 4: centered crops + confidence score
python notebooks/exp4.py

---

## Acknowledgements

Supervisor: Anna Ritz, Reed College  
Dataset: Aubreville et al., MIDOG++ (Scientific Data, 2023)  
SAM: Kirillov et al., Meta AI Research (ICCV, 2023)
