# MedSAM BRISC – Quantum Classification

This repository contains the code used for brain tumor MRI segmentation and classification on the BRISC dataset.

## Pipeline

The project was done in two main steps:

**1. MedSAM fine-tuning**

`medsam_full_fine_tuned2.py`

* Starts from the pretrained MedSAM model.
* Fine-tunes MedSAM for tumor segmentation.
* Uses the Multi-BBox approach.
* Produces the best fine-tuned checkpoint:
  `medsam_full_v2_best.pth`

**2. Classification head and ablation**

`medsam_head_ablation_compare.py`

* Starts from `medsam_full_v2_best.pth`.
* Adds a classification head after the MedSAM image embedding.
* Compares different classification heads/configurations, including MLP and quantum heads (Q4, Q6, Q8).

## Workflow

```text
Pretrained MedSAM
       ↓
medsam_full_fine_tuned2.py
       ↓
medsam_full_v2_best.pth
       ↓
medsam_head_ablation_compare.py
       ↓
MLP / Q4 / Q6 / Q8
```

The `.pth` checkpoints are not included in the repository because they are large files.
