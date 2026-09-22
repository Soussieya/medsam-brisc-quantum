# MedSAM BRISC – Quantum Classification

This repository contains the code developed for brain tumor **MRI segmentation and classification** on the **BRISC dataset** using **MedSAM** and an ablation study comparing a classical **MLP classification head** with a **quantum variational classification head (VQC)**.

The project is organized into two main stages:

1. **MedSAM full fine-tuning for tumor segmentation**
2. **Classification head ablation: MLP vs. Quantum**

---

## 1. Project Pipeline

The complete workflow is:

```text
                    BRISC MRI Dataset
                           │
                           ▼
              Pretrained MedSAM ViT-B
                           │
                           ▼
          ┌────────────────────────────┐
          │  Full MedSAM Fine-Tuning   │
          │                            │
          │ medsam_full_fine_tuned2.py │
          └────────────────────────────┘
                           │
                           ▼
             medsam_full_v2_best.pth
                           │
                           ▼
          ┌────────────────────────────┐
          │ Classification Head        │
          │ Ablation Study              │
          │                            │
          │ medsam_head_ablation_      │
          │ compare.py                 │
          └────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
          MLP Head                 Quantum VQC
                                    Q4 / Q6 / Q8
              │                         │
              └────────────┬────────────┘
                           ▼
                 Comparative Metrics
```

The segmentation model is trained first. The resulting fine-tuned MedSAM checkpoint is then used as the starting point for the classification-head ablation experiments.

---

# 2. Repository Structure

```text
medsam-brisc-quantum/
│
├── medsam_full_fine_tuned2.py
│
├── medsam_head_ablation_compare.py
│
├── requirements.txt
│
├── README.md
│
└── report/
    └── PFE_Mitacs_Report_Eya_Soussi.pdf
```

### `medsam_full_fine_tuned2.py`

Full MedSAM fine-tuning pipeline.

It:

* loads pretrained MedSAM ViT-B;
* fine-tunes the image encoder, prompt encoder and mask decoder;
* uses the BRISC segmentation dataset;
* uses a **multi-bounding-box** strategy;
* extracts one bounding box for each connected tumor component;
* uses the same multi-BBox strategy during training, validation and testing;
* evaluates segmentation performance;
* saves the best fine-tuned checkpoint.

The script uses:

* Batch size: `2`
* Gradient accumulation: `2`
* Effective batch size: `4`
* Number of epochs: `50`
* Early stopping patience: `10`
* Image size: `1024 × 1024`

The three MedSAM components use different learning rates:

```text
Image encoder  : 1e-5
Prompt encoder : 5e-5
Mask decoder   : 1e-4
```

The best checkpoint is saved as:

```text
medsam_full_v2_best.pth
```

---

### `medsam_head_ablation_compare.py`

Classification-head ablation pipeline.

It starts from the fine-tuned MedSAM checkpoint and compares:

```text
MedSAM embedding
       │
       ▼
   Classification Head
       │
       ├── MLP
       │
       └── Quantum VQC
```

The quantum head can be configured with different numbers of qubits, such as:

```text
Q4
Q6
Q8
```

The current default configuration is:

```text
N_QUBITS = 6
N_LAYERS = 3
```

The script forces the image gate to **OFF** for this ablation. Therefore, the experiment focuses specifically on the classification head rather than on the image quantum gate.

---

# 3. Environment Setup

The experiments were developed and executed in the following environment:

```text
/home/esoussi/stage/medsam_env
```

Activate the environment:

```bash
source /home/esoussi/stage/medsam_env/bin/activate
```

Then move to the MedSAM directory:

```bash
cd /home/esoussi/stage/MedSAM/
```

Install the required Python packages:

```bash
pip install -r /path/to/medsam-brisc-quantum/requirements.txt
```

The repository also assumes that the MedSAM source code is available at:

```text
/home/esoussi/stage/MedSAM/
```

---

# 4. Dataset

The experiments use the **BRISC brain MRI dataset**.

The expected dataset structure is:

```text
brisc2025/
│
├── segmentation_task/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   │
│   └── test/
│       ├── images/
│       └── masks/
│
└── classification_task/
    ├── train/
    └── test/
```

The current scripts expect the dataset at:

```text
/home/esoussi/stage/Eya PFE/archive/brisc2025/
```

The segmentation paths used by the scripts are:

```text
/home/esoussi/stage/Eya PFE/archive/brisc2025/segmentation_task/train/images
/home/esoussi/stage/Eya PFE/archive/brisc2025/segmentation_task/train/masks
/home/esoussi/stage/Eya PFE/archive/brisc2025/segmentation_task/test/images
/home/esoussi/stage/Eya PFE/archive/brisc2025/segmentation_task/test/masks
```

---

# 5. Stage 1 — MedSAM Full Fine-Tuning

## 5.1 Purpose

The first stage fine-tunes the pretrained MedSAM model on BRISC for brain tumor segmentation.

Unlike a single global bounding box, the implementation uses **one bounding box per connected tumor component**.

This multi-BBox strategy is used during:

* training;
* validation;
* testing.

The image encoder is computed once per image, while the mask decoder is evaluated for each connected component.

---

## 5.2 Check the pretrained checkpoint

The script expects the pretrained MedSAM checkpoint at:

```text
/home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_vit_b.pth
```

Check that it exists:

```bash
ls -lh /home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_vit_b.pth
```

---

## 5.3 Start the fine-tuning

Activate the environment:

```bash
source /home/esoussi/stage/medsam_env/bin/activate
```

Move to the MedSAM directory:

```bash
cd /home/esoussi/stage/MedSAM/
```

Run:

```bash
nohup python medsam_full_fine_tuned2.py > medsam_full_fine_tuned2.log 2>&1 &
```

Get the process ID:

```bash
echo "PID: $!"
```

Monitor the training:

```bash
tail -f medsam_full_fine_tuned2.log
```

The script prints information such as:

```text
Device
Batch size
Effective batch size
Number of epochs
Learning rates
Training / validation progress
```

---

## 5.4 Fine-tuned checkpoint

At the end of training, the best checkpoint is saved as:

```text
/home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_full_v2_best.pth
```

The last checkpoint is saved as:

```text
/home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_full_v2_last.pth
```

Check the best checkpoint:

```bash
ls -lh /home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_full_v2_best.pth
```

This checkpoint is required for Stage 2.

---

# 6. Stage 2 — Classification Head Ablation

The second stage starts from:

```text
medsam_full_v2_best.pth
```

The goal is to compare a classical MLP classification head with a quantum VQC classification head while keeping the segmentation backbone configuration fixed.

The eight main reported metrics are:

### Segmentation

* Dice
* IoU
* Precision
* Recall
* Specificity

### Classification

* Accuracy
* Macro-F1
* Balanced Accuracy

---

# 7. Run the MLP Baseline

Activate the environment:

```bash
source /home/esoussi/stage/medsam_env/bin/activate
```

Move to:

```bash
cd /home/esoussi/stage/MedSAM/
```

Run the MLP experiment:

```bash
CUDA_VISIBLE_DEVICES=0 HEAD_MODE=mlp N_QUBITS=6 \
nohup python medsam_head_ablation_compare.py > head_mlp.log 2>&1 &
```

Get the process ID:

```bash
echo "PID MLP: $!"
```

Monitor the experiment:

```bash
tail -f head_mlp.log
```

The resulting metrics are stored in:

```text
head_ablation_compare/metrics_mlp.json
```

---

# 8. Run the Quantum Classification Head

The quantum experiment uses a variational quantum circuit.

For the 6-qubit configuration:

```bash
CUDA_VISIBLE_DEVICES=0 HEAD_MODE=quantum N_QUBITS=6 \
nohup python medsam_head_ablation_compare.py > head_quantum.log 2>&1 &
```

Get the process ID:

```bash
echo "PID QUANTUM: $!"
```

Monitor the experiment:

```bash
tail -f head_quantum.log
```

The resulting metrics are stored in:

```text
head_ablation_compare/metrics_quantum.json
```

---

# 9. Quantum Configuration

The number of qubits can be changed using the `N_QUBITS` environment variable.

For example:

### 4 qubits

```bash
HEAD_MODE=quantum N_QUBITS=4 python medsam_head_ablation_compare.py
```

### 6 qubits

```bash
HEAD_MODE=quantum N_QUBITS=6 python medsam_head_ablation_compare.py
```

### 8 qubits

```bash
HEAD_MODE=quantum N_QUBITS=8 python medsam_head_ablation_compare.py
```

The number of circuit layers can also be changed:

```bash
HEAD_MODE=quantum N_QUBITS=6 N_LAYERS=3 \
python medsam_head_ablation_compare.py
```

The default configuration is:

```text
N_QUBITS = 6
N_LAYERS = 3
```

---

# 10. Quantum Backend

The script automatically tests the available PennyLane backend.

The default behavior is:

```text
lightning.qubit
       │
       │ if unavailable
       ▼
default.qubit
```

For the amplitude-embedding configuration, the script forces:

```text
default.qubit + backprop
```

The quantum backend can also be specified manually through:

```bash
QUANTUM_BACKEND=default.qubit
```

or:

```bash
QUANTUM_BACKEND=lightning.qubit
```

---

# 11. Generate the Comparison Table

After both the MLP and quantum experiments are completed, the comparison can be generated without running another training process.

Run:

```bash
COMPARE_ONLY=1 python medsam_head_ablation_compare.py
```

The script generates:

```text
head_ablation_compare/
│
├── metrics_mlp.json
├── metrics_quantum.json
├── comparison_table.md
├── comparison_table.csv
└── comparison_table.png
```

### `comparison_table.md`

Human-readable Markdown comparison.

### `comparison_table.csv`

Structured results that can be imported into Excel, Python, MATLAB, etc.

### `comparison_table.png`

Visual comparison of the metrics.

---

# 12. Monitoring Long Experiments

Because MedSAM and the quantum experiments can require significant computation time, `nohup` is used to allow the process to continue after disconnecting from the terminal.

Check running processes:

```bash
ps aux | grep python
```

Monitor MLP:

```bash
tail -f head_mlp.log
```

Monitor quantum:

```bash
tail -f head_quantum.log
```

Monitor MedSAM fine-tuning:

```bash
tail -f medsam_full_fine_tuned2.log
```

---

# 13. Experiment Order

For a complete reproduction of the pipeline, follow this order:

```text
1. Activate medsam_env
          │
          ▼
2. Prepare BRISC dataset
          │
          ▼
3. Run medsam_full_fine_tuned2.py
          │
          ▼
4. Obtain medsam_full_v2_best.pth
          │
          ▼
5. Run MLP head
          │
          ▼
6. Run Quantum head
          │
          ▼
7. Generate comparison table
```

In commands:

```bash
source /home/esoussi/stage/medsam_env/bin/activate
cd /home/esoussi/stage/MedSAM/
```

Then:

```bash
nohup python medsam_full_fine_tuned2.py > medsam_full_fine_tuned2.log 2>&1 &
```

After the fine-tuning is complete:

```bash
CUDA_VISIBLE_DEVICES=0 HEAD_MODE=mlp N_QUBITS=6 \
nohup python medsam_head_ablation_compare.py > head_mlp.log 2>&1 &
```

Then:

```bash
CUDA_VISIBLE_DEVICES=0 HEAD_MODE=quantum N_QUBITS=6 \
nohup python medsam_head_ablation_compare.py > head_quantum.log 2>&1 &
```

Finally:

```bash
COMPARE_ONLY=1 python medsam_head_ablation_compare.py
```

---

# 14. Results

The classification-head ablation produces a standardized comparison based on:

| Category       | Metrics                                   |
| -------------- | ----------------------------------------- |
| Segmentation   | Dice, IoU, Precision, Recall, Specificity |
| Classification | Accuracy, Macro-F1, Balanced Accuracy     |

The individual results are stored as JSON files, while the final comparison is available in Markdown, CSV and PNG formats.

---

# 15. Important Configuration Paths

Before running the experiments on another machine or server, verify the following paths in the Python scripts:

```text
MedSAM root
/home/esoussi/stage/MedSAM/

BRISC dataset
/home/esoussi/stage/Eya PFE/archive/brisc2025/

Pretrained MedSAM checkpoint
/home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_vit_b.pth

Fine-tuned MedSAM checkpoint
/home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_full_v2_best.pth
```

These paths are currently configured for the original experimental environment and may need to be modified for another machine.

Also verify the normalization setting before reproducing the classification experiments:

```text
NORM_MODE = "medsam"
```

The normalization must match the one used during MedSAM fine-tuning.

---

# 16. Report

The complete PFE internship report describing the methodology, experiments and results is available here:

[**PFE Internship Report – Eya Soussi**](report/PFE_Mitacs_Report_Eya_Soussi.pdf)

---

# 17. Main Files

| File                                      | Description                                          |
| ----------------------------------------- | ---------------------------------------------------- |
| `medsam_full_fine_tuned2.py`              | Full MedSAM fine-tuning with multi-BBox segmentation |
| `medsam_head_ablation_compare.py`         | MLP vs. quantum classification-head ablation         |
| `requirements.txt`                        | Python dependencies                                  |
| `report/PFE_Mitacs_Report_Eya_Soussi.pdf` | Complete PFE internship report                       |
| `README.md`                               | Project documentation                                |

---

## Summary

This repository implements a two-stage MedSAM-based pipeline:

```text
BRISC
  │
  ▼
MedSAM Full Fine-Tuning
  │
  │  Multi-BBox segmentation
  ▼
Fine-Tuned MedSAM
  │
  ▼
Classification Head Ablation
  │
  ├──────────────┐
  ▼              ▼
MLP             VQC
                 │
          Q4 / Q6 / Q8
                 │
                 ▼
        Comparative Evaluation
```

The main purpose of the ablation study is to evaluate different classification-head configurations while using the fine-tuned MedSAM segmentation model as the common foundation.
