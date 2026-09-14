
r"""
================================================================================
MedSAM (FULL FINE-TUNED) — HEAD ABLATION : MLP vs VQC (6 qubits)
================================================================================
La gate image est FORCEE a OFF (IdentityGate). Le seul facteur etudie est la
tete de classification : MLP classique vs tete quantique (VQC).

NOUVEAU DANS CETTE VERSION
--------------------------------------------------------------------------------
A la fin de chaque run, le script ecrit un fichier de metriques normalise dans
COMPARE_DIR :

    COMPARE_DIR/metrics_mlp.json
    COMPARE_DIR/metrics_quantum.json

contenant EXACTEMENT les 8 metriques demandees :
    Segmentation : Dice, IoU, Precision, Recall, Specificity
    Classification : Accuracy, Macro-F1, Balanced Accuracy

Puis il tente de construire automatiquement le tableau comparatif :
    COMPARE_DIR/comparison_table.md    (Markdown, colonnes MLP | N-Qubit Quantum)
    COMPARE_DIR/comparison_table.csv
    COMPARE_DIR/comparison_table.png   (barres groupees, theme sombre)

Si un seul des deux runs est termine, le tableau n'est pas ecrit et un message
l'indique. Lance l'autre mode, puis relance en mode COMPARE_ONLY (voir plus bas)
si tu veux juste regenerer le tableau.

--------------------------------------------------------------------------------
!!! A VERIFIER AVANT DE LANCER !!!
--------------------------------------------------------------------------------
(0) SECTION 1 (CONFIG) : ces constantes avaient disparu de la version que tu m'as
    envoyee (SEED, IMG_SIZE, N_QUBITS, N_LAYERS, CLASSES, LR_*, LAMBDA_CLS,
    GRAD_ACCUM, MARGIN_BBOX, MIN_AREA, DEVICE...). Je les ai restaurees avec des
    valeurs coherentes avec ta pipeline. VERIFIE-LES contre ton script d'origine
    avant de lancer, sinon les runs ne seront pas comparables a tes resultats
    precedents.
(1) MEDSAM_FT_CKPT  -> chemin de TON checkpoint MedSAM full fine-tuned.
    find /home/esoussi/stage -iname "*.pth" -exec ls -lh {} \; | sort -k5 -h
(2) NORM_MODE       -> DOIT correspondre a la normalisation de ton fine-tuning.
        "medsam"   = min-max par image vers [0,1]
        "imagenet" = normalisation ImageNet
(3) N_QUBITS=6 par defaut (le tableau demande "6-Qubit Quantum").

--------------------------------------------------------------------------------
LANCEMENT (les deux bras, sequentiellement, meme GPU)
--------------------------------------------------------------------------------
source /home/esoussi/stage/medsam_env/bin/activate
cd /home/esoussi/stage/MedSAM/

# Bras 1 : tete MLP
CUDA_VISIBLE_DEVICES=0 HEAD_MODE=mlp N_QUBITS=6 \
  nohup python medsam_head_ablation_compare.py > head_mlp.log 2>&1 &
echo "PID MLP: $!"

# Bras 2 : tete quantique 6 qubits (apres la fin du bras 1, ou sur un autre GPU)
CUDA_VISIBLE_DEVICES=1 HEAD_MODE=quantum N_QUBITS=6 \
  nohup python medsam_head_ablation_compare.py > head_quantum.log 2>&1 &
echo "PID QUANTUM: $!"

# Regenerer uniquement le tableau (aucun entrainement, aucun GPU requis)
COMPARE_ONLY=1 python medsam_head_ablation_compare.py

tail -f head_quantum.log
================================================================================
"""

import os
import cv2
import csv
import json
import math
import time
import random
import datetime
import logging
import socket
import platform
import traceback
import signal
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from PIL import Image
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Construction directe de l'architecture SAM ViT-B (contourne build_sam.py)
from segment_anything.modeling import (
    ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer,
)

import pennylane as qml

# ============================================================================
# 0. LOGGING / DIAGNOSTICS CONFIG
# ============================================================================
LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_EVERY_N_STEPS = int(os.environ.get("LOG_EVERY_N_STEPS", "50"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
QUANTUM_DIAG_ENABLED = int(os.environ.get("QUANTUM_DIAG_ENABLED", "1"))
GRAD_DIAG_ENABLED = int(os.environ.get("GRAD_DIAG_ENABLED", "1"))
BASELINE_DICE = float(os.environ.get("BASELINE_DICE", "0.0"))

_LOGGER_NAMES = ["setup", "backend", "data", "model", "quantum", "train", "eval", "ckpt", "compare"]
_LOGGERS = {name: logging.getLogger(name) for name in _LOGGER_NAMES}
_WARN_ERR_MESSAGES = Counter()
_WARN_ERR_COUNTS = Counter()
_RUN_LOG_STATE = {"log_file": None, "jsonl_file": None, "digest_file": None, "timestamp": None}


class JsonlHandler(logging.Handler):
    def __init__(self, jsonl_path):
        super().__init__()
        self.fp = open(jsonl_path, "a", encoding="utf-8")

    def emit(self, record):
        payload = {
            "ts": datetime.datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        details = getattr(record, "details", None)
        if event is not None:
            payload["event"] = event
        if details is not None:
            payload["details"] = details
        try:
            self.fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self.fp.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.fp.close()
        except Exception:
            pass
        super().close()


def _count_warn_error(level_name, msg):
    if level_name in ("WARNING", "ERROR", "CRITICAL"):
        _WARN_ERR_COUNTS[level_name] += 1
        _WARN_ERR_MESSAGES[msg] += 1


def log_event(logger_name, level, message, event=None, **details):
    lg = _LOGGERS[logger_name]
    extra = {}
    if event is not None:
        extra["event"] = event
    if details:
        extra["details"] = details
    _count_warn_error(logging.getLevelName(level), message)
    lg.log(level, message, extra=extra)


def _bootstrap_fallback_logging():
    root = logging.getLogger()
    if root.handlers:
        return
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s"
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    h.setFormatter(logging.Formatter(fmt))
    root.setLevel(logging.DEBUG)
    root.addHandler(h)
    logging.captureWarnings(True)


def configure_logging(run_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{run_name}_{ts}.log")
    jsonl_path = os.path.join(LOG_DIR, f"{run_name}_{ts}.jsonl")
    digest_path = os.path.join(LOG_DIR, f"{run_name}_digest.txt")
    _RUN_LOG_STATE["log_file"] = log_path
    _RUN_LOG_STATE["jsonl_file"] = jsonl_path
    _RUN_LOG_STATE["digest_file"] = digest_path
    _RUN_LOG_STATE["timestamp"] = ts

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    jh = JsonlHandler(jsonl_path)
    jh.setLevel(logging.DEBUG)

    root.addHandler(ch)
    root.addHandler(fh)
    root.addHandler(jh)
    logging.captureWarnings(True)

    def _excepthook(exc_type, exc, tb):
        log_event(
            "setup", logging.ERROR, "Unhandled exception in process", event="UNHANDLED_EXCEPTION",
            traceback="".join(traceback.format_exception(exc_type, exc, tb)),
        )
        for hh in root.handlers:
            try:
                hh.flush()
            except Exception:
                pass

    sys.excepthook = _excepthook

    def _signal_handler(signum, _frame):
        log_event("setup", logging.WARNING, f"Signal received: {signum}", event="SIGNAL", signal=int(signum))
        for hh in root.handlers:
            try:
                hh.flush()
            except Exception:
                pass
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    log_event("setup", logging.INFO, f"Logging configured | log={log_path} | jsonl={jsonl_path}", event="LOGGING_READY")


def flush_log_handlers():
    root = logging.getLogger()
    for h in root.handlers:
        try:
            h.flush()
        except Exception:
            pass


_bootstrap_fallback_logging()


# ==============================================================================
# 1. CONFIG
# ==============================================================================
# --- Modes d'ablation -------------------------------------------------------
# GATE_MODE : conserve pour compatibilite, mais FORCE a "off" plus bas.
GATE_MODE = os.environ.get("GATE_MODE", "quantum").strip().lower()
if GATE_MODE not in {"quantum", "classical", "off"}:
    raise ValueError(f"GATE_MODE invalide: {GATE_MODE}. Utiliser quantum|classical|off")

HEAD_MODE = os.environ.get("HEAD_MODE", "mlp").strip().lower()  # mlp | quantum
if HEAD_MODE not in {"mlp", "quantum", "quantum_parallel"}:
    raise ValueError(f"HEAD_MODE invalide: {HEAD_MODE}. Utiliser mlp|quantum|quantum_parallel")

# Encodage classique -> quantique. Ablation demandee :
#   "angle"     : AngleEmbedding + data re-uploading a chaque couche (defaut)
#   "amplitude" : AmplitudeEmbedding, encodage unique de 2**N_QUBITS valeurs
EMBED_MODE = os.environ.get("EMBED_MODE", "angle").strip().lower()
if EMBED_MODE not in {"angle", "amplitude"}:
    raise ValueError(f"EMBED_MODE invalide: {EMBED_MODE}. Utiliser angle|amplitude")

DATA_FRACTION = float(os.environ.get("DATA_FRACTION", "1.0"))
if not (0.0 < DATA_FRACTION <= 1.0):
    raise ValueError(f"DATA_FRACTION invalide: {DATA_FRACTION}. Utiliser 0 < DATA_FRACTION <= 1")

SELECT_METRIC = os.environ.get("SELECT_METRIC", "val_acc").strip().lower()  # val_acc|val_dice|combined
if SELECT_METRIC not in {"val_acc", "val_dice", "combined"}:
    raise ValueError(f"SELECT_METRIC invalide: {SELECT_METRIC}. Utiliser val_acc|val_dice|combined")

# Dans ce script, la gate n'est plus l'objet d'etude : on la force a OFF.
REQUESTED_GATE_MODE = GATE_MODE
GATE_MODE = "off"

_RUN_SUFFIX = HEAD_MODE if HEAD_MODE == "mlp" else f"{HEAD_MODE}_{EMBED_MODE}"
RUN_NAME = os.environ.get("RUN_NAME", f"medsam_head_{_RUN_SUFFIX}")
ENFORCE_CUDA_VISIBLE = os.environ.get("ENFORCE_CUDA_VISIBLE", "1") == "1"
COMPARE_ONLY = os.environ.get("COMPARE_ONLY", "0") == "1"

# --- Hyperparametres (RESTAURES : verifie-les contre ton script d'origine) ---
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = int(os.environ.get("SEED", "42"))
IMG_SIZE     = 1024          # ViT-B : positional embeddings figes a 1024, NE PAS CHANGER
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "50"))
PATIENCE     = int(os.environ.get("PATIENCE", "10"))
LR_ENCODER   = float(os.environ.get("LR_ENCODER", "1e-5"))   # encoder gele ici, garde pour tracabilite
LR_DECODER   = float(os.environ.get("LR_DECODER", "5e-6"))
LR_QUANTUM   = float(os.environ.get("LR_QUANTUM", "5e-4"))
LR_CLASSIF   = float(os.environ.get("LR_CLASSIF", "1e-4"))
LAMBDA_CLS   = float(os.environ.get("LAMBDA_CLS", "0.5"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))

# --- Quantique --------------------------------------------------------------
N_QUBITS = int(os.environ.get("N_QUBITS", "6"))   # <-- 6 qubits (colonne "6-Qubit Quantum")
N_LAYERS = int(os.environ.get("N_LAYERS", "3"))

# --- Bbox / classes ---------------------------------------------------------
MARGIN_BBOX = int(os.environ.get("MARGIN_BBOX", "10"))
MIN_AREA    = int(os.environ.get("MIN_AREA", "50"))
CLASSES     = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
N_CLASSES   = len(CLASSES)

# --- NORMALISATION : DOIT correspondre a ton fine-tuning MedSAM -------------
NORM_MODE = "medsam"   # "medsam" (min-max [0,1]) ou "imagenet"
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_ROOT       = "/home/esoussi/stage/Eya PFE/archive/brisc2025"
SEG_TRAIN_IMG   = os.path.join(DATA_ROOT, "segmentation_task", "train", "images")
SEG_TRAIN_MSK   = os.path.join(DATA_ROOT, "segmentation_task", "train", "masks")
SEG_TEST_IMG    = os.path.join(DATA_ROOT, "segmentation_task", "test", "images")
SEG_TEST_MSK    = os.path.join(DATA_ROOT, "segmentation_task", "test", "masks")

CLS_TRAIN_ROOT  = os.path.join(DATA_ROOT, "classification_task", "train")
CLS_TEST_ROOT   = os.path.join(DATA_ROOT, "classification_task", "test")
CLS_FOLDER_TO_IDX = {"glioma": 0, "meningioma": 1, "pituitary": 2, "no_tumor": 3}

WORK_DIR  = "/home/esoussi/stage/MedSAM/work_dir"
os.makedirs(WORK_DIR, exist_ok=True)

# (1) <-- A VERIFIER : ton checkpoint MedSAM full fine-tuned
MEDSAM_FT_CKPT = "/home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_full_v2_best.pth"

BEST_CKPT = os.path.join(WORK_DIR, f"medsam_head_ablation_{RUN_NAME}_best.pth")
LAST_CKPT = os.path.join(WORK_DIR, f"medsam_head_ablation_{RUN_NAME}_last.pth")
RESULTS_DIR = os.path.join("/home/esoussi/stage/MedSAM", f"finetune_results_medsam_head_ablation_{RUN_NAME}")
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_METADATA_JSON = os.path.join(RESULTS_DIR, f"run_summary_{RUN_NAME}.json")

SOTA_DICE = 0.948

# ==============================================================================
# 1B. COMPARAISON MLP vs QUANTUM — chemins et definition du tableau
# ==============================================================================
COMPARE_DIR = os.environ.get("COMPARE_DIR", "/home/esoussi/stage/MedSAM/head_ablation_compare")
os.makedirs(COMPARE_DIR, exist_ok=True)

# Un fichier par bras. Le nom NE contient PAS n_qubits : un seul bras quantique
# a la fois dans le tableau. Change COMPARE_DIR si tu veux comparer 4 vs 6 qubits.
def head_metrics_path(head_mode):
    if head_mode == "quantum_parallel":
        fname = "metrics_quantum_parallel.json"
    else:
        fname = f"metrics_{head_mode}.json"
    return os.path.join(COMPARE_DIR, fname)

COMPARISON_TABLE_MD  = os.path.join(COMPARE_DIR, "comparison_table.md")
COMPARISON_TABLE_CSV = os.path.join(COMPARE_DIR, "comparison_table.csv")
COMPARISON_TABLE_PNG = os.path.join(COMPARE_DIR, "comparison_table.png")

# Les 8 metriques demandees, dans l'ordre exact du tableau.
COMPARISON_METRICS = [
    ("seg_dice",              "Segmentation Dice"),
    ("seg_iou",               "Segmentation IoU"),
    ("seg_precision",         "Segmentation Precision"),
    ("seg_recall",            "Segmentation Recall"),
    ("seg_specificity",       "Segmentation Specificity"),
    ("cls_accuracy",          "Classification Accuracy"),
    ("cls_macro_f1",          "Classification Macro-F1"),
    ("cls_balanced_accuracy", "Classification Balanced Accuracy"),
]

DIAG_STATE = {
    "global_step": 0,
    "optimizer_steps": 0,
    "epoch": 0,
    "epoch_rows": [],
    "quantum_epoch": {},
    "dtype_logged_batches": 0,
    "vqc_first_nonzero_grad_step": None,
    "vqc_prev_weights": None,
    "vqc_weight_change": [],
    "scaler_skipped_steps_epoch": 0,
    "warnings": _WARN_ERR_COUNTS,
    "epoch_grad_diag": {},
}

BBOX_STATS = {
    "images": 0,
    "zero": 0,
    "one": 0,
    "two": 0,
    "three_plus": 0,
    "dropped_components": 0,
    "zero_box_paths": [],
}


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


# ==============================================================================
# 2. QUANTUM BACKEND
# ==============================================================================
# NOTE : lightning.gpu est volontairement EXCLU par defaut (cuStateVec ne peut
# plus allouer son handle quand MedSAM ViT-B @1024 sature la VRAM). Avec 6 qubits
# (64 amplitudes), le CPU reste largement suffisant.
QUANTUM_BACKEND = os.environ.get("QUANTUM_BACKEND", "auto")  # auto | lightning.qubit | lightning.gpu | default.qubit


def _probe_backend(name, diff_method, n_qubits, n_layers):
    """Cree le device ET execute reellement circuit + jacobienne."""
    log_event("backend", logging.INFO, f"[QuantumProbe] Trying backend={name} diff_method={diff_method}",
              event="BACKEND_CANDIDATE", backend=name, diff_method=diff_method,
              n_qubits=int(n_qubits), n_layers=int(n_layers))
    dev = qml.device(name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def _probe(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.Rot(*weights[l, q], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    x = torch.zeros(n_qubits, dtype=torch.float32)
    w = torch.zeros(n_layers, n_qubits, 3, dtype=torch.float32, requires_grad=True)
    out = _probe(x, w)
    out = torch.stack(out) if isinstance(out, (list, tuple)) else out
    out.sum().backward()          # <-- l'erreur cuStateVec surgit ici, pas avant
    log_event("backend", logging.INFO, f"[QuantumProbe] Backend OK: {name} ({diff_method})",
              event="BACKEND_OK", backend=name, diff_method=diff_method)
    return dev


def get_quantum_device(n_qubits, n_layers=N_LAYERS):
    # CRITIQUE : adjoint ne differencie pas a travers une preparation d'etat.
    # En amplitude embedding, le gradient doit remonter jusqu'a la couche `pre`,
    # donc on impose default.qubit + backprop.
    if EMBED_MODE == "amplitude":
        log_event("backend", logging.INFO,
                  "[Quantum] EMBED_MODE=amplitude -> backend force a default.qubit/backprop "
                  "(adjoint ne supporte pas la differentiation d'une state preparation)",
                  event="BACKEND_FORCED_AMPLITUDE")
        return _probe_backend("default.qubit", "backprop", n_qubits, n_layers), "backprop"
    if QUANTUM_BACKEND == "auto":
        candidates = [("lightning.qubit", "adjoint"), ("default.qubit", "backprop")]
    else:
        forced = (QUANTUM_BACKEND,
                  "backprop" if QUANTUM_BACKEND == "default.qubit" else "adjoint")
        candidates = [forced, ("lightning.qubit", "adjoint"), ("default.qubit", "backprop")]

    last_err = None
    for name, diff in candidates:
        try:
            return _probe_backend(name, diff, n_qubits, n_layers), diff
        except Exception as e:
            last_err = e
            log_event(
                "backend", logging.WARNING,
                f"[Quantum] backend '{name}' inutilisable ({type(e).__name__}: {str(e)[:140]}) -> backend suivant",
                event="BACKEND_FAIL", backend=name, diff_method=diff,
                exc_type=type(e).__name__, exc_message=str(e),
            )
    raise RuntimeError(f"Aucun backend quantique utilisable. Derniere erreur : {last_err}")


QDEV, QDIFF_METHOD = get_quantum_device(N_QUBITS)
log_event(
    "backend", logging.INFO,
    f"[Quantum] Backend : {QDEV.name} (diff_method={QDIFF_METHOD}) | N_QUBITS={N_QUBITS} N_LAYERS={N_LAYERS} | HEAD_MODE={HEAD_MODE}",
    event="BACKEND_SELECTED", backend=QDEV.name, diff_method=QDIFF_METHOD,
    n_qubits=int(N_QUBITS), n_layers=int(N_LAYERS), head_mode=HEAD_MODE,
)


def _vqc_circuit_amplitude(inputs, weights):
    """Amplitude embedding : 2**N_QUBITS valeurs encodees dans les amplitudes.
    L'encodage est une PREPARATION D'ETAT : il ne peut apparaitre qu'une fois,
    en tete de circuit. Il n'y a donc PAS de data re-uploading ici, contrairement
    au bras `angle`. normalize=True renormalise le vecteur (contrainte physique :
    la somme des carres des amplitudes vaut 1). pad_with couvre le cas ou la
    dimension d'entree serait inferieure a 2**N_QUBITS."""
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True, pad_with=0.0)
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.Rot(*weights[l, q], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    z_terms = [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]
    zz_terms = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % N_QUBITS)) for q in range(N_QUBITS)]
    return z_terms + zz_terms


def _vqc_circuit(inputs, weights):
    # Data re-uploading (Perez-Salinas et al. 2020) : on re-embarque les MEMES
    # features classiques a chaque couche au lieu d'une seule fois au depart.
    for l in range(N_LAYERS):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        for q in range(N_QUBITS):
            qml.Rot(*weights[l, q], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    # Readout enrichi : Z simples + correlateurs ZZ voisins.
    z_terms = [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]
    zz_terms = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % N_QUBITS)) for q in range(N_QUBITS)]
    return z_terms + zz_terms


_ACTIVE_CIRCUIT = _vqc_circuit_amplitude if EMBED_MODE == "amplitude" else _vqc_circuit
_vqc_qnode = qml.QNode(_ACTIVE_CIRCUIT, QDEV, interface="torch", diff_method=QDIFF_METHOD)
VQC_OUT_DIM = 2 * N_QUBITS  # Z terms + ZZ correlators
VQC_IN_DIM = (2 ** N_QUBITS) if EMBED_MODE == "amplitude" else N_QUBITS
log_event("quantum", logging.INFO,
          f"[Quantum] EMBED_MODE={EMBED_MODE} | entree circuit={VQC_IN_DIM} valeurs | "
          f"sortie={VQC_OUT_DIM} | re-uploading={'non' if EMBED_MODE == 'amplitude' else 'oui'}",
          event="EMBED_MODE_INFO", embed_mode=EMBED_MODE,
          vqc_in_dim=int(VQC_IN_DIM), vqc_out_dim=int(VQC_OUT_DIM))


class VQCLayer(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(_vqc_qnode, {"weights": (n_layers, n_qubits, 3)})

    def forward(self, x):
        return self.qlayer(x)


# ==============================================================================
# 3. GATES IMAGE (conservees ; forcees a IdentityGate dans ce script)
# ==============================================================================
class ImageQuantumGate(nn.Module):
    def __init__(self, embed_dim=256, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, n_qubits),
        )
        self.vqc = VQCLayer(n_qubits, n_layers)
        self.post = nn.Sequential(nn.Linear(2 * n_qubits, embed_dim))
        nn.init.zeros_(self.post[-1].weight)
        nn.init.constant_(self.post[-1].bias, 0.0)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_emb):
        t0 = time.perf_counter()
        B, C, H, W = image_emb.shape
        avg = F.adaptive_avg_pool2d(image_emb, 1).flatten(1)
        mx = F.adaptive_max_pool2d(image_emb, 1).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)
        pre_tanh = self.pre(pooled)
        angles = torch.tanh(pre_tanh) * math.pi
        t_vqc0 = time.perf_counter()
        q_out = self.vqc(angles.float())
        t_vqc1 = time.perf_counter()
        q_out = q_out.to(dtype=image_emb.dtype)
        gate = torch.sigmoid(self.post(q_out)).view(B, C, 1, 1)
        gated = image_emb + self.alpha * gate * image_emb
        gated_avg = F.adaptive_avg_pool2d(gated, 1).flatten(1)

        if QUANTUM_DIAG_ENABLED:
            with torch.no_grad():
                ep = DIAG_STATE.get("epoch", 0)
                qd = DIAG_STATE["quantum_epoch"].setdefault(ep, {
                    "qubit_outputs": [], "gate_vectors": [], "angles": [],
                    "pre_tanh": [], "vqc_time": [], "step_time": [], "dtype_samples": [],
                })
                qd["qubit_outputs"].append(q_out.detach().float().cpu())
                qd["gate_vectors"].append(gate.detach().float().view(B, C).cpu())
                qd["angles"].append(angles.detach().float().cpu())
                qd["pre_tanh"].append(pre_tanh.detach().float().cpu())
                qd["vqc_time"].append(t_vqc1 - t_vqc0)
                qd["step_time"].append(time.perf_counter() - t0)
                if DIAG_STATE["dtype_logged_batches"] < 3:
                    qd["dtype_samples"].append({
                        "angles": str(angles.dtype), "q_out": str(q_out.dtype),
                        "gate": str(gate.dtype), "gated": str(gated.dtype),
                    })
        return gated, gated_avg


class ImageClassicalGate(nn.Module):
    """Baseline classique : meme budget d'entree/sortie que la VQC."""
    def __init__(self, embed_dim=256, n_qubits=N_QUBITS):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, n_qubits),
        )
        self.core = nn.Sequential(
            nn.Linear(n_qubits, 4 * n_qubits), nn.Tanh(),
            nn.Linear(4 * n_qubits, 4 * n_qubits), nn.Tanh(),
            nn.Linear(4 * n_qubits, 2 * n_qubits),
        )
        self.post = nn.Sequential(nn.Linear(2 * n_qubits, embed_dim))
        nn.init.zeros_(self.post[-1].weight)
        nn.init.constant_(self.post[-1].bias, 0.0)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_emb):
        B, C, H, W = image_emb.shape
        avg = F.adaptive_avg_pool2d(image_emb, 1).flatten(1)
        mx = F.adaptive_max_pool2d(image_emb, 1).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)
        angles = torch.tanh(self.pre(pooled)) * math.pi
        c_out = self.core(angles)
        gate = torch.sigmoid(self.post(c_out)).view(B, C, 1, 1)
        gated = image_emb + self.alpha * gate * image_emb
        gated_avg = F.adaptive_avg_pool2d(gated, 1).flatten(1)
        return gated, gated_avg


class IdentityGate(nn.Module):
    """Ablation OFF : aucune modulation, mais meme interface."""
    def forward(self, image_emb):
        avg = F.adaptive_avg_pool2d(image_emb, 1).flatten(1)
        return image_emb, avg


class ParallelVQCLayer(nn.Module):
    """
    16 parallel VQC circuits, each 8 qubits.
    Input: 128 dims (16 chunks × 8 dims)
    Output: 256 dims (16 circuits × 16 output dims)
    """
    def __init__(self, n_circuits=16, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_per_circuit = 2 * n_qubits  # <Z> + <ZZ>
        
        # Create QNodes for each circuit
        self.qnodes = nn.ModuleList([
            nn.Module()  # Placeholder; we'll store QNode outside ModuleList
            for _ in range(n_circuits)
        ])
        
        # Build actual QNodes (not in ModuleList, managed manually)
        self._qnodes = []
        for i in range(n_circuits):
            qnode = build_vqc_circuit_angle(n_qubits, n_layers)
            self._qnodes.append(qnode)
        
        # Learnable weights for each circuit
        # Shape: (n_circuits, n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.randn(n_circuits, n_layers, n_qubits, 3) * 0.1
        )
    
    def forward(self, x):
        """
        Forward pass through all circuits in parallel.
        
        Args:
            x: (batch_size, 128) — 16 chunks of 8 dims each
        
        Returns:
            (batch_size, 256) — concatenated outputs from all circuits
        """
        batch_size = x.shape[0]
        outputs = []
        
        # Process each circuit
        for i in range(self.n_circuits):
            # Extract chunk for this circuit: dims [8*i : 8*(i+1)]
            chunk = x[:, i*self.n_qubits : (i+1)*self.n_qubits]  # (batch, 8)            # Apply circuit to each sample in batch
            circuit_outputs = []
            for b in range(batch_size):
                # default.qubit runs on CPU internally — move data there first.
                # .cpu()/.to() stay differentiable, so gradients still flow back.
                inp_cpu = chunk[b].cpu()
                w_cpu = self.weights[i].cpu()
                out = self._qnodes[i](inp_cpu, w_cpu)
                if isinstance(out, (list, tuple)):
                    out = torch.stack(out)
                out = out.to(x.device)
                circuit_outputs.append(out)
                
            # Stack batch: (batch_size, output_per_circuit)
            circuit_out = torch.stack(circuit_outputs, dim=0)
            outputs.append(circuit_out)
        
        # Concatenate all circuit outputs: (batch_size, 16*16) = (batch, 256)
        all_outputs = torch.cat(outputs, dim=1)
        return all_outputs.float()


# ============================================================================
# PARALLEL VQC CLASSIFICATION HEAD
# ============================================================================

class ParallelVQCClassificationHead(nn.Module):
    """
    Classification head with parallel multi-VQC.
    
    Architecture:
      256 (pooled) → 128 (linear) → 16 parallel VQC (8q each) → 256 → 4 classes
    
    Information flow:
      - 256→128: 50% compression (acceptable)
      - Each VQC encodes 8 dims fully (angle + re-uploading)
      - No per-circuit bottleneck
      - Output 256 dims → full capacity for classification
    
    Parameters:
      - Projection 256→128: 33k params
      - 16 VQCs (8q, 3L each): 16 * 8 * 3 * 3 = 1,152 params (tiny!)
      - Final linear 256→4: 1,028 params
      - Total: ~35k (similar to MLP)
    """
    
    def __init__(self, embed_dim=256, n_circuits=16, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Pre-processing: layer norm + projection 256 → 128
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, n_circuits * n_qubits)
        nn.init.kaiming_uniform_(self.projection.weight, a=math.sqrt(5))
        
        # Parallel VQC circuits
        self.vqc = ParallelVQCLayer(n_circuits, n_qubits, n_layers)
        
        # Post-processing: final classification layer
        vqc_output_dim = n_circuits * (2 * n_qubits)  # 16 * 16 = 256
        self.classifier = nn.Sequential(
            nn.LayerNorm(vqc_output_dim),
            nn.Linear(vqc_output_dim, n_classes)
        )
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, pooled_features):
        """
        Args:
            pooled_features: (batch_size, 256) — pooled image embeddings from MedSAM
        
        Returns:
            logits: (batch_size, 4) — class logits
        """
        # Normalize
        x = self.norm(pooled_features)  # (batch, 256)
        
        # Project to 128 dims (16 chunks of 8)
        x = self.projection(x)  # (batch, 128)
        
        # Apply parallel VQCs
        x = self.vqc(x)  # (batch, 256)
        
        # Classify
        logits = self.classifier(x)  # (batch, 4)
        return logits


# ============================================================================
# PARAMETER COUNTING
# ============================================================================

def count_vqc_params(head):
    """Count parameters in parallel VQC head."""
    total = sum(p.numel() for p in head.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    
    # Breakdown
    proj_params = sum(p.numel() for p in head.projection.parameters())
    vqc_params = sum(p.numel() for p in head.vqc.parameters())
    class_params = sum(p.numel() for p in head.classifier.parameters())
    
    return {
        "total": total,
        "trainable": trainable,
        "projection_256_to_128": proj_params,
        "vqc_circuits_16x8q_3L": vqc_params,
        "classifier": class_params,
    }


# ============================================================================
# DEMO / TESTING
# ============================================================================

def get_quantum_device_for_parallel(n_qubits=8, n_layers=3):
    """Get device for parallel VQC."""
    # For 8 qubits, default.qubit + backprop is safe and fast enough
    dev = qml.device("default.qubit", wires=n_qubits)
    return dev


def build_vqc_circuit_angle(n_qubits=8, n_layers=3):
    """
    VQC circuit with angle embedding and data re-uploading.
    
    Input: n_qubits values (typically 8)
    Output: n_qubits <Z> + n_qubits <ZZ> correlators = 2*n_qubits values
    """
    dev = get_quantum_device_for_parallel(n_qubits, n_layers)
    
    def _circuit(inputs, weights):
        # inputs: (n_qubits,) — data re-uploading at each layer
        # weights: (n_layers, n_qubits, 3) — Rot parameters
        for l in range(n_layers):
            # Re-upload data at each layer
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            # Parameterized rotations
            for q in range(n_qubits):
                qml.Rot(*weights[l, q], wires=q)
            # Ring CNOT entanglement
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        
        # Readout: single-qubit <Z> + two-qubit <ZZ> correlators
        z_terms = [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        zz_terms = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q+1) % n_qubits)) 
                    for q in range(n_qubits)]
        return z_terms + zz_terms
    
    qnode = qml.QNode(_circuit, dev, interface="torch", diff_method="backprop")
    return qnode


# ============================================================================
# PARALLEL VQC MODULE
# ============================================================================

class ParallelVQCLayer(nn.Module):
    """
    16 parallel VQC circuits, each 8 qubits.
    Input: 128 dims (16 chunks × 8 dims)
    Output: 256 dims (16 circuits × 16 output dims)
    """
    def __init__(self, n_circuits=16, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_per_circuit = 2 * n_qubits  # <Z> + <ZZ>
        
        # Create QNodes for each circuit
        self.qnodes = nn.ModuleList([
            nn.Module()  # Placeholder; we'll store QNode outside ModuleList
            for _ in range(n_circuits)
        ])
        
        # Build actual QNodes (not in ModuleList, managed manually)
        self._qnodes = []
        for i in range(n_circuits):
            qnode = build_vqc_circuit_angle(n_qubits, n_layers)
            self._qnodes.append(qnode)
        
        # Learnable weights for each circuit
        # Shape: (n_circuits, n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.randn(n_circuits, n_layers, n_qubits, 3) * 0.1
        )
    
    def forward(self, x):
        """
        Forward pass through all circuits in parallel.
        
        Args:
            x: (batch_size, 128) — 16 chunks of 8 dims each
        
        Returns:
            (batch_size, 256) — concatenated outputs from all circuits
        """
        batch_size = x.shape[0]
        outputs = []
        
        # Process each circuit
        for i in range(self.n_circuits):
            # Extract chunk for this circuit: dims [8*i : 8*(i+1)]
            chunk = x[:, i*self.n_qubits : (i+1)*self.n_qubits]  # (batch, 8)
            
            # Apply circuit to each sample in batch
            circuit_outputs = []
            for b in range(batch_size):
                # default.qubit runs on CPU internally -- move data there first.
                # .cpu()/.to() stay differentiable, so gradients still flow back.
                inp_cpu = chunk[b].cpu()
                w_cpu = self.weights[i].cpu()
                out = self._qnodes[i](inp_cpu, w_cpu)
                if isinstance(out, (list, tuple)):
                    out = torch.stack(out)
                out = out.to(x.device)
                circuit_outputs.append(out)
            
            # Stack batch: (batch_size, output_per_circuit)
            circuit_out = torch.stack(circuit_outputs, dim=0)
            outputs.append(circuit_out)
        
        # Concatenate all circuit outputs: (batch_size, 16*16) = (batch, 256)
        all_outputs = torch.cat(outputs, dim=1)
        return all_outputs.float()


# ============================================================================
# PARALLEL VQC CLASSIFICATION HEAD
# ============================================================================

class ParallelVQCClassificationHead(nn.Module):
    """
    Classification head with parallel multi-VQC.
    
    Architecture:
      256 (pooled) → 128 (linear) → 16 parallel VQC (8q each) → 256 → 4 classes
    
    Information flow:
      - 256→128: 50% compression (acceptable)
      - Each VQC encodes 8 dims fully (angle + re-uploading)
      - No per-circuit bottleneck
      - Output 256 dims → full capacity for classification
    
    Parameters:
      - Projection 256→128: 33k params
      - 16 VQCs (8q, 3L each): 16 * 8 * 3 * 3 = 1,152 params (tiny!)
      - Final linear 256→4: 1,028 params
      - Total: ~35k (similar to MLP)
    """
    
    def __init__(self, embed_dim=256, n_circuits=16, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Pre-processing: layer norm + projection 256 → 128
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, n_circuits * n_qubits)
        nn.init.kaiming_uniform_(self.projection.weight, a=math.sqrt(5))
        
        # Parallel VQC circuits
        self.vqc = ParallelVQCLayer(n_circuits, n_qubits, n_layers)
        
        # Post-processing: final classification layer
        vqc_output_dim = n_circuits * (2 * n_qubits)  # 16 * 16 = 256
        self.classifier = nn.Sequential(
            nn.LayerNorm(vqc_output_dim),
            nn.Linear(vqc_output_dim, n_classes)
        )
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, pooled_features):
        """
        Args:
            pooled_features: (batch_size, 256) — pooled image embeddings from MedSAM
        
        Returns:
            logits: (batch_size, 4) — class logits
        """
        # Normalize
        x = self.norm(pooled_features)  # (batch, 256)
        
        # Project to 128 dims (16 chunks of 8)
        x = self.projection(x)  # (batch, 128)
        
        # Apply parallel VQCs
        x = self.vqc(x)  # (batch, 256)
        
        # Classify
        logits = self.classifier(x)  # (batch, 4)
        return logits


# ============================================================================
# PARAMETER COUNTING
# ============================================================================

def count_vqc_params(head):
    """Count parameters in parallel VQC head."""
    total = sum(p.numel() for p in head.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    
    # Breakdown
    proj_params = sum(p.numel() for p in head.projection.parameters())
    vqc_params = sum(p.numel() for p in head.vqc.parameters())
    class_params = sum(p.numel() for p in head.classifier.parameters())
    
    return {
        "total": total,
        "trainable": trainable,
        "projection_256_to_128": proj_params,
        "vqc_circuits_16x8q_3L": vqc_params,
        "classifier": class_params,
    }


# ============================================================================
# DEMO / TESTING
# ============================================================================

def get_quantum_device_for_parallel(n_qubits=8, n_layers=3):
    """Get device for parallel VQC."""
    # For 8 qubits, default.qubit + backprop is safe and fast enough
    dev = qml.device("default.qubit", wires=n_qubits)
    return dev


def build_vqc_circuit_angle(n_qubits=8, n_layers=3):
    """
    VQC circuit with angle embedding and data re-uploading.
    
    Input: n_qubits values (typically 8)
    Output: n_qubits <Z> + n_qubits <ZZ> correlators = 2*n_qubits values
    """
    dev = get_quantum_device_for_parallel(n_qubits, n_layers)
    
    def _circuit(inputs, weights):
        # inputs: (n_qubits,) — data re-uploading at each layer
        # weights: (n_layers, n_qubits, 3) — Rot parameters
        for l in range(n_layers):
            # Re-upload data at each layer
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            # Parameterized rotations
            for q in range(n_qubits):
                qml.Rot(*weights[l, q], wires=q)
            # Ring CNOT entanglement
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        
        # Readout: single-qubit <Z> + two-qubit <ZZ> correlators
        z_terms = [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        zz_terms = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q+1) % n_qubits)) 
                    for q in range(n_qubits)]
        return z_terms + zz_terms
    
    qnode = qml.QNode(_circuit, dev, interface="torch", diff_method="backprop")
    return qnode


# ============================================================================
# PARALLEL VQC MODULE
# ============================================================================

class ParallelVQCLayer(nn.Module):
    """
    16 parallel VQC circuits, each 8 qubits.
    Input: 128 dims (16 chunks × 8 dims)
    Output: 256 dims (16 circuits × 16 output dims)
    """
    def __init__(self, n_circuits=16, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_per_circuit = 2 * n_qubits  # <Z> + <ZZ>
        
        # Create QNodes for each circuit
        self.qnodes = nn.ModuleList([
            nn.Module()  # Placeholder; we'll store QNode outside ModuleList
            for _ in range(n_circuits)
        ])
        
        # Build actual QNodes (not in ModuleList, managed manually)
        self._qnodes = []
        for i in range(n_circuits):
            qnode = build_vqc_circuit_angle(n_qubits, n_layers)
            self._qnodes.append(qnode)
        
        # Learnable weights for each circuit
        # Shape: (n_circuits, n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.randn(n_circuits, n_layers, n_qubits, 3) * 0.1
        )
    
    def forward(self, x):
        """
        Forward pass through all circuits in parallel.
        
        Args:
            x: (batch_size, 128) — 16 chunks of 8 dims each
        
        Returns:
            (batch_size, 256) — concatenated outputs from all circuits
        """
        batch_size = x.shape[0]
        outputs = []
        
        # Process each circuit
        for i in range(self.n_circuits):
            # Extract chunk for this circuit: dims [8*i : 8*(i+1)]
            chunk = x[:, i*self.n_qubits : (i+1)*self.n_qubits]  # (batch, 8)
            
            # Apply circuit to each sample in batch
            circuit_outputs = []
            for b in range(batch_size):
                # default.qubit runs on CPU internally -- move data there first.
                # .cpu()/.to() stay differentiable, so gradients still flow back.
                inp_cpu = chunk[b].cpu()
                w_cpu = self.weights[i].cpu()
                out = self._qnodes[i](inp_cpu, w_cpu)
                if isinstance(out, (list, tuple)):
                    out = torch.stack(out)
                out = out.to(x.device)
                circuit_outputs.append(out)
            
            # Stack batch: (batch_size, output_per_circuit)
            circuit_out = torch.stack(circuit_outputs, dim=0)
            outputs.append(circuit_out)
        
        # Concatenate all circuit outputs: (batch_size, 16*16) = (batch, 256)
        all_outputs = torch.cat(outputs, dim=1)
        return all_outputs.float()


# ============================================================================
# PARALLEL VQC CLASSIFICATION HEAD
# ============================================================================

class ParallelVQCClassificationHead(nn.Module):
    """
    Classification head with parallel multi-VQC.
    
    Architecture:
      256 (pooled) → 128 (linear) → 16 parallel VQC (8q each) → 256 → 4 classes
    
    Information flow:
      - 256→128: 50% compression (acceptable)
      - Each VQC encodes 8 dims fully (angle + re-uploading)
      - No per-circuit bottleneck
      - Output 256 dims → full capacity for classification
    
    Parameters:
      - Projection 256→128: 33k params
      - 16 VQCs (8q, 3L each): 16 * 8 * 3 * 3 = 1,152 params (tiny!)
      - Final linear 256→4: 1,028 params
      - Total: ~35k (similar to MLP)
    """
    
    def __init__(self, embed_dim=256, n_circuits=16, n_qubits=8, n_layers=3, n_classes=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Pre-processing: layer norm + projection 256 → 128
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, n_circuits * n_qubits)
        nn.init.kaiming_uniform_(self.projection.weight, a=math.sqrt(5))
        
        # Parallel VQC circuits
        self.vqc = ParallelVQCLayer(n_circuits, n_qubits, n_layers)
        
        # Post-processing: final classification layer
        vqc_output_dim = n_circuits * (2 * n_qubits)  # 16 * 16 = 256
        self.classifier = nn.Sequential(
            nn.LayerNorm(vqc_output_dim),
            nn.Linear(vqc_output_dim, n_classes)
        )
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, pooled_features):
        """
        Args:
            pooled_features: (batch_size, 256) — pooled image embeddings from MedSAM
        
        Returns:
            logits: (batch_size, 4) — class logits
        """
        # Normalize
        x = self.norm(pooled_features)  # (batch, 256)
        
        # Project to 128 dims (16 chunks of 8)
        x = self.projection(x)  # (batch, 128)
        
        # Apply parallel VQCs
        x = self.vqc(x)  # (batch, 256)
        
        # Classify
        logits = self.classifier(x)  # (batch, 4)
        return logits


# ============================================================================
# PARAMETER COUNTING
# ============================================================================

def count_vqc_params(head):
    """Count parameters in parallel VQC head."""
    total = sum(p.numel() for p in head.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    
    # Breakdown
    proj_params = sum(p.numel() for p in head.projection.parameters())
    vqc_params = sum(p.numel() for p in head.vqc.parameters())
    class_params = sum(p.numel() for p in head.classifier.parameters())
    
    return {
        "total": total,
        "trainable": trainable,
        "projection_256_to_128": proj_params,
        "vqc_circuits_16x8q_3L": vqc_params,
        "classifier": class_params,
    }


# ============================================================================
# DEMO / TESTING
# ============================================================================

class ClassificationHead(nn.Module):
    """Tete MLP classique (bras 'MLP' du tableau)."""
    def __init__(self, embed_dim=256, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, pooled_features):
        return self.net(pooled_features)


class QuantumClassificationHead(nn.Module):
    """Tete quantique sur le chemin critique de classification (sans bypass).
    Bras 'N-Qubit Quantum' du tableau."""
    def __init__(self, embed_dim=256, n_qubits=N_QUBITS, n_classes=N_CLASSES):
        """angle     : projection 256 -> N_QUBITS puis tanh*pi (data re-uploading).
        amplitude : projection 256 -> 2**N_QUBITS. Pour N_QUBITS=8, 2**8=256 donc
        AUCUNE projection (Identity) : l'embedding poole entre integralement dans
        le circuit. Normalisation L2 assuree par AmplitudeEmbedding."""
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_mode = EMBED_MODE
        if EMBED_MODE == "amplitude":
            target = 2 ** n_qubits
            if target == embed_dim:
                self.pre = nn.Identity()   # 8 qubits : 2**8 = 256 = embed_dim
            else:
                self.pre = nn.Linear(embed_dim, target)
        else:
            self.pre = nn.Linear(embed_dim, n_qubits)
        self.vqc = VQCLayer(n_qubits, N_LAYERS)
        self.out = nn.Linear(2 * n_qubits, n_classes)

    def forward(self, pooled_features):
        x = self.norm(pooled_features)
        x = self.pre(x)
        if self.embed_mode == "amplitude":
            # Garde-fou : un vecteur de norme nulle rendrait la normalisation
            # indefinie (NaN silencieux). On borne la norme par le bas.
            x = x.float()
            nrm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
            circuit_in = x / nrm
        else:
            circuit_in = (torch.tanh(x) * math.pi).float()
        q_out = self.vqc(circuit_in).to(dtype=pooled_features.dtype)
        return self.out(q_out)


def stratified_subset_indices(dataset, indices, fraction, seed):
    """Sous-echantillonne une liste d'indices en preservant la distribution de classes."""
    if fraction >= 1.0:
        return list(indices)

    by_class = {k: [] for k in range(N_CLASSES)}
    for idx in indices:
        sample = dataset[idx]
        by_class[int(sample["class_idx"])].append(int(idx))

    rng = np.random.default_rng(seed)
    selected = []
    for k in range(N_CLASSES):
        cls_idx = by_class[k]
        if not cls_idx:
            continue
        arr = np.array(cls_idx, dtype=np.int64)
        rng.shuffle(arr)
        n_take = max(1, int(round(len(arr) * fraction)))
        n_take = min(n_take, len(arr))
        selected.extend(arr[:n_take].tolist())

    rng.shuffle(selected)
    return selected


# ==============================================================================
# 4. DATASETS — DEUX SOURCES DISTINCTES, FUSIONNEES
# ==============================================================================
def extract_multi_bboxes(mask_np, margin=MARGIN_BBOX, min_area=MIN_AREA):
    BBOX_STATS["images"] += 1
    if mask_np.sum() == 0:
        BBOX_STATS["zero"] += 1
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), connectivity=8)
    bboxes = []
    H, W = mask_np.shape
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            BBOX_STATS["dropped_components"] += 1
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append(np.array([max(0, x - margin), max(0, y - margin),
                                min(W - 1, x + w + margin), min(H - 1, y + h + margin)], dtype=np.float32))
    nb = len(bboxes)
    if nb == 0:
        BBOX_STATS["zero"] += 1
    elif nb == 1:
        BBOX_STATS["one"] += 1
    elif nb == 2:
        BBOX_STATS["two"] += 1
    else:
        BBOX_STATS["three_plus"] += 1
    return bboxes


def class_idx_from_seg_path(path):
    """Pour segmentation_task : uniquement gl/me/pi (jamais no_tumor la-dedans)."""
    name = os.path.basename(path).lower()
    if "_gl_" in name or "glioma" in name: return 0
    if "_me_" in name or "meningioma" in name: return 1
    if "_pi_" in name or "pituitary" in name: return 2
    raise ValueError(f"Nom de fichier inattendu dans segmentation_task : {path}")


class SegClsDataset(Dataset):
    """Images AVEC masque (segmentation_task) — perte seg + perte cls (3 classes)."""
    def __init__(self, img_dir, mask_dir, img_size=IMG_SIZE, augment=False):
        self.img_paths = sorted(Path(img_dir).glob("*.jpg"))
        self.mask_dir = mask_dir; self.img_size = img_size; self.augment = augment

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = Path(self.mask_dir) / (img_path.stem + ".png")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        if self.augment:
            # NB: pas de flip vertical (anatomiquement invalide pour l'IRM cerebrale)
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1, :]); mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1.0)
                image = cv2.warpAffine(image, M, (self.img_size, self.img_size),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            if random.random() < 0.4:
                factor = random.uniform(0.85, 1.15)
                image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        bboxes = extract_multi_bboxes(mask)
        if not bboxes:
            H, W = mask.shape
            cx, cy = W // 2, H // 2
            bboxes = [np.array([cx - 20, cy - 20, cx + 20, cy + 20], dtype=np.float32)]
            BBOX_STATS["zero_box_paths"].append(str(img_path))
            log_event("data", logging.WARNING, f"Image yielding 0 bbox after filtering: {img_path}",
                      event="ZERO_BBOX_IMAGE", path=str(img_path))
        class_idx = class_idx_from_seg_path(str(img_path))
        return {"image": image, "mask": mask.astype(np.float32), "bboxes": bboxes,
                "class_idx": class_idx, "has_mask": True, "path": str(img_path)}


class ClsOnlyDataset(Dataset):
    """Images SANS masque (classification_task/<classe>) — perte cls UNIQUEMENT."""
    def __init__(self, class_dir, class_idx, img_size=IMG_SIZE, augment=False):
        self.img_paths = sorted(Path(class_dir).glob("*.jpg"))
        self.class_idx = class_idx; self.img_size = img_size; self.augment = augment

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1, :])
            if random.random() < 0.4:
                factor = random.uniform(0.85, 1.15)
                image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        dummy_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        return {"image": image, "mask": dummy_mask, "bboxes": [],
                "class_idx": self.class_idx, "has_mask": False, "path": str(img_path)}


def collate_single(batch): return batch[0]


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1); targets = targets.view(-1)
        inter = (probs * targets).sum()
        return 1 - (2 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class SegLoss(nn.Module):
    def __init__(self): super().__init__(); self.dice = DiceLoss(); self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets): return 0.5 * self.dice(logits, targets) + 0.5 * self.bce(logits, targets)


def compute_seg_metrics(logits, targets, eps=1e-6):
    """Dice, IoU, Precision, Recall, Specificity — les 5 metriques de segmentation
    du tableau comparatif."""
    probs = torch.sigmoid(logits); preds = (probs > 0.5).float(); targets = targets.float()
    tp = (preds * targets).sum(); fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum(); tn = ((1 - preds) * (1 - targets)).sum()
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps); iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps); recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {"dice": dice.item(), "iou": iou.item(), "precision": precision.item(),
            "recall": recall.item(), "specificity": specificity.item()}


# ==============================================================================
# 5. CONSTRUCTION + CHARGEMENT MedSAM
# ==============================================================================
def build_sam_vit_b_no_ckpt(device):
    """Construit l'architecture SAM ViT-B DIRECTEMENT (sans build_sam.py)."""
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=12, embed_dim=768, img_size=image_size, mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12,
            patch_size=vit_patch_size, qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11], window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8,
            ),
            transformer_dim=prompt_embed_dim, iou_head_depth=3, iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    return sam.to(device)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_param_report(medsam_model, gate_module, classifier):
    parts = {
        "image_encoder": count_params(medsam_model.image_encoder),
        "prompt_encoder": count_params(medsam_model.prompt_encoder),
        "mask_decoder": count_params(medsam_model.mask_decoder),
        "gate": count_params(gate_module),
        "classifier": count_params(classifier),
    }
    all_total = sum(v[0] for v in parts.values())
    all_trainable = sum(v[1] for v in parts.values())
    log_event("model", logging.INFO, "\n[Params] Breakdown (total | trainable)", event="PARAM_BREAKDOWN_BEGIN")
    for k, (tot, trn) in parts.items():
        pct = 100.0 * tot / max(all_total, 1)
        log_event("model", logging.INFO, f"  - {k:14s}: {tot:,} | {trn:,} | frozen={tot-trn:,} | {pct:.4f}%",
                  event="PARAM_PART", part=k, total=int(tot), trainable=int(trn), frozen=int(tot - trn), pct=float(pct))
    log_event("model", logging.INFO, f"  - {'ALL':14s}: {all_total:,} | {all_trainable:,} | frozen={all_total-all_trainable:,}",
              event="PARAM_TOTAL", total=int(all_total), trainable=int(all_trainable), frozen=int(all_total - all_trainable))

    vqc_count = N_LAYERS * N_QUBITS * 3
    log_event("quantum", logging.INFO,
              f"[Params] isolated VQC circuit params (N_LAYERS*N_QUBITS*3) = {vqc_count}",
              event="VQC_PARAM_COUNT", vqc_param_count=int(vqc_count))
    if hasattr(gate_module, "core"):
        core_total, core_train = count_params(gate_module.core)
        log_event("quantum", logging.INFO,
                  f"[Params] ImageClassicalGate.core params = {core_total:,} (trainable={core_train:,})",
                  event="CLASSICAL_CORE_PARAM_COUNT", core_total=int(core_total), core_trainable=int(core_train))

    cls_t, cls_tr = count_params(classifier)
    log_event("model", logging.INFO, f"  - classification_head: {cls_t:,} | {cls_tr:,}",
              event="CLS_HEAD_PART", total=int(cls_t), trainable=int(cls_tr))
    log_event("model", logging.INFO,
              f"[HeadParams] mode={HEAD_MODE} total={cls_t:,} trainable={cls_tr:,}",
              event="HEAD_PARAM_COUNT", head_mode=HEAD_MODE, total=int(cls_t), trainable=int(cls_tr))
    return {
        "parts": {k: {"total": int(v[0]), "trainable": int(v[1])} for k, v in parts.items()},
        "all": {"total": int(all_total), "trainable": int(all_trainable)},
        "head": {"total": int(cls_t), "trainable": int(cls_tr)},
        "vqc_circuit_params": int(vqc_count) if HEAD_MODE == "quantum" else 0,
    }


def load_model(device=DEVICE):
    medsam_model = build_sam_vit_b_no_ckpt(device)

    # Gate forcee en OFF : on etudie uniquement l'effet de la tete de classification.
    quantum_gate = IdentityGate().to(device)
    if HEAD_MODE == "quantum":
        classifier = QuantumClassificationHead(256, N_QUBITS, N_CLASSES).to(device)
    elif HEAD_MODE == "quantum_parallel":
        classifier = ParallelVQCClassificationHead(embed_dim=256, n_circuits=16, n_qubits=8, n_layers=3, n_classes=N_CLASSES).to(device)
    else:
        classifier = ClassificationHead(256, N_CLASSES).to(device)

    if not os.path.exists(MEDSAM_FT_CKPT):
        raise FileNotFoundError(
            f"\n[FATAL] MEDSAM_FT_CKPT introuvable : {MEDSAM_FT_CKPT}\n"
            f"        Sans checkpoint MedSAM fine-tuned, le modele serait ALEATOIRE.\n"
            f"        find /home/esoussi/stage -iname '*.pth' -exec ls -lh {{}} \\; | sort -k5 -h\n"
        )

    ckpt_size = os.path.getsize(MEDSAM_FT_CKPT)
    log_event("ckpt", logging.INFO, f"[Init] Loading checkpoint {MEDSAM_FT_CKPT} ({ckpt_size/1024/1024:.2f} MB)",
              event="CKPT_LOAD_BEGIN", path=MEDSAM_FT_CKPT, size_bytes=int(ckpt_size))

    ckpt = torch.load(MEDSAM_FT_CKPT, map_location=device, weights_only=False)
    state = _extract_state_dict(ckpt)
    if isinstance(ckpt, dict):
        log_event("ckpt", logging.INFO, f"[Init] Checkpoint keys: {sorted(list(ckpt.keys()))}",
                  event="CKPT_KEYS", keys=sorted(list(ckpt.keys())))
    missing, unexpected = medsam_model.load_state_dict(state, strict=False)
    log_event("ckpt", logging.INFO, f"[Init] MedSAM full fine-tuned charge depuis {MEDSAM_FT_CKPT}", event="CKPT_LOADED")
    lvl = logging.WARNING if (len(missing) > 0 or len(unexpected) > 0) else logging.INFO
    log_event("ckpt", lvl, f"       missing={len(missing)}  unexpected={len(unexpected)}",
              event="CKPT_KEY_MISMATCH", missing_count=len(missing), unexpected_count=len(unexpected),
              missing_names=missing[:200], unexpected_names=unexpected[:200])
    if len(missing) > 50:
        log_event("ckpt", logging.WARNING,
                  f"       [ATTENTION] beaucoup de poids manquants ({len(missing)}) — verifie que c'est bien un checkpoint MedSAM ViT-B complet.",
                  event="CKPT_HIGH_MISSING", examples=missing[:5])
    if isinstance(ckpt, dict):
        log_event("ckpt", logging.INFO,
                  f"       (checkpoint epoch={ckpt.get('epoch', 'NA')}, best_dice={ckpt.get('best_dice', ckpt.get('best', 'NA'))})",
                  event="CKPT_META", epoch=ckpt.get("epoch", "NA"), best_dice=ckpt.get("best_dice", ckpt.get("best", "NA")))
        if "quantum_state" in ckpt:
            try:
                quantum_gate.load_state_dict(ckpt["quantum_state"], strict=False)
                log_event("ckpt", logging.INFO, "       -> quantum_state trouve : etat gate charge (strict=False).",
                          event="QUANTUM_STATE_LOADED")
            except Exception as e:
                log_event("ckpt", logging.WARNING,
                          f"       -> quantum_state ignore (gate forcee OFF): {type(e).__name__}",
                          event="QUANTUM_STATE_INCOMPAT", exc_type=type(e).__name__, exc_message=str(e))
        # IMPORTANT : la tete de classification N'EST PAS rechargee ici. Les deux
        # bras (MLP / quantum) doivent partir du MEME etat initial (aleatoire) au
        # dessus du meme backbone, sinon la comparaison est biaisee.
        if "classifier_state" in ckpt:
            log_event("ckpt", logging.WARNING,
                      "       -> classifier_state present mais IGNORE volontairement : les deux bras du tableau "
                      "doivent partir d'une tete non entrainee pour etre comparables.",
                      event="CLASSIFIER_STATE_SKIPPED")
        else:
            log_event("ckpt", logging.INFO, "       -> classifier_state absent : tete initialisee (attendu).",
                      event="CLASSIFIER_STATE_MISSING")

    medsam_model.image_encoder.train(True)
    medsam_model.prompt_encoder.train(True)
    medsam_model.mask_decoder.train(True)
    for p in medsam_model.image_encoder.parameters(): p.requires_grad = False
    for p in medsam_model.prompt_encoder.parameters(): p.requires_grad = True
    for p in medsam_model.mask_decoder.parameters(): p.requires_grad = True

    n_cls = sum(p.numel() for p in classifier.parameters())
    n_cls_train = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log_event("model", logging.INFO,
              f"[Classifier] HEAD_MODE={HEAD_MODE} Params: {n_cls:,} (trainable={n_cls_train:,}) — {N_CLASSES} classes: {CLASSES}",
              event="CLASSIFIER_PARAMS", params=int(n_cls), trainable=int(n_cls_train),
              classes=CLASSES, head_mode=HEAD_MODE)
    log_event("model", logging.INFO, f"[Preproc] NORM_MODE = '{NORM_MODE}'", event="NORM_MODE", norm_mode=NORM_MODE)
    log_event("model", logging.INFO,
              f"[Ablation] Requested GATE_MODE='{REQUESTED_GATE_MODE}' -> FORCED GATE_MODE='off' (IdentityGate)",
              event="GATE_MODE_FORCED_OFF", requested_gate_mode=REQUESTED_GATE_MODE, gate_mode="off")
    enc_total, enc_train = count_params(medsam_model.image_encoder)
    tr_total = (sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
                + sum(p.numel() for p in classifier.parameters() if p.requires_grad))
    log_event("model", logging.INFO,
              f"[Freeze] image_encoder frozen: trainable {enc_train:,}/{enc_total:,} | total trainable (model+head)={tr_total:,}",
              event="ENCODER_FROZEN", image_encoder_total=int(enc_total),
              image_encoder_trainable=int(enc_train), total_trainable=int(tr_total))
    return medsam_model, quantum_gate, classifier


def medsam_preprocess(image_np, device):
    img = image_np.astype(np.float32)
    if NORM_MODE == "medsam":
        img = (img - img.min()) / np.clip(img.max() - img.min(), 1e-8, None)
    elif NORM_MODE == "imagenet":
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    else:
        raise ValueError(f"NORM_MODE inconnu : {NORM_MODE}")
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return img_t


def medsam_forward_joint(medsam_model, quantum_gate, image_np, bboxes_list, has_mask, device=DEVICE):
    img_t = medsam_preprocess(image_np, device)
    image_emb = medsam_model.image_encoder(img_t)                 # [1, 256, 64, 64]
    gated_img_emb, pooled = quantum_gate(image_emb)

    if not has_mask:
        return None, pooled

    dense_pe = medsam_model.prompt_encoder.get_dense_pe()
    fused_prob = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    for bbox_np in bboxes_list:
        box_torch = torch.as_tensor(bbox_np, dtype=torch.float32, device=device).reshape(1, 1, 4)
        sparse_emb, dense_emb = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)
        low_res_masks, _ = medsam_model.mask_decoder(
            image_embeddings=gated_img_emb,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        logits_full = F.interpolate(low_res_masks, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        fused_prob = torch.maximum(fused_prob, torch.sigmoid(logits_full))
    fused_prob = torch.clamp(fused_prob, 1e-6, 1 - 1e-6)
    seg_logits = torch.log(fused_prob / (1 - fused_prob)).squeeze(0)
    return seg_logits, pooled


# ==============================================================================
# 6. TRAIN / VALIDATE
# ==============================================================================
def _grad_l2_norm(params):
    s = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        s += float(torch.sum(g * g).item())
    return math.sqrt(max(s, 0.0))


def _flatten_grads(params):
    vecs = []
    for p in params:
        if p.grad is None:
            continue
        vecs.append(p.grad.detach().float().reshape(-1))
    if not vecs:
        return None
    return torch.cat(vecs, dim=0)


def _flatten_weights(params):
    vecs = []
    for p in params:
        vecs.append(p.detach().float().reshape(-1).cpu())
    if not vecs:
        return None
    return torch.cat(vecs, dim=0)


def _vqc_params(module):
    """Parametres du circuit variationnel, quel que soit le module (gate ou head)."""
    return [p for n, p in module.named_parameters() if "vqc" in n or "qlayer" in n]


def _log_head_quantum_epoch(epoch, classifier):
    """Diagnostics de la tete quantique (angles, sorties de qubits, poids VQC)."""
    if HEAD_MODE != "quantum" or not QUANTUM_DIAG_ENABLED:
        return {}
    out = {"epoch": int(epoch)}
    vqc_params = _vqc_params(classifier)
    if vqc_params:
        w = _flatten_weights(vqc_params)
        if DIAG_STATE["vqc_prev_weights"] is None:
            dw = 0.0
        else:
            dw = float(torch.norm(w - DIAG_STATE["vqc_prev_weights"], p=2).item())
        DIAG_STATE["vqc_prev_weights"] = w
        out["vqc_weight_change_norm"] = dw
        out["vqc_weight_l2"] = float(torch.norm(w, p=2).item())
        if dw == 0.0 and epoch > 1:
            log_event("quantum", logging.ERROR,
                      "VQC head weights are frozen — gradient never arrives",
                      event="HEAD_VQC_FROZEN", epoch=int(epoch))
        log_event("quantum", logging.INFO,
                  f"[HeadVQC][epoch={epoch}] weight-change L2={dw:.6e} | weight L2={out['vqc_weight_l2']:.6e}",
                  event="HEAD_VQC_WEIGHT_CHANGE", epoch=int(epoch), dw=float(dw))
    grad_diag = DIAG_STATE.get("epoch_grad_diag", {}).get(int(epoch), {})
    out["vqc_grad_norm_mean"] = float(grad_diag.get("vqc_grad_norm_mean", 0.0))
    out["vqc_grad_var_mean"] = float(grad_diag.get("vqc_grad_var_mean", 0.0))
    out["scaler_skipped_steps_epoch"] = int(grad_diag.get("scaler_skipped_steps_epoch", 0))
    return out


def train_one_epoch(medsam_model, quantum_gate, classifier, loader, optimizer, scaler, seg_loss_fn, device):
    medsam_model.train(); quantum_gate.train(); classifier.train()
    epoch = DIAG_STATE.get("epoch", 0)
    total_seg_loss = 0.0; n_seg = 0
    total_cls_loss = 0.0
    seg_metrics_sum = {"dice": 0.0, "iou": 0.0}
    correct_cls = 0
    vqc_grad_norm_epoch = []
    vqc_grad_var_epoch = []
    ratio_vqc_mask_epoch = []
    optimizer.zero_grad()
    t0 = time.time(); n_total = len(loader)

    for i, batch in enumerate(loader):
        DIAG_STATE["global_step"] += 1
        gs = DIAG_STATE["global_step"]
        image = batch["image"]; bboxes = batch["bboxes"]; has_mask = batch["has_mask"]
        class_idx = torch.tensor([batch["class_idx"]], device=device)
        source = "segmentation_task" if has_mask else "classification_task/no_tumor"

        if i < 3:
            with torch.no_grad():
                img_t_dbg = medsam_preprocess(image, device)
                bbox_valid = True
                for bb in bboxes:
                    if len(bb) != 4:
                        bbox_valid = False
                        break
                    x0, y0, x1, y1 = [float(v) for v in bb]
                    if not (0 <= x0 <= x1 < IMG_SIZE and 0 <= y0 <= y1 < IMG_SIZE):
                        bbox_valid = False
                        break
                mask_ratio = float(np.mean(batch["mask"])) if has_mask else 0.0
                log_event(
                    "data", logging.INFO,
                    f"[BatchInspect] i={i} src={source} img_shape={tuple(img_t_dbg.shape)} dtype={img_t_dbg.dtype} "
                    f"min={img_t_dbg.min().item():.6f} max={img_t_dbg.max().item():.6f} mean={img_t_dbg.mean().item():.6f} "
                    f"std={img_t_dbg.std(unbiased=False).item():.6f} mask_pos_ratio={mask_ratio:.6f} bbox_valid={bbox_valid}",
                    event="BATCH_INSPECT", batch=int(i), source=source,
                    img_shape=list(img_t_dbg.shape), dtype=str(img_t_dbg.dtype),
                    min=float(img_t_dbg.min().item()), max=float(img_t_dbg.max().item()),
                    mean=float(img_t_dbg.mean().item()), std=float(img_t_dbg.std(unbiased=False).item()),
                    mask_pos_ratio=mask_ratio, bbox_valid=bool(bbox_valid),
                )
                if not bbox_valid:
                    log_event("data", logging.WARNING, f"Invalid bbox coordinates at batch {i}",
                              event="BBOX_INVALID", batch=int(i),
                              bboxes=[bb.tolist() if hasattr(bb, "tolist") else list(bb) for bb in bboxes])

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            seg_logits, pooled = medsam_forward_joint(medsam_model, quantum_gate, image, bboxes, has_mask, device)
            class_logits = classifier(pooled)
            loss_cls = F.cross_entropy(class_logits, class_idx)

            if has_mask:
                mask = torch.from_numpy(batch["mask"]).unsqueeze(0).to(device)
                loss_seg = seg_loss_fn(seg_logits, mask)
                loss = (loss_seg + LAMBDA_CLS * loss_cls) / GRAD_ACCUM
            else:
                loss_seg = None
                loss = (LAMBDA_CLS * loss_cls) / GRAD_ACCUM

        scaler.scale(loss).backward()
        if (i + 1) % GRAD_ACCUM == 0:
            old_scale = float(scaler.get_scale())
            scaler.unscale_(optimizer)
            params = list(medsam_model.parameters()) + list(quantum_gate.parameters()) + list(classifier.parameters())
            grad_before = _grad_l2_norm(params)
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            grad_after = _grad_l2_norm(params)

            # Le VQC vit maintenant dans la TETE, pas dans la gate.
            vqc_params = _vqc_params(classifier)
            mask_params = list(medsam_model.mask_decoder.parameters())
            vqc_grad_norm = _grad_l2_norm(vqc_params) if vqc_params else 0.0
            mask_grad_norm = _grad_l2_norm(mask_params)
            ratio = vqc_grad_norm / max(mask_grad_norm, 1e-12)
            vqc_grad_norm_epoch.append(vqc_grad_norm)
            ratio_vqc_mask_epoch.append(ratio)
            vqc_flat_grad = _flatten_grads(vqc_params)
            if vqc_flat_grad is not None:
                gvar = float(vqc_flat_grad.var(unbiased=False).item())
                vqc_grad_var_epoch.append(gvar)
                if DIAG_STATE["vqc_first_nonzero_grad_step"] is None and vqc_grad_norm > 0.0:
                    DIAG_STATE["vqc_first_nonzero_grad_step"] = int(gs)
                    log_event("quantum", logging.INFO,
                              f"[HeadVQC] first non-zero VQC gradient at global_step={gs}",
                              event="HEAD_FIRST_NONZERO_VQC_GRAD", step=int(gs))
            else:
                gvar = 0.0

            scaler.step(optimizer)
            scaler.update()
            new_scale = float(scaler.get_scale())
            step_skipped = new_scale < old_scale
            if step_skipped:
                DIAG_STATE["scaler_skipped_steps_epoch"] += 1
            DIAG_STATE["optimizer_steps"] += 1

            lrs = [float(g["lr"]) for g in optimizer.param_groups]
            if (DIAG_STATE["optimizer_steps"] % max(LOG_EVERY_N_STEPS, 1) == 0) or (i + 1 == n_total):
                log_event(
                    "train", logging.INFO,
                    f"step={DIAG_STATE['optimizer_steps']} epoch={epoch} batch={i+1}/{n_total} gs={gs} src={source} "
                    f"has_mask={has_mask} loss_seg={loss_seg.item() if loss_seg is not None else 'NA'} "
                    f"loss_cls={loss_cls.item():.6f} total_loss={loss.item():.6f} "
                    f"grad_before={grad_before:.6e} grad_after={grad_after:.6e} "
                    f"vqc_grad={vqc_grad_norm:.6e} scale={old_scale:.1f}->{new_scale:.1f} skipped={step_skipped} lrs={lrs}",
                    event="TRAIN_STEP", epoch=int(epoch), batch=int(i), global_step=int(gs),
                    optimizer_step=int(DIAG_STATE["optimizer_steps"]), source=source, has_mask=bool(has_mask),
                    loss_seg=float(loss_seg.item()) if loss_seg is not None else None,
                    loss_cls=float(loss_cls.item()), total_loss=float(loss.item()),
                    grad_norm_before=float(grad_before), grad_norm_after=float(grad_after),
                    scaler_old=float(old_scale), scaler_new=float(new_scale), scaler_step_skipped=bool(step_skipped),
                    lr_groups=lrs, vqc_grad_norm=float(vqc_grad_norm), vqc_grad_var=float(gvar),
                    grad_ratio_vqc_mask=float(ratio),
                )
            optimizer.zero_grad()

        total_cls_loss += loss_cls.item()
        if has_mask:
            total_seg_loss += loss_seg.item(); n_seg += 1
            with torch.no_grad():
                m = compute_seg_metrics(seg_logits.detach(), mask)
                seg_metrics_sum["dice"] += m["dice"]; seg_metrics_sum["iou"] += m["iou"]
        pred_class = class_logits.argmax(dim=1).item()
        correct_cls += int(pred_class == class_idx.item())

        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            log_event("train", logging.INFO,
                      f"    [progres] {i+1}/{n_total} ({rate:.2f} img/s, ETA={((n_total-(i+1))/rate)/60:.1f}min, "
                      f"train_acc_cls={correct_cls/(i+1):.4f})",
                      event="TRAIN_PROGRESS", batch=int(i + 1), total=int(n_total),
                      rate=float(rate), train_acc=float(correct_cls / (i + 1)))

    if vqc_grad_var_epoch:
        mean_var = float(np.mean(vqc_grad_var_epoch))
        mean_ratio = float(np.mean(ratio_vqc_mask_epoch)) if ratio_vqc_mask_epoch else 0.0
        lvl_v = logging.WARNING if mean_var < 1e-6 else logging.INFO
        lvl_r = logging.WARNING if mean_ratio < 1e-4 else logging.INFO
        log_event("quantum", lvl_v, f"[HeadVQC][epoch={epoch}] mean VQC grad var={mean_var:.6e}",
                  event="HEAD_GRAD_VAR", epoch=int(epoch), value=mean_var)
        log_event("quantum", lvl_r,
                  f"[HeadVQC][epoch={epoch}] mean ||grad_vqc||/||grad_mask_decoder||={mean_ratio:.6e}",
                  event="HEAD_GRAD_RATIO", epoch=int(epoch), value=mean_ratio)

    DIAG_STATE["epoch_grad_diag"][int(epoch)] = {
        "vqc_grad_norm_mean": float(np.mean(vqc_grad_norm_epoch)) if vqc_grad_norm_epoch else 0.0,
        "vqc_grad_var_mean": float(np.mean(vqc_grad_var_epoch)) if vqc_grad_var_epoch else 0.0,
        "vqc_mask_ratio_mean": float(np.mean(ratio_vqc_mask_epoch)) if ratio_vqc_mask_epoch else 0.0,
        "scaler_skipped_steps_epoch": int(DIAG_STATE["scaler_skipped_steps_epoch"]),
    }

    seg_dice = seg_metrics_sum["dice"] / max(n_seg, 1)
    seg_iou = seg_metrics_sum["iou"] / max(n_seg, 1)
    return (total_seg_loss / max(n_seg, 1), total_cls_loss / n_total,
            {"dice": seg_dice, "iou": seg_iou}, correct_cls / n_total)


@torch.no_grad()
def validate(medsam_model, quantum_gate, classifier, loader, seg_loss_fn, device):
    medsam_model.eval(); quantum_gate.eval(); classifier.eval()
    total_seg_loss = 0.0; n_seg = 0
    total_cls_loss = 0.0
    seg_metrics_sum = {"dice": 0.0, "iou": 0.0}
    correct_cls = 0; n_total = len(loader)

    for batch in loader:
        image = batch["image"]; bboxes = batch["bboxes"]; has_mask = batch["has_mask"]
        class_idx = torch.tensor([batch["class_idx"]], device=device)
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            seg_logits, pooled = medsam_forward_joint(medsam_model, quantum_gate, image, bboxes, has_mask, device)
            class_logits = classifier(pooled)
            loss_cls = F.cross_entropy(class_logits, class_idx)
            if has_mask:
                mask = torch.from_numpy(batch["mask"]).unsqueeze(0).to(device)
                loss_seg = seg_loss_fn(seg_logits, mask)

        total_cls_loss += loss_cls.item()
        if has_mask:
            total_seg_loss += loss_seg.item(); n_seg += 1
            m = compute_seg_metrics(seg_logits, mask)
            seg_metrics_sum["dice"] += m["dice"]; seg_metrics_sum["iou"] += m["iou"]
        pred_class = class_logits.argmax(dim=1).item()
        correct_cls += int(pred_class == class_idx.item())

    seg_dice = seg_metrics_sum["dice"] / max(n_seg, 1)
    seg_iou = seg_metrics_sum["iou"] / max(n_seg, 1)
    log_event("train", logging.INFO,
              f"[Validate] seg_loss={total_seg_loss/max(n_seg,1):.6f} cls_loss={total_cls_loss/n_total:.6f} "
              f"dice={seg_dice:.6f} iou={seg_iou:.6f} acc={correct_cls/n_total:.6f}",
              event="VAL_EPOCH")
    return (total_seg_loss / max(n_seg, 1), total_cls_loss / n_total,
            {"dice": seg_dice, "iou": seg_iou}, correct_cls / n_total)


def log_startup_header():
    gpu_name = None
    gpu_mem_gb = None
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        gpu_name = prop.name
        gpu_mem_gb = float(prop.total_memory / (1024 ** 3))
    cfg = {
        "DEVICE": DEVICE, "IMG_SIZE": IMG_SIZE, "NUM_EPOCHS": NUM_EPOCHS, "PATIENCE": PATIENCE,
        "LR_ENCODER": LR_ENCODER, "LR_DECODER": LR_DECODER, "LR_QUANTUM": LR_QUANTUM,
        "LR_CLASSIF": LR_CLASSIF, "LAMBDA_CLS": LAMBDA_CLS, "GRAD_ACCUM": GRAD_ACCUM,
        "WEIGHT_DECAY": WEIGHT_DECAY, "SEED": SEED, "N_QUBITS": N_QUBITS, "N_LAYERS": N_LAYERS,
        "GATE_MODE": GATE_MODE, "REQUESTED_GATE_MODE": REQUESTED_GATE_MODE, "HEAD_MODE": HEAD_MODE,
        "RUN_NAME": RUN_NAME, "NORM_MODE": NORM_MODE, "DATA_FRACTION": DATA_FRACTION,
        "SELECT_METRIC": SELECT_METRIC, "SOTA_DICE": SOTA_DICE, "BASELINE_DICE": BASELINE_DICE,
        "COMPARE_DIR": COMPARE_DIR,
    }
    log_event("setup", logging.INFO,
              f"Startup host={socket.gethostname()} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')} "
              f"gpu={gpu_name} gpu_mem_gb={gpu_mem_gb}",
              event="STARTUP_HOST", hostname=socket.gethostname(),
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET"),
              gpu_name=gpu_name, gpu_mem_gb=gpu_mem_gb)
    log_event("setup", logging.INFO,
              f"Versions python={platform.python_version()} torch={torch.__version__} pennylane={qml.__version__} "
              f"numpy={np.__version__} opencv={cv2.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}",
              event="STARTUP_VERSIONS", python=platform.python_version(), torch=torch.__version__,
              pennylane=qml.__version__, numpy=np.__version__, opencv=cv2.__version__,
              cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    log_event("setup", logging.INFO, f"Config JSON: {json.dumps(cfg, ensure_ascii=True)}", event="CONFIG_JSON", config=cfg)


def log_dataset_stats(seg_ds, notumor_ds, train_ds, val_ds):
    total = len(seg_ds) + len(notumor_ds)
    log_event("data", logging.INFO,
              f"Source 1 (avec masque, seg+cls) : {len(seg_ds)} images — {SEG_TRAIN_IMG}",
              event="DATA_SOURCE1", count=int(len(seg_ds)), path=SEG_TRAIN_IMG)
    log_event("data", logging.INFO,
              f"Source 2 (sans masque, cls seule, No Tumor) : {len(notumor_ds)} images",
              event="DATA_SOURCE2", count=int(len(notumor_ds)), path=os.path.join(CLS_TRAIN_ROOT, "no_tumor"))
    log_event("data", logging.INFO,
              f"Total combine : {total} | Train: {len(train_ds)} | Val: {len(val_ds)}",
              event="DATA_SPLIT", total=int(total), train=int(len(train_ds)), val=int(len(val_ds)))

    cls_counts_all = Counter(); cls_counts_train = Counter(); cls_counts_val = Counter()
    src_train = Counter(); src_val = Counter()

    for i in range(len(seg_ds)):
        cls_counts_all[class_idx_from_seg_path(str(seg_ds.img_paths[i]))] += 1
    cls_counts_all[3] += len(notumor_ds)

    for idx in train_ds.indices:
        if idx < len(seg_ds):
            cls = class_idx_from_seg_path(str(seg_ds.img_paths[idx])); src_train["segmentation_task"] += 1
        else:
            cls = 3; src_train["classification_task/no_tumor"] += 1
        cls_counts_train[cls] += 1

    for idx in val_ds.indices:
        if idx < len(seg_ds):
            cls = class_idx_from_seg_path(str(seg_ds.img_paths[idx])); src_val["segmentation_task"] += 1
        else:
            cls = 3; src_val["classification_task/no_tumor"] += 1
        cls_counts_val[cls] += 1

    def _fmt_counts(cnt, denom):
        return {CLASSES[k]: {"count": int(v), "pct": float(100.0 * v / max(denom, 1))} for k, v in sorted(cnt.items())}

    log_event("data", logging.INFO, f"Class distribution all: {_fmt_counts(cls_counts_all, total)}",
              event="CLASS_DIST_ALL", data=_fmt_counts(cls_counts_all, total))
    log_event("data", logging.INFO, f"Class distribution train: {_fmt_counts(cls_counts_train, len(train_ds))}",
              event="CLASS_DIST_TRAIN", data=_fmt_counts(cls_counts_train, len(train_ds)))
    log_event("data", logging.INFO, f"Class distribution val: {_fmt_counts(cls_counts_val, len(val_ds))}",
              event="CLASS_DIST_VAL", data=_fmt_counts(cls_counts_val, len(val_ds)))
    log_event("data", logging.INFO, f"Source in train: {dict(src_train)} | Source in val: {dict(src_val)}",
              event="SOURCE_SPLIT", train=dict(src_train), val=dict(src_val))


def _dice_distribution(arr):
    if len(arr) == 0:
        return {}
    x = np.array(arr, dtype=np.float64)
    return {
        "min": float(np.min(x)), "p5": float(np.percentile(x, 5)), "p25": float(np.percentile(x, 25)),
        "median": float(np.percentile(x, 50)), "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)), "max": float(np.max(x)),
    }


def write_digest(summary, warning_counts, warning_messages):
    lines = []
    lines.append("=" * 100)
    lines.append(f"RUN DIGEST | {summary.get('run_name')} | head_mode={summary.get('head_mode')} | "
                 f"n_qubits={summary.get('n_qubits')} | ts={summary.get('timestamp')}")
    lines.append("=" * 100)
    lines.append(f"startup: backend={summary.get('backend', {})}")
    lines.append(f"params: {summary.get('params', {})}")
    lines.append("\nPer-epoch metrics:")
    lines.append("epoch | train_dice | train_acc | val_dice | val_acc | val_iou | combined")
    for r in summary.get("epoch_rows", []):
        lines.append(
            f"{r['epoch']:5d} | {r['train_dice']:.6f} | {r['train_acc']:.6f} | {r['val_dice']:.6f} | "
            f"{r['val_acc']:.6f} | {r['val_iou']:.6f} | {r['combined']:.6f}"
        )
    lines.append("\nComparison metrics (this arm):")
    for key, label in COMPARISON_METRICS:
        lines.append(f"{label:36s} = {summary.get('comparison_metrics', {}).get(key, float('nan')):.6f}")
    lines.append("\nFinal test metrics:")
    lines.append(f"segmentation={summary.get('test_segmentation', {})}")
    lines.append(f"classification={summary.get('test_classification', {})}")
    lines.append("\nWarnings/Errors (deduplicated):")
    lines.append(f"Counts by level: {dict(warning_counts)}")
    for msg, cnt in warning_messages.most_common():
        lines.append(f"[{cnt}] {msg}")
    lines = lines[:250]
    if _RUN_LOG_STATE["digest_file"]:
        with open(_RUN_LOG_STATE["digest_file"], "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log_event("setup", logging.INFO, f"[Digest] {_RUN_LOG_STATE['digest_file']}", event="DIGEST_WRITTEN")


# ==============================================================================
# 6B. EXPORT DES METRIQUES + TABLEAU COMPARATIF MLP vs QUANTUM
# ==============================================================================
def build_comparison_metrics(seg_test, cls_test):
    """Extrait EXACTEMENT les 8 metriques du tableau, dans un dict a plat."""
    return {
        "seg_dice":              float(seg_test["dice"]),
        "seg_iou":               float(seg_test["iou"]),
        "seg_precision":         float(seg_test["precision"]),
        "seg_recall":            float(seg_test["recall"]),
        "seg_specificity":       float(seg_test["specificity"]),
        "cls_accuracy":          float(cls_test["accuracy"]),
        "cls_macro_f1":          float(cls_test["macro_f1"]),
        "cls_balanced_accuracy": float(cls_test["balanced_accuracy"]),
    }


def write_head_metrics(metrics, extra=None):
    """Ecrit le fichier de metriques normalise du bras courant dans COMPARE_DIR."""
    payload = {
        "head_mode": HEAD_MODE,
        "label": "MLP" if HEAD_MODE == "mlp" else f"{N_QUBITS}-Qubit Quantum",
        "n_qubits": int(N_QUBITS) if HEAD_MODE == "quantum" else None,
        "n_layers": int(N_LAYERS) if HEAD_MODE == "quantum" else None,
        "run_name": RUN_NAME,
        "seed": int(SEED),
        "gate_mode": GATE_MODE,
        "select_metric": SELECT_METRIC,
        "data_fraction": float(DATA_FRACTION),
        "norm_mode": NORM_MODE,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    path = head_metrics_path(HEAD_MODE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log_event("compare", logging.INFO, f"[Compare] Metriques du bras '{HEAD_MODE}' ecrites -> {path}",
              event="HEAD_METRICS_WRITTEN", path=path, head_mode=HEAD_MODE, metrics=metrics)
    return path


def _load_head_metrics(head_mode):
    path = head_metrics_path(head_mode)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event("compare", logging.WARNING, f"[Compare] Lecture impossible de {path}: {type(e).__name__}: {e}",
                  event="HEAD_METRICS_READ_FAIL", path=path)
        return None


def plot_comparison(labels_mlp, labels_q, m_mlp, m_q):
    plt.style.use("dark_background")
    names = [lbl for _, lbl in COMPARISON_METRICS]
    vals_mlp = [m_mlp[k] for k, _ in COMPARISON_METRICS]
    vals_q = [m_q[k] for k, _ in COMPARISON_METRICS]
    y = np.arange(len(names))
    h = 0.38

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    b1 = ax.barh(y - h / 2, vals_mlp, height=h, color="#2E86AB", label=labels_mlp)
    b2 = ax.barh(y + h / 2, vals_q, height=h, color="#7B2FF7", label=labels_q)
    for bars, vals in ((b1, vals_mlp), (b2, vals_q)):
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.012, 1.005), bar.get_y() + bar.get_height() / 2,
                    f"{100*v:.2f}%", va="center", color="white", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names, color="white")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score", color="white")
    ax.set_title(f"Head ablation — {labels_mlp} vs {labels_q} (MedSAM, gate OFF)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.grid(axis="x", color="#30363d", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(COMPARISON_TABLE_PNG, dpi=150, facecolor="#0d1117")
    plt.close()
    log_event("compare", logging.INFO, f"[Compare] Figure -> {COMPARISON_TABLE_PNG}",
              event="COMPARISON_FIGURE_WRITTEN", path=COMPARISON_TABLE_PNG)


def build_comparison_table():
    """Construit le tableau MLP vs Quantum si les deux fichiers de metriques existent."""
    d_mlp = _load_head_metrics("mlp")
    d_q = _load_head_metrics("quantum")

    if d_mlp is None or d_q is None:
        missing = [m for m, d in (("mlp", d_mlp), ("quantum", d_q)) if d is None]
        log_event("compare", logging.WARNING,
                  f"[Compare] Tableau NON genere : bras manquant(s) = {missing}. "
                  f"Lance l'autre bras avec HEAD_MODE=<mode>, puis relance avec COMPARE_ONLY=1.",
                  event="COMPARISON_INCOMPLETE", missing=missing, compare_dir=COMPARE_DIR)
        return None

    m_mlp = d_mlp["metrics"]; m_q = d_q["metrics"]
    label_mlp = d_mlp.get("label", "MLP")
    label_q = d_q.get("label", "Quantum")

    # --- Markdown ---
    lines = []
    lines.append(f"# Head ablation — {label_mlp} vs {label_q}")
    lines.append("")
    lines.append(f"MedSAM full fine-tuned, image gate = OFF (IdentityGate). "
                 f"Seed {d_mlp.get('seed')} / {d_q.get('seed')}. "
                 f"Circuit : {d_q.get('n_qubits')} qubits x {d_q.get('n_layers')} couches.")
    lines.append("")
    lines.append(f"| Metric | {label_mlp} | {label_q} | Delta (Q - MLP) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key, label in COMPARISON_METRICS:
        a = m_mlp.get(key, float("nan")); b = m_q.get(key, float("nan"))
        lines.append(f"| **{label}** | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    lines.append("")
    lines.append(f"_Genere le {datetime.datetime.now().isoformat(timespec='seconds')} — "
                 f"runs : {d_mlp.get('run_name')} / {d_q.get('run_name')}._")
    md = "\n".join(lines) + "\n"
    with open(COMPARISON_TABLE_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # --- CSV ---
    with open(COMPARISON_TABLE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", label_mlp, label_q, "delta_q_minus_mlp"])
        for key, label in COMPARISON_METRICS:
            a = m_mlp.get(key, float("nan")); b = m_q.get(key, float("nan"))
            w.writerow([label, f"{a:.6f}", f"{b:.6f}", f"{b - a:+.6f}"])

    # --- Figure ---
    try:
        plot_comparison(label_mlp, label_q, m_mlp, m_q)
    except Exception as e:
        log_event("compare", logging.WARNING, f"[Compare] Figure non generee: {type(e).__name__}: {e}",
                  event="COMPARISON_FIGURE_FAIL")

    # --- Log lisible dans le .log ---
    log_event("compare", logging.INFO, "\n" + "=" * 80, event="COMPARISON_TABLE")
    log_event("compare", logging.INFO, "TABLEAU COMPARATIF — HEAD ABLATION", event="COMPARISON_TABLE")
    log_event("compare", logging.INFO, "=" * 80, event="COMPARISON_TABLE")
    log_event("compare", logging.INFO,
              f"{'Metric':36s} | {label_mlp:>14s} | {label_q:>18s} | {'Delta':>9s}", event="COMPARISON_TABLE")
    log_event("compare", logging.INFO, "-" * 86, event="COMPARISON_TABLE")
    for key, label in COMPARISON_METRICS:
        a = m_mlp.get(key, float("nan")); b = m_q.get(key, float("nan"))
        log_event("compare", logging.INFO,
                  f"{label:36s} | {a:14.4f} | {b:18.4f} | {b - a:+9.4f}",
                  event="COMPARISON_ROW", metric=label, mlp=float(a), quantum=float(b), delta=float(b - a))
    log_event("compare", logging.INFO, "=" * 80, event="COMPARISON_TABLE")
    log_event("compare", logging.INFO,
              f"[Compare] Fichiers -> {COMPARISON_TABLE_MD} | {COMPARISON_TABLE_CSV} | {COMPARISON_TABLE_PNG}",
              event="COMPARISON_TABLE_WRITTEN")
    return md


# ==============================================================================
# 7. MAIN
# ==============================================================================
def main():
    configure_logging(RUN_NAME)

    # Mode "tableau seulement" : aucun entrainement, aucun GPU requis.
    if COMPARE_ONLY:
        log_event("compare", logging.INFO, "[COMPARE_ONLY=1] Regeneration du tableau depuis les JSON existants.",
                  event="COMPARE_ONLY_MODE", compare_dir=COMPARE_DIR)
        build_comparison_table()
        flush_log_handlers()
        return

    log_startup_header()
    train_t0 = time.time()

    log_event("setup", logging.INFO, f"[Paths] BEST_CKPT={BEST_CKPT} | RESULTS_DIR={RESULTS_DIR} | COMPARE_DIR={COMPARE_DIR}",
              event="PATHS", best_ckpt=BEST_CKPT, results_dir=RESULTS_DIR, compare_dir=COMPARE_DIR)

    if torch.cuda.is_available() and ENFORCE_CUDA_VISIBLE and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES absent. Sur obiwan, lance toujours le script avec "
            "CUDA_VISIBLE_DEVICES=N nohup python script.py > run.log 2>&1 &"
        )

    log_event("setup", logging.INFO, "=" * 80, event="BANNER")
    log_event("setup", logging.INFO,
              f"MedSAM full fine-tuned — HEAD ABLATION : {'MLP' if HEAD_MODE=='mlp' else f'{N_QUBITS}-Qubit Quantum'}",
              event="BANNER")
    log_event("setup", logging.INFO, "=" * 80, event="BANNER")
    log_event("setup", logging.INFO, f"Run name: {RUN_NAME}", event="RUN_NAME", run_name=RUN_NAME)
    log_event("setup", logging.INFO,
              f"Requested gate mode: {REQUESTED_GATE_MODE} | Effective gate mode: {GATE_MODE}",
              event="RUN_GATE_MODE", requested_gate_mode=REQUESTED_GATE_MODE, gate_mode=GATE_MODE)
    log_event("setup", logging.INFO, f"Head mode: {HEAD_MODE} | N_QUBITS={N_QUBITS} | N_LAYERS={N_LAYERS}",
              event="RUN_HEAD_MODE", head_mode=HEAD_MODE, n_qubits=int(N_QUBITS), n_layers=int(N_LAYERS))
    log_event("setup", logging.INFO, f"Select metric: {SELECT_METRIC} | Data fraction: {DATA_FRACTION:.4f}",
              event="RUN_SELECT", select_metric=SELECT_METRIC, data_fraction=float(DATA_FRACTION))

    # ---- Source 1 : images AVEC masque ----
    seg_ds = SegClsDataset(SEG_TRAIN_IMG, SEG_TRAIN_MSK, augment=True)
    # ---- Source 2 : images SANS masque (No Tumor) ----
    no_tumor_dir = os.path.join(CLS_TRAIN_ROOT, "no_tumor")
    notumor_ds = ClsOnlyDataset(no_tumor_dir, class_idx=3, augment=True)

    full_ds = ConcatDataset([seg_ds, notumor_ds])
    n_val = int(0.15 * len(full_ds)); n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val],
                                                     generator=torch.Generator().manual_seed(SEED))

    if DATA_FRACTION < 1.0:
        sub_indices = stratified_subset_indices(full_ds, train_ds.indices, DATA_FRACTION, SEED)
        train_ds = torch.utils.data.Subset(full_ds, sub_indices)
        frac_counts = Counter()
        for idx in train_ds.indices:
            frac_counts[int(full_ds[idx]["class_idx"])] += 1
        frac_counts_fmt = {CLASSES[k]: int(frac_counts.get(k, 0)) for k in range(N_CLASSES)}
        log_event("data", logging.INFO,
                  f"[DATA_FRACTION] train subset active: fraction={DATA_FRACTION:.4f} | n_train={len(train_ds)} | per_class={frac_counts_fmt}",
                  event="TRAIN_SUBSAMPLE", data_fraction=float(DATA_FRACTION),
                  n_train=int(len(train_ds)), per_class=frac_counts_fmt)

    log_dataset_stats(seg_ds, notumor_ds, train_ds, val_ds)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_single)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_single)

    medsam_model, quantum_gate, classifier = load_model(DEVICE)
    param_report = print_param_report(medsam_model, quantum_gate, classifier)
    seg_loss_fn = SegLoss()
    optimizer = torch.optim.AdamW([
        {"params": medsam_model.prompt_encoder.parameters(), "lr": LR_DECODER},
        {"params": medsam_model.mask_decoder.parameters(), "lr": LR_DECODER},
        {"params": quantum_gate.parameters(), "lr": LR_QUANTUM},
        {"params": [p for n, p in classifier.named_parameters() if not ("vqc" in n or "qlayer" in n)], "lr": LR_CLASSIF},
        {"params": [p for n, p in classifier.named_parameters() if ("vqc" in n or "qlayer" in n)], "lr": LR_QUANTUM},
    ], weight_decay=WEIGHT_DECAY)
    for gi, g in enumerate(optimizer.param_groups):
        n_tensors = len(g["params"]); n_params = int(sum(p.numel() for p in g["params"]))
        log_event("train", logging.INFO,
                  f"[OptimGroup {gi}] lr={g['lr']} wd={g.get('weight_decay', WEIGHT_DECAY)} tensors={n_tensors} params={n_params}",
                  event="OPTIM_GROUP", group_index=int(gi), lr=float(g["lr"]),
                  weight_decay=float(g.get("weight_decay", WEIGHT_DECAY)),
                  tensors=int(n_tensors), params=int(n_params))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler(device="cuda", enabled=torch.cuda.is_available())

    best_score = -1e18; patience_counter = 0
    best_epoch = -1; best_val_dice = 0.0; best_val_acc = 0.0
    best_select_value = -1e18
    history = {"train_dice": [], "train_acc": [], "train_seg_loss": [], "train_cls_loss": [],
               "val_dice": [], "val_acc": [], "val_seg_loss": [], "val_cls_loss": []}
    quantum_traj = []
    scaler_skipped_total = 0

    for epoch in range(NUM_EPOCHS):
        DIAG_STATE["epoch"] = epoch + 1
        DIAG_STATE["scaler_skipped_steps_epoch"] = 0
        epoch_t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tr_seg_loss, tr_cls_loss, tr_seg_m, tr_acc = train_one_epoch(
            medsam_model, quantum_gate, classifier, train_loader, optimizer, scaler, seg_loss_fn, DEVICE)
        val_seg_loss, val_cls_loss, val_seg_m, val_acc = validate(
            medsam_model, quantum_gate, classifier, val_loader, seg_loss_fn, DEVICE)
        scheduler.step()
        history["train_dice"].append(tr_seg_m["dice"]); history["train_acc"].append(tr_acc)
        history["train_seg_loss"].append(tr_seg_loss); history["train_cls_loss"].append(tr_cls_loss)
        history["val_dice"].append(val_seg_m["dice"]); history["val_acc"].append(val_acc)
        history["val_seg_loss"].append(val_seg_loss); history["val_cls_loss"].append(val_cls_loss)

        combined_score = 0.5 * val_seg_m["dice"] + 0.5 * val_acc
        if SELECT_METRIC == "val_acc":
            select_value = float(val_acc)
        elif SELECT_METRIC == "val_dice":
            select_value = float(val_seg_m["dice"])
        else:
            select_value = float(combined_score)
        epoch_sec = time.time() - epoch_t0
        max_mem = float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
        scaler_skipped_total += DIAG_STATE["scaler_skipped_steps_epoch"]

        log_event("train", logging.INFO,
                  f"[Epoch {epoch+1}/{NUM_EPOCHS}] train_dice={tr_seg_m['dice']:.4f} train_acc={tr_acc:.4f} | "
                  f"val_dice={val_seg_m['dice']:.4f} val_acc={val_acc:.4f} val_iou={val_seg_m['iou']:.4f} "
                  f"(combine={combined_score:.4f})",
                  event="EPOCH_METRICS", epoch=int(epoch + 1), n_epochs=int(NUM_EPOCHS),
                  train_dice=float(tr_seg_m["dice"]), train_iou=float(tr_seg_m["iou"]), train_acc=float(tr_acc),
                  val_dice=float(val_seg_m["dice"]), val_iou=float(val_seg_m["iou"]), val_acc=float(val_acc),
                  combined=float(combined_score), select_value=float(select_value), select_metric=SELECT_METRIC,
                  epoch_sec=float(epoch_sec), max_memory_gb=float(max_mem),
                  patience_counter=int(patience_counter),
                  scaler_skipped_steps_epoch=int(DIAG_STATE["scaler_skipped_steps_epoch"]))

        ckpt_data = {"epoch": epoch, "model_state": medsam_model.state_dict(),
                     "quantum_state": quantum_gate.state_dict(), "classifier_state": classifier.state_dict(),
                     "optim_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                     "scaler_state": scaler.state_dict(), "best_score": best_score,
                     "best_select_value": best_select_value, "history": history,
                     "select_metric": SELECT_METRIC, "head_mode": HEAD_MODE, "n_qubits": N_QUBITS}
        torch.save(ckpt_data, LAST_CKPT)
        if select_value > best_select_value:
            best_select_value = select_value
            best_score = combined_score
            ckpt_data["best_score"] = best_score
            ckpt_data["best_select_value"] = best_select_value
            best_epoch = epoch + 1
            best_val_dice = val_seg_m["dice"]; best_val_acc = val_acc
            torch.save(ckpt_data, BEST_CKPT); patience_counter = 0
            log_event("ckpt", logging.INFO,
                      f"  -> Nouveau meilleur ({SELECT_METRIC}={best_select_value:.4f}, combined={best_score:.4f}, "
                      f"Dice={val_seg_m['dice']:.4f}, Acc={val_acc:.4f})",
                      event="CKPT_BEST_SAVED", path=BEST_CKPT, epoch=int(best_epoch), score=float(best_score),
                      val_dice=float(best_val_dice), val_acc=float(best_val_acc),
                      select_metric=SELECT_METRIC, select_value=float(best_select_value))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log_event("train", logging.INFO, f"[Early stopping] a l'epoch {epoch+1}",
                          event="EARLY_STOP", epoch=int(epoch + 1), best_epoch=int(best_epoch),
                          patience=int(PATIENCE), select_metric=SELECT_METRIC,
                          best_select_value=float(best_select_value))
                break

        qe = _log_head_quantum_epoch(epoch + 1, classifier)
        if qe:
            quantum_traj.append(qe)

        DIAG_STATE["epoch_rows"].append({
            "epoch": int(epoch + 1), "train_dice": float(tr_seg_m["dice"]), "train_acc": float(tr_acc),
            "val_dice": float(val_seg_m["dice"]), "val_acc": float(val_acc), "val_iou": float(val_seg_m["iou"]),
            "combined": float(combined_score), "select_value": float(select_value),
        })
        flush_log_handlers()

    plot_training_curves(history)
    total_train_sec = float(time.time() - train_t0)
    log_event("train", logging.INFO,
              f"[Termine] best combined={best_score:.4f} | temps total={total_train_sec/3600.0:.3f} h",
              event="TRAIN_DONE", best_score=float(best_score), best_epoch=int(best_epoch),
              total_sec=total_train_sec)
    val_dice_std_last5 = float(np.std(history["val_dice"][-5:])) if len(history["val_dice"]) >= 5 else (
        float(np.std(history["val_dice"])) if history["val_dice"] else 0.0)
    log_event("train", logging.INFO, f"[Noise floor] val_dice std sur les 5 dernieres epochs = {val_dice_std_last5:.6e}",
              event="VAL_NOISE_FLOOR", value=val_dice_std_last5)

    # ---------------- EVALUATIONS FINALES (source des 8 metriques) -----------
    seg_test = evaluate_segmentation_test(medsam_model, quantum_gate, DEVICE)
    cls_test = evaluate_classification_test(medsam_model, quantum_gate, classifier, DEVICE)

    comparison_metrics = build_comparison_metrics(seg_test, cls_test)
    log_event("eval", logging.INFO, "\n" + "-" * 80, event="ARM_METRICS_HEADER")
    log_event("eval", logging.INFO,
              f"METRIQUES DU BRAS '{HEAD_MODE.upper()}' ({'MLP' if HEAD_MODE=='mlp' else f'{N_QUBITS}-Qubit Quantum'})",
              event="ARM_METRICS_HEADER")
    log_event("eval", logging.INFO, "-" * 80, event="ARM_METRICS_HEADER")
    for key, label in COMPARISON_METRICS:
        log_event("eval", logging.INFO, f"{label:36s} = {comparison_metrics[key]:.4f}",
                  event="ARM_METRIC", metric=label, value=float(comparison_metrics[key]))

    write_head_metrics(comparison_metrics, extra={
        "best_epoch": int(best_epoch),
        "best_val_dice": float(best_val_dice),
        "best_val_acc": float(best_val_acc),
        "head_params_total": int(param_report["head"]["total"]),
        "head_params_trainable": int(param_report["head"]["trainable"]),
        "vqc_circuit_params": int(param_report["vqc_circuit_params"]),
        "total_train_time_sec": total_train_sec,
        "val_dice_std_last5": val_dice_std_last5,
    })

    summary = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME, "gate_mode": GATE_MODE, "requested_gate_mode": REQUESTED_GATE_MODE,
        "head_mode": HEAD_MODE, "n_qubits": int(N_QUBITS), "n_layers": int(N_LAYERS),
        "select_metric": SELECT_METRIC, "data_fraction": float(DATA_FRACTION), "device": DEVICE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        "norm_mode": NORM_MODE, "seed": SEED, "n_epochs_configured": NUM_EPOCHS,
        "best_epoch": best_epoch, "best_score": best_score, "best_select_value": best_select_value,
        "best_val_dice": best_val_dice, "best_val_acc": best_val_acc,
        "backend": {"name": QDEV.name, "diff_method": QDIFF_METHOD},
        "epoch_rows": DIAG_STATE["epoch_rows"], "quantum_traj": quantum_traj,
        "vqc_first_nonzero_grad_step": DIAG_STATE["vqc_first_nonzero_grad_step"],
        "scaler_skipped_total": scaler_skipped_total, "val_dice_std_last5": val_dice_std_last5,
        "test_segmentation": {k: v for k, v in seg_test.items() if k != "worst20"},
        "test_classification": {
            "accuracy": cls_test["accuracy"], "macro_f1": cls_test["macro_f1"],
            "balanced_accuracy": cls_test["balanced_accuracy"],
            "per_class_accuracy": cls_test["per_class_accuracy"],
            "per_class_precision": cls_test["per_class_precision"],
            "per_class_recall": cls_test["per_class_recall"],
            "per_class_f1": cls_test["per_class_f1"],
        },
        "comparison_metrics": comparison_metrics,
        "params": param_report, "warning_counts": dict(_WARN_ERR_COUNTS),
        "total_train_time_sec": total_train_sec,
    }
    with open(RUN_METADATA_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log_event("setup", logging.INFO, f"[Run Summary] {RUN_METADATA_JSON}", event="RUN_SUMMARY_WRITTEN", path=RUN_METADATA_JSON)

    write_digest(summary, _WARN_ERR_COUNTS, _WARN_ERR_MESSAGES)

    # ---- Tableau comparatif (si les deux bras sont disponibles) ----
    build_comparison_table()
    flush_log_handlers()


# ==============================================================================
# 8A. EVALUATION SEGMENTATION — segmentation_task/test (3 classes tumorales)
# ==============================================================================
@torch.no_grad()
def evaluate_segmentation_test(medsam_model, quantum_gate, device):
    log_event("eval", logging.INFO, "\n" + "=" * 80, event="EVAL_SEG_HEADER")
    log_event("eval", logging.INFO, "EVALUATION SEGMENTATION — segmentation_task/test (3 classes)", event="EVAL_SEG_HEADER")
    log_event("eval", logging.INFO, "=" * 80, event="EVAL_SEG_HEADER")
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    medsam_model.load_state_dict(ckpt["model_state"]); quantum_gate.load_state_dict(ckpt["quantum_state"])
    medsam_model.eval(); quantum_gate.eval()

    test_ds = SegClsDataset(SEG_TEST_IMG, SEG_TEST_MSK, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_single)

    all_m = []; per_type = {"Glioma": [], "Meningioma": [], "Pituitary": []}
    per_image_dice = []
    for batch in test_loader:
        image = batch["image"]; mask = torch.from_numpy(batch["mask"]).unsqueeze(0).to(device); bboxes = batch["bboxes"]
        ttype = CLASSES[batch["class_idx"]]
        pth = batch.get("path", "")
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            seg_logits, _ = medsam_forward_joint(medsam_model, quantum_gate, image, bboxes, has_mask=True, device=device)
        m = compute_seg_metrics(seg_logits, mask)
        all_m.append(m); per_type[ttype].append(m["dice"])
        per_image_dice.append((pth, float(m["dice"])))

    # Moyennes par image des 5 metriques -> lignes "Segmentation *" du tableau.
    avg = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}
    log_event("eval", logging.INFO,
              f"Dice={avg['dice']:.4f} IoU={avg['iou']:.4f} Prec={avg['precision']:.4f} "
              f"Rec={avg['recall']:.4f} Spec={avg['specificity']:.4f}",
              event="SEG_GLOBAL", **avg)
    for t, v in per_type.items():
        if v:
            log_event("eval", logging.INFO, f"  {t}: {np.mean(v):.4f} (n={len(v)})",
                      event="SEG_PER_TYPE", tumor_type=t, dice=float(np.mean(v)), n=int(len(v)))

    dice_vals = [d for _, d in per_image_dice]
    dist = _dice_distribution(dice_vals)
    n = max(len(dice_vals), 1)
    gt097 = sum(d > 0.97 for d in dice_vals)
    gt095 = sum(d > 0.95 for d in dice_vals)
    lt090 = sum(d < 0.90 for d in dice_vals)
    worst20 = sorted(per_image_dice, key=lambda x: x[1])[:20]
    log_event("eval", logging.INFO,
              f"[Dist] min={dist.get('min',0):.4f} p5={dist.get('p5',0):.4f} p25={dist.get('p25',0):.4f} "
              f"median={dist.get('median',0):.4f} p75={dist.get('p75',0):.4f} p95={dist.get('p95',0):.4f} max={dist.get('max',0):.4f}",
              event="DICE_DISTRIBUTION", **dist)
    log_event("eval", logging.INFO,
              f"[Dist] Dice>0.97: {gt097}/{n} | >0.95: {gt095}/{n} | <0.90: {lt090}/{n}",
              event="DICE_THRESHOLDS", gt097=int(gt097), gt095=int(gt095), lt090=int(lt090), n=int(n))

    delta_sota = avg["dice"] - SOTA_DICE
    delta_baseline = avg["dice"] - BASELINE_DICE
    log_event("eval", logging.INFO,
              f"Seg Dice delta vs SOTA({SOTA_DICE:.6f})={delta_sota:+.6f} | vs BASELINE({BASELINE_DICE:.6f})={delta_baseline:+.6f}",
              event="SEG_DELTAS", sota_dice=float(SOTA_DICE), baseline_dice=float(BASELINE_DICE),
              delta_sota=float(delta_sota), delta_baseline=float(delta_baseline))

    with open(os.path.join(RESULTS_DIR, "segmentation_test_results.json"), "w") as f:
        json.dump({
            "global": avg,
            "per_type": {t: float(np.mean(v)) if v else None for t, v in per_type.items()},
            "dice_distribution": dist,
            "thresholds": {
                ">0.97": {"count": int(gt097), "fraction": float(gt097 / n)},
                ">0.95": {"count": int(gt095), "fraction": float(gt095 / n)},
                "<0.90": {"count": int(lt090), "fraction": float(lt090 / n)},
            },
            "worst20": [{"path": p, "dice": d} for p, d in worst20],
            "delta_vs_sota": float(delta_sota),
            "delta_vs_baseline": float(delta_baseline),
        }, f, indent=2)

    out = dict(avg)
    out["dice_distribution"] = dist
    out["thresholds"] = {
        ">0.97": {"count": int(gt097), "fraction": float(gt097 / n)},
        ">0.95": {"count": int(gt095), "fraction": float(gt095 / n)},
        "<0.90": {"count": int(lt090), "fraction": float(lt090 / n)},
    }
    out["worst20"] = [{"path": p, "dice": d} for p, d in worst20]
    out["delta_vs_sota"] = float(delta_sota)
    out["delta_vs_baseline"] = float(delta_baseline)
    out["per_type_dice"] = {t: (float(np.mean(v)) if v else None) for t, v in per_type.items()}
    return out


def classification_metrics_from_confusion(conf_mat):
    per_class_acc = []; per_class_f1 = []; per_class_precision = []; per_class_recall = []
    for i in range(conf_mat.shape[0]):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp
        fn = conf_mat[i, :].sum() - tp
        denom_p = tp + fp
        denom_r = tp + fn
        precision = tp / denom_p if denom_p > 0 else 0.0
        recall = tp / denom_r if denom_r > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc_i = tp / denom_r if denom_r > 0 else 0.0
        per_class_precision.append(float(precision))
        per_class_recall.append(float(recall))
        per_class_acc.append(float(acc_i))
        per_class_f1.append(float(f1))
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0   # -> "Classification Macro-F1"
    bal_acc = float(np.mean(per_class_acc)) if per_class_acc else 0.0  # -> "Classification Balanced Accuracy"
    return per_class_acc, per_class_precision, per_class_recall, per_class_f1, macro_f1, bal_acc


# ==============================================================================
# 8B. EVALUATION CLASSIFICATION — classification_task/test (4 classes COMPLETES)
# ==============================================================================
@torch.no_grad()
def evaluate_classification_test(medsam_model, quantum_gate, classifier, device):
    log_event("eval", logging.INFO, "\n" + "=" * 80, event="EVAL_CLS_HEADER")
    log_event("eval", logging.INFO, "EVALUATION CLASSIFICATION — classification_task/test (4 classes)", event="EVAL_CLS_HEADER")
    log_event("eval", logging.INFO, "=" * 80, event="EVAL_CLS_HEADER")
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    medsam_model.load_state_dict(ckpt["model_state"]); quantum_gate.load_state_dict(ckpt["quantum_state"])
    classifier.load_state_dict(ckpt["classifier_state"])
    medsam_model.eval(); quantum_gate.eval(); classifier.eval()

    all_true, all_pred = [], []
    for folder, cls_idx in CLS_FOLDER_TO_IDX.items():
        class_dir = os.path.join(CLS_TEST_ROOT, folder)
        img_paths = sorted(Path(class_dir).glob("*.jpg"))
        log_event("eval", logging.INFO, f"  {CLASSES[cls_idx]:12s}: {len(img_paths)} images de test",
                  event="CLS_TEST_COUNT", class_name=CLASSES[cls_idx], n=int(len(img_paths)))
        for img_path in img_paths:
            image = np.array(Image.open(img_path).convert("RGB"))
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                _, pooled = medsam_forward_joint(medsam_model, quantum_gate, image, [], has_mask=False, device=device)
                class_logits = classifier(pooled)
            pred_class = class_logits.argmax(dim=1).item()
            all_true.append(cls_idx); all_pred.append(pred_class)

    accuracy = float(np.mean([t == p for t, p in zip(all_true, all_pred)]))
    conf_mat = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(all_true, all_pred):
        conf_mat[t, p] += 1

    (per_class_acc_list, per_class_prec_list, per_class_rec_list,
     per_class_f1_list, macro_f1, bal_acc) = classification_metrics_from_confusion(conf_mat)

    log_event("eval", logging.INFO, f"\nAccuracy globale (4 classes) : {accuracy:.4f}", event="CLS_ACC", accuracy=float(accuracy))
    log_event("eval", logging.INFO, f"Macro-F1 (4 classes)         : {macro_f1:.4f}", event="CLS_MACRO_F1", macro_f1=float(macro_f1))
    log_event("eval", logging.INFO, f"Balanced Accuracy (4 classes): {bal_acc:.4f}", event="CLS_BAL_ACC", balanced_accuracy=float(bal_acc))
    log_event("eval", logging.INFO, "\nMatrice de confusion (lignes=vrai, colonnes=predit) :", event="CLS_CONFUSION_TABLE")
    log_event("eval", logging.INFO, "             " + "  ".join(f"{c[:8]:>8s}" for c in CLASSES), event="CLS_CONFUSION_TABLE")
    for i, c in enumerate(CLASSES):
        log_event("eval", logging.INFO, f"{c[:12]:12s} " + "  ".join(f"{conf_mat[i,j]:8d}" for j in range(N_CLASSES)),
                  event="CLS_CONFUSION_TABLE")

    per_class_acc = {CLASSES[i]: per_class_acc_list[i] for i in range(N_CLASSES)}
    per_class_precision = {CLASSES[i]: per_class_prec_list[i] for i in range(N_CLASSES)}
    per_class_recall = {CLASSES[i]: per_class_rec_list[i] for i in range(N_CLASSES)}
    per_class_f1 = {CLASSES[i]: per_class_f1_list[i] for i in range(N_CLASSES)}
    for i, cname in enumerate(CLASSES):
        log_event("eval", logging.INFO,
                  f"[PerClass] {cname}: precision={per_class_prec_list[i]:.4f} recall={per_class_rec_list[i]:.4f} "
                  f"f1={per_class_f1_list[i]:.4f} acc={per_class_acc_list[i]:.4f}",
                  event="CLS_PER_CLASS", class_name=cname,
                  precision=float(per_class_prec_list[i]), recall=float(per_class_rec_list[i]),
                  f1=float(per_class_f1_list[i]), acc=float(per_class_acc_list[i]))

    with open(os.path.join(RESULTS_DIR, "classification_test_results.json"), "w") as f:
        json.dump({"accuracy": accuracy, "macro_f1": macro_f1, "balanced_accuracy": bal_acc,
                   "per_class_accuracy": per_class_acc, "per_class_precision": per_class_precision,
                   "per_class_recall": per_class_recall, "per_class_f1": per_class_f1,
                   "confusion_matrix": conf_mat.tolist(), "classes": CLASSES}, f, indent=2)

    plot_confusion_and_histograms(conf_mat, accuracy)
    return {
        "accuracy": accuracy, "macro_f1": macro_f1, "balanced_accuracy": bal_acc,
        "per_class_accuracy": per_class_acc, "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall, "per_class_f1": per_class_f1,
        "confusion_matrix": conf_mat,
    }


def plot_training_curves(history):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d1117")
    epochs = range(1, len(history["val_dice"]) + 1)
    axes[0].plot(epochs, history["train_dice"], color="#2ea043", label="Train Dice")
    axes[0].plot(epochs, history["val_dice"], color="#3fb950", label="Val Dice")
    axes[0].plot(epochs, history["train_acc"], color="#a371f7", label="Train Acc")
    axes[0].plot(epochs, history["val_acc"], color="#d2a8ff", label="Val Acc")
    axes[0].set_title(f"Train vs Val — head={HEAD_MODE}", color="white"); axes[0].legend()
    axes[1].plot(epochs, history["train_seg_loss"], color="#1f6feb", label="Train Seg Loss")
    axes[1].plot(epochs, history["val_seg_loss"], color="#58a6ff", label="Val Seg Loss")
    axes[1].plot(epochs, history["train_cls_loss"], color="#ea6045", label="Train Cls Loss")
    axes[1].plot(epochs, history["val_cls_loss"], color="#f78166", label="Val Cls Loss")
    axes[1].set_title("Train vs Val Losses", color="white"); axes[1].legend()
    for ax in axes:
        ax.set_facecolor("#0d1117"); ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, facecolor="#0d1117")
    plt.close()


def plot_confusion_and_histograms(conf_mat, accuracy):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0d1117")

    axes[0].imshow(conf_mat, cmap="Purples")
    axes[0].set_xticks(range(N_CLASSES)); axes[0].set_yticks(range(N_CLASSES))
    axes[0].set_xticklabels(CLASSES, rotation=30, ha="right", color="white")
    axes[0].set_yticklabels(CLASSES, color="white")
    axes[0].set_xlabel("Predit", color="white"); axes[0].set_ylabel("Vrai", color="white")
    axes[0].set_title(f"Confusion — head={HEAD_MODE} (Acc={accuracy:.4f})", color="white")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            axes[0].text(j, i, str(conf_mat[i, j]), ha="center", va="center",
                         color="white" if conf_mat[i, j] < conf_mat.max() / 2 else "black", fontsize=12)

    per_class_acc = [conf_mat[i, i] / conf_mat[i].sum() if conf_mat[i].sum() > 0 else 0 for i in range(N_CLASSES)]
    colors_bar = ["#7B2FF7", "#2E86AB", "#3fb950", "#f78166"]
    bars = axes[1].bar(CLASSES, per_class_acc, color=colors_bar)
    for bar, a in zip(bars, per_class_acc):
        axes[1].text(bar.get_x() + bar.get_width() / 2, a + 0.01, f"{a:.3f}",
                     ha="center", color="white", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Accuracy de classification par classe", color="white")
    axes[1].set_facecolor("#0d1117"); axes[1].tick_params(colors="white")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "confusion_and_histograms.png")
    plt.savefig(out_path, dpi=150, facecolor="#0d1117")
    plt.close()
    log_event("eval", logging.INFO, f"[Figure] {out_path}", event="FIGURE_WRITTEN", path=out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_event("setup", logging.ERROR, f"Fatal error: {type(e).__name__}: {e}",
                  event="MAIN_EXCEPTION", traceback=traceback.format_exc())
        flush_log_handlers()
        raise
r"""
================================================================================
PARALLEL MULTI-VQC CLASSIFICATION HEAD
================================================================================
Architecture to avoid information loss:
  1. Linear 256 → 128 (50% compression, manageable)
  2. Split 128 into 16 chunks of 8
  3. 16 parallel VQC circuits (8 qubits each, 3 layers, angle embedding)
  4. Each circuit outputs 16 dims (8 <Z> + 8 <ZZ>)
  5. Concatenate: 16 * 16 = 256 dims
  6. Linear 256 → 4 classes

Information preservation:
  - 256 → 128: 50% loss (acceptable)
  - Each 8-dim chunk fully encoded in VQC (angle embedding, data re-uploading)
  - No per-circuit bottleneck
  - Parallel batched execution for speed
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from functools import partial


# ============================================================================
# QUANTUM BACKEND + CIRCUIT
# ============================================================================

def get_quantum_device_for_parallel(n_qubits=8, n_layers=3):
    """Get device for parallel VQC."""
    # For 8 qubits, default.qubit + backprop is safe and fast enough
    dev = qml.device("default.qubit", wires=n_qubits)
    return dev


def build_vqc_circuit_angle(n_qubits=8, n_layers=3):
    """
    VQC circuit with angle embedding and data re-uploading.
    
    Input: n_qubits values (typically 8)
    Output: n_qubits <Z> + n_qubits <ZZ> correlators = 2*n_qubits values
    """
    dev = get_quantum_device_for_parallel(n_qubits, n_layers)
    
    def _circuit(inputs, weights):
        # inputs: (n_qubits,) — data re-uploading at each layer
        # weights: (n_layers, n_qubits, 3) — Rot parameters
        for l in range(n_layers):
            # Re-upload data at each layer
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            # Parameterized rotations
            for q in range(n_qubits):
                qml.Rot(*weights[l, q], wires=q)
            # Ring CNOT entanglement
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        
        # Readout: single-qubit <Z> + two-qubit <ZZ> correlators
        z_terms = [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        zz_terms = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q+1) % n_qubits)) 
                    for q in range(n_qubits)]
        return z_terms + zz_terms
    
    qnode = qml.QNode(_circuit, dev, interface="torch", diff_method="backprop")
    return qnode


# ============================================================================
# PARALLEL VQC MODULE
# ============================================================================

