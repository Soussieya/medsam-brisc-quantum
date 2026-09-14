# ============================================================
# medsam_full_fine_tuned2.py
# MedSAM FULL FINE-TUNE v2 — MULTI-BBOX in TRAIN + VAL + TEST
# ============================================================
#
# Key difference vs medsam_full_fine_tune.py (v1):
#   v1 : training uses LARGEST bbox only (1 bbox per image)
#        test uses multi-bbox (one per connected component)
#        → model learns on noisy signal (global bbox includes empty space
#          between fragmented tumor regions), then corrects only at test time
#
#   v2 : training uses MULTI-BBOX from the start (one per connected component)
#        same as MedSAM v3 lesson: training on multi-bbox directly teaches
#        the decoder to segment each region precisely from epoch 1
#        → gradient flows through ALL bboxes simultaneously (differentiable
#          OR fusion: fused_prob = max(sigmoid(R1), sigmoid(R2), ...))
#        → the image encoder is called ONCE per image regardless of bbox count
#          (only the mask decoder is called once per connected component)
#
# Why this should improve Dice:
#   MedSAM v1 (global bbox) → v3 (multi-bbox): +0.0037 Dice on test set
#   This version applies the same principle to full fine-tune (93.7M params)
#   Expected gain: +0.002 to +0.005 Dice vs medsam_full_fine_tune.py (0.9390)
#   Estimated test Dice: 0.940 - 0.945
#
# Training strategy:
#   Image encoder (89.7M)  : TRAINED (LR=1e-5, low to avoid forgetting)
#   Prompt encoder (0.0M)  : TRAINED (LR=5e-5)
#   Mask decoder (4.06M)   : TRAINED (LR=1e-4)
#   Total: 93.7M / 93.7M = 100%
#   Batch size: 2 (reduced for 24GB VRAM)
#   Gradient accumulation: 2 steps (effective batch = 4)
#
# Run:
#   source /home/esoussi/stage/medsam_env/bin/activate
#   cd /home/esoussi/stage/MedSAM/
#   nohup python medsam_full_fine_tuned2.py > medsam_full_fine_tuned2.log 2>&1 &
#   echo "PID: $!"
#   tail -f medsam_full_fine_tuned2.log
# ============================================================

import os, sys, time, random, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MEDSAM_ROOT = "/home/esoussi/stage/MedSAM"
if MEDSAM_ROOT not in sys.path:
    sys.path.insert(0, MEDSAM_ROOT)
from segment_anything import sam_model_registry

# ============================================================
# CONFIG
# ============================================================
DEVICE       = torch.device("cuda:1")
DATA_ROOT    = "/home/esoussi/stage/Eya PFE/archive/brisc2025/segmentation_task"
CKPT_PATH    = "/home/esoussi/stage/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
SAVE_DIR     = "/home/esoussi/stage/MedSAM/full_finetune_v2_results"
CKPT_SAVE    = "/home/esoussi/stage/MedSAM/work_dir/MedSAM"
BEST_CKPT    = os.path.join(CKPT_SAVE, "medsam_full_v2_best.pth")
LAST_CKPT    = os.path.join(CKPT_SAVE, "medsam_full_v2_last.pth")

IMG_SIZE     = 1024
BBOX_MARGIN  = 10
MIN_AREA     = 50          # ignore components < 50px (noise)
BATCH_SIZE   = 2           # reduced for 24GB VRAM with full fine-tune
GRAD_ACCUM   = 2           # effective batch = BATCH_SIZE * GRAD_ACCUM = 4
NUM_EPOCHS   = 50
PATIENCE     = 10
VAL_RATIO    = 0.15
NUM_WORKERS  = 4
SEED         = 42

# Three LR groups — critical to avoid catastrophic forgetting
LR_ENCODER   = 1e-5        # very low — preserve MedSAM medical knowledge
LR_PROMPT    = 5e-5        # medium
LR_DECODER   = 1e-4        # standard — decoder learns fastest
WEIGHT_DECAY = 1e-4

# Reference values for figures
SOTA_DICE         = 0.948
MEDSAM_V3_DICE    = 0.9307
PROMPT_VQC_DICE   = 0.9323
MEDSAM_FULL_DICE  = 0.9390  # v1 result to beat
SABERNET_DICE     = 0.8881

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_SAVE, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",      "ytick.color": "#8b949e",
    "text.color": "white",         "grid.color": "#30363d",
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
    "legend.labelcolor": "white",
})

print("=" * 70)
print("  MedSAM FULL FINE-TUNE v2 — MULTI-BBOX in TRAIN + VAL + TEST")
print("  Strategy: Abbas et al. 2025 + MedSAM v3 multi-bbox lesson")
print("=" * 70)
print(f"  Device       : {DEVICE}")
print(f"  Batch size   : {BATCH_SIZE} (grad accum={GRAD_ACCUM}, effective={BATCH_SIZE*GRAD_ACCUM})")
print(f"  Epochs       : {NUM_EPOCHS} | Patience : {PATIENCE}")
print(f"  LR encoder   : {LR_ENCODER}  (low — no catastrophic forgetting)")
print(f"  LR decoder   : {LR_DECODER}")
print(f"  Multi-bbox   : YES — train + val + test (key difference vs v1)")
print("=" * 70)


# ============================================================
# MULTI-BBOX EXTRACTION
# ============================================================
def extract_multi_bboxes(mask_np, margin=BBOX_MARGIN, min_area=MIN_AREA):
    """
    Extract one bounding box per connected component.

    This is the MedSAM v3 lesson applied to full fine-tune:
    A single global bbox over two separate tumor regions includes the
    empty space between them, causing FP in the empty zone and FN on
    small regions. One bbox per component gives precise localization.

    Returns: list of [x1, y1, x2, y2] arrays, empty if no tumor.
    """
    if mask_np.sum() == 0:
        return []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_np.astype(np.uint8), connectivity=8)
    bboxes = []
    H, W = mask_np.shape
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT];  y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append(np.array([
            max(0,   x - margin),
            max(0,   y - margin),
            min(W-1, x + w + margin),
            min(H-1, y + h + margin),
        ], dtype=np.float32))
    return bboxes


def get_training_bboxes(mask_np):
    """
    For training/validation: always return at least one bbox so the
    forward pass has a gradient. If no tumor (No Tumor class), use a
    small dummy bbox at image center — model learns to predict empty mask.
    """
    bboxes = extract_multi_bboxes(mask_np)
    if not bboxes:
        cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
        bboxes = [np.array([cx-20, cy-20, cx+20, cy+20], dtype=np.float32)]
    return bboxes


def tumor_type_from_path(path_str):
    name = Path(path_str).stem
    if "_gl_" in name: return "Glioma"
    if "_me_" in name: return "Meningioma"
    if "_pi_" in name: return "Pituitary"
    if "_no_" in name: return "No Tumor"
    return "Unknown"


# ============================================================
# DATASET — returns list of bboxes per image
# ============================================================
class BRISCMultiBboxDataset(Dataset):
    """
    Returns bboxes as a Python list (one array per connected component).
    DataLoader uses a custom collate to handle variable-length bbox lists.

    Key difference vs v1 dataset:
        v1: returns single largest bbox as tensor
        v2: returns list of all component bboxes
    """
    def __init__(self, img_paths, mask_dir, augment=False, train_mode=True):
        self.img_paths  = [str(p) for p in img_paths]
        self.mask_dir   = Path(mask_dir)
        self.augment    = augment
        self.train_mode = train_mode  # True: dummy bbox for No Tumor; False: empty list

    def __len__(self):
        return len(self.img_paths)

    def _augment(self, image, mask):
        # Horizontal flip only — vertical flip is anatomically invalid for brain MRI
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask  = np.ascontiguousarray(mask[:, ::-1])
        # Small rotation ±15°
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w  = image.shape[:2]
            M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask  = cv2.warpAffine(mask,  M, (w, h),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        # Brightness jitter (scanner gain variation)
        if random.random() < 0.4:
            image = np.clip(
                image.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255
            ).astype(np.uint8)
        return image, mask

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = str(self.mask_dir / (Path(img_path).stem + ".png"))

        image    = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path).convert("L"))
        mask_bin = (mask_raw > 127).astype(np.uint8)

        image = cv2.resize(image,    (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask_bin, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # IMPORTANT: augment first, THEN recompute bboxes
            # (rotation/flip moves the connected components)
            image, mask = self._augment(image, mask)

        # Multi-bbox after augmentation
        if self.train_mode:
            bboxes = get_training_bboxes(mask)  # always >= 1 bbox
        else:
            bboxes = extract_multi_bboxes(mask)  # can be empty for No Tumor

        return {
            "image"    : torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1),
            "image_np" : image,                  # uint8 HWC for visualization
            "mask"     : torch.from_numpy(mask).float(),
            "bboxes"   : bboxes,                 # list of np.array([x1,y1,x2,y2])
            "img_path" : img_path,
        }


def collate_multi_bbox(batch):
    """
    Custom collate: stack images and masks normally,
    but keep bboxes as a list of lists (variable length per image).
    """
    images   = torch.stack([b["image"] for b in batch])
    masks    = torch.stack([b["mask"]  for b in batch])
    bboxes   = [b["bboxes"]   for b in batch]
    img_paths= [b["img_path"] for b in batch]
    images_np= [b["image_np"] for b in batch]
    return {
        "image"    : images,
        "image_np" : images_np,
        "mask"     : masks,
        "bboxes"   : bboxes,
        "img_path" : img_paths,
    }


# ============================================================
# LOSS & METRICS
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        tgts  = targets.view(-1)
        inter = (probs * tgts).sum()
        return 1.0 - (2.0*inter + self.smooth) / (probs.sum() + tgts.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.bce(logits, targets)


def compute_metrics(pred_bin, gt_bin, eps=1e-6):
    p  = pred_bin.float().view(-1)
    g  = gt_bin.float().view(-1)
    TP = (p * g).sum(); FP = (p*(1-g)).sum()
    FN = ((1-p)*g).sum(); TN = ((1-p)*(1-g)).sum()
    return {
        "dice"        : ((2*TP+eps)/(2*TP+FP+FN+eps)).item(),
        "iou"         : ((TP+eps)/(TP+FP+FN+eps)).item(),
        "precision"   : ((TP+eps)/(TP+FP+eps)).item(),
        "recall"      : ((TP+eps)/(TP+FN+eps)).item(),
        "specificity" : ((TN+eps)/(TN+FP+eps)).item(),
    }


# ============================================================
# MULTI-BBOX FORWARD PASS — differentiable OR fusion
# ============================================================
def forward_multi_bbox(sam, images, bboxes_per_image):
    """
    Forward pass with multi-bbox and differentiable OR fusion.

    For each image in the batch:
        1. Image encoder called ONCE (shared across all bboxes)
        2. Mask decoder called ONCE PER connected component
        3. Probabilities fused by OR: fused_prob = max(sig(R1), sig(R2), ...)
        4. Fused prob converted back to logits for loss computation

    This fusion is differentiable (torch.maximum) so gradients flow
    correctly through ALL bboxes back to the encoder.

    Dimension flow per image:
        image:        [3, 1024, 1024]
        img_emb:      [1, 256, 64, 64]  (computed once)
        bbox_i:       [4]               (x1, y1, x2, y2)
        sparse_emb_i: [1, 2, 256]
        dense_emb_i:  [1, 256, 64, 64]
        logits_i:     [1, 1, 256, 256]
        fused_prob:   [1, 1, 256, 256]  (max over all component probs)
        fused_logit:  [1, 1, 256, 256]  (back to logits for loss)
    """
    batch_logits = []

    for i in range(images.shape[0]):
        img   = images[i:i+1]           # [1, 3, 1024, 1024]
        bboxes = bboxes_per_image[i]    # list of np arrays

        # Image encoder — called ONCE per image
        img_emb = sam.image_encoder(img)  # [1, 256, 64, 64]

        # Initialize fused probability map
        fused_prob = torch.zeros(1, 1, 256, 256, device=DEVICE)

        for bbox_np in bboxes:
            bbox_t = torch.from_numpy(
                np.array(bbox_np, dtype=np.float32).reshape(4)
            ).to(DEVICE).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None, boxes=bbox_t, masks=None)

            logits_i, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )  # [1, 1, 256, 256]

            # OR fusion — differentiable via torch.maximum
            fused_prob = torch.maximum(fused_prob, torch.sigmoid(logits_i))

        # Convert fused probability back to logits for loss computation
        fused_prob  = torch.clamp(fused_prob, 1e-6, 1 - 1e-6)
        fused_logit = torch.log(fused_prob / (1 - fused_prob))  # [1, 1, 256, 256]
        batch_logits.append(fused_logit.squeeze(0))             # [1, 256, 256]

    return torch.stack(batch_logits, dim=0)  # [B, 1, 256, 256]


@torch.no_grad()
def forward_multi_bbox_test(sam, image_single, bboxes_list):
    """Test-only forward (no gradient needed)."""
    if not bboxes_list:
        return torch.full((1, 1, 256, 256), -10.0, device=DEVICE)

    img_emb    = sam.image_encoder(image_single)
    fused_prob = torch.zeros(1, 1, 256, 256, device=DEVICE)

    for bbox_np in bboxes_list:
        bbox_t = torch.from_numpy(
            np.array(bbox_np, dtype=np.float32).reshape(4)
        ).to(DEVICE).unsqueeze(0).unsqueeze(0)

        sparse_emb, dense_emb = sam.prompt_encoder(
            points=None, boxes=bbox_t, masks=None)

        logits_i, _ = sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        fused_prob = torch.maximum(fused_prob, torch.sigmoid(logits_i))

    fused_prob = torch.clamp(fused_prob, 1e-6, 1 - 1e-6)
    return torch.log(fused_prob / (1 - fused_prob))


# ============================================================
# MODEL LOADING
# ============================================================
def load_medsam_full(ckpt_path):
    print("Loading MedSAM ViT-B (full fine-tune) ...")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    sam.to(DEVICE)

    # Unfreeze ALL parameters — 100% trainable
    for p in sam.parameters():
        p.requires_grad = True

    n_enc = sum(p.numel() for p in sam.image_encoder.parameters())
    n_pmt = sum(p.numel() for p in sam.prompt_encoder.parameters())
    n_dec = sum(p.numel() for p in sam.mask_decoder.parameters())
    n_tot = n_enc + n_pmt + n_dec

    print(f"  Image encoder  : {n_enc/1e6:.1f}M  LR={LR_ENCODER}")
    print(f"  Prompt encoder : {n_pmt/1e6:.2f}M  LR={LR_PROMPT}")
    print(f"  Mask decoder   : {n_dec/1e6:.2f}M  LR={LR_DECODER}")
    print(f"  TOTAL trainable: {n_tot/1e6:.1f}M / {n_tot/1e6:.1f}M (100%)")
    return sam


# ============================================================
# TRAIN / VAL
# ============================================================
def train_one_epoch(sam, loader, optimizer, loss_fn, scaler):
    sam.train()
    total_loss = 0.0; dice_list = []
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        images    = batch["image"].to(DEVICE)
        masks_gt  = batch["mask"].to(DEVICE)        # [B, 1024, 1024]
        bboxes_per = batch["bboxes"]                # list of lists

        with autocast(device_type="cuda"):
            logits = forward_multi_bbox(sam, images, bboxes_per)  # [B, 1, 256, 256]
            logits = logits.squeeze(1)              # [B, 256, 256]

            # Resize GT mask to match logit resolution (256x256)
            masks_256 = F.interpolate(
                masks_gt.unsqueeze(1).float(),
                size=(256, 256), mode="nearest"
            ).squeeze(1)

            loss = loss_fn(logits, masks_256) / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sam.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).long()
            for p, g in zip(preds, masks_256.long()):
                dice_list.append(compute_metrics(p, g)["dice"])

    return total_loss / len(loader), float(np.mean(dice_list))


@torch.no_grad()
def validate(sam, loader, loss_fn):
    sam.eval()
    total_loss = 0.0; all_metrics = []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        images    = batch["image"].to(DEVICE)
        masks_gt  = batch["mask"].to(DEVICE)
        bboxes_per = batch["bboxes"]

        with autocast(device_type="cuda"):
            logits = forward_multi_bbox(sam, images, bboxes_per).squeeze(1)
            masks_256 = F.interpolate(
                masks_gt.unsqueeze(1).float(), size=(256, 256), mode="nearest"
            ).squeeze(1)
            loss = loss_fn(logits, masks_256)

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).long()
        for p, g in zip(preds, masks_256.long()):
            all_metrics.append(compute_metrics(p, g))

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    return total_loss / len(loader), avg


@torch.no_grad()
def evaluate_test(sam, loader):
    sam.eval()
    results = []
    n_multi = 0

    for batch in tqdm(loader, desc="  Test "):
        image     = batch["image"].to(DEVICE)       # [1, 3, 1024, 1024]
        mask_1024 = batch["mask"][0]                # [1024, 1024]
        img_path  = batch["img_path"][0]

        mask_np     = mask_1024.numpy().astype(np.uint8)
        bboxes_list = extract_multi_bboxes(mask_np)  # real test behavior (no dummy)

        if len(bboxes_list) >= 2:
            n_multi += 1

        with autocast(device_type="cuda"):
            fused_logit = forward_multi_bbox_test(sam, image, bboxes_list)

        logit_1024 = F.interpolate(
            fused_logit, size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze()
        pred_bin = (torch.sigmoid(logit_1024) > 0.5).long().cpu()

        m = compute_metrics(pred_bin, mask_1024.long())
        m["img_path"]   = img_path
        m["tumor_type"] = tumor_type_from_path(img_path)
        m["pred"]       = pred_bin.numpy().astype(np.uint8)
        m["gt"]         = mask_np
        m["image"]      = batch["image"][0].permute(1,2,0).cpu().numpy()
        m["bboxes"]     = bboxes_list
        m["n_bboxes"]   = len(bboxes_list)
        results.append(m)

    print(f"\n  Multi-region images (>=2 bboxes): {n_multi} / {len(results)}")
    return results


# ============================================================
# DATALOADERS
# ============================================================
def build_loaders():
    train_img_dir  = os.path.join(DATA_ROOT, "train", "images")
    train_mask_dir = os.path.join(DATA_ROOT, "train", "masks")
    test_img_dir   = os.path.join(DATA_ROOT, "test",  "images")
    test_mask_dir  = os.path.join(DATA_ROOT, "test",  "masks")

    all_train = sorted(Path(train_img_dir).glob("*.jpg"))
    random.shuffle(all_train)
    n_val      = int(len(all_train) * VAL_RATIO)
    val_imgs   = all_train[:n_val]
    train_imgs = all_train[n_val:]
    test_imgs  = sorted(Path(test_img_dir).glob("*.jpg"))

    print(f"  Train: {len(train_imgs)}  |  Val: {len(val_imgs)}  |  Test: {len(test_imgs)}")

    kw = dict(collate_fn=collate_multi_bbox, pin_memory=True)
    return (
        DataLoader(BRISCMultiBboxDataset(train_imgs, train_mask_dir, augment=True,  train_mode=True),
                   batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, **kw),
        DataLoader(BRISCMultiBboxDataset(val_imgs,   train_mask_dir, augment=False, train_mode=True),
                   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, **kw),
        DataLoader(BRISCMultiBboxDataset(test_imgs,  test_mask_dir,  augment=False, train_mode=False),
                   batch_size=1,          shuffle=False, num_workers=NUM_WORKERS, **kw),
    )


# ============================================================
# VISUALISATION
# ============================================================
def plot_curves(history, best_dice, save_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.patch.set_facecolor("#0d1117")

    REFS_DICE = [
        (SOTA_DICE,        "#3fb950", "--", f"SOTA {SOTA_DICE}"),
        (MEDSAM_FULL_DICE, "#f78166", "-.", f"MedSAM FULL v1 {MEDSAM_FULL_DICE}"),
        (PROMPT_VQC_DICE,  "#bc8cff", ":",  f"PROMPT-VQC {PROMPT_VQC_DICE}"),
        (MEDSAM_V3_DICE,   "#ffa657", ":",  f"MedSAM v3 {MEDSAM_V3_DICE}"),
    ]
    REFS_IOU = [
        (0.903,  "#3fb950", "--", "SOTA 0.903"),
        (0.8875, "#f78166", "-.", "MedSAM FULL v1 0.8875"),
    ]

    def _panel(ax, tr_k, vl_k, title, refs):
        ax.set_facecolor("#161b22")
        ax.plot(epochs, history[tr_k], color="#58a6ff", lw=2.2,
                marker="o", markersize=2.5, label="Train")
        ax.plot(epochs, history[vl_k], color="#ffa657", lw=2.2,
                marker="s", markersize=2.5, linestyle="--", label="Val")
        ax.fill_between(epochs, history[tr_k], history[vl_k],
                        alpha=0.07, color="#ffa657")
        for rv, rc, rls, rl in refs:
            ax.axhline(rv, color=rc, lw=1.2, linestyle=rls, alpha=0.7, label=rl)
        if title == "Dice":
            # Overfitting indicator
            gap = [t - v for t, v in zip(history[tr_k], history[vl_k])]
            if max(gap) > 0.02:
                peak_ep = gap.index(max(gap)) + 1
                ax.axvline(peak_ep, color="#f78166", lw=1, linestyle=":", alpha=0.5)
                ax.text(peak_ep, min(history[vl_k]),
                        f"  gap={max(gap):.3f}", color="#f78166", fontsize=7)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.legend(fontsize=7); ax.grid(alpha=0.2)
        for s in ax.spines.values(): s.set_edgecolor("#30363d")

    _panel(axes[0], "train_loss", "val_loss", "Loss", [])
    _panel(axes[1], "train_dice", "val_dice", "Dice", REFS_DICE)
    _panel(axes[2], "train_iou",  "val_iou",  "IoU",  REFS_IOU)

    ax = axes[3]; ax.set_facecolor("#161b22")
    if history.get("val_precision"):
        ax.plot(epochs, history["val_precision"], color="#bc8cff", lw=2.2,
                marker="^", markersize=3, label="Val Precision")
        ax.plot(epochs, history["val_recall"],    color="#00d4aa", lw=2.2,
                marker="v", markersize=3, linestyle="--", label="Val Recall")
        ax.fill_between(epochs, history["val_precision"], history["val_recall"],
                        alpha=0.07, color="#bc8cff")
        ax.legend(fontsize=8)
    ax.set_title("Precision & Recall", color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.grid(alpha=0.2)
    for s in ax.spines.values(): s.set_edgecolor("#30363d")

    plt.suptitle(
        f"MedSAM FULL FINE-TUNE v2 (Multi-BBox Train+Val+Test) — BRISC 2025  |  "
        f"Best Val Dice: {best_dice:.4f}",
        color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Curves saved: {save_path}")


def plot_qualitative(results, save_path):
    TUMOR_TYPES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    TYPE_COLORS = {"Glioma":"#f78166","Meningioma":"#3fb950","Pituitary":"#58a6ff","No Tumor":"#e5c07b"}
    BBOX_COLORS = ["cyan","#ffa657","#3fb950","#bc8cff"]
    by_type = {t:[r for r in results if r["tumor_type"]==t] for t in TUMOR_TYPES}

    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    fig.patch.set_facecolor("#0d1117")
    for col, t in enumerate(["Image MRI + BBox(es)","Ground Truth","Prediction","Overlay TP/FN/FP"]):
        axes[0,col].set_title(t, color="white", fontsize=11, fontweight="bold", pad=8)

    alpha = 0.45
    for row, ttype in enumerate(TUMOR_TYPES):
        cands = by_type[ttype]
        if not cands:
            for col in range(4): axes[row,col].axis("off")
            axes[row,0].text(0.5,0.5,f"No {ttype}", transform=axes[row,0].transAxes,
                             color="white", ha="center", va="center")
            continue

        s = max(random.sample(cands, min(5,len(cands))), key=lambda r: r["dice"])
        img_d=np.clip(s["image"],0,1); gt=s["gt"]; pred=s["pred"]
        ov=img_d.copy()
        TP=(gt==1)&(pred==1); FN=(gt==1)&(pred==0); FP=(gt==0)&(pred==1)
        ov[TP]=ov[TP]*(1-alpha)+np.array([0.,1.,0.])*alpha
        ov[FN]=ov[FN]*(1-alpha)+np.array([1.,0.,0.])*alpha
        ov[FP]=ov[FP]*(1-alpha)+np.array([0.,.4,1.])*alpha
        ov=np.clip(ov,0,1); border=TYPE_COLORS[ttype]

        for col,(data,cmap,vmin,vmax) in enumerate([
            (img_d,None,None,None),(gt,"gray",0,1),(pred,"gray",0,1),(ov,None,None,None)]):
            ax=axes[row,col]; ax.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax); ax.axis("off")
            for sp in ax.spines.values(): sp.set_edgecolor(border); sp.set_linewidth(2.5); sp.set_visible(True)

        ax=axes[row,0]
        for bi,bn in enumerate(s["bboxes"]):
            arr=np.array(bn,dtype=np.float32).reshape(4); x1,y1,x2,y2=arr.astype(int)
            c=BBOX_COLORS[bi%len(BBOX_COLORS)]
            ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,lw=2,edgecolor=c,facecolor="none",linestyle="--"))
            ax.text(x1,max(0,y1-8),f"R{bi+1}",color=c,fontsize=8,fontweight="bold")

        axes[row,0].set_ylabel(ttype,color=border,fontsize=11,fontweight="bold",rotation=90,labelpad=8)
        axes[row,3].set_xlabel(
            f"Dice={s['dice']:.3f}  IoU={s['iou']:.3f}  "
            f"Prec={s['precision']:.3f}  Rec={s['recall']:.3f}  ({s['n_bboxes']} bbox)",
            color="#00d4aa",fontsize=9,labelpad=6)

    fig.legend(handles=[
        mpatches.Patch(color="#00ff00",alpha=0.8,label="True Positive (TP)"),
        mpatches.Patch(color="#ff0000",alpha=0.8,label="False Negative (FN)"),
        mpatches.Patch(color="#0066ff",alpha=0.8,label="False Positive (FP)")],
        loc="lower center",ncol=3,fontsize=10,bbox_to_anchor=(0.5,-0.01))
    plt.suptitle("MedSAM FULL v2 (Multi-BBox) — Qualitative Results | BRISC 2025",
                 color="white",fontsize=13,fontweight="bold")
    plt.tight_layout(rect=[0,0.04,1,1])
    plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#0d1117")
    plt.close()
    print(f"  Qualitative saved: {save_path}")


def plot_metrics_summary(results, dice_mu, iou_mu, dice_sd, iou_sd, save_path):
    dice_vals   = [r["dice"] for r in results]
    TUMOR_TYPES = ["Glioma","Meningioma","Pituitary","No Tumor"]
    TYPE_COLORS = ["#f78166","#3fb950","#58a6ff","#e5c07b"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#0d1117")

    ax=axes[0]; ax.set_facecolor("#161b22")
    n,bins,patches=ax.hist(dice_vals,bins=35,edgecolor="black",linewidth=0.5)
    for patch,left in zip(patches,bins[:-1]):
        if left>=0.92: patch.set_facecolor("#3fb950")
        elif left>=0.80: patch.set_facecolor("#ffa657")
        else: patch.set_facecolor("#f78166")
    ax.axvline(dice_mu,color="white",lw=2,linestyle="--",label=f"Mean={dice_mu:.3f}")
    ax.axvline(MEDSAM_FULL_DICE,color="#f78166",lw=1.3,linestyle="-.",label=f"v1 {MEDSAM_FULL_DICE}")
    ax.axvline(SOTA_DICE,color="#3fb950",lw=1.3,linestyle=":",label=f"SOTA {SOTA_DICE}")
    ax.set_title("Dice Distribution (test set)",color="white",fontsize=12,fontweight="bold")
    ax.set_xlabel("Dice Score"); ax.set_ylabel("Number of images")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    for s in ax.spines.values(): s.set_edgecolor("#30363d")

    ax=axes[1]; ax.set_facecolor("#161b22")
    td=[float(np.mean([r["dice"] for r in results if r["tumor_type"]==t] or [0])) for t in TUMOR_TYPES]
    bars=ax.bar(TUMOR_TYPES,td,color=TYPE_COLORS,edgecolor="black",linewidth=0.8,width=0.6)
    ax.set_ylim(0,1.05)
    for bar,val in zip(bars,td):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f"{val:.3f}",
                ha="center",va="bottom",color="white",fontsize=10,fontweight="bold")
    ax.set_title("Dice by Tumor Type",color="white",fontsize=12,fontweight="bold")
    ax.grid(alpha=0.2,axis="y")
    for s in ax.spines.values(): s.set_edgecolor("#30363d")

    ax=axes[2]; ax.set_facecolor("#161b22")
    models=["SABERNet\nauto","MedSAM v3\npartial","MedSAM FULL\nv1","MedSAM FULL\nv2 (this)","SOTA\nAbbas 2025"]
    dices=[SABERNET_DICE,MEDSAM_V3_DICE,MEDSAM_FULL_DICE,dice_mu,SOTA_DICE]
    ious= [0.8123,0.8735,0.8875,iou_mu,0.903]
    colors=["#e5c07b","#79c7e3","#f78166","#3fb950","#2a78d6"]
    x=np.arange(len(models)); w=0.35
    b1=ax.bar(x-w/2,dices,w,color=colors,edgecolor="black",linewidth=0.8,label="Dice",alpha=0.9)
    b2=ax.bar(x+w/2,ious, w,color=colors,edgecolor="black",linewidth=0.8,label="IoU", alpha=0.6)
    for bar in [b1[3],b2[3]]: bar.set_edgecolor("cyan"); bar.set_linewidth(2.5)
    ax.set_ylim(0.75,1.02); ax.set_xticks(x); ax.set_xticklabels(models,fontsize=8)
    for bar,val in zip(list(b1)+list(b2),dices+ious):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f"{val:.3f}",
                ha="center",va="bottom",color="white",fontsize=7)
    ax.set_title("Model Comparison",color="white",fontsize=12,fontweight="bold")
    ax.set_ylabel("Score"); ax.legend(fontsize=9); ax.grid(alpha=0.2,axis="y")
    for s in ax.spines.values(): s.set_edgecolor("#30363d")

    plt.suptitle(
        f"MedSAM FULL v2 (Multi-BBox) — Dice={dice_mu:.4f}±{dice_sd:.4f}  IoU={iou_mu:.4f}±{iou_sd:.4f}",
        color="white",fontsize=12,fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#0d1117")
    plt.close()
    print(f"  Metrics saved: {save_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\nBuilding data loaders ...")
    train_loader, val_loader, test_loader = build_loaders()

    print("\nInitializing MedSAM FULL FINE-TUNE v2 ...")
    sam = load_medsam_full(CKPT_PATH)

    optimizer = torch.optim.Adam([
        {"params": sam.image_encoder.parameters(),  "lr": LR_ENCODER},
        {"params": sam.prompt_encoder.parameters(), "lr": LR_PROMPT},
        {"params": sam.mask_decoder.parameters(),   "lr": LR_DECODER},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    loss_fn   = CombinedLoss().to(DEVICE)
    scaler    = GradScaler(device="cuda")

    best_dice   = 0.0; patience_ct = 0; start_epoch = 1
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou" : [], "val_iou" : [],
        "val_precision": [], "val_recall": [],
    }

    if os.path.exists(LAST_CKPT):
        print(f"\nResuming from {LAST_CKPT} ...")
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE, weights_only=False)
        sam.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt["best_dice"]
        history     = ckpt["history"]
        patience_ct = ckpt.get("patience_ct", 0)
        print(f"  Resuming epoch {ckpt['epoch']}  |  Best Dice: {best_dice:.4f}")
    else:
        print("\nStarting new training ...\n")

    print(f"\n{'='*70}")
    print(f"  MedSAM FULL v2 — {NUM_EPOCHS} epochs | MULTI-BBOX TRAIN+VAL+TEST")
    print(f"  100% params | grad_accum={GRAD_ACCUM} | effective batch={BATCH_SIZE*GRAD_ACCUM}")
    print(f"{'='*70}")

    t_start = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[2]["lr"]
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}  "
              f"[LR encoder={lr_enc:.2e} | decoder={lr_dec:.2e}]")

        tr_loss, tr_dice = train_one_epoch(sam, train_loader, optimizer, loss_fn, scaler)
        vl_loss, vl_m    = validate(sam, val_loader, loss_fn)
        scheduler.step()

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_dice); history["val_dice"].append(vl_m["dice"])
        history["train_iou"].append(tr_dice);  history["val_iou"].append(vl_m["iou"])
        history["val_precision"].append(vl_m["precision"])
        history["val_recall"].append(vl_m["recall"])

        gap = tr_dice - vl_m["dice"]
        overfit_warn = "  ⚠ gap > 0.03 — overfitting" if gap > 0.03 else ""

        is_best = vl_m["dice"] > best_dice
        if is_best: best_dice = vl_m["dice"]; patience_ct = 0
        else: patience_ct += 1

        print(f"  Train  loss={tr_loss:.4f}  dice={tr_dice:.4f}")
        print(f"  Val    loss={vl_loss:.4f}  dice={vl_m['dice']:.4f}  "
              f"iou={vl_m['iou']:.4f}  prec={vl_m['precision']:.4f}  "
              f"rec={vl_m['recall']:.4f}  gap={gap:+.4f}{overfit_warn}")
        print(f"  Best={best_dice:.4f}  Patience={patience_ct}/{PATIENCE}"
              + ("  ★ [BEST SAVED]" if is_best else ""))

        ckpt = {
            "epoch": epoch, "model_state": sam.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_dice": best_dice, "patience_ct": patience_ct, "history": history,
        }
        torch.save(ckpt, LAST_CKPT)
        if is_best: torch.save(ckpt, BEST_CKPT)
        if patience_ct >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}"); break

    elapsed = (time.time() - t_start) / 60
    print(f"\nTraining completed in {elapsed:.1f} min | Best Val Dice: {best_dice:.4f}")

    best_data = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
    sam.load_state_dict(best_data["model_state"])
    print(f"  Best checkpoint: epoch {best_data['epoch']}")

    plot_curves(history, best_dice,
                os.path.join(SAVE_DIR, "full_v2_curves.png"))

    print("\nEvaluating on test set (860 images, multi-bbox) ...")
    results = evaluate_test(sam, test_loader)

    dice_vals = [r["dice"] for r in results]
    iou_vals  = [r["iou"]  for r in results]
    dice_mu, dice_sd = float(np.mean(dice_vals)), float(np.std(dice_vals))
    iou_mu,  iou_sd  = float(np.mean(iou_vals)),  float(np.std(iou_vals))
    prec_mu  = float(np.mean([r["precision"]   for r in results]))
    rec_mu   = float(np.mean([r["recall"]      for r in results]))
    spec_mu  = float(np.mean([r["specificity"] for r in results]))

    print("\n" + "=" * 70)
    print("  RESULTS — MedSAM FULL FINE-TUNE v2 (Multi-BBox) — BRISC 2025")
    print("=" * 70)
    print(f"  Dice        : {dice_mu:.4f} ± {dice_sd:.4f}")
    print(f"  IoU         : {iou_mu:.4f} ± {iou_sd:.4f}")
    print(f"  Precision   : {prec_mu:.4f}")
    print(f"  Recall      : {rec_mu:.4f}")
    print(f"  Specificity : {spec_mu:.4f}")
    print(f"  Avg bboxes  : {np.mean([r['n_bboxes'] for r in results]):.2f} / image")
    print("-" * 70)
    print(f"  MedSAM FULL v1 (largest bbox in train) : Dice=0.9390")
    print(f"  MedSAM FULL v2 (multi-bbox in train)   : Dice={dice_mu:.4f}  ← THIS RUN")
    print(f"  SOTA Abbas et al. 2025                 : Dice=0.9480")
    print("=" * 70)

    plot_qualitative(results, os.path.join(SAVE_DIR, "full_v2_qualitative.png"))
    plot_metrics_summary(results, dice_mu, iou_mu, dice_sd, iou_sd,
                         os.path.join(SAVE_DIR, "full_v2_metrics.png"))

    print(f"\nAll results in: {SAVE_DIR}/")
    print(f"  Best checkpoint: {BEST_CKPT}")

