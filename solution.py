"""
Solar Filament Segmentation Challenge 2026 — solution pipeline
==============================================================
Competition : https://www.kaggle.com/competitions/filament-segmentation-2026
Dataset     : MAGFiLO v1.0 (Ahmadzadeh et al., Scientific Data, 2024,
              doi:10.1038/s41597-024-03876-y)
Metric      : Panoptic Quality (PQ = SQ x RQ, IoU matching at 0.5)
              (Kirillov et al., CVPR 2019)

Usage
-----
    python solution.py train  --fold 0                 # train one GroupKFold split
    python solution.py eval   --fold 0                 # local PQ of a trained fold
    python solution.py tune   --fold 0                 # grid-search post-proc on local PQ
    python solution.py submit --ckpts runs/fold0.pt runs/fold1.pt ...

Method summary
--------------
1. Anti-leakage split ....... GroupKFold on the *physical* observation name: the
                               same image is annotated by 2-3 independent annotators
                               in the COCO file, so splitting by COCO id leaks
                               near-duplicates across folds.
2. Limb mask ................ annotations only exist within +/-70 deg of the
                               central meridian -> masked in the loss and zeroed
                               at inference.
3. Backbone ................. timm encoder (features_only) U-Net, additive
                               attention gates on skip connections (Oktay et al.,
                               2018), dilated residual bottleneck (zero-init, so
                               it starts as the identity map).
4. Training target .......... segmentation masks only, as required by the
                               competition restriction on other ground-truth metadata.
5. Loss ..................... BCE + soft-Dice + soft-clDice (Shit et al., CVPR
                               2021; connectivity-aware — plain Dice is blind to
                               a filament predicted as disjoint pieces).
6. Training ................. filament-centred crops, dihedral augmentations
                               (flips/rot90 are physically valid: the disk is
                               isotropic), AMP, cosine LR, weight EMA.
7. Inference ................ overlap-add sliding window, 8-way dihedral TTA,
                               optional ensembling of several fold checkpoints.
8. Instances ................ distance-transform watershed (splits touching
                               filaments — the classic PQ killer), then GT-style
                               cleanup: fill holes, keep largest component, drop
                               small objects (<5% islands are dropped in MAGFiLO).
9. Threshold tuning ......... small grid searched on LOCAL Panoptic Quality —
                               the only metric that mirrors the leaderboard.

Reproduction: run `train`, `eval`, `tune`, and `submit` in sequence, or execute
the notebook pipeline. The `--limit` option caps smoke-test train and validation
images and caps training to two epochs.
"""

import argparse
import json
import math
import random
import time
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as maskutils
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from PIL import Image
from scipy import ndimage
from skimage.draw import line as draw_line
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes
from skimage.segmentation import watershed
from sklearn.model_selection import GroupKFold

# --------------------------------------------------------------------------- #
# 1. CONFIG
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    # paths
    data_dir:   str = "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    train_json: str = "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    test_dir:   str = "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    work_dir:   str = "/kaggle/working/runs"
    progress_log: str = ""
    # data
    img_size:     int = 2048    # fixed by the competition for every image
    crop_size:    int = 768     # training crops
    oversample_p: float = 0.75  # prob. a crop is centred on filament pixels
    disk_radius:  int = 950     # approx. solar-disk radius (images are centred)
    limb_max_deg: float = 70.0  # nothing is annotated beyond +-70 deg
    limb_geometry: str = "longitude"  # B0=0 projected longitude; legacy_wedge for audits
    # model
    encoder:      str = "resnet18"
    pretrained:   bool = False
    decoder_ch:   tuple = (256, 128, 64, 32, 16)
    bn_ch:        int = 256
    bn_dilations: tuple = (1, 2, 4)
    # optimisation
    epochs:       int = 60
    eval_every:   int = 2
    batch_size:   int = 1
    grad_accum_steps: int = 8
    num_workers:  int = 0
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    amp:          bool = True
    ema_decay:    float = 0.999
    init_ckpt: str = ""  # warm start, not an optimizer resume
    eval_max_images: int = 0  # fixed random monitoring subset; 0 means full fold
    preview_count: int = 3
    # loss weights
    w_bce:   float = 0.3
    w_dice:  float = 0.3
    w_cld:   float = 0.4
    w_spine: float = 0.0
    pos_weight_cap: float = 5.0
    # inference / post-processing
    tta:          bool = True
    tta_batch_size: int = 2
    window:       int = 1024
    overlap:      int = 256
    thresh:       float = 0.5
    min_area:     int = 50
    min_dist_ws:  int = 25
    close_kernel: int = 3
    instance_method: str = "components"
    # tuning grid (local PQ)
    tune_thresh:  tuple = (0.3, 0.4, 0.5, 0.6, 0.7)
    tune_mdist:   tuple = (15, 25, 40)
    tune_marea:   tuple = (30, 50, 100)
    # validation
    n_folds: int = 5
    fold:    int = 0
    seed:    int = 2026
    limit:   int = 0      # >0: cap train/val images (smoke tests)
    device:  str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# 2. DATA
# --------------------------------------------------------------------------- #
def base_name(file_name: str) -> str:
    """Physical observation name, e.g. '20260901165702Bh' from
    '010101-20260901165702Bh.jpeg' — the anti-leakage grouping key."""
    return Path(file_name).stem.split("-")[-1]


def polygon_to_mask(ann, h, w):
    """COCO polygon -> bool mask. MAGFiLO stores exactly one polygon per filament."""
    rle = maskutils.frPyObjects(ann["segmentation"], h, w)
    decoded = maskutils.decode(rle)
    return decoded.any(axis=2) if decoded.ndim == 3 else decoded.astype(bool)


def spine_to_distmap(ann, h, w):
    """Euclidean distance of every pixel to the filament spine (a polyline).
    Rasterise the spine, then EDT on the complement. Returned values are later
    log-squashed by the dataset (long tail)."""
    spine = np.asarray(ann["spine"], dtype=np.float32).reshape(-1, 2)
    img = np.zeros((h, w), dtype=bool)
    xs, ys = spine[:, 0].astype(int), spine[:, 1].astype(int)
    for (x0, y0, x1, y1) in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        rr, cc = draw_line(y0, x0, y1, x1)          # skimage: (row, col) = (y, x)
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        img[rr[ok], cc[ok]] = True
    if img.sum() == 0:                              # degenerate spine -> bbox centre
        x, y, bw, bh = ann["bbox"]
        img[min(h - 1, int(y + bh / 2)), min(w - 1, int(x + bw / 2))] = True
    return ndimage.distance_transform_edt(~img).astype(np.float32)


class FilamentDataset(torch.utils.data.Dataset):
    """Crops for training (x, mask, auxiliary placeholder, valid-mask); full-image access
    via _load_full() for validation/inference."""

    def __init__(self, coco: COCO, img_ids, cfg: Cfg, train: bool, limb: np.ndarray):
        self.coco, self.ids, self.cfg, self.train, self.limb = coco, img_ids, cfg, train, limb
        self._cache = {}

    def __len__(self):
        return len(self.ids)

    def _load_full(self, img_id):
        if img_id in self._cache:
            return self._cache[img_id]
        info = self.coco.loadImgs([img_id])[0]
        h, w = info["height"], info["width"]
        img = np.asarray(Image.open(Path(self.cfg.data_dir) / "train_images" / info["file_name"]),
                         dtype=np.float32) / 255.0
        m = np.zeros((h, w), bool)
        d = np.zeros((h, w), np.float32)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id])):
            m |= polygon_to_mask(ann, h, w)
        if len(self._cache) > 16:                      # bound RAM (2048^2 floats)
            self._cache.clear()
        self._cache[img_id] = (img, m, d)
        return img, m, d

    def __getitem__(self, i):
        img, m, d = self._load_full(self.ids[i])
        H, W = img.shape
        c = self.cfg.crop_size
        if c > min(H, W):
            raise ValueError("crop_size exceeds image dimensions")
        if self.train and random.random() < self.cfg.oversample_p and (m & self.limb).any():
            ys, xs = np.where(m & self.limb)
            point = random.randrange(len(ys))  # x and y MUST describe the same pixel
            y0 = np.clip(int(ys[point]) - c // 2, 0, H - c)
            x0 = np.clip(int(xs[point]) - c // 2, 0, W - c)
        else:
            y0, x0 = random.randint(0, H - c), random.randint(0, W - c)
        sl = np.s_[y0:y0 + c, x0:x0 + c]
        x  = torch.from_numpy(img[sl][None])
        y  = torch.from_numpy(m[sl][None].astype(np.float32))
        sd = torch.from_numpy(d[sl][None])
        vm = torch.from_numpy(self.limb[sl][None].astype(np.float32))
        if self.train:                                 # physics-valid dihedral aug
            if random.random() < 0.5: x, y, sd, vm = x.flip(-1), y.flip(-1), sd.flip(-1), vm.flip(-1)
            if random.random() < 0.5: x, y, sd, vm = x.flip(-2), y.flip(-2), sd.flip(-2), vm.flip(-2)
            k = random.randrange(4)
            if k: x, y, sd, vm = (torch.rot90(t, k, (-2, -1)) for t in (x, y, sd, vm))
            if random.random() < 0.3:                  # photometric jitter
                x = (x * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05)).clamp(0, 1)
                x = x ** random.uniform(0.8, 1.2)
        return x, y, sd, vm


# --------------------------------------------------------------------------- #
# 3. LIMB MASK
# --------------------------------------------------------------------------- #
def build_limb_mask(cfg: Cfg) -> np.ndarray:
    """1 inside the annotated region (disk & |meridian angle| <= 70 deg), else 0."""
    h = w = cfg.img_size
    y, x = np.ogrid[:h, :w]
    cx = cy = cfg.img_size // 2
    r = np.hypot(x - cx, y - cy)
    if cfg.limb_geometry == "legacy_wedge":
        ang = np.degrees(np.arcsin(np.clip(np.abs(x - cx) / (r + 1e-6), 0, 1)))
        return (r <= cfg.disk_radius) & (ang <= cfg.limb_max_deg)
    if cfg.limb_geometry != "longitude":
        raise ValueError(f"Unknown limb geometry: {cfg.limb_geometry}")
    # Orthographic projection at B0=0: x=R*cos(latitude)*sin(longitude).
    # The previous polar-angle wedge wrongly removed the entire equatorial row.
    half_width = np.sqrt(np.maximum(cfg.disk_radius ** 2 - (y - cy) ** 2, 0))
    return (r <= cfg.disk_radius) & (np.abs(x - cx) <= half_width * np.sin(np.deg2rad(cfg.limb_max_deg)))


# --------------------------------------------------------------------------- #
# 4. MODEL
# --------------------------------------------------------------------------- #
def norm_layer(channels: int) -> nn.Module:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def replace_batch_norm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, norm_layer(child.num_features))
        else:
            replace_batch_norm(child)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm_layer(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), norm_layer(cout), nn.ReLU(inplace=True))

    def forward(self, x): return self.b(x)


class DilatedBottleneck(nn.Module):
    """Residual dilated stack; zero-init BN => exact identity at initialisation."""

    def __init__(self, ch, mid, dils=(1, 2, 4)):
        super().__init__()
        self.red = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), norm_layer(mid), nn.ReLU(inplace=True))
        self.cvs = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=d, dilation=d, bias=False),
            norm_layer(mid), nn.ReLU(inplace=True)) for d in dils])
        self.exp = nn.Sequential(nn.Conv2d(mid, ch, 1, bias=False), norm_layer(ch))
        nn.init.zeros_(self.exp[1].weight)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x): return self.act(x + self.exp(self.cvs(self.red(x))))


class AttentionGate(nn.Module):
    """Additive gate (Oktay et al., 2018): soft per-pixel mask on a skip connection."""

    def __init__(self, gc, xc, ic):
        super().__init__()
        self.Wg = nn.Conv2d(gc, ic, 1); self.Wx = nn.Conv2d(xc, ic, 1, stride=2, bias=False)
        self.psi = nn.Conv2d(ic, 1, 1)
        self.bg = norm_layer(ic); self.bx = norm_layer(ic)

    def forward(self, g, x):
        g1, x1 = self.bg(self.Wg(g)), self.bx(self.Wx(x))
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, x1.shape[-2:], mode="nearest")
        a = torch.sigmoid(self.psi(F.relu(g1 + x1, inplace=True)))
        return x * F.interpolate(a, x.shape[-2:], mode="bilinear", align_corners=False)


class FilamentUNet(nn.Module):
    """timm features_only encoder + U-Net decoder with attention-gated skips."""

    def __init__(self, cfg: Cfg, pretrained=True):
        super().__init__()
        self.enc = timm.create_model(cfg.encoder, pretrained=pretrained, features_only=True, in_chans=1)
        replace_batch_norm(self.enc)
        ech = self.enc.feature_info.channels(); ered = self.enc.feature_info.reduction()
        r2c = dict(zip(ered, ech)); deepest = max(ered)
        n_up = int(math.log2(deepest))
        self.bn = DilatedBottleneck(ech[-1], cfg.bn_ch, cfg.bn_dilations)
        blocks, gates, self.has_gate, self.skip_red = [], [], [], []
        cin = ech[-1]
        for i in range(n_up):
            red = deepest // 2 ** (i + 1); sc = r2c.get(red, 0)
            cout = cfg.decoder_ch[min(i, len(cfg.decoder_ch) - 1)]
            if sc:
                gates.append(AttentionGate(cin, sc, max(sc // 2, 8))); self.has_gate.append(True)
            else:
                gates.append(nn.Identity()); self.has_gate.append(False)
            blocks.append(ConvBlock(cin + sc, cout))
            self.skip_red.append(red if sc else None); cin = cout
        self.blocks, self.gates = nn.ModuleList(blocks), nn.ModuleList(gates)
        self.head_seg, self.head_spine = nn.Conv2d(cin, 1, 1), nn.Conv2d(cin, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = self.enc(x)
        r2f = dict(zip(self.enc.feature_info.reduction(), feats))
        out = self.bn(feats[-1])
        for blk, gate, act, red in zip(self.blocks, self.gates, self.has_gate, self.skip_red):
            g = out                                    # decoder signal, one stage coarser
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
            if red is not None:
                sk = gate(g, r2f[red]) if act else r2f[red]
                out = blk(torch.cat([out, sk], 1))
            else:
                out = blk(out)
        seg, spn = self.head_seg(out), self.head_spine(out)
        if seg.shape[-2:] != (H, W):
            seg = F.interpolate(seg, (H, W), mode="bilinear", align_corners=False)
            spn = F.interpolate(spn, (H, W), mode="bilinear", align_corners=False)
        return seg, spn


# --------------------------------------------------------------------------- #
# 5. LOSS
# --------------------------------------------------------------------------- #
def soft_dice(pred, tgt, vm, eps=1e-6):
    p, t = (pred * vm).flatten(1), (tgt * vm).flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def soft_cldice(pred, tgt, vm, iters=8, smooth=1.0):
    """Mask-only topology surrogate, not a differentiable PQ objective.

    Morphological skeleton and topology precision/sensitivity follow Shit et al.
    https://github.com/jocpae/clDice . Reduce per image in float32 for AMP safety.
    """
    def erode(x):
        return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)),
                             -F.max_pool2d(-x, (1, 3), 1, (0, 1)))

    def skel(x):
        skeleton = F.relu(x - F.max_pool2d(erode(x), 3, 1, 1))
        for _ in range(iters):
            x = erode(x)
            delta = F.relu(x - F.max_pool2d(erode(x), 3, 1, 1))
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton

    p, t = pred.float() * vm, tgt.float() * vm
    sp, st = skel(p), skel(t)
    dims = tuple(range(1, p.ndim))
    prec = ((sp * t).sum(dims) + smooth) / (sp.sum(dims) + smooth)
    rec = ((st * p).sum(dims) + smooth) / (st.sum(dims) + smooth)
    return (1 - 2 * prec * rec / (prec + rec).clamp_min(1e-7)).mean()


def criterion(model, x, y, sd, vm, cfg: Cfg):
    logits, spn = model(x)
    logits = logits.float()
    prob = torch.sigmoid(logits)
    n = vm.sum().clamp(min=1.0)
    positives = (y * vm).sum().clamp(min=1.0)
    negatives = ((1.0 - y) * vm).sum()
    pos_weight = (negatives / positives).clamp(min=1.0, max=cfg.pos_weight_cap)
    bce_map = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=pos_weight, reduction="none"
    )
    bce = (bce_map * vm).sum() / n
    loss = (cfg.w_bce * bce
            + cfg.w_dice * soft_dice(prob, y, vm)
            + cfg.w_cld * soft_cldice(prob, y, vm))
    if cfg.w_spine:
        loss = loss + cfg.w_spine * F.mse_loss(spn * vm, sd * vm)
    return loss, prob


# --------------------------------------------------------------------------- #
# 6. SPLIT (anti-leakage) + LOADERS
# --------------------------------------------------------------------------- #
def get_fold(cfg: Cfg):
    """Returns (train_ids, val_ids) where val holds ONE annotation set per
    physical observation (the leaderboard GT is also one set per image)."""
    coco = COCO(cfg.train_json)
    ids = coco.getImgIds()
    groups = np.array([base_name(coco.loadImgs([i])[0]["file_name"]) for i in ids])
    gkf = GroupKFold(n_splits=cfg.n_folds)
    tr_idx, va_idx = list(gkf.split(ids, groups=groups))[cfg.fold]
    tr_ids = [ids[i] for i in tr_idx]
    va_ids, seen = [], set()
    for i in va_idx:                                   # dedup duplicated observations
        b = groups[i]
        if b not in seen:
            seen.add(b); va_ids.append(ids[i])
    if cfg.limit:
        tr_ids, va_ids = tr_ids[:cfg.limit], va_ids[: min(cfg.limit, len(va_ids))]
    return coco, tr_ids, va_ids


def make_loader(coco, ids, cfg, train):
    ds = FilamentDataset(coco, ids, cfg, train=train, limb=build_limb_mask(cfg))
    return torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                                       num_workers=cfg.num_workers,
                                       pin_memory=cfg.device == "cuda", drop_last=train), ds


# --------------------------------------------------------------------------- #
# 7. TRAIN (AMP + cosine LR + EMA)
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model, decay):
        self.m = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.d = decay

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.m[k].mul_(self.d).add_(v.detach(), alpha=1 - self.d)
            else:
                self.m[k] = v.detach().clone()

    def copy_to(self, model): model.load_state_dict(self.m)


def train(cfg: Cfg, epoch_callback=None):
    seed_everything(cfg.seed + cfg.fold)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    if (Path(cfg.work_dir) / f"fold{cfg.fold}.pt").exists():
        raise FileExistsError("Use a new work_dir to preserve the existing training run")
    progress_path = Path(cfg.progress_log) if cfg.progress_log else Path(cfg.work_dir) / "training_progress.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        f"training_started fold={cfg.fold} epochs={cfg.epochs} "
        f"batch_size={cfg.batch_size} effective_batch_size={cfg.batch_size * cfg.grad_accum_steps}\n",
        encoding="utf-8",
    )
    coco, tr_ids, va_ids = get_fold(cfg)
    if cfg.eval_max_images and len(va_ids) > cfg.eval_max_images:
        va_ids = sorted(np.random.default_rng(cfg.seed).choice(
            va_ids, cfg.eval_max_images, replace=False).tolist())
    _write_json_atomically(Path(cfg.work_dir) / "split.json", {
        "train_ids": tr_ids, "monitor_ids": va_ids,
        "monitor_only": bool(cfg.eval_max_images), "cfg": vars(cfg),
    })
    tr, _ = make_loader(coco, tr_ids, cfg, train=True)
    va, vds = make_loader(coco, va_ids, cfg, train=False)
    model = FilamentUNet(cfg, pretrained=cfg.pretrained).to(cfg.device)
    if cfg.init_ckpt:
        initial = torch.load(cfg.init_ckpt, map_location=cfg.device, weights_only=False)
        if initial["cfg"]["fold"] != cfg.fold:
            raise ValueError("Warm-start checkpoint must belong to the same held-out fold")
        model.load_state_dict(initial["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    ema = EMA(model, cfg.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.device == "cuda")
    best = -1.0
    history = []
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); run = 0.0
        with progress_path.open("a", encoding="utf-8") as log:
            log.write(f"epoch_started {ep + 1}/{cfg.epochs}\n")
        print(f"[fold {cfg.fold}] epoch {ep + 1:03d}/{cfg.epochs:03d} started", flush=True)
        opt.zero_grad(set_to_none=True)
        for batch_index, (x, y, sd, vm) in enumerate(tr):
            x, y, sd, vm = (t.to(cfg.device, non_blocking=True) for t in (x, y, sd, vm))
            with torch.amp.autocast("cuda", enabled=cfg.amp and cfg.device == "cuda"):
                loss, _ = criterion(model, x, y, sd, vm, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {ep + 1}, batch {batch_index}")
            group_start = (batch_index // cfg.grad_accum_steps) * cfg.grad_accum_steps
            group_size = min(cfg.grad_accum_steps, len(tr) - group_start)
            scaler.scale(loss / group_size).backward()
            should_step = ((batch_index + 1) % cfg.grad_accum_steps == 0
                           or batch_index + 1 == len(tr))
            if should_step:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); ema.update(model)
            run += loss.item()
        sch.step()
        should_eval = bool(cfg.eval_every) and (
            (ep + 1) % cfg.eval_every == 0 or ep + 1 == cfg.epochs
        )
        report = evaluate_detailed(model, vds, cfg, ema) if should_eval else None
        pq = report["aggregate"]["pq"]["mean"] if report else None
        improved = pq is not None and pq > best
        if improved:
            best = pq
        elapsed = time.time() - t0
        message = (f"[fold {cfg.fold}] epoch {ep + 1:03d}/{cfg.epochs:03d} "
               f"loss {run/len(tr):.4f} val_PQ {pq if pq is not None else 'not_evaluated'} "
               f"validation={'run' if should_eval else 'skipped'} elapsed_s {elapsed:.0f}")
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": model.state_dict(),
            "ema": ema.m,
            "cfg": vars(cfg),
            "epoch": ep,
            "completed_epochs": ep + 1,
            "best_pq": best,
            "validation_pq": pq,
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "scaler": scaler.state_dict(),
            "status": "running" if ep + 1 < cfg.epochs else "completed",
        }
        torch.save(checkpoint, f"{cfg.work_dir}/fold{cfg.fold}.pt")
        if improved:
            torch.save(checkpoint, f"{cfg.work_dir}/fold{cfg.fold}_best.pt")
        row = {"epoch": ep + 1, "loss": run / len(tr), "pq": pq,
               "elapsed_seconds": elapsed, "lr": sch.get_last_lr()[0]}
        if report:
            row.update(report["aggregate"])
            row["pq"] = pq
            _write_json_atomically(Path(cfg.work_dir) / f"validation_epoch{ep + 1:03d}.json", report)
        history.append(row)
        _write_json_atomically(Path(cfg.work_dir) / "history.json", history)
        if should_eval:
            preview_model = copy.deepcopy(model)
            ema.copy_to(preview_model)
            save_training_figures(preview_model, vds, cfg, ep + 1, history)
            del preview_model
        with progress_path.open("a", encoding="utf-8") as log:
            log.write(message + "\n")
        print(message, flush=True)
        if epoch_callback is not None:
            epoch_callback(ep + 1, Path(cfg.work_dir) / f"fold{cfg.fold}.pt")
    with progress_path.open("a", encoding="utf-8") as log:
        log.write(f"training_completed best_PQ {best:.4f}\n")
    print(f"[fold {cfg.fold}] best local PQ {best:.4f}")


# --------------------------------------------------------------------------- #
# 8. INFERENCE (sliding window + dihedral TTA + optional fold ensembling)
# --------------------------------------------------------------------------- #
def load_image(path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def dihedral_tta_pairs(enabled: bool):
    identity = lambda tensor: tensor
    if not enabled:
        return [(identity, identity)]
    return [
        (identity, identity),
        (lambda tensor: tensor.flip(-1), lambda tensor: tensor.flip(-1)),
        (lambda tensor: tensor.flip(-2), lambda tensor: tensor.flip(-2)),
        (lambda tensor: tensor.flip(-1).flip(-2), lambda tensor: tensor.flip(-1).flip(-2)),
        (lambda tensor: tensor.transpose(-1, -2), lambda tensor: tensor.transpose(-1, -2)),
        (lambda tensor: tensor.transpose(-1, -2).flip(-1), lambda tensor: tensor.flip(-1).transpose(-1, -2)),
        (lambda tensor: tensor.transpose(-1, -2).flip(-2), lambda tensor: tensor.flip(-2).transpose(-1, -2)),
        (lambda tensor: tensor.transpose(-1, -2).flip(-1).flip(-2),
         lambda tensor: tensor.transpose(-1, -2).flip(-1).flip(-2)),
    ]


@torch.no_grad()
def predict_full(models, img, cfg: Cfg) -> np.ndarray:
    """Overlap-add sliding window; 8-way dihedral TTA; probability map averaged
    over models (fold ensemble) and transforms."""
    for m in models: m.eval()
    H, W = img.shape; w = cfg.window; s = w - cfg.overlap
    prob = np.zeros((H, W), np.float32); cnt = np.zeros_like(prob)
    tta = dihedral_tta_pairs(cfg.tta)
    tta_batch_size = max(1, min(cfg.tta_batch_size, len(tta)))
    y_starts = sorted(set(range(0, max(H - w, 0) + 1, s)) | {max(H - w, 0)})
    x_starts = sorted(set(range(0, max(W - w, 0) + 1, s)) | {max(W - w, 0)})
    for y0 in y_starts:
        for x0 in x_starts:
            tile = torch.from_numpy(img[y0:y0 + w, x0:x0 + w][None, None]).to(cfg.device)
            acc = 0.0
            for start in range(0, len(tta), tta_batch_size):
                tta_chunk = tta[start:start + tta_batch_size]
                transformed_tiles = torch.cat([augment(tile) for augment, _ in tta_chunk])
                for m in models:
                    logits, _ = m(transformed_tiles)
                    for logit, (_, invert) in zip(logits, tta_chunk):
                        acc = acc + invert(torch.sigmoid(logit.float())).squeeze().cpu().numpy()
            acc /= len(tta) * len(models)
            prob[y0:y0 + w, x0:x0 + w] += acc; cnt[y0:y0 + w, x0:x0 + w] += 1
    return prob / np.maximum(cnt, 1)


# --------------------------------------------------------------------------- #
# 9. INSTANCES (watershed separation + GT-style cleanup)
# --------------------------------------------------------------------------- #
def postprocess_one(mask: np.ndarray) -> np.ndarray:
    """MAGFiLO rules: no holes, single connected piece."""
    mask = remove_small_holes(mask, area_threshold=64)
    lab, n = ndimage.label(mask)
    if n == 0: return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def prob_to_instances(prob, cfg: Cfg, limb) -> list:
    """Threshold -> closing -> EDT -> seeded watershed -> per-instance cleanup.
    The watershed is what prevents two touching filaments from being submitted
    as one blob (a one-to-many relation that PQ penalises twice)."""
    m = (prob > cfg.thresh) & limb
    if cfg.close_kernel > 1:
        m = ndimage.binary_closing(m, structure=np.ones((cfg.close_kernel,) * 2))
    m &= limb
    components, n = ndimage.label(m)
    if cfg.instance_method == "components":
        lab = components
    elif cfg.instance_method == "watershed":
        dist = ndimage.distance_transform_edt(m)
        seeds = peak_local_max(dist, min_distance=cfg.min_dist_ws, exclude_border=False)
        markers = np.zeros_like(components)
        markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
        # Nearby disconnected objects must not disappear due to peak suppression.
        next_marker = len(seeds) + 1
        for i, box in enumerate(ndimage.find_objects(components), 1):
            if box is None:
                continue
            region = components[box] == i
            if not markers[box][region].any():
                local = np.where(region, dist[box], -1)
                point = np.unravel_index(local.argmax(), local.shape)
                markers[tuple(p + s.start for p, s in zip(point, box))] = next_marker
                next_marker += 1
        lab = watershed(-dist, markers, mask=m)
    else:
        raise ValueError(f"Unknown instance method: {cfg.instance_method}")
    instances = []
    for i, box in enumerate(ndimage.find_objects(lab), 1):
        if box is None:
            continue
        local = postprocess_one(lab[box] == i) & limb[box]
        if local.sum() >= cfg.min_area:
            mask = np.zeros_like(m)
            mask[box] = local
            instances.append(mask)
    return instances


# --------------------------------------------------------------------------- #
# 10. LOCAL PANOPTIC QUALITY (leaderboard mirror)
# --------------------------------------------------------------------------- #
def instance_iou_matrix(pred, gt):
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    # COCO's compiled RLE intersection supports overlapping GT masks too.
    encode = lambda masks: [maskutils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
    return maskutils.iou(encode(pred), encode(gt), [0] * len(gt))


def match_instances(pred, gt, iou_thr=0.5):
    """Greedily match instances by descending IoU, as required by PQ."""
    iou = instance_iou_matrix(pred, gt)
    mp, mg, matches = set(), set(), []
    for flat_index in np.argsort(iou.ravel())[::-1]:
        v = float(iou.ravel()[flat_index])
        if v < iou_thr: break
        i, j = np.unravel_index(flat_index, iou.shape)
        if i not in mp and j not in mg:
            mp.add(i); mg.add(j); matches.append((int(i), int(j), v))
    return iou, matches


def panoptic_quality(pred, gt, iou_thr=0.5):
    """Greedy IoU matching (standard PQ protocol): pairs sorted by IoU."""
    _, matches = match_instances(pred, gt, iou_thr)
    tps = [match[2] for match in matches]
    tp = len(tps); fp, fn = len(pred) - tp, len(gt) - tp
    sq = float(np.mean(tps)) if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
    return sq * rq, sq, rq


def _pixel_scores(pred, gt):
    pred_union = np.logical_or.reduce(pred) if pred else None
    gt_union = np.logical_or.reduce(gt) if gt else None
    if pred_union is None and gt_union is None:
        return 1.0, 1.0
    if pred_union is None or gt_union is None:
        return 0.0, 0.0
    inter = np.logical_and(pred_union, gt_union).sum()
    union = np.logical_or(pred_union, gt_union).sum()
    dice = 2.0 * inter / (pred_union.sum() + gt_union.sum()) if pred_union.sum() + gt_union.sum() else 1.0
    return float(dice), float(inter / union) if union else 1.0


def image_metrics(pred, gt, iou_thr=0.5, elapsed_seconds=0.0):
    """Return all score components requested by the competition overview."""
    iou, matches = match_instances(pred, gt, iou_thr)
    matched_dice = []
    for pi, gi, _ in matches:
        inter = np.logical_and(pred[pi], gt[gi]).sum()
        denom = pred[pi].sum() + gt[gi].sum()
        matched_dice.append(float(2.0 * inter / denom) if denom else 1.0)
    tp = len(matches)
    fp, fn = len(pred) - tp, len(gt) - tp
    sq = float(np.mean([m[2] for m in matches])) if matches else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
    pixel_dice, pixel_iou = _pixel_scores(pred, gt)
    gt_overlap_counts = (iou > 0).sum(axis=0) if len(gt) else np.zeros(0, dtype=int)
    pred_overlap_counts = (iou > 0).sum(axis=1) if len(pred) else np.zeros(0, dtype=int)
    return {
        "pq": float(sq * rq), "sq": sq, "rq": float(rq),
        "pixel_dice": pixel_dice, "pixel_iou": pixel_iou,
        "matched_dice": matched_dice,
        "matched_iou": [float(m[2]) for m in matches],
        "tp": tp, "fp": fp, "fn": fn,
        "num_pred": len(pred), "num_gt": len(gt),
        "one_to_many_gt": int((gt_overlap_counts >= 2).sum()),
        "many_to_one_pred": int((pred_overlap_counts >= 2).sum()),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95))}


def _write_json_atomically(path: Path, payload) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _evaluation_signature(cfg: Cfg, checkpoint_path: Path) -> dict:
    checkpoint_stat = checkpoint_path.stat()
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "fold": cfg.fold,
        "tta": cfg.tta,
        "window": cfg.window,
        "overlap": cfg.overlap,
        "thresh": cfg.thresh,
        "min_area": cfg.min_area,
        "min_dist_ws": cfg.min_dist_ws,
        "close_kernel": cfg.close_kernel,
        "instance_method": cfg.instance_method,
        "limb_geometry": cfg.limb_geometry,
        "metric_version": "masked_gt_v2",
    }


@torch.no_grad()
def evaluate_detailed(model, valset: FilamentDataset, cfg: Cfg, ema=None,
                      completed_reports=None, partial_path: Path | None = None,
                      partial_signature=None):
    if ema is not None:
        # Never overwrite live training parameters (Adam moments refer to them).
        model = copy.deepcopy(model)
        ema.copy_to(model)
    completed_reports = completed_reports or {}
    reports = []
    total_images = len(valset.ids)
    for image_index, img_id in enumerate(valset.ids, start=1):
        existing_report = completed_reports.get(str(img_id))
        if existing_report is not None:
            reports.append(existing_report)
            print(f"[eval] {image_index:03d}/{total_images:03d} reused", flush=True)
            continue
        img, _, _ = valset._load_full(img_id)
        started = time.perf_counter()
        pred = prob_to_instances(predict_full([model], img, cfg), cfg, valset.limb)
        elapsed = time.perf_counter() - started
        gt = ground_truth_instances(valset, img_id)
        report = image_metrics(pred, gt, elapsed_seconds=elapsed)
        report["image_id"] = img_id
        report["file_name"] = valset.coco.imgs[img_id]["file_name"]
        reports.append(report)
        completed_reports[str(img_id)] = report
        if partial_path is not None:
            _write_json_atomically(partial_path, {
                "signature": partial_signature,
                "reports_by_image_id": completed_reports,
            })
        print(
            f"[eval] {image_index:03d}/{total_images:03d} PQ {report['pq']:.4f} "
            f"pred {report['num_pred']} gt {report['num_gt']} elapsed_s {elapsed:.1f}",
            flush=True,
        )
    scalar_keys = ("pq", "sq", "rq", "pixel_dice", "pixel_iou", "tp", "fp", "fn",
                   "one_to_many_gt", "many_to_one_pred", "elapsed_seconds")
    aggregate = {key: _summary([row[key] for row in reports]) for key in scalar_keys}
    aggregate["mean_instances_dice"] = _summary(
        [value for row in reports for value in row["matched_dice"]]
    )
    aggregate["mean_instances_iou"] = _summary(
        [value for row in reports for value in row["matched_iou"]]
    )
    aggregate["images_per_second"] = (
        len(reports) / sum(row["elapsed_seconds"] for row in reports)
        if reports and sum(row["elapsed_seconds"] for row in reports) else 0.0
    )
    total_tp = sum(row["tp"] for row in reports)
    total_fp = sum(row["fp"] for row in reports)
    total_fn = sum(row["fn"] for row in reports)
    matched_ious = [value for row in reports for value in row["matched_iou"]]
    dataset_sq = float(np.mean(matched_ious)) if matched_ious else 0.0
    dataset_rq = (
        total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        if total_tp + total_fp + total_fn else 0.0
    )
    aggregate["dataset_pq"] = float(dataset_sq * dataset_rq)
    aggregate["dataset_sq"] = dataset_sq
    aggregate["dataset_rq"] = float(dataset_rq)
    aggregate["total_tp"] = total_tp
    aggregate["total_fp"] = total_fp
    aggregate["total_fn"] = total_fn
    return {"aggregate": aggregate, "per_image": reports}


@torch.no_grad()
def evaluate(model, valset: FilamentDataset, cfg: Cfg, ema=None) -> float:
    return evaluate_detailed(model, valset, cfg, ema)["aggregate"]["pq"]["mean"]


def ground_truth_instances(valset, img_id):
    info = valset.coco.imgs[img_id]
    masks = [polygon_to_mask(a, info["height"], info["width"]) & valset.limb
             for a in valset.coco.loadAnns(valset.coco.getAnnIds(imgIds=[img_id]))]
    return [m for m in masks if m.any()]


@torch.no_grad()
def save_training_figures(model, valset, cfg, epoch, history):
    """Fixed examples and actual validation curves, usable from CLI or notebook."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(cfg.work_dir) / "previews"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["epoch"] for r in history], [r["loss"] for r in history], "o-")
    axes[0].set(title="Training loss", xlabel="Epoch")
    scored = [r for r in history if r["pq"] is not None]
    axes[1].plot([r["epoch"] for r in scored], [r["pq"] for r in scored], "o-", label="PQ")
    axes[1].plot([r["epoch"] for r in scored], [r["rq"]["mean"] for r in scored], "o-", label="RQ")
    axes[1].set(title=f"Fixed validation monitor (n={len(valset.ids)})", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout(); fig.savefig(Path(cfg.work_dir) / "learning_curves.png", dpi=140); plt.close(fig)
    for img_id in valset.ids[:cfg.preview_count]:
        img, _, _ = valset._load_full(img_id)
        prob = predict_full([model], img, cfg)
        pred = prob_to_instances(prob, cfg, valset.limb)
        gt = ground_truth_instances(valset, img_id)
        metrics = image_metrics(pred, gt)
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for ax in axes:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[0].contour(valset.limb, levels=[0.5], colors=["cyan"], linewidths=0.5)
        axes[0].set_title("Image and annotation support")
        for ax, masks, title in [(axes[1], gt, f"GT: {len(gt)} instances"),
                                  (axes[3], pred, f"Prediction: {len(pred)} instances")]:
            labels = np.zeros(img.shape, dtype=np.int32)
            for i, mask in enumerate(masks, 1): labels[mask] = i
            ax.imshow(np.ma.masked_equal(labels, 0), cmap="tab20", alpha=0.8, interpolation="nearest")
            ax.set_title(title)
        axes[2].imshow(np.ma.masked_where(~valset.limb, prob), cmap="magma", vmin=0, vmax=1, alpha=0.85)
        axes[2].set_title("Foreground probability")
        fig.suptitle(f"Epoch {epoch} | image {img_id} | PQ={metrics['pq']:.3f} SQ={metrics['sq']:.3f} "
                     f"RQ={metrics['rq']:.3f} | TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        fig.tight_layout(); fig.savefig(folder / f"epoch{epoch:03d}_image{img_id}.png", dpi=130); plt.close(fig)
    table = "".join(f"<tr><td>{r['epoch']}</td><td>{r['loss']:.4f}</td><td>{r['pq']}</td></tr>" for r in history)
    pictures = "".join(f'<figure><img src="previews/{p.name}"><figcaption>{p.name}</figcaption></figure>'
                       for p in sorted(folder.glob(f"epoch{epoch:03d}_*.png")))
    html = ('<!doctype html><meta charset="utf-8"><title>Filament PQ training</title>'
            '<style>body{font:16px system-ui;margin:32px;max-width:1600px}img{max-width:100%}'
            'td,th{padding:8px 20px;text-align:right}figure{margin:20px 0}</style>'
            f'<h1>Fold {cfg.fold}: PQ training, epoch {epoch}</h1>'
            f'<p>Fixed validation monitor: {len(valset.ids)} observations. Reload to see new epochs. '
            'This monitoring score is used for model selection; use the separate audit for comparison.</p>'
            '<img src="learning_curves.png"><table><tr><th>Epoch</th><th>Loss</th><th>PQ</th></tr>'
            + table + '</table>' + pictures)
    (Path(cfg.work_dir) / "dashboard.html").write_text(html, encoding="utf-8")


def evaluate_folds(cfg: Cfg):
    coco, _, va_ids = get_fold(cfg)
    if getattr(cfg, "positive_only", False):
        va_ids = [image_id for image_id in va_ids
                  if coco.getAnnIds(imgIds=[image_id])]
        if not va_ids:
            raise ValueError("No annotated validation images found for this fold")
    _, vds = make_loader(coco, va_ids, cfg, train=False)
    ckpt = Path(cfg.work_dir) / f"fold{cfg.fold}.pt"
    best_ckpt = Path(cfg.work_dir) / f"fold{cfg.fold}_best.pt"
    if best_ckpt.exists():
        ckpt = best_ckpt
    print(f"evaluating checkpoint: {ckpt}", flush=True)
    model = FilamentUNet(cfg, pretrained=False).to(cfg.device)
    model.load_state_dict(torch.load(ckpt, map_location=cfg.device)["ema"])
    partial_path = Path(cfg.work_dir) / f"evaluation_fold{cfg.fold}.partial.json"
    signature = _evaluation_signature(cfg, ckpt)
    completed_reports = {}
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("signature") == signature:
            completed_reports = {
                str(image_id): report
                for image_id, report in partial.get("reports_by_image_id", {}).items()
            }
            print(f"resuming evaluation with {len(completed_reports)} completed images", flush=True)
        else:
            print("ignoring evaluation partial with a different configuration", flush=True)
    report = evaluate_detailed(
        model,
        vds,
        cfg,
        completed_reports=completed_reports,
        partial_path=partial_path,
        partial_signature=signature,
    )
    report_path = Path(cfg.work_dir) / f"evaluation_fold{cfg.fold}.json"
    report_path.write_text(json.dumps(report, indent=2))
    partial_path.unlink(missing_ok=True)
    print(json.dumps(report["aggregate"], indent=2))
    print(f"saved detailed evaluation: {report_path}")


# --------------------------------------------------------------------------- #
# 11. POST-PROCESS TUNING ON LOCAL PQ
# --------------------------------------------------------------------------- #
def tune(cfg: Cfg):
    """Grid-search (thresh, min_dist_ws, min_area) on the held-out fold.
    Cheap and decisive: these three knobs move PQ more than most arch changes."""
    coco, _, va_ids = get_fold(cfg)
    _, vds = make_loader(coco, va_ids, cfg, train=False)
    model = FilamentUNet(cfg, pretrained=False).to(cfg.device)
    model.load_state_dict(torch.load(f"{cfg.work_dir}/fold{cfg.fold}.pt", map_location=cfg.device)["ema"])
    limb = vds.limb
    probs, gts = [], []
    for img_id in vds.ids:                              # cache prob maps once
        img, _, _ = vds._load_full(img_id)
        probs.append(predict_full([model], img, cfg))
        gts.append(ground_truth_instances(vds, img_id))
    best = (-float("inf"), None)
    for th in cfg.tune_thresh:
        for md in cfg.tune_mdist:
            for ma in cfg.tune_marea:
                cfg.thresh, cfg.min_dist_ws, cfg.min_area = th, md, ma
                pq = np.mean([panoptic_quality(prob_to_instances(p, cfg, limb), g)[0]
                              for p, g in zip(probs, gts)])
                print(f"thresh={th:.1f} min_dist={md:2d} min_area={ma:3d}  PQ {pq:.4f}", flush=True)
                if pq > best[0]: best = (pq, (th, md, ma))
    pq0, (th, md, ma) = best
    print(f"BEST: PQ {pq0:.4f} with thresh={th} min_dist_ws={md} min_area={ma}")
    Path(f"{cfg.work_dir}/best_postproc.json").write_text(
        json.dumps({"thresh": th, "min_dist_ws": md, "min_area": ma, "local_pq": pq0}, indent=2))


# --------------------------------------------------------------------------- #
# 12. SUBMISSION (RLE CSV, one row per predicted filament)
# --------------------------------------------------------------------------- #
def mask_to_rle(mask: np.ndarray) -> str:
    rle = maskutils.encode(np.asfortranarray(mask.astype(np.uint8)))
    c = rle["counts"]
    return c.decode() if isinstance(c, bytes) else c


def submit(cfg: Cfg, ckpts):
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints supplied. Train final folds first or pass --ckpts explicitly; "
            f"searched work directory: {cfg.work_dir}"
        )
    models = []
    for c in ckpts:
        payload = torch.load(c, map_location=cfg.device, weights_only=False)
        if payload.get("cfg", {}).get("limit", 0):
            raise ValueError(f"Refusing smoke checkpoint for submission: {c}")
        m = FilamentUNet(cfg, pretrained=False).to(cfg.device)
        m.load_state_dict(payload["ema"])
        models.append(m)
    bp = Path(f"{cfg.work_dir}/best_postproc.json")
    if bp.exists():                                   # apply tuned post-proc if available
        for k, v in json.loads(bp.read_text()).items():
            if hasattr(cfg, k): setattr(cfg, k, v)
        print("using tuned post-processing:", bp.read_text())
    limb = build_limb_mask(cfg)
    rows = []
    files = sorted(Path(cfg.test_dir).glob("*.jpeg"))
    if not files:
        raise FileNotFoundError(f"No test JPEGs found in {cfg.test_dir}")
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    image_counts = {}
    for f in files:
        prob = predict_full(models, load_image(f), cfg)
        inst = prob_to_instances(prob, cfg, limb)
        image_counts[f.name] = len(inst)
        for k, msk in enumerate(inst, 1):
            if msk.shape != (cfg.img_size, cfg.img_size):
                raise ValueError(f"Unexpected mask shape for {f.name}: {msk.shape}")
            if not np.any(msk):
                raise ValueError(f"Empty predicted instance for {f.name}_{k}")
            encoded = mask_to_rle(msk)
            decoded = maskutils.decode({"size": [cfg.img_size, cfg.img_size], "counts": encoded})
            if decoded.shape != (cfg.img_size, cfg.img_size) or not np.array_equal(decoded.astype(bool), msk):
                raise ValueError(f"RLE round-trip failed for {f.name}_{k}")
            rows.append({"filament_id": f"{f.stem}_{k}", "segmentation_rle": encoded})
        print(f"{f.name}: {len(inst)} filaments", flush=True)
    out = Path(cfg.work_dir) / "submission.csv"
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if list(submission.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError("Submission columns do not match the competition schema")
    if submission["filament_id"].isna().any() or not submission["filament_id"].is_unique:
        raise ValueError("Submission filament IDs must be non-empty and unique")
    submission.to_csv(out, index=False)
    _write_json_atomically(Path(cfg.work_dir) / "submission_manifest.json", {
        "checkpoints": ckpts, "config": vars(cfg), "image_instance_counts": image_counts,
        "n_images": len(files), "n_instances": len(rows), "rle_round_trip": "all passed",
    })
    print(f"saved {out} ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# ENTRY POINT
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Solar Filament Segmentation Challenge 2026")
    ap.add_argument("cmd", choices=["train", "eval", "tune", "submit"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap images (smoke test)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--progress-log", default=None, help="durable per-epoch progress log")
    ap.add_argument("--eval-every", type=int, default=None, help="run full validation every N epochs")
    ap.add_argument("--ckpts", nargs="*", default=None)
    ap.add_argument("--no-tta", action="store_true", help="disable test-time augmentation for CPU smoke evaluation")
    ap.add_argument("--window", type=int, default=None, help="inference window size")
    ap.add_argument("--positive-only", action="store_true", help="evaluate only images with ground-truth instances")
    ap.add_argument("--pretrained", action="store_true", help="download/use pretrained encoder weights")
    ap.add_argument("--init-ckpt", default="", help="same-fold EMA warm start; fresh optimizer")
    ap.add_argument("--eval-max-images", type=int, default=0, help="fixed monitoring subset, not final validation")
    ap.add_argument("--preview-count", type=int, default=3)
    ap.add_argument("--instance-method", choices=["components", "watershed"], default="components")
    ap.add_argument("--limb-geometry", choices=["longitude", "legacy_wedge"], default="longitude")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ema-decay", type=float, default=None)
    ap.add_argument("--crop-size", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    a = ap.parse_args()

    cfg = Cfg()
    cfg.fold, cfg.limit = a.fold, a.limit
    if a.epochs is not None: cfg.epochs = a.epochs
    if a.data_dir:
        cfg.data_dir = a.data_dir
        local_json = Path(cfg.data_dir) / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        if local_json.exists():
            cfg.train_json = str(local_json)
    if a.test_dir: cfg.test_dir = a.test_dir
    if a.work_dir: cfg.work_dir = a.work_dir
    if a.progress_log: cfg.progress_log = a.progress_log
    if a.eval_every is not None: cfg.eval_every = a.eval_every
    if a.no_tta: cfg.tta = False
    if a.window is not None: cfg.window = a.window
    cfg.positive_only = a.positive_only
    cfg.pretrained = a.pretrained
    cfg.init_ckpt = a.init_ckpt
    cfg.eval_max_images = a.eval_max_images
    cfg.preview_count = a.preview_count
    cfg.instance_method = a.instance_method
    cfg.limb_geometry = a.limb_geometry
    for field in ("lr", "ema_decay", "crop_size", "thresh", "min_area"):
        if getattr(a, field) is not None: setattr(cfg, field, getattr(a, field))
    if cfg.limit: cfg.epochs = min(cfg.epochs, 2)

    if a.cmd == "train":
        train(cfg)
    elif a.cmd == "eval":
        evaluate_folds(cfg)
    elif a.cmd == "tune":
        tune(cfg)
    elif a.cmd == "submit":
        ckpts = a.ckpts or [f"{cfg.work_dir}/fold{cfg.fold}.pt"]
        missing = [c for c in ckpts if not Path(c).exists()]
        if missing:
            raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
        submit(cfg, ckpts)


if __name__ == "__main__":
    main()
