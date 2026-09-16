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
4. Multi-task heads ......... filament logit + log-distance-to-spine regression.
                               MAGFiLO ships one polyline spine per filament:
                               free auxiliary supervision of the medial axis.
5. Loss ..................... BCE + soft-Dice + soft-clDice (Shit et al., CVPR
                               2021; connectivity-aware — plain Dice is blind to
                               a filament predicted as disjoint pieces) + spine MSE.
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

Reproduction: `python solution.py train --fold 0 --epochs 2 --limit 8` runs a
full smoke test (train + eval + tune + submit dry-run) on a handful of images.
"""

import argparse
import json
import math
import random
import time
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
    # data
    img_size:     int = 2048    # fixed by the competition for every image
    crop_size:    int = 768     # training crops
    oversample_p: float = 0.75  # prob. a crop is centred on filament pixels
    disk_radius:  int = 950     # approx. solar-disk radius (images are centred)
    limb_max_deg: float = 70.0  # nothing is annotated beyond +-70 deg
    # model
    encoder:      str = "resnet18"
    decoder_ch:   tuple = (256, 128, 64, 32, 16)
    bn_ch:        int = 256
    bn_dilations: tuple = (1, 2, 4)
    # optimisation
    epochs:       int = 60
    batch_size:   int = 8
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    amp:          bool = True
    ema_decay:    float = 0.999
    # loss weights
    w_bce:   float = 0.3
    w_dice:  float = 0.3
    w_cld:   float = 0.4
    w_spine: float = 0.0
    # inference / post-processing
    tta:          bool = True
    window:       int = 1024
    overlap:      int = 256
    thresh:       float = 0.5
    min_area:     int = 50
    min_dist_ws:  int = 25
    close_kernel: int = 3
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
    torch.backends.cudnn.benchmark = True


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
    return maskutils.decode(rle).astype(bool)


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
    """Crops for training (x, mask, spine-distmap, valid-mask); full-image access
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
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            m |= polygon_to_mask(ann, h, w)
        if len(self._cache) > 16:                      # bound RAM (2048^2 floats)
            self._cache.clear()
        self._cache[img_id] = (img, m, d)
        return img, m, d

    def __getitem__(self, i):
        img, m, d = self._load_full(self.ids[i])
        H, W = img.shape
        c = self.cfg.crop_size
        if self.train and random.random() < self.cfg.oversample_p and m.sum() > 0:
            ys, xs = np.where(m)                       # centre on a filament pixel
            y0 = np.clip(int(ys[random.randrange(len(ys))]) - c // 2, 0, H - c)
            x0 = np.clip(int(xs[random.randrange(len(xs))]) - c // 2, 0, W - c)
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
    ang = np.degrees(np.arccos(np.clip((x - cx) / (r + 1e-6), -1, 1)))
    return ((r <= cfg.disk_radius) & (ang <= cfg.limb_max_deg))


# --------------------------------------------------------------------------- #
# 4. MODEL
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x): return self.b(x)


class DilatedBottleneck(nn.Module):
    """Residual dilated stack; zero-init BN => exact identity at initialisation."""

    def __init__(self, ch, mid, dils=(1, 2, 4)):
        super().__init__()
        self.red = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.cvs = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=d, dilation=d, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)) for d in dils])
        self.exp = nn.Sequential(nn.Conv2d(mid, ch, 1, bias=False), nn.BatchNorm2d(ch))
        nn.init.zeros_(self.exp[1].weight)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x): return self.act(x + self.exp(self.cvs(self.red(x))))


class AttentionGate(nn.Module):
    """Additive gate (Oktay et al., 2018): soft per-pixel mask on a skip connection."""

    def __init__(self, gc, xc, ic):
        super().__init__()
        self.Wg = nn.Conv2d(gc, ic, 1); self.Wx = nn.Conv2d(xc, ic, 1, stride=2, bias=False)
        self.psi = nn.Conv2d(ic, 1, 1)
        self.bg = nn.BatchNorm2d(ic); self.bx = nn.BatchNorm2d(ic)

    def forward(self, g, x):
        g1, x1 = self.bg(self.Wg(g)), self.bx(self.Wx(x))
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, x1.shape[-2:], mode="nearest")
        a = torch.sigmoid(self.psi(F.relu(g1 + x1, inplace=True)))
        return x * F.interpolate(a, x.shape[-2:], mode="bilinear", align_corners=False)


class FilamentUNet(nn.Module):
    """timm features_only encoder + U-Net decoder with attention-gated skips.
    Heads: filament segmentation logit + log-distance-to-spine regression."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.enc = timm.create_model(cfg.encoder, pretrained=True, features_only=True, in_chans=1)
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
    """Soft clDice (Shit et al., 2021): Dice on differentiable skeletons obtained
    by iterative contour stripping. Penalises fragmented filament predictions
    that plain Dice cannot see — directly aligned with the PQ metric."""
    def skel(x):
        for _ in range(iters):
            minp = -F.max_pool2d(-x, 3, 1, 1)
            contour = F.relu(F.max_pool2d(x, 3, 1, 1) - minp)
            x = F.relu(x - contour)
        return x
    p, t = pred * vm, tgt * vm
    sp, st = skel(p), skel(t)
    prec = (st * p).sum() / (sp.sum() + smooth)
    rec  = (sp * t).sum() / (st.sum() + smooth)
    return 1 - 2 * prec * rec / (prec + rec + smooth)


def criterion(model, x, y, sd, vm, cfg: Cfg):
    logits, spn = model(x)
    prob = torch.sigmoid(logits)
    n = vm.sum().clamp(min=1.0)
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * vm).sum() / n
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
                                       num_workers=4, pin_memory=True, drop_last=train), ds


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


def train(cfg: Cfg):
    seed_everything(cfg.seed + cfg.fold)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    coco, tr_ids, va_ids = get_fold(cfg)
    tr, _ = make_loader(coco, tr_ids, cfg, train=True)
    va, vds = make_loader(coco, va_ids, cfg, train=False)
    model = FilamentUNet(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    ema = EMA(model, cfg.ema_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    best = 0.0
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for x, y, sd, vm in tr:
            x, y, sd, vm = (t.to(cfg.device, non_blocking=True) for t in (x, y, sd, vm))
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                loss, _ = criterion(model, x, y, sd, vm, cfg)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); ema.update(model)
            run += loss.item()
        sch.step()
        pq = evaluate(model, vds, cfg, ema)
        best = max(best, pq)
        print(f"[fold {cfg.fold}] epoch {ep:03d}  loss {run/len(tr):.4f}  val PQ {pq:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.m, "cfg": vars(cfg)},
                   f"{cfg.work_dir}/fold{cfg.fold}.pt")
    print(f"[fold {cfg.fold}] best local PQ {best:.4f}")


# --------------------------------------------------------------------------- #
# 8. INFERENCE (sliding window + dihedral TTA + optional fold ensembling)
# --------------------------------------------------------------------------- #
def load_image(path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


@torch.no_grad()
def predict_full(models, img, cfg: Cfg) -> np.ndarray:
    """Overlap-add sliding window; 8-way dihedral TTA; probability map averaged
    over models (fold ensemble) and transforms."""
    for m in models: m.eval()
    H, W = img.shape; w = cfg.window; s = w - cfg.overlap
    prob = np.zeros((H, W), np.float32); cnt = np.zeros_like(prob)
    tta = [lambda t: t]
    if cfg.tta:
        tta = [lambda t: t, lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.flip(-1).flip(-2),
               lambda t: t.transpose(-1, -2), lambda t: t.transpose(-1, -2).flip(-1),
               lambda t: t.transpose(-1, -2).flip(-2), lambda t: t.transpose(-1, -2).flip(-1).flip(-2)]
    y_starts = sorted(set(range(0, max(H - w, 0) + 1, s)) | {max(H - w, 0)})
    x_starts = sorted(set(range(0, max(W - w, 0) + 1, s)) | {max(W - w, 0)})
    for y0 in y_starts:
        for x0 in x_starts:
            tile = torch.from_numpy(img[y0:y0 + w, x0:x0 + w][None, None]).to(cfg.device)
            acc = 0.0
            for tf in tta:
                tt = tf(tile)
                for m in models:
                    logits, _ = m(tt)
                    acc = acc + tf(torch.sigmoid(logits.float())).squeeze().cpu().numpy()
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
    dist = ndimage.distance_transform_edt(m)
    seeds = peak_local_max(dist, min_distance=cfg.min_dist_ws, exclude_border=False)
    markers = np.zeros_like(dist, int); markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
    lab = watershed(-dist, markers, mask=m)
    return [postprocess_one(lab == i) for i in range(1, lab.max() + 1)
            if (lab == i).sum() >= cfg.min_area]


# --------------------------------------------------------------------------- #
# 10. LOCAL PANOPTIC QUALITY (leaderboard mirror)
# --------------------------------------------------------------------------- #
def panoptic_quality(pred, gt, iou_thr=0.5):
    """Greedy IoU matching (standard PQ protocol): pairs sorted by IoU, both
    partners must be unmatched; unmatched predictions are FP, GT are FN."""
    iou = np.zeros((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            u = np.logical_or(p, g).sum()
            iou[i, j] = np.logical_and(p, g).sum() / u if u else 0.0
    mp, mg, tps = set(), set(), []
    for v in sorted(iou.flatten(), reverse=True):
        if v < iou_thr: break
        i, j = np.argwhere(iou == v)[0]
        if i not in mp and j not in mg:
            mp.add(i); mg.add(j); tps.append(v)
    tp = len(tps); fp, fn = len(pred) - tp, len(gt) - tp
    sq = float(np.mean(tps)) if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
    return sq * rq, sq, rq


@torch.no_grad()
def evaluate(model, valset: FilamentDataset, cfg: Cfg, ema=None) -> float:
    """Mean PQ over full 2048x2048 validation images (EMA weights, TTA on)."""
    if ema is not None: ema.copy_to(model)
    saved_tta = cfg.tta; cfg.tta = True
    limb = valset.limb
    pq_list = []
    for img_id in valset.ids:
        img, _, _ = valset._load_full(img_id)
        pred = prob_to_instances(predict_full([model], img, cfg), cfg, limb)
        gt = [polygon_to_mask(a, img.shape[0], img.shape[1])
              for a in valset.coco.loadAnns(valset.coco.getAnnIds(imgIds=img_id))]
        pq_list.append(panoptic_quality(pred, gt)[0])
    cfg.tta = saved_tta
    return float(np.mean(pq_list))


def evaluate_folds(cfg: Cfg):
    coco, _, va_ids = get_fold(cfg)
    _, vds = make_loader(coco, va_ids, cfg, train=False)
    ckpt = f"{cfg.work_dir}/fold{cfg.fold}.pt"
    model = FilamentUNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load(ckpt, map_location=cfg.device)["ema"])
    pq = evaluate(model, vds, cfg)
    print(f"fold {cfg.fold}: local PQ = {pq:.4f}  (checkpoint {ckpt})")


# --------------------------------------------------------------------------- #
# 11. POST-PROCESS TUNING ON LOCAL PQ
# --------------------------------------------------------------------------- #
def tune(cfg: Cfg):
    """Grid-search (thresh, min_dist_ws, min_area) on the held-out fold.
    Cheap and decisive: these three knobs move PQ more than most arch changes."""
    coco, _, va_ids = get_fold(cfg)
    _, vds = make_loader(coco, va_ids, cfg, train=False)
    model = FilamentUNet(cfg).to(cfg.device)
    model.load_state_dict(torch.load(f"{cfg.work_dir}/fold{cfg.fold}.pt", map_location=cfg.device)["ema"])
    limb = vds.limb
    probs, gts = [], []
    for img_id in vds.ids:                              # cache prob maps once
        img, _, _ = vds._load_full(img_id)
        probs.append(predict_full([model], img, cfg))
        gts.append([polygon_to_mask(a, img.shape[0], img.shape[1])
                    for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))])
    best = (0.0, None)
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
        m = FilamentUNet(cfg).to(cfg.device)
        m.load_state_dict(torch.load(c, map_location=cfg.device)["ema"])
        models.append(m)
    bp = Path(f"{cfg.work_dir}/best_postproc.json")
    if bp.exists():                                   # apply tuned post-proc if available
        for k, v in json.loads(bp.read_text()).items():
            if hasattr(cfg, k): setattr(cfg, k, v)
        print("using tuned post-processing:", bp.read_text())
    limb = build_limb_mask(cfg)
    rows = []
    for f in sorted(Path(cfg.test_dir).glob("*.jpeg")):
        prob = predict_full(models, load_image(f), cfg)
        inst = prob_to_instances(prob, cfg, limb)
        for k, msk in enumerate(inst, 1):
            rows.append({"filament_id": f"{f.stem}_{k}", "segmentation_rle": mask_to_rle(msk)})
        print(f"{f.name}: {len(inst)} filaments", flush=True)
    out = Path(cfg.work_dir) / "submission.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
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
    ap.add_argument("--ckpts", nargs="*", default=None)
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
