# -*- coding: utf-8 -*-
"""Solar 暗条 U-Net 训练（GPU 版）：2048² 共识标签 → 任意 crop 尺寸。

与本地快照 code_snapshot/unet_train.py（512² 固定）的关系：
  * --crop（默认 1024）：图像 2048² jpeg → crop² 双线性；标签 support2048/
    votes2048 用面积平均（k×k 块均值，k=2048//crop，crop 须整除 2048）降到
    crop²，再取比例 t = s_avg / s_avg.max() —— 面积平均投票数再取比例，
    不用最近邻、不丢投票（本地 512 套标签就是用同思路的面积平均生成的，
    见 code_snapshot/build_pixel_labels.py）。
  * 类别 CE 目标：面积平均后的 votes 取 argmax（只在 s_avg>0 处），
    未标注/背景像素 ignore_index=-1（口径同 code_snapshot/unet_train.py:227-235）。
  * 模型 FilamentUNet5：本地 FilamentUNet（code_snapshot/unet_train.py:169-217）
    的 base16/depth4 → base32/depth5 加宽加深版，仍是 support+4 类双头 U-Net。
  * 损失沿用本地口径：soft BCE（正类软加权 w=1+9·t）+ 0.5·CE
    （code_snapshot/unet_train.py:25-26 与 compute_loss）。
  * 划分：{data-root}/labels/val_split.json 的 train/val；val 不进训练；
    实际名单 = split ∩ 磁盘图像 ∩ 707 标签，丢弃名单如实落盘。
  * V100 (sm_70)：fp16 autocast + GradScaler（Volta 有 fp16 tensor core）。

用法（冒烟）：
  python unet_train_crop.py --gpu 0 --crop 1024 --epochs 2 \
      --tag smoke1024
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

LAMBDA_CLASS = 0.5   # loss = bce + LAMBDA_CLASS * ce（同本地口径）
POS_W = 9.0          # soft 正类加权 w = 1 + POS_W * t


# ----------------------------------------------------------------- 参数
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="solar U-Net crop-size trainer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--crop", type=int, default=1024,
                   help="工作分辨率，2048 的约数（1024/512）")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--out", default=os.environ.get("SOLAR_WORK_DIR", "./out"),
                   help="工作输出目录（默认 $SOLAR_WORK_DIR，再退 ./out）")
    p.add_argument("--data-root",
                   default=os.environ.get("SOLAR_DATA_ROOT", "./data"),
                   help="数据根目录（默认 $SOLAR_DATA_ROOT，再退 ./data）")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--depth", type=int, default=5, choices=(4, 5))
    p.add_argument("--tag", default="run")
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args(argv)


# 公开仓库版：被 unet_infer1024.py 复用 FilamentUNet5 时以空参数走默认值，
# 不消费进程的 argv（作为脚本运行时行为与原版完全一致）。
ARGS = parse_args() if __name__ == "__main__" else parse_args([])
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)   # 必须在 import torch 前

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CONS = os.path.join(ARGS.data_root, "labels", "consensus")
IMG_DIR = os.path.join(ARGS.data_root, "MAGFiLO_1.0_Kaggle_2026",
                       "train", "train_images")
SPLIT = os.path.join(ARGS.data_root, "labels", "val_split.json")
CACHE_DIR = os.path.join(ARGS.out, "cache")
CROP = ARGS.crop


# ------------------------------------------------------- 标签/图像预处理
def process_one(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单图：2048² 标签面积平均到 crop²（投票数均值→比例），图像双线性缩放。

    返回 (img_u8[crop,crop], t_f16[crop,crop]∈[0,1], cls_i8[crop,crop]∈0..4)。
    """
    stem = os.path.splitext(name)[0]
    with np.load(os.path.join(CONS, stem + ".npz")) as z:
        s = z["support2048"].astype(np.float32)      # (2048,2048) 投票覆盖数
        v = z["votes2048"].astype(np.float32)        # (4,2048,2048) 各类票数
    k = s.shape[0] // CROP
    assert s.shape[0] == CROP * k, f"crop={CROP} 必须整除 2048"
    s_low = s.reshape(CROP, k, CROP, k).mean(axis=(1, 3))          # 面积平均
    v_low = v.reshape(4, CROP, k, CROP, k).mean(axis=(2, 4))
    m = float(s_low.max())
    t = s_low / m if m > 0 else np.zeros_like(s_low)               # 取比例
    cls = np.zeros((CROP, CROP), np.int8)
    pos = s_low > 0
    cls[pos] = v_low.argmax(axis=0)[pos].astype(np.int8) + 1       # 1..4
    img = Image.open(os.path.join(IMG_DIR, name)).convert("L") \
               .resize((CROP, CROP), Image.BILINEAR)
    return np.asarray(img, np.uint8), t.astype(np.float16), cls


def process_one_safe(name: str):
    """解码失败（如另一进程正在写入的半截 jpeg）→ 记原因并丢弃，不炸整个缓存。"""
    try:
        img, t, cls = process_one(name)
        return name, img, t, cls, ""
    except Exception as e:                      # noqa: BLE001
        return name, None, None, None, f"{type(e).__name__}: {e}"


def build_cache(names: List[str], tag: str) -> Tuple[str, List[str]]:
    """把 (图像, soft 目标, 类别) 预处理成一份堆叠 npz，多进程保序。

    返回 (cache_path, bad_names)；bad_names = 解码失败名单（如实外报）。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.perf_counter()
    import multiprocessing as mp
    n_proc = max(1, min(ARGS.workers, os.cpu_count() or 2))
    with mp.get_context("fork").Pool(n_proc) as pool:
        out = pool.map(process_one_safe, names, chunksize=4)  # map 保序 → 不串图
    keep: List[str] = []
    bad: List[Tuple[str, str]] = []
    for name, img, t, cls, err in out:            # out 与 names 同序
        if img is None:
            bad.append((name, err))
        else:
            keep.append(name)
    import hashlib
    key = hashlib.sha1("\n".join(sorted(keep)).encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"{tag}_crop{CROP}_{key}.npz")
    if os.path.exists(path):
        print(f"[cache] 复用已有 {path}", flush=True)
        return path, [b[0] for b in bad]
    imgs = np.stack([o[1] for o in out if o[1] is not None])
    tgts = np.stack([o[2] for o in out if o[1] is not None])
    clss = np.stack([o[3] for o in out if o[1] is not None])
    np.savez(path, imgs=imgs, tgt=tgts, cls=clss,
             names=np.asarray(keep))
    print(f"[cache] {len(keep)}/{len(names)} 图 crop={CROP} → {path} "
          f"({(imgs.nbytes+tgts.nbytes+clss.nbytes)/2**20:.0f} MiB, "
          f"{time.perf_counter()-t0:.1f}s, {n_proc} 进程保序)", flush=True)
    if bad:
        print(f"[cache] 解码失败 {len(bad)} 张（如实记录）: "
              f"{[b[0] for b in bad][:5]}{'...' if len(bad) > 5 else ''}",
              flush=True)
    return path, [b[0] for b in bad]


# ------------------------------------------------------------------ 数据
class CropDS:
    """RAM 内堆叠数组 + 联合增广（img/t/cls 共用同一套翻转/抖动参数）。"""

    def __init__(self, cache_path: str, augment: bool, seed: int):
        with np.load(cache_path, allow_pickle=False) as z:
            stored = [str(x) for x in z["names"]]
            self.names = stored
            self.img = z["imgs"]
            self.t = z["tgt"].astype(np.float32)
            self.c = z["cls"]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        img = self.img[i].astype(np.float32) / 255.0
        t = self.t[i]
        c = self.c[i]
        if self.augment:
            if self.rng.random() < 0.5:
                img, t, c = img[::-1], t[::-1], c[::-1]
            if self.rng.random() < 0.5:
                img, t, c = img[:, ::-1], t[:, ::-1], c[:, ::-1]
            gain = float(self.rng.uniform(0.8, 1.2))
            bias = float(self.rng.uniform(-0.1, 0.1))
            img = np.clip(img * gain + bias, 0.0, 1.0)
        w = 1.0 + POS_W * t
        return (np.ascontiguousarray(img[None]),
                np.ascontiguousarray(t[None]),
                np.ascontiguousarray(c.astype(np.int64)),
                np.ascontiguousarray(w[None]))


# ------------------------------------------------------------------ 模型
class _Dbl(nn.Module):
    """(Conv3x3-ReLU)×2，与 code_snapshot/unet_train.py 的 _Dbl 相同。"""

    def __init__(self, i: int, o: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(o, o, 3, padding=1),
                               nn.ReLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class FilamentUNet5(nn.Module):
    """U-Net，depth=5（4 次下采样，瓶颈 1/16）或 depth=4，base 可调；
    support 单通道头 + 4 类头（Left/Right/Unidentifiable/Ambiguous），
    结构沿 code_snapshot/unet_train.py FilamentUNet（base16/depth4）。"""

    def __init__(self, base: int = 32, depth: int = 5, in_ch: int = 1):
        super().__init__()
        b = base
        chs = [in_ch] + [b * (2 ** i) for i in range(depth)]   # 1,32..512
        self.enc = nn.ModuleList(_Dbl(chs[i], chs[i + 1])
                                 for i in range(depth))
        self.p = nn.MaxPool2d(2)
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.dec.append(nn.ModuleDict({
                "up": nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2),
                "cat": _Dbl(chs[i] * 2, chs[i]),
            }))
        self.out_sup = nn.Conv2d(b, 1, 1)
        self.out_cls = nn.Conv2d(b, 4, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        skips = []
        h = x
        for i, blk in enumerate(self.enc):
            h = blk(h if i == 0 else self.p(h))
            if i < len(self.enc) - 1:
                skips.append(h)
        for j, d in enumerate(self.dec):
            h = d["up"](h)
            h = d["cat"](torch.cat([h, skips[-1 - j]], 1))
        return self.out_sup(h), self.out_cls(h)


# ------------------------------------------------------------------ 损失
def compute_loss(out_sup, out_cls, target, cls
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """soft BCE（w=1+9t）+ 0.5·CE（cls==0 → ignore），口径同本地 compute_loss。"""
    weight = 1.0 + POS_W * target
    bce = F.binary_cross_entropy_with_logits(out_sup, target, weight=weight)
    labels = (cls - 1).clamp_min(-1)          # 0→-1(ignore)，1..4→0..3
    if int((labels >= 0).sum()) > 0:
        ce = F.cross_entropy(out_cls, labels, ignore_index=-1)
    else:
        ce = out_cls.sum() * 0.0
    loss = bce + LAMBDA_CLASS * ce
    return loss, {"bce": float(bce.detach()), "ce": float(ce.detach())}


# ------------------------------------------------------------------ 训练
def pick_names(split_names: List[str]) -> Tuple[List[str], List[str]]:
    """split ∩ 磁盘图像 ∩ 标签，返回 (保留, 丢弃)。"""
    imgs = set(os.listdir(IMG_DIR))
    labs = {os.path.splitext(f)[0] + ".jpeg" for f in os.listdir(CONS)}
    keep = [n for n in split_names if n in imgs and n in labs]
    drop = [n for n in split_names if n not in set(keep)]
    return keep, drop


def run_epoch(model, ds: CropDS, device, bs, train, opt=None, scaler=None):
    idx = np.random.permutation(len(ds)) if train else np.arange(len(ds))
    sums = {"loss": 0.0, "bce": 0.0, "ce": 0.0}
    dice_num = dice_den = 0
    steps = 0
    step_ms: List[float] = []
    first_loss = last_loss = float("nan")
    ctx = torch.enable_grad() if train else torch.no_grad()
    t_d = 0.0
    with ctx:
        for k in range(0, len(idx), bs):
            if train:
                t0 = time.perf_counter()
            batch = [ds[int(i)] for i in idx[k:k + bs]]
            x, t, c, w = (torch.from_numpy(np.stack(col)).to(device)
                          for col in zip(*batch))
            if train:
                t_d += time.perf_counter() - t0
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out_sup, out_cls = model(x)
            out_sup, out_cls = out_sup.float(), out_cls.float()
            loss, parts = compute_loss(out_sup, out_cls, t, c)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            l = float(loss.detach())
            first_loss = l if steps == 0 else first_loss
            last_loss = l
            sums["loss"] += l
            sums["bce"] += parts["bce"]
            sums["ce"] += parts["ce"]
            steps += 1
            if train:
                step_ms.append((time.perf_counter() - t0) * 1000)
            with torch.no_grad():
                p = torch.sigmoid(out_sup) > 0.5
                g = t > 0.5
                dice_num += int((p & g).sum()) * 2
                dice_den += int(p.sum() + g.sum())
    n = max(1, steps)
    dice = dice_num / dice_den if dice_den else 0.0
    return (sums["loss"] / n, {"bce": sums["bce"] / n, "ce": sums["ce"] / n,
                               "dice50": dice}, steps, first_loss, last_loss,
            step_ms, t_d)


def main():
    a = ARGS
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    print(f"[env] device={dev} name={dev_name} torch={torch.__version__} "
          f"cuda_avail={torch.cuda.is_available()} seed={a.seed} "
          f"crop={a.crop} bs={a.bs} base={a.base} depth={a.depth} "
          f"amp=fp16 tag={a.tag}", flush=True)

    with open(SPLIT, encoding="utf-8") as f:
        split = json.load(f)
    tr_keep, tr_drop = pick_names(split["train"])
    va_keep, va_drop = pick_names(split["val"])
    print(f"[data] root={a.data_root} val_split.json: train {len(tr_keep)}/"
          f"{len(split['train'])}（缺图/缺标签丢弃 {len(tr_drop)}） val "
          f"{len(va_keep)}/{len(split['val'])}（丢 {len(va_drop)}）；"
          f"val 不进训练", flush=True)
    with open(os.path.join(a.out, f"dropped_{a.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"train_dropped": tr_drop, "val_dropped": va_drop}, f,
                  ensure_ascii=False, indent=1)

    tr_cache, tr_bad = build_cache(tr_keep, "train")
    va_cache, va_bad = build_cache(va_keep, "val")
    if tr_bad or va_bad:
        with open(os.path.join(a.out, f"decode_failed_{a.tag}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"train": tr_bad, "val": va_bad,
                       "note": "图像解码失败（半截文件/损坏），本次训练弃用并如实记录"},
                      f, ensure_ascii=False, indent=1)
        print(f"[data] 解码失败: train {len(tr_bad)} val {len(va_bad)} → "
              f"decode_failed_{a.tag}.json", flush=True)
    ds_tr = CropDS(tr_cache, augment=True, seed=a.seed)
    ds_va = CropDS(va_cache, augment=False, seed=a.seed)

    model = FilamentUNet5(base=a.base, depth=a.depth).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] FilamentUNet5 base={a.base} depth={a.depth} "
          f"params={n_par} ({n_par/1e6:.2f}M)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    hist: List[Dict] = []
    ckpt_last = os.path.join(a.out, f"unet_{a.tag}_last.pt")
    best = float("inf")
    t_all = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        t0 = time.perf_counter()
        tr, trp, ntr, fl, ll, sms, t_d = run_epoch(
            model, ds_tr, dev, a.bs, True, opt, scaler)
        t_tr = time.perf_counter() - t0
        t0 = time.perf_counter()
        va, vap, nva, _, _, _, _ = run_epoch(model, ds_va, dev, a.bs, False)
        t_va = time.perf_counter() - t0
        sched.step()
        sms_sorted = sorted(sms)
        p50 = sms_sorted[len(sms_sorted)//2] if sms_sorted else 0
        rec = {"epoch": ep, "train_loss": tr, "train_bce": trp["bce"],
               "train_ce": trp["ce"], "val_loss": va, "val_dice50":
               vap["dice50"], "train_sec": t_tr, "val_sec": t_va,
               "train_steps": ntr, "step_ms_p50": p50,
               "first_step_loss": fl, "last_step_loss": ll}
        hist.append(rec)
        print(f"[ep{ep}] train_loss={tr:.5f} (bce {trp['bce']:.5f} + "
              f"0.5·ce {trp['ce']:.5f}) 首步={fl:.5f} 末步={ll:.5f} "
              f"dice50={trp['dice50']:.4f} | {t_tr:.1f}s ({ntr} 步, "
              f"p50 {p50:.0f}ms/步, 取数 {t_d:.1f}s) | val_loss={va:.5f} "
              f"val_dice50={vap['dice50']:.4f} ({t_va:.1f}s)", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(a),
                    "epoch": ep, "hist": hist}, ckpt_last)
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "args": vars(a),
                        "epoch": ep, "hist": hist},
                       os.path.join(a.out, f"unet_{a.tag}_best.pt"))
    total = time.perf_counter() - t_all
    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[mem] torch.max_memory_allocated={peak:.2f} GiB", flush=True)
    print(f"[summary] tag={a.tag} epochs={a.epochs} 总耗时={total:.1f}s "
          f"纯训练={sum(h['train_sec'] for h in hist):.1f}s "
          f"首epoch训练秒={hist[0]['train_sec']:.1f} "
          f"末epoch训练秒={hist[-1]['train_sec']:.1f} "
          f"loss首末(逐epoch)={hist[0]['train_loss']:.5f}→"
          f"{hist[-1]['train_loss']:.5f}", flush=True)
    with open(os.path.join(a.out, f"history_{a.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"args": vars(a), "hist": hist, "total_sec": total,
                   "device": dev_name, "params": n_par}, f,
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
