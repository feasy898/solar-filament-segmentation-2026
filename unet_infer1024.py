# -*- coding: utf-8 -*-
"""Solar · 1024² U-Net 推理：checkpoint → 逐图概率缓存（prob1024 / cls1024 npz）。

出处说明（诚实边界）
------------------------------------------------------------------------------
原内部工作区里这一步跑在一台远端 V100S 上；早期快照缺失时本文件曾按**下游
契约**重建，后来拿回了当时的原始脚本并对拍，两种口径已对齐：

* 缓存格式：每图一个 ``<stem>.npz``，键 ``prob1024`` (float16, 1024×1024)、
  ``cls1024`` (uint8, 1024×1024，0=背景，1..4=Left/Right/Unidentifiable/
  Ambiguous)；
* 图像解码 = 训练时的解码（unet_train_crop.process_one）：2048² jpeg →
  ``convert("L").resize((1024,1024), BILINEAR) / 255``，**与训练严格同分布**；
* 模型 = unet_train_crop.FilamentUNet5（base32/depth5，7,759,653 参数），
  结构超参从 checkpoint 的 ``args`` 里读，不用手工对齐；
* TTA = 翻转变体集合逐个前向、概率与**类别 softmax 一并累加取平均**再翻回：
  none=1 次、hflip=原图+水平、vflip=原图+垂直、**hvflip=原图+水平+垂直+
  （水平∘垂直）共 4 次等权平均**（⚠ hvflip 含对角变体，不是 3 次——当年
  产出提交件的原始脚本即此口径；实测用本文件对冻结概率缓存**逐位复现**
  |Δ|=0，见 README §7）。

验证：用本文件的 4 变体 hvflip 口径重算当年冻结 test 概率缓存中的图，
float16 存储后逐位一致；下游 ``postproc_sweep.py``（读同一格式）与
``unet_sweep.eval_fast`` 的评测对拍见 README §7。

用法
------------------------------------------------------------------------------
    # val 概率缓存（--split 限定 val 名单）
    python unet_infer1024.py --ckpt out/unet_s0_best.pt \
        --images $DATA/MAGFiLO_1.0_Kaggle_2026/train/train_images \
        --out prob_val_s0 --split $WORK/labels/val_split.json --tta hflip
    # test 概率缓存（全集 180 张）
    python unet_infer1024.py --ckpt out/unet_s0_best.pt \
        --images $DATA/MAGFiLO_1.0_Kaggle_2026/test/test_images \
        --out prob_test_s0 --tta hflip
    # 多 seed 集成：对每个 seed 各跑一次，把多个 --out 目录一起交给
    # postproc_sweep.py（它在概率上取平均），本脚本不做集成。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CROP = 1024
HEIGHT = WIDTH = 2048


# --------------------------------------------------------------------- 模型
def load_model1024(ckpt_path: str, device: Optional[str] = None):
    """加载 unet_train_crop.FilamentUNet5 权重（base/depth 取自 ckpt 的 args）。"""
    import torch
    from unet_train_crop import FilamentUNet5
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {}) or {}
    model = FilamentUNet5(base=int(args.get("base", 32)),
                          depth=int(args.get("depth", 5))).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck, device


def _forward(model, device, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(N,1024,1024) float32 [0,1] → (prob N×1024², cls_softmax N×4×1024²)。

    与训练同精度口径：cuda 上 fp16 autocast。
    """
    import torch
    x = torch.from_numpy(batch[:, None].astype(np.float32)).to(device)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=(str(device) == "cuda")):
            sup_logit, cls_logit = model(x)
        prob = torch.sigmoid(sup_logit.float())
        cls_prob = torch.softmax(cls_logit.float(), dim=1)
    p = prob[:, 0].cpu().numpy().astype(np.float32)
    c = cls_prob.cpu().numpy().astype(np.float32)
    return p, c


# --------------------------------------------------------------------- 解码
def decode1024(path: str) -> np.ndarray:
    """2048² jpeg → 1024² float32 [0,1]（与 unet_train_crop.process_one 同分布）。"""
    from PIL import Image
    img = Image.open(path).convert("L").resize((CROP, CROP), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


# TTA 翻转变体集合（("h","v") = 先垂直再水平的对角变体，即 180° 旋转输入）
TTA_VARIANTS = {
    "none": ((),),
    "hflip": ((), ("h",)),
    "vflip": ((), ("v",)),
    "hvflip": ((), ("h",), ("v",), ("h", "v")),
}


def prob_for_image1024(path: str, model, device,
                       tta: str = "none") -> Tuple[np.ndarray, np.ndarray]:
    """单图 → (prob1024 float32, cls1024 uint8)。

    TTA：变体集合里每个翻转各前向一次，概率与类别 softmax **累加取等权平均**
    后翻回原方向（hvflip = 原图+h+v+h∘v 共 4 次——当年产出提交件的原始口径；
    类别 = 平均后的 softmax 一次性 argmax，不做逐次投票）。
    """
    img = decode1024(path)
    variants = TTA_VARIANTS[tta]
    acc_p = np.zeros((CROP, CROP), np.float32)
    acc_c = np.zeros((4, CROP, CROP), np.float32)
    for var in variants:
        xx = img
        if "v" in var:
            xx = xx[::-1, :]
        if "h" in var:
            xx = xx[:, ::-1]
        p, cp = _forward(model, device, np.ascontiguousarray(xx)[None])
        p, cp = p[0], cp[0]                      # 单图：(1024²), (4,1024²)
        if "v" in var:
            p = p[::-1, :]
            cp = cp[:, ::-1, :]
        if "h" in var:
            p = p[:, ::-1]
            cp = cp[:, :, ::-1]
        acc_p += p
        acc_c += cp
    n = len(variants)
    prob = acc_p / n
    cls = (acc_c.argmax(axis=0) + 1).astype(np.uint8)   # 1..4
    return prob, cls


# ----------------------------------------------------------------------- CLI
def _list_images(d: str) -> List[str]:
    exts = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
    names = sorted(n for n in os.listdir(d) if n.lower().endswith(exts))
    return [os.path.join(d, n) for n in names]


def _restrict(names: List[str], split: Optional[str],
              names_file: Optional[str]) -> List[str]:
    """按 val_split.json 的 val 名单或逐行名单文件过滤（stem 匹配）。"""
    keep: Optional[set] = None
    if split:
        with open(split, encoding="utf-8") as f:
            obj = json.load(f)
        keep = {os.path.splitext(n)[0] for n in obj["val"]}
    if names_file:
        with open(names_file, encoding="utf-8") as f:
            keep = {ln.strip() for ln in f if ln.strip()}
    if keep is None:
        return names
    return [p for p in names if os.path.splitext(os.path.basename(p))[0] in keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="unet_train_crop.py 产出的 .pt")
    ap.add_argument("--images", required=True, help="2048² jpeg 图像目录")
    ap.add_argument("--out", required=True, help="概率缓存输出目录")
    ap.add_argument("--tta", choices=["none", "hflip", "vflip", "hvflip"],
                    default="none")
    ap.add_argument("--split", default=None, help="val_split.json：只推理 val 名单")
    ap.add_argument("--names-file", default=None, help="逐行 stem 名单（与 --split 叠加）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model, ck, device = load_model1024(args.ckpt)
    n_par = sum(p.numel() for p in model.parameters())
    paths = _restrict(_list_images(args.images), args.split, args.names_file)
    if args.limit:
        paths = paths[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"[infer1024] device={device} ckpt_epoch={ck.get('epoch')} "
          f"params={n_par} tta={args.tta} n={len(paths)}", flush=True)

    t0 = time.perf_counter()
    for i, p in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(args.out, stem + ".npz")
        if os.path.exists(dst) and not args.overwrite:
            continue
        prob, cls = prob_for_image1024(p, model, device, tta=args.tta)
        np.savez_compressed(dst, prob1024=prob.astype(np.float16),
                            cls1024=cls.astype(np.uint8))
        if i % 20 == 0 or i == len(paths):
            print(f"[infer1024] {i}/{len(paths)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    meta = {"ckpt": os.path.abspath(args.ckpt), "ckpt_epoch": ck.get("epoch"),
            "tta": args.tta, "n_images": len(paths), "device": str(device),
            "crop": CROP, "params": n_par}
    with open(os.path.join(args.out, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[infer1024] {len(paths)} 张 → {args.out} "
          f"({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
