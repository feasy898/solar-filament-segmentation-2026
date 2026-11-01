# -*- coding: utf-8 -*-
"""val 上的实例抽取/阈值/形状过滤扫描（快评器）。

用途：在 val（labels/val_split.json 的固定划分）上比较「概率图 → 掩膜 → 实例」
的不同配方，挑出给最终候选文件用的配置。快评器复刻 solar_eval_local 的两套
口径（官方池化 PQ + 逐行 Dice），但用 numpy 的 flat-index 交并替代 frozenset
运算；**最终数字一律以官方 solar_eval_local.py 复核**（本脚本只做搜索）。

快评器自检：对官方 v3 基线 CSV 跑出的 PQ/Dice 与 eval_v3_val.json 逐位一致
（见 --crosscheck）。

用法：
    ./.venv-solar-gpu/bin/python unet_sweep.py --crosscheck preds_v3_val.csv
    ./.venv-solar-gpu/bin/python unet_sweep.py --prob output/unet_val_prob_frame \
        --grid a1 --out output/sweep_a1.json
    ./.venv-solar-gpu/bin/python unet_sweep.py --prob output/unet_val_prob_frame \
        --grid a2 --base output/sweep_a1.json --out output/sweep_a2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 公开仓库版：脚本平铺同一目录（rle.py / solar_eval_local.py / unet_infer.py
# 都在 HERE），原内部工作区的 "../../solar" 路径 shim 不再需要。
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rle import HEIGHT, WIDTH  # noqa: E402
from solar_eval_local import load_gt_rows  # noqa: E402
from unet_infer import (build_prob2048, extract_instances,  # noqa: E402
                        extract_instances_pipeline, labels_to_flat)

GTAnn = Tuple[str, str]     # (标注者, 图像stem)
FlatMasks = Dict[str, List[np.ndarray]]


# ------------------------------------------------------------------ GT / 预测
def _flat(mask: frozenset) -> np.ndarray:
    """frozenset[(row,col)] → 升序 flat index 数组（c*H + r，COCO 列主序）。"""
    if not mask:
        return np.zeros(0, np.int64)
    a = np.fromiter((c * HEIGHT + r for r, c in mask), np.int64, len(mask))
    a.sort()
    return a


def load_gt_flat(ann_path: str, names: Sequence[str]
                 ) -> Tuple[Dict[GTAnn, List[np.ndarray]], Dict]:
    rows, meta = load_gt_rows(ann_path, names)
    return {k: [_flat(m) for m in v] for k, v in rows.items()}, meta


def load_preds_csv_flat(path: str) -> FlatMasks:
    import csv
    from rle import rle_to_mask
    out: FlatMasks = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            fid, rle = row[0], row[1]
            img = fid.rsplit("_", 1)[0]
            out.setdefault(img, []).append(_flat(rle_to_mask(rle, HEIGHT, WIDTH)))
    return out


# ------------------------------------------------------------------ 快评器
def _iou_mat(gts: List[np.ndarray], preds: List[np.ndarray]) -> np.ndarray:
    m = np.zeros((len(gts), len(preds)), np.float64)
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            inter = np.intersect1d(g, p, assume_unique=True).size
            union = g.size + p.size - inter
            if union:
                m[i, j] = inter / union
    return m


def eval_fast(gt: Dict[GTAnn, List[np.ndarray]], preds: FlatMasks,
              rows: Optional[Sequence[GTAnn]] = None
              ) -> Dict[str, float]:
    """官方口径池化 PQ + 逐行 Dice（与 solar_eval_local 同语义）。

    rows 给定时只统计这些 (标注者, 图像) 行（用于分半稳健性检查）。
    """
    pred_union: Dict[str, np.ndarray] = {}
    for img, lst in preds.items():
        pred_union[img] = (np.unique(np.concatenate(lst)) if lst
                           else np.zeros(0, np.int64))
    tp_iou_sum = 0.0
    tp = fp = fn = 0
    d_num = d_den = 0
    row_dice: List[float] = []
    for key, gts in gt.items():
        if rows is not None and key not in rows:
            continue
        ann, img = key
        plist = preds.get(img, [])
        if gts and plist:
            ious = _iou_mat(gts, plist)
            hits = ious > 0.5
            tp_iou_sum += float(ious[hits].sum())
            tp += int(hits.sum())
            fp += int((~hits.any(axis=0)).sum())
            fn += int((~hits.any(axis=1)).sum())
        else:
            fp += len(plist)
            fn += len(gts)
        A = np.unique(np.concatenate(gts)) if gts else np.zeros(0, np.int64)
        B = pred_union.get(img, np.zeros(0, np.int64))
        inter = np.intersect1d(A, B, assume_unique=True).size
        den = A.size + B.size
        row_dice.append((2.0 * inter / den) if den else 0.0)
        d_num += 2 * inter
        d_den += den
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": (tp_iou_sum / denom) if denom else 0.0,
        "dice_pooled": (d_num / d_den) if d_den else 0.0,
        "dice_macro_per_row": float(np.mean(row_dice)) if row_dice else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "tp_iou_sum": tp_iou_sum,
        "pred_masks": int(sum(len(v) for v in preds.values())),
        "pred_images": len(preds),
    }


# ------------------------------------------------------------------ 概率缓存
def load_prob_cache(prob_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
    for fn in sorted(os.listdir(prob_dir)):
        if not fn.endswith(".npz") or fn.startswith("_"):
            continue
        with np.load(os.path.join(prob_dir, fn)) as z:
            scheme = str(z["scheme"]) if "scheme" in z.files else "frame"
            out[os.path.splitext(fn)[0]] = (z["prob512"].astype(np.float32),
                                           z["cls512"].astype(np.uint8),
                                           z["origin"].astype(np.int32),
                                           scheme)
    return out


def run_config(cache, cfg: Dict) -> FlatMasks:
    preds: FlatMasks = {}
    for stem, (prob512, cls512, origin, scheme) in cache.items():
        prob, cls = build_prob2048(prob512, cls512, origin,
                                   mode=cfg.get("upsample", "bilinear"),
                                   native_tiles=(scheme == "tile"))
        c = dict(thr=cfg["thr"], method=cfg["method"],
                 class_split=cfg.get("class_split", False),
                 connectivity=cfg.get("connectivity", 4),
                 marker_min_distance=cfg.get("marker_min_distance", 12),
                 marker_sigma=cfg.get("marker_sigma", 4.0),
                 marker_thr=cfg.get("marker_thr"),
                 min_area=cfg.get("min_area", 0), max_area=cfg.get("max_area", 0),
                 min_elo=cfg.get("min_elo", 0.0), min_len=cfg.get("min_len", 0.0))
        lab, _n, _tag = extract_instances_pipeline(prob, cls, c)
        preds[stem] = labels_to_flat(lab)
    return preds


# ------------------------------------------------------------------ 网格
def grid_a1() -> List[Dict]:
    """上采样方式 × 阈值 × 最小面积。"""
    out = []
    for up in ("nearest", "bilinear"):
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
            for area in (0, 200, 320, 600, 1000):
                out.append({"upsample": up, "thr": thr, "method": "cc",
                            "class_split": False, "min_area": area})
    return out


def grid_a1b() -> List[Dict]:
    """细化：上采样 × 阈值 × 最小面积（a1 的高分邻域）。"""
    out = []
    for up in ("nearest", "bilinear"):
        for thr in (0.60, 0.65, 0.70, 0.75):
            for area in (400, 600, 800, 1000, 1300, 1800):
                out.append({"upsample": up, "thr": thr, "method": "cc",
                            "class_split": False, "min_area": area})
    return out


def grid_a2(base: List[Dict]) -> List[Dict]:
    """在 a1 最优几个点上扫形状过滤。"""
    tops = sorted(base, key=lambda r: -r["pq"])[:3]
    out = []
    for t in tops:
        c = dict(t["cfg"])
        for max_area in (0, 6000, 10000, 20000):
            for min_elo in (0.0, 1.5, 2.0, 2.5):
                for min_len in (0.0, 20.0, 30.0, 45.0):
                    d = dict(c)
                    d.update({"max_area": max_area, "min_elo": min_elo,
                              "min_len": min_len})
                    out.append(d)
    return out


def grid_ws(base: List[Dict]) -> List[Dict]:
    """分水岭：标记阈值 / 间距 / sigma 扫描。"""
    tops = sorted(base, key=lambda r: -r["pq"])[:2]
    out = []
    for t in tops:
        c = dict(t["cfg"])
        for mthr in (None, 0.3, 0.4, 0.5):
            for md in (4, 8, 12, 20):
                for sig in (2.0, 4.0, 8.0):
                    d = dict(c)
                    d.update({"method": "watershed", "marker_thr": mthr,
                              "marker_min_distance": md, "marker_sigma": sig})
                    out.append(d)
    return out


# ------------------------------------------------------------------ CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prob", default=None, help="概率缓存目录")
    ap.add_argument("--gt-ann", default=os.path.join(
        HERE, "MAGFiLO_1.0_Kaggle_2026", "train",
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    ap.add_argument("--split", default=os.path.join(HERE, "labels", "val_split.json"))
    ap.add_argument("--crosscheck", default=None,
                    help="用官方 CSV 交叉验证快评器（与 solar_eval_local 数字对比）")
    ap.add_argument("--grid", default="a1", choices=["a1", "a1b", "a2", "ws"])
    ap.add_argument("--base", default=None, help="上一阶段的扫描结果 JSON")
    ap.add_argument("--out", default=None, help="扫描结果 JSON 落盘")
    args = ap.parse_args()

    with open(args.split, encoding="utf-8") as f:
        names = list(json.load(f)["val"])
    gt, meta = load_gt_flat(args.gt_ann, names)
    print(f"[gt] rows={meta['gt_rows']} images={meta['gt_images']} "
          f"annotators={meta['gt_annotators']} masks={meta['gt_masks']}")

    if args.crosscheck:
        preds = load_preds_csv_flat(args.crosscheck)
        rep = eval_fast(gt, preds)
        print(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True))
        return 0

    cache = load_prob_cache(args.prob)
    print(f"[prob] {len(cache)} 张已入内存")
    if args.grid == "a1":
        grid = grid_a1()
    elif args.grid == "a1b":
        grid = grid_a1b()
    else:
        base = json.load(open(args.base, encoding="utf-8"))
        grid = grid_a2(base) if args.grid == "a2" else grid_ws(base)

    rows = []
    # 分半稳健性：按图像名排序后的奇偶位切成两半，分别算 PQ
    imgs = sorted({img for (_a, img) in gt})
    half = {img: (i % 2) for i, img in enumerate(imgs)}
    half_rows = {0: [k for k in gt if half[k[1]] == 0],
                 1: [k for k in gt if half[k[1]] == 1]}
    t0 = time.perf_counter()
    for i, cfg in enumerate(grid, 1):
        preds = run_config(cache, cfg)
        rep = eval_fast(gt, preds)
        rep["cfg"] = cfg
        rep["sec"] = round(time.perf_counter() - t0, 1)
        for h in (0, 1):
            rep[f"pq_half{h}"] = eval_fast(gt, preds, half_rows[h])["pq"]
        rows.append(rep)
        print(f"[{i}/{len(grid)}] " + json.dumps(
            {k: rep[k] for k in ("pq", "dice_pooled", "tp", "fp", "fn",
                                 "pred_masks", "pq_half0", "pq_half1")},
            ensure_ascii=False),
            "|", json.dumps(cfg, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: -r["pq"])
    print("\n=== top 10 by pq ===")
    for r in rows[:10]:
        print(json.dumps({k: r[k] for k in
                          ("pq", "dice_pooled", "tp", "fp", "fn", "pred_masks", "cfg")},
                         ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
