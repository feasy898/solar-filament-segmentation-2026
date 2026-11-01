# -*- coding: utf-8 -*-
"""生成**固定**的 val 划分：labels/val_split.json。

划分口径（写死在这里，之后训练与评测都只认这一份文件）：
  * 宇宙 = 官方训练集的 **707 张物理图**（GT JSON 里出现过的 file_name）。
  * val  = 其中「**多标注者**」（≥2 位标注者，共 296 张）的**分层抽样**子集：
           分层键 = (年代=file_name 前 4 位, 该图标注者数)，层内按 file_name 排序后
           用固定 seed 的 random.Random 做确定性洗牌，再取前 ceil(ratio·层大小) 个。
           —— 只用**公开训练标注**的信息（标注者数、年代、文件名），**不碰任何测试图**。
  * train= 宇宙减去 val（其余全部训练图，含单标注者图与未入 val 的多标注者图）。
  * 附加  train_local / val_evaluable：当前磁盘上真正有 jpeg 的子集（方便立刻开跑）。

用法：
    ./.venv-solar-gpu/bin/python make_val_split.py --out labels/val_split.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANN = os.path.join(_HERE, "MAGFiLO_1.0_Kaggle_2026", "train",
                           "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
IMAGE_DIRS = [
    os.path.join(_HERE, "MAGFiLO_1.0_Kaggle_2026", "train", "train_images"),
    os.path.join(_HERE, "MAGFiLO_1.0_Kaggle_2026", "MAGFiLO_1.0_Kaggle_2026",
                 "train", "train_images"),
]


def local_images(extra_dirs=None) -> set:
    """磁盘上真正存在的训练图集合。

    公开仓库版：数据在仓库外，用 --images 传入 train_images 目录（可多次）；
    不传时回落到原内部工作区的默认路径（对第三方通常为空）。
    """
    out = set()
    dirs = list(IMAGE_DIRS) + list(extra_dirs or [])
    for d in dirs:
        if os.path.isdir(d):
            out |= {f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png"))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=DEFAULT_ANN)
    ap.add_argument("--out", default=os.path.join(_HERE, "labels", "val_split.json"))
    ap.add_argument("--images", action="append", default=[],
                    help="train_images 目录（--require-local 判定本地有哪些图；可重复传）")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--ratio", type=float, default=1.0 / 3.0, help="val 占多标注者图的比例")
    ap.add_argument("--min-annotators", type=int, default=2)
    ap.add_argument("--require-local", action="store_true", default=True,
                    help="val 候选池只取本地已有 jpeg 的多标注者图（默认开）")
    ap.add_argument("--allow-missing-local", dest="require_local", action="store_false",
                    help="val 候选池含本地缺图的多标注者图")
    args = ap.parse_args()

    coco = json.load(open(args.ann, encoding="utf-8"))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    annotators = defaultdict(set)
    for ann in coco["annotations"]:
        f = id2file[ann["image_id"]]
        annotators[f].add(ann["image_id"].split("-", 1)[0])

    universe = sorted(annotators)                     # 707 张
    multi = sorted(f for f in universe if len(annotators[f]) >= args.min_annotators)
    have = local_images(args.images)
    multi_all = list(multi)
    if args.require_local:
        # val 候选池只留本地已有 jpeg 的图 —— 保证「val 划分」与「能跑的评测集」完全一致
        multi = [f for f in multi if f in have]

    strata = defaultdict(list)
    for f in multi:
        strata[(f[:4], len(annotators[f]))].append(f)

    rng = random.Random(args.seed)
    val: list = []
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        k = math.ceil(args.ratio * len(members))
        val.extend(members[:k])
    val = sorted(val)

    val_set = set(val)
    train = [f for f in universe if f not in val_set]
    val_eval = [f for f in val if f in have]
    train_local = [f for f in train if f in have]

    digest = hashlib.sha256("\n".join(val).encode("utf-8")).hexdigest()
    obj = {
        "created": "2026-10-31",
        "seed": args.seed,
        "ratio_val_of_multi_annotator": args.ratio,
        "min_annotators_for_val": args.min_annotators,
        "criteria": ("val = 官方训练集中多标注者(>=2)物理图，按 (年代, 标注者数) 分层、"
                     "层内 file_name 排序后固定 seed 洗牌取前 ceil(ratio·层大小)；"
                     "train = 全部 707 张训练图减去 val。仅用公开训练标注信息，不含任何测试图。"),
        "gt_ann": os.path.abspath(args.ann),
        "universe_size": len(universe),
        "multi_annotator_size": len(multi_all),
        "multi_annotator_local_size": len(multi),
        "val_sha256": digest,
        "val_size": len(val),
        "val_annotator_rows": sum(len(annotators[f]) for f in val),
        "val_local_size": len(val_eval),
        "train_size": len(train),
        "train_local_size": len(train_local),
        "val": val,
        "val_local": val_eval,
        "train": train,
        "train_local": train_local,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"[split] universe={len(universe)} multi={len(multi)} "
          f"val={len(val)} (本地有图 {len(val_eval)}) train={len(train)} "
          f"(本地有图 {len(train_local)}) sha256={digest[:16]}… → {args.out}")


if __name__ == "__main__":
    main()
