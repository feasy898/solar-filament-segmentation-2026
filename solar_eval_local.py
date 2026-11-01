# -*- coding: utf-8 -*-
"""双口径本地评测台：Solar Filament（IEEE BigData Cup 2026）PQ + Dice。

口径定义（官方自评 notebook 复刻件：poc/metrics/metrics_local/solar_pq.py）
--------------------------------------------------------------------------
* **行（row）= (标注者, 图像)**。GT 按「标注者-图像」逐行；预测按图像共享，
  同一张图的预测被该图所有标注者的行复用。
* **命中判定** IoU > 0.5（严格大于，0.5 本身不算命中）。
* **TP 数的是「对」**：一根预测可同时命中多根 GT，一根 GT 也可被多根预测命中，
  每个命中组合各计一个 TP 对；**分子累加命中对的 IoU 本身**（不是计数 1）。
* **FP** = 该行内没命中任何 GT 的预测根数；**FN** = 该行内没被任何预测命中的 GT 根数。
* **PQ（池化，官方口径）** = ΣIoU(TP) / (|TP| + 0.5·|FP| + 0.5·|FN|)，全部行池化。
* **PQ（宏平均，按标注者）** = 先按标注者各自池化出 PQ，再对标注者取平均
  （复刻件 per_annotator_pq() 的口径）。
* **PQ（宏平均，按行）** = 逐行 PQ 的算术平均（额外提供的第三种口径）。

Dice 口径（与 PQ 同构：同样按 (标注者, 图像) 逐行）
--------------------------------------------------------------------------
* 每行取该标注者在该图上的 **GT 掩膜集合的并集 A** 与 **该图预测集合的并集 B**；
* **行 Dice** = 2·|A∩B| / (|A|+|B|)（分母 0 ⇒ 0）。
* **Dice 行池化（micro，像素级池化）** = Σ行 2·|A∩B| / Σ行 (|A|+|B|)。
* **Dice 宏平均** = 逐行 Dice 的算术平均（另给按标注者宏平均作对照）。

GT 只来自**公开训练标注**（train/MAGFiLO_1.0_Annotations_kaggle2026_train.json，
polygon → pycocotools 光栅化，官方语义，见下方 polygon_to_mask）。
测试图 GT 不存在，本脚本**不接受**任何测试图标注作为输入。

用法
------------------------------------------------------------------------------
    # 1) 预测来自官方 CSV（filament_id,segmentation_rle，RLE 解码）
    ./.venv-solar-gpu/bin/python solar_eval_local.py \
        --gt-ann MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --split labels/val_split.json \
        --preds preds_v3_val.csv --pred-format csv --out eval_v3_val.json

    # 2) 预测来自掩膜目录（<stem>.npz，含 int32 实例标签图 labels[2048,2048]）
    ./.venv-solar-gpu/bin/python solar_eval_local.py \
        --gt-ann ... --split labels/val_split.json \
        --preds preds_v3_val_npz --pred-format npz --out eval_v3_val.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rle import HEIGHT, WIDTH, rle_to_mask  # noqa: E402

Mask = frozenset  # frozenset[(row, col)]
GTKey = Tuple[str, str]  # (annotator, image_stem)


def polygon_to_mask(seg) -> Mask:
    """COCO polygon → 像素集合（pycocotools frPyObjects+merge，官方光栅化语义）。

    公开仓库版注：原内部工作区从 build_threshold_baseline.py 导入同名函数，
    本仓库未收录该基线脚本，故将其 8 行实现原样内联（逐字一致）。
    """
    from pycocotools import mask as maskUtils
    import numpy as np
    rles = maskUtils.frPyObjects(seg, HEIGHT, WIDTH)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    rows, cols = np.nonzero(m)
    return frozenset(zip(rows.tolist(), cols.tolist()))



# --------------------------------------------------------------------- 基础
def mask_to_pixels(mask: Mask) -> set:
    """像素集合 → set（Dice 用；solar_pq 的 frozenset 语义等价）。"""
    return set(mask)


def iou(a: Mask, b: Mask) -> float:
    """逐像素二值掩膜 IoU；双空集定义为 0.0。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------- GT 载入
def load_gt_rows(ann_path: str, names: Sequence[str] | None = None
                 ) -> Tuple[Dict[GTKey, List[Mask]], Dict[str, object]]:
    """官方训练标注 → {(标注者, 图像stem): [GT 掩膜, ...]}。

    标注者取 image.id 的 "-" 前缀（image.id 形如 "040301-20140609195854Bh"），
    与 build_threshold_baseline.gt_entries_for_file 同口径。
    names 为 None ⇒ 全部物理图；否则只取给定 file_name（且必须在标注里存在）。
    """
    coco = json.load(open(ann_path, encoding="utf-8"))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    want = set(names) if names is not None else None
    rows: Dict[GTKey, List[Mask]] = {}
    n_ann_total = 0
    for ann in coco["annotations"]:
        file_name = id2file[ann["image_id"]]
        if want is not None and file_name not in want:
            continue
        annotator = ann["image_id"].split("-", 1)[0]
        stem = os.path.splitext(file_name)[0]
        rows.setdefault((annotator, stem), []).append(polygon_to_mask(ann["segmentation"]))
        n_ann_total += 1
    meta = {
        "gt_images": len({k[1] for k in rows}),
        "gt_rows": len(rows),
        "gt_annotators": len({k[0] for k in rows}),
        "gt_masks": n_ann_total,
    }
    return rows, meta


# ----------------------------------------------------------------- 预测载入
def load_preds_csv(path: str) -> Dict[str, List[Mask]]:
    """官方提交 CSV（filament_id,segmentation_rle）→ {image_stem: [掩膜, ...]}。

    filament_id = "<image_id>_<k>"；image_id 与图像文件名词干一致。
    RLE 用项目内 rle.py 解码（与 pycocotools 交叉对拍过）。
    """
    out: Dict[str, List[Mask]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if [h.strip() for h in header[:2]] != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"CSV 表头应为 filament_id,segmentation_rle，实际 {header}")
        for row in reader:
            if not row:
                continue
            fid, rle = row[0], row[1]
            image_id = fid.rsplit("_", 1)[0]
            out.setdefault(image_id, []).append(rle_to_mask(rle, HEIGHT, WIDTH))
    return out


def load_preds_npz_dir(path: str) -> Dict[str, List[Mask]]:
    """掩膜目录 → {image_stem: [掩膜, ...]}。

    支持两种 npz 内容：
      * "labels"：int32 (H, W) 实例标签图，0 = 背景，k>0 = 第 k 个实例；
      * "masks" ：object 数组，每项一个 (H, W) bool 掩膜。
    """
    import numpy as np
    out: Dict[str, List[Mask]] = {}
    for fn in sorted(os.listdir(path)):
        if not fn.lower().endswith(".npz"):
            continue
        stem = os.path.splitext(fn)[0]
        with np.load(os.path.join(path, fn), allow_pickle=True) as z:
            masks: List[Mask] = []
            if "labels" in z:
                lab = z["labels"]
                for k in range(1, int(lab.max()) + 1):
                    rr, cc = np.nonzero(lab == k)
                    if rr.size:
                        masks.append(frozenset(zip(rr.tolist(), cc.tolist())))
            elif "masks" in z:
                for m in z["masks"]:
                    m = np.asarray(m).astype(bool)
                    rr, cc = np.nonzero(m)
                    if rr.size:
                        masks.append(frozenset(zip(rr.tolist(), cc.tolist())))
            else:
                raise ValueError(f"{fn}: npz 里既没有 labels 也没有 masks")
        if masks:
            out[stem] = masks
    return out


# ------------------------------------------------------------------- PQ 口径
def _row_stats(gt_masks: Sequence[Mask], pred_masks: Sequence[Mask],
               iou_threshold: float) -> Tuple[float, int, int, int]:
    """一行（标注者-图像）的 (ΣIoU(TP), |TP|, |FP|, |FN|)。

    逻辑与官方复刻件 solar_pq._row_stats 一致：TP 数「对」，分子累加命中对的 IoU。
    """
    if len(gt_masks) == 0:
        return (0.0, 0, len(pred_masks), 0)
    if len(pred_masks) == 0:
        return (0.0, 0, 0, len(gt_masks))
    ious = [[iou(g, p) for p in pred_masks] for g in gt_masks]
    hits = [[v > iou_threshold for v in row] for row in ious]
    tp_iou_sum = sum(ious[i][j] for i in range(len(gt_masks))
                     for j in range(len(pred_masks)) if hits[i][j])
    tp = sum(1 for row in hits for h in row if h)
    fp = sum(1 for j in range(len(pred_masks))
             if not any(hits[i][j] for i in range(len(hits))))
    fn = sum(1 for row in hits if not any(row))
    return (tp_iou_sum, tp, fp, fn)


def pq_report(gt_rows: Dict[GTKey, List[Mask]], preds: Dict[str, List[Mask]],
              iou_threshold: float = 0.5) -> Dict[str, object]:
    """PQ 四种口径 + 计数。

    * pq                 池化（官方口径）：全部行池进同一分子/分母
    * pq_macro_per_annotator  **按标注者分组，先在标注者内对「逐行 PQ」取平均，
                        再对标注者取平均** —— 与官方复刻件 per_annotator_pq() 完全同义
                        （该函数返回 {标注者: 其各行 PQ 的均值}）
    * pq_macro_per_row   逐行 PQ 的算术平均
    * pq_macro_per_image 每张图的各行先池化成该图一个 PQ，再对图取平均
                         （= 本工作区历史日志里的「macroPQ（各图池化后平均）」口径，
                          见 WORKSPACE.md §6.3）
    """
    tp_iou_sum = 0.0
    tp = fp = fn = 0
    per_annot: Dict[str, List[Tuple[float, int, int, int]]] = {}
    per_image: Dict[str, List[Tuple[float, int, int, int]]] = {}
    per_row_pq: List[float] = []
    for (annotator, image), gt_masks in gt_rows.items():
        s, t, f_pos, f_neg = _row_stats(gt_masks, preds.get(image, []), iou_threshold)
        tp_iou_sum += s
        tp += t
        fp += f_pos
        fn += f_neg
        per_annot.setdefault(annotator, []).append((s, t, f_pos, f_neg))
        per_image.setdefault(image, []).append((s, t, f_pos, f_neg))
        denom = t + 0.5 * f_pos + 0.5 * f_neg
        per_row_pq.append(s / denom if denom > 0 else 0.0)
    denom = tp + 0.5 * fp + 0.5 * fn
    annot_pqs = []
    for ann, rows in per_annot.items():
        # 官方复刻件 per_annotator_pq 的语义：**标注者内先对逐行 PQ 取平均**
        # （不是把该标注者的所有行池化成同一个 PQ）
        row_pqs = []
        for (s, t, f_pos, f_neg) in rows:
            d = t + 0.5 * f_pos + 0.5 * f_neg
            row_pqs.append(s / d if d > 0 else 0.0)
        annot_pqs.append(sum(row_pqs) / len(row_pqs))
    image_pqs = []
    for image, rows in per_image.items():
        i_s = sum(r[0] for r in rows)
        i_t = sum(r[1] for r in rows)
        i_f = sum(r[2] for r in rows)
        i_n = sum(r[3] for r in rows)
        d = i_t + 0.5 * i_f + 0.5 * i_n
        image_pqs.append(i_s / d if d > 0 else 0.0)
    return {
        "pq": (tp_iou_sum / denom) if denom > 0 else 0.0,
        "pq_macro_per_annotator": (sum(annot_pqs) / len(annot_pqs)) if annot_pqs else 0.0,
        "pq_macro_per_row": (sum(per_row_pq) / len(per_row_pq)) if per_row_pq else 0.0,
        "pq_macro_per_image": (sum(image_pqs) / len(image_pqs)) if image_pqs else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_iou_sum": tp_iou_sum,
        "n_rows": len(gt_rows),
        "iou_threshold": iou_threshold,
    }


# ----------------------------------------------------------------- Dice 口径
def dice_report(gt_rows: Dict[GTKey, List[Mask]], preds: Dict[str, List[Mask]]
                ) -> Dict[str, object]:
    """Dice：逐行 Dice + 行池化（micro）+ 两种宏平均。"""
    pred_union_cache: Dict[str, set] = {}

    def pred_union(image: str) -> set:
        u = pred_union_cache.get(image)
        if u is None:
            u = set()
            for m in preds.get(image, []):
                u |= m
            pred_union_cache[image] = u
        return u

    row_dice: List[float] = []
    inter_total = 0
    denom_total = 0
    per_annot: Dict[str, List[Tuple[int, int]]] = {}
    for (annotator, image), gt_masks in gt_rows.items():
        a: set = set()
        for m in gt_masks:
            a |= m
        b = pred_union(image)
        inter = len(a & b)
        da, db = len(a), len(b)
        den = da + db
        row_dice.append((2.0 * inter / den) if den > 0 else 0.0)
        inter_total += 2 * inter
        denom_total += den
        per_annot.setdefault(annotator, []).append((2 * inter, den))
    annot_dice = []
    for ann, rows in per_annot.items():
        i_s = sum(r[0] for r in rows)
        d_s = sum(r[1] for r in rows)
        annot_dice.append(i_s / d_s if d_s > 0 else 0.0)
    return {
        "dice_pooled": (inter_total / denom_total) if denom_total > 0 else 0.0,
        "dice_macro_per_row": (sum(row_dice) / len(row_dice)) if row_dice else 0.0,
        "dice_macro_per_annotator": (sum(annot_dice) / len(annot_dice)) if annot_dice else 0.0,
        "dice_num_pixels": inter_total,   # Σ行 2·|A∩B|（池化 Dice 的分子）
        "dice_den_pixels": denom_total,   # Σ行 (|A|+|B|)（池化 Dice 的分母）
        "n_rows": len(gt_rows),
    }


# ------------------------------------------------------------------ 汇总入口
def evaluate(gt_rows: Dict[GTKey, List[Mask]], preds: Dict[str, List[Mask]],
             iou_threshold: float = 0.5) -> Dict[str, object]:
    """PQ + Dice 一次性汇总。"""
    out: Dict[str, object] = {}
    out.update(pq_report(gt_rows, preds, iou_threshold))
    out.update(dice_report(gt_rows, preds))
    out["gt_images"] = len({k[1] for k in gt_rows})
    out["gt_annotators"] = len({k[0] for k in gt_rows})
    out["pred_images"] = len(preds)
    out["pred_masks"] = sum(len(v) for v in preds.values())
    return out


def _read_split(path: str) -> Tuple[List[str], Dict[str, object]]:
    """读 val_split.json → (val 的 file_name 列表, 其余元信息)。"""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    val = obj["val"] if isinstance(obj, dict) else obj
    return list(val), obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Solar 双口径本地评测台（PQ + Dice）")
    ap.add_argument("--gt-ann", required=True, help="公开训练标注（COCO JSON）")
    ap.add_argument("--split", default=None, help="labels/val_split.json（取 val 列表）")
    ap.add_argument("--names", default=None, help="file_name 名单（一行一个），与 --split 二选一")
    ap.add_argument("--preds", required=True, help="预测：官方 CSV 或掩膜 npz 目录")
    ap.add_argument("--pred-format", choices=["auto", "csv", "npz"], default="auto")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="报告 JSON 落盘路径")
    args = ap.parse_args()

    if args.split and args.names:
        sys.exit("--split 与 --names 只能给一个")
    if args.split:
        names, split_meta = _read_split(args.split)
    elif args.names:
        names = [ln.strip() for ln in open(args.names, encoding="utf-8") if ln.strip()]
        split_meta = None
    else:
        names, split_meta = None, None

    fmt = args.pred_format
    if fmt == "auto":
        fmt = "csv" if args.preds.lower().endswith(".csv") else "npz"
    preds = load_preds_csv(args.preds) if fmt == "csv" else load_preds_npz_dir(args.preds)

    gt_rows, gt_meta = load_gt_rows(args.gt_ann, names)
    report = evaluate(gt_rows, preds, args.iou_threshold)
    report.update(gt_meta)
    report["pred_format"] = fmt
    report["pred_source"] = os.path.abspath(args.preds)
    report["gt_ann"] = os.path.abspath(args.gt_ann)
    if split_meta is not None:
        report["split_file"] = os.path.abspath(args.split)
        report["split_val_size"] = len(split_meta.get("val", [])) if isinstance(split_meta, dict) else None

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[eval] 报告 → {args.out}")


if __name__ == "__main__":
    main()
