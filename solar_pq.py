# -*- coding: utf-8 -*-
"""太阳暗条分割（Solar Filament Segmentation Challenge 2026, IEEE BigData Cup）Panoptic Quality 本地复刻。

定义出处：赛会主办 Azim Ahmadzadeh 官方自评 notebook 存档
<工作区>/查漏R1-22-Solar Filament Segmentation Challenge 2026（IEEE BigD/raw/samples-self-evaluation-notebook.md
（源 https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook ，In[3]/In[4]）：

- GT 按"标注者-图像"逐行（gt_annotator_image_ids.unique()），每行取该标注者在该
  图像上的全部暗条；预测取该图像的全部暗条（所有标注者共用同一组预测）。
- 逐对算 IoU；hit 判定 iou_matrix > 0.5（严格大于，0.5 本身不命中）。
- TP = 所有命中"对"（不做一对一唯一匹配：一根预测可与多根 GT 同时命中，反之亦然），
  分子累加命中对的 IoU 本身（不是计数 1）。
- FP = 该行内列和为 0 的预测；FN = 行和为 0 的 GT。
- PQ = Σ IoU(TP) / (|TP| + 0.5·|FP| + 0.5·|FN|)；分母为 0 → 0.0。
- 全部行池化为一个全局 PQ。

歧义点：
1. "逐标注者计分取平均"还是池化：官方 notebook 将所有标注者-图像行累加进同一个
   分子/分母（池化，micro）。本实现默认池化，另提供 per_annotator_pq() 可输出
   宏平均口径作对照——若后续官方榜单被证实按标注者宏平均，需切换。
2. 官方 notebook 中 filament_id 解析同时出现 "_"（get_overlap_df）与 "-"（process_entry）
   两种分隔符，且 GT 行键="标注者-图像"而预测键="图像"。本实现直接接受结构化元组
   (annotator, image, mask) / (image, mask)，把该解析歧义挡在适配层之外。
3. 官方 notebook 用 pycocotools COCO 压缩 RLE 解码 2048×2048 掩膜。本实现用像素集合
   frozenset[(row,col)] 表达掩膜，语义等价（IoU 对逐像素二值掩膜精确成立）；
   接 RLE 时需先经 pycocotools 解码再转像素集合（见 README 差异点）。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

Mask = frozenset  # frozenset[(int, int)] 像素坐标集合
GTEntry = Tuple[str, str, Mask]   # (annotator_id, image_id, mask)
PredEntry = Tuple[str, Mask]      # (image_id, mask)


def iou(a: Mask, b: Mask) -> float:
    """逐像素二值掩膜 IoU；双空集定义为 0.0（不崩溃）。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _row_stats(gt_masks: Sequence[Mask], pred_masks: Sequence[Mask],
               iou_threshold: float) -> Tuple[float, int, int, int]:
    """一行（标注者-图像）的 (ΣIoU(TP), TP, FP, FN)，逻辑逐行复刻 get_pq_score 循环体。"""
    if len(gt_masks) == 0:
        return (0.0, 0, len(pred_masks), 0)
    if len(pred_masks) == 0:
        return (0.0, 0, 0, len(gt_masks))
    hit = [[iou(g, p) > iou_threshold for p in pred_masks] for g in gt_masks]
    tp_iou_sum = sum(iou(g, p)
                     for g_row, g in zip(hit, gt_masks)
                     for h, p in zip(g_row, pred_masks) if h)
    tp = sum(1 for row in hit for h in row if h)
    fp = sum(1 for j in range(len(pred_masks)) if not any(hit[i][j] for i in range(len(hit))))
    fn = sum(1 for row in hit if not any(row))
    return (tp_iou_sum, tp, fp, fn)


def _group_gt(gt: Iterable[GTEntry]) -> Dict[Tuple[str, str], List[Mask]]:
    grouped: Dict[Tuple[str, str], List[Mask]] = {}
    for annotator, image, mask in gt:
        grouped.setdefault((annotator, image), []).append(mask)
    return grouped


def _group_pred(pred: Iterable[PredEntry]) -> Dict[str, List[Mask]]:
    grouped: Dict[str, List[Mask]] = {}
    for image, mask in pred:
        grouped.setdefault(image, []).append(mask)
    return grouped


def solar_pq(gt: Iterable[GTEntry], pred: Iterable[PredEntry],
             iou_threshold: float = 0.5) -> float:
    """官方口径 Panoptic Quality（全标注者-图像行池化）。分母 0 → 0.0。"""
    gt_rows = _group_gt(gt)
    pred_rows = _group_pred(pred)
    tp_iou_sum = 0.0
    tp = fp = fn = 0
    for (annotator, image), gt_masks in gt_rows.items():
        s, t, f_pos, f_neg = _row_stats(gt_masks, pred_rows.get(image, []), iou_threshold)
        tp_iou_sum += s
        tp += t
        fp += f_pos
        fn += f_neg
    denom = tp + 0.5 * fp + 0.5 * fn
    return tp_iou_sum / denom if denom > 0 else 0.0


def per_annotator_pq(gt: Iterable[GTEntry], pred: Iterable[PredEntry],
                     iou_threshold: float = 0.5) -> Dict[str, float]:
    """歧义点 1 的对照口径：按标注者分别池化该标注者全部图像行。"""
    gt_rows = _group_gt(gt)
    pred_rows = _group_pred(pred)
    per: Dict[str, List[float]] = {}
    for (annotator, image), gt_masks in gt_rows.items():
        s, t, f_pos, f_neg = _row_stats(gt_masks, pred_rows.get(image, []), iou_threshold)
        denom = t + 0.5 * f_pos + 0.5 * f_neg
        pq = s / denom if denom > 0 else 0.0
        per.setdefault(annotator, []).append(pq)
    return {a: sum(v) / len(v) for a, v in per.items()}
