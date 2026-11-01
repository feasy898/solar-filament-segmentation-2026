# -*- coding: utf-8 -*-
"""Solar · 官方提交 CSV writer + 提交前校验器。

格式契约（raw/official.md "Submission File"）：
  - 单个 CSV，两列：filament_id,segmentation_rle
  - 每行一根预测暗条；filament_id = "<image_id>_<k>"，k 只为去重（_1/_2/...），
    image_id 部分必须与官方图像 id 逐字一致；
  - segmentation_rle = COCO 压缩 RLE counts 字符串（2048×2048 定尺，无需 Size）；
  - **不要引号**（官方原话 "Do not include quotations"）；
  - 行数 = 预测暗条总数，可与 GT 数不同（按重叠匹配）。

writer 从 {(image_id): [mask, ...]} 生成 CSV 文本；校验器独立可对任何来源的
CSV 文本做提交前硬闸检查（D1 通管线时，红队可用它故意构造坏提交做验收）。
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Sequence, Tuple

from rle import HEIGHT, WIDTH, Mask, mask_to_rle, rle_to_mask

HEADER = "filament_id,segmentation_rle"


def write_submission_csv(preds: Dict[str, Sequence[Mask]], eol: str = "\n") -> str:
    """{image_id: [mask,...]} → 官方 CSV 文本（无引号、无 Size 列）。

    同一 image_id 内按输入顺序编号 _1.._k；mask 必须非空（空暗条无意义且
    浪费一行 FP 风险，直接 ValueError）。
    """
    buf = io.StringIO()
    buf.write(HEADER + eol)
    for image_id, masks in preds.items():
        if not image_id or any(ch in image_id for ch in ",\"\n\r"):
            raise ValueError(f"非法 image_id: {image_id!r}")
        for k, mask in enumerate(masks, start=1):
            if not mask:
                raise ValueError(f"{image_id} 第 {k} 根掩膜为空——空预测只造 FP，禁止提交")
            rle = mask_to_rle(mask, HEIGHT, WIDTH)
            buf.write(f"{image_id}_{k},{rle}{eol}")
    return buf.getvalue()


def validate_submission_csv(text: str, expect_images: "Sequence[str] | None" = None,
                            height: int = HEIGHT, width: int = WIDTH
                            ) -> List[str]:
    """提交前硬闸。返回问题清单（空 = 通过）。

    检查项：表头、两列、filament_id 唯一、image_id 前缀合法（可选白名单）、
    RLE 无引号/引号类字符、RLE 可解码且恰覆盖 2048×2048、掩膜非空。
    """
    issues: List[str] = []
    lines = text.splitlines()
    if not lines:
        return ["CSV 为空"]
    if lines[0].strip() != HEADER:
        issues.append(f"表头应为 {HEADER!r}，实际 {lines[0].strip()!r}")
    seen: set = set()
    n_rows = 0
    for ln in lines[1:]:
        if not ln.strip():
            issues.append(f"第 {n_rows + 2} 行为空行")
            continue
        parts = ln.split(",")
        n_rows += 1
        if len(parts) != 2:
            issues.append(f"第 {n_rows + 1} 行列数={len(parts)}（应为 2）: {ln[:40]}...")
            continue
        fid, rle = parts
        if fid in seen:
            issues.append(f"filament_id 重复: {fid}")
        seen.add(fid)
        if "_" not in fid:
            issues.append(f"filament_id 缺行尾序号: {fid}")
        image_id = fid.rsplit("_", 1)[0]
        if expect_images is not None and image_id not in expect_images:
            issues.append(f"未知 image_id: {image_id}")
        if any(q in rle for q in "'\""):
            issues.append(f"{fid}: RLE 含引号（官方禁止）")
        if any(ch in rle for ch in " \t"):
            issues.append(f"{fid}: RLE 含空白")
        try:
            mask = rle_to_mask(rle, height, width)
        except ValueError as e:
            issues.append(f"{fid}: RLE 解码失败（{e}）")
            continue
        if not mask:
            issues.append(f"{fid}: 空掩膜")
    return issues


def submission_to_pred_entries(text: str, height: int = HEIGHT, width: int = WIDTH
                               ) -> List[Tuple[str, Mask]]:
    """提交 CSV 文本 → solar_pq 的 PredEntry 列表（本地自评入口）。"""
    entries: List[Tuple[str, Mask]] = []
    for i, ln in enumerate(text.splitlines()[1:]):
        if not ln.strip():
            continue
        fid, rle = ln.split(",")
        entries.append((fid.rsplit("_", 1)[0], rle_to_mask(rle, height, width)))
    return entries
