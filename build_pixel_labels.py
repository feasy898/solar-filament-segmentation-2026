# -*- coding: utf-8 -*-
"""Solar · 多标注者多边形标注 → 像素级训练素材（2026-10-31）。

输入只有 train 目录的官方 COCO JSON（绝不读 test/ 测试图目录的任何标注或标签）：
  MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  （1,154 条目 = 1,154 个**互不重复**的 (标注者前缀, file_name) 对；37 位标注者；
   每物理图条目数 M∈{1:411, 2:145, 3:151}；全部 2048×2048、单 polygon、iscrowd=0）

产出（落盘 data/solar/labels/）：
  (A) gt_per_annotator/{stem}.npz —— 按标注者分行的评测 GT。每行 = (标注者 id,
      file_name, 该标注者在这张图上的全部实例掩膜集合)，掩膜 2048×2048 二值
      uint8。键名：file_name / annotators（行序）/ masks_{i:03d}=(n_i,2048,2048)
      / categories_{i:03d}=(n_i,)int16 / ann_ids_{i:03d}=(n_i,)U / image_ids_{i:03d}。
      行清单见 gt_index.json。
  (B) consensus/{stem}.npz —— 共识训练标签。所有标注者的实例光栅化到同一
      2048×2048 画布后逐像素投票：
        support2048  u8  该像素被多少个标注者的实例（任意类别）覆盖，0..M
        votes2048    u8×4通道 按类别的标注者覆盖票数（通道序=CLASS_IDS）
        class2048    u8  0=背景，1..4=argmax(votes)（平票取通道序：
                         Left>Right>Unidentifiable>Ambiguous）
        support512   f16 support 的 4×4 块均值（强度图）
        votes512     u8×4 votes 的 4×4 块和（真实上限 M·16=48）
        class512     u8  同 class2048 语义
        meta_json    元数据（M/scale/orig_h/orig_w/各类实例数/平票像素数…）
      Unidentifiable/Ambiguous **单列通道**（通道 2/3），不丢：本批数据实测
      Ambiguous(id=4) 实例数为 0（categories 表有该类但 8,199 实例全落在
      Left/Right/Unidentifiable），通道照常保留、meta 如实记录。
      行清单见 consensus_index.json，总账见 build_manifest.json。

口径对齐：光栅化用 pycocotools frPyObjects+merge（=官方语义，与
annot_consensus_probe2.poly_to_arr 逐字节一致）；support 的定义与 probe2 的
投票数组（逐标注者全实例并集 → 跨标注者求和）严格同构，测试
test_vote_matches_probe2_on_real_multiannotator_files 在真实多标注者图上对拍。

官方 PQ 口径（poc/metrics/metrics_local/solar_pq.py）：GT 按「标注者-图像」逐行、
预测按图像共享、IoU>0.5 命中、TP 计对数不一一匹配、全行池化 —— 因此 (A) 按
标注者分行落盘、(B) 的共识形状（实验结论 support≥2 最优）是逐图最优单预测。

用法：
    .venv-solar-gpu/bin/python build_pixel_labels.py build [--limit N]
    .venv-solar-gpu/bin/python build_pixel_labels.py verify
    .venv-solar-gpu/bin/python -m unittest test_build_pixel_labels -v
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

# ------------------------------------------------------------------ 常量与路径
HERE = os.path.dirname(os.path.abspath(__file__))
HEIGHT = WIDTH = 2048                    # 官方定尺（rle.py 同款常量语义）
CLASS_IDS: Tuple[int, ...] = (1, 2, 3, 4)
CLASS_NAMES: Dict[int, str] = {1: "Left", 2: "Right",
                               3: "Unidentifiable", 4: "Ambiguous"}
DOWNSAMPLE_FACTOR = 4
DOWNSAMPLE_SIZE = HEIGHT // DOWNSAMPLE_FACTOR   # 512

DEFAULT_ANN_PATH = os.path.join(
    HERE, "MAGFiLO_1.0_Kaggle_2026", "train",
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json")   # 只在 train 目录，无 test
DEFAULT_LABELS_DIR = os.path.join(HERE, "labels")
TIE_POLICY = "argmax-first-channel: Left>Right>Unidentifiable>Ambiguous"


# ------------------------------------------------------------------- 读取与行

def class_vocabulary(ann_path: str = DEFAULT_ANN_PATH) -> Dict[int, str]:
    """JSON categories → {category_id: name}；类别集合必须恰好 4 类。"""
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    vocab = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    if set(vocab.values()) != set(CLASS_NAMES.values()):
        raise ValueError(f"类别集合不是预期 4 类: {vocab}")
    if set(vocab) != set(CLASS_IDS):
        raise ValueError(f"category_id 集合不是 {CLASS_IDS}: {sorted(vocab)}")
    return vocab


def rows_from_coco(coco: dict) -> List[dict]:
    """COCO JSON → 标注者-图像行列表（每行 = 一个 (标注者前缀, file_name) 对）。

    1,154 条目互不重复（test_gt_row_count_equals_annotator_image_pairs 守卫）；
    行内 anns 只挂本条目 image_id；行序按 (file_name, annotator) 确定化。
    """
    anns_by_img: Dict[str, List[dict]] = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    rows: List[dict] = []
    seen = set()
    for im in coco["images"]:
        pre, sep, stem = str(im["id"]).partition("-")
        if not sep or stem != os.path.splitext(im["file_name"])[0]:
            raise ValueError(f"image.id 不符合 '标注者-图名' 形态: {im['id']}")
        key = (pre, im["file_name"])
        if key in seen:
            raise ValueError(f"(标注者,文件) 对重复: {key}")
        seen.add(key)
        rows.append({
            "annotator": pre,
            "file_name": im["file_name"],
            "image_id": str(im["id"]),
            "height": int(im["height"]),
            "width": int(im["width"]),
            "anns": anns_by_img.get(str(im["id"]), []),
        })
    rows.sort(key=lambda r: (r["file_name"], r["annotator"]))
    return rows


def load_rows(ann_path: str = DEFAULT_ANN_PATH) -> Tuple[Dict[int, str], List[dict]]:
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    vocab = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    if set(vocab.values()) != set(CLASS_NAMES.values()) or set(vocab) != set(CLASS_IDS):
        raise ValueError(f"类别集合不是预期 4 类: {vocab}")
    return vocab, rows_from_coco(coco)


# ------------------------------------------------------------------ 光栅化/共识

def rasterize(seg, h: int, w: int) -> np.ndarray:
    """COCO polygon → (h,w) 二值 uint8（pycocotools 官方语义，=probe2.poly_to_arr）。"""
    from pycocotools import mask as maskUtils
    rles = maskUtils.frPyObjects(seg, h, w)
    return maskUtils.decode(maskUtils.merge(rles)).astype(np.uint8)


def consensus_for_file(rows: List[dict]) -> Dict[str, np.ndarray]:
    """一个物理图的全部标注者行 → 逐像素投票（support + 4 类别票 + 类别图）。

    support[p]   = 覆盖 p 的标注者数（每位标注者按其全部实例的并集计 1 票，
                   与 annot_consensus_probe2.py:166-178 的 votes 数组同构）。
    votes[c][p]  = 在类别 c 上覆盖 p 的标注者数（每位标注者每类最多 1 票）。
    class[p]     = argmax(votes)+1；全 0 → 0；平票取通道序（TIE_POLICY）。
    """
    hs = {r["height"] for r in rows}
    ws = {r["width"] for r in rows}
    if len(hs) != 1 or len(ws) != 1:
        raise ValueError(f"同图各条目尺寸不一致: {hs}x{ws}")
    h, w = hs.pop(), ws.pop()
    n_class = len(CLASS_IDS)
    support = np.zeros((h, w), dtype=np.uint8)
    votes = np.zeros((n_class, h, w), dtype=np.uint8)
    intra_multiclass_px = 0
    for r in rows:
        union_ann = np.zeros((h, w), dtype=np.uint8)
        class_cov = np.zeros((n_class, h, w), dtype=np.uint8)
        for a in r["anns"]:
            m = rasterize(a["segmentation"], h, w)
            union_ann |= m
            class_cov[int(a["category_id"]) - 1] |= m
        votes += class_cov
        support += (union_ann > 0).astype(np.uint8)
        intra_multiclass_px += int(((class_cov > 0).sum(0) > 1).sum())
    tot = votes.sum(0)
    class_map = (np.argmax(votes, 0) + 1).astype(np.uint8)
    class_map[tot == 0] = 0
    mx = votes.max(0)
    n_tie = int((((votes == mx).sum(0) > 1) & (tot > 0)).sum())
    return {"support2048": support, "votes2048": votes, "class2048": class_map,
            "n_tie_pixels": n_tie, "intra_multiclass_px": intra_multiclass_px,
            "height": h, "width": w}


def downsample(support: np.ndarray, votes: np.ndarray,
               factor: int = DOWNSAMPLE_FACTOR) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(support,(4,h,w)) → (块均值 f16 support, 块和 u8 votes, 类别图 u8)。"""
    h, w = support.shape
    if h % factor or w % factor:
        raise ValueError(f"尺寸 {h}x{w} 不被 factor={factor} 整除")
    s = support.reshape(h // factor, factor, w // factor, factor) \
               .sum(axis=(1, 3), dtype=np.float32) / float(factor * factor)
    v = votes.reshape(4, h // factor, factor, w // factor, factor) \
             .sum(axis=(2, 4), dtype=np.uint32)
    v = np.minimum(v, 255).astype(np.uint8)
    c = (np.argmax(v, 0) + 1).astype(np.uint8)
    c[v.sum(0) == 0] = 0
    return s.astype(np.float16), v, c


# ------------------------------------------------------------------- (A) GT npz

def _save_gt_npz(entries: List[Tuple[str, dict]], file_name: str,
                 h: int, w: int, out_dir: str) -> dict:
    """entries=[(标注者, 行数组dict)]（按标注者序）→ gt_per_annotator/{stem}.npz。"""
    os.makedirs(out_dir, exist_ok=True)
    entries = sorted(entries, key=lambda e: e[0])
    payload: Dict[str, np.ndarray] = {
        "file_name": np.array(file_name),
        "height": np.array(h, dtype=np.int32),
        "width": np.array(w, dtype=np.int32),
        "annotators": np.array([ann for ann, _ in entries]),
    }
    for i, (ann, arrs) in enumerate(entries):
        payload[f"masks_{i:03d}"] = arrs["masks"]
        payload[f"categories_{i:03d}"] = arrs["categories"]
        payload[f"ann_ids_{i:03d}"] = arrs["ann_ids"]
        payload[f"image_ids_{i:03d}"] = arrs["image_ids"]
    stem = os.path.splitext(file_name)[0]
    path = os.path.join(out_dir, stem + ".npz")
    np.savez_compressed(path, **payload)
    n_inst = int(sum(e[1]["masks"].shape[0] for e in entries))
    rows_meta = [{"row_key": f"masks_{i:03d}", "annotator": ann,
                  "file_name": file_name,
                  "image_id": f"{ann}-{stem}",
                  "n_instances": int(arrs["masks"].shape[0]),
                  "category_ids": [int(c) for c in arrs["categories"]],
                  "ann_ids": [str(x) for x in arrs["ann_ids"]]}
                 for i, (ann, arrs) in enumerate(entries)]
    return {"npz": path, "file_name": file_name, "n_rows": len(entries),
            "n_instances": n_inst, "rows": rows_meta}


def _row_to_entry(row: dict, h: int, w: int) -> Tuple[str, dict]:
    """标注者行 → (标注者, npz 行数组 dict)（光栅化全部实例）。"""
    masks = np.stack([rasterize(a["segmentation"], h, w)
                      for a in row["anns"]], 0) if row["anns"] \
        else np.zeros((0, h, w), dtype=np.uint8)
    return (row["annotator"], {
        "masks": masks,
        "categories": np.array([int(a["category_id"]) for a in row["anns"]],
                               dtype=np.int16),
        "ann_ids": np.array([str(a["id"]) for a in row["anns"]]),
        "image_ids": np.array([str(a.get("image_id", row.get("image_id", "")))
                               for a in row["anns"]]),
    })


def _write_gt_npz(rows: List[dict], out_dir: str) -> dict:
    """同一物理图的全部标注者行 → labels/gt_per_annotator/{stem}.npz（覆盖写）。"""
    fn = rows[0]["file_name"]
    hs = {r["height"] for r in rows}
    ws = {r["width"] for r in rows}
    if len(hs) != 1 or len(ws) != 1:
        raise ValueError(f"{fn} 各条目尺寸不一致: {hs}x{ws}")
    h, w = hs.pop(), ws.pop()
    return _save_gt_npz([_row_to_entry(r, h, w) for r in rows], fn, h, w, out_dir)


def write_gt_row_npz(row: dict, out_dir: str) -> dict:
    """单行写入（同文件多行时在 npz 载荷级合并；供测试与增量使用）。"""
    stem = os.path.splitext(row["file_name"])[0]
    path = os.path.join(out_dir, stem + ".npz")
    entries: List[Tuple[str, dict]] = []
    if os.path.exists(path):                      # 已有 → 原样读回旧行（不重光栅化）
        with np.load(path, allow_pickle=False) as z:
            fn, h, w = str(z["file_name"]), int(z["height"]), int(z["width"])
            for i, ann in enumerate(str(x) for x in z["annotators"]):
                entries.append((ann, {"masks": z[f"masks_{i:03d}"],
                                      "categories": z[f"categories_{i:03d}"],
                                      "ann_ids": z[f"ann_ids_{i:03d}"],
                                      "image_ids": z[f"image_ids_{i:03d}"]}))
    else:
        fn, h, w = row["file_name"], row["height"], row["width"]
    if row["file_name"] != fn:
        raise ValueError(f"file_name 不一致: {row['file_name']} vs {fn}")
    entries = [e for e in entries if e[0] != row["annotator"]]   # 同标注者幂等覆盖
    entries.append(_row_to_entry(row, h, w))
    return _save_gt_npz(entries, fn, h, w, out_dir)


# ------------------------------------------------------------------- (B) npz

def write_consensus_npz(rows: List[dict], out_dir: str) -> dict:
    """一个物理图 → labels/consensus/{stem}.npz（2048 原生 + 512 训练友好）。"""
    os.makedirs(out_dir, exist_ok=True)
    cons = consensus_for_file(rows)
    s512, v512, c512 = downsample(cons["support2048"], cons["votes2048"])
    fn = rows[0]["file_name"]
    stem = os.path.splitext(fn)[0]
    per_class = collections.Counter(int(a["category_id"]) for r in rows
                                    for a in r["anns"])
    meta = {
        "file_name": fn,
        "annotators": sorted(r["annotator"] for r in rows),
        "M": len(rows),
        "orig_h": cons["height"], "orig_w": cons["width"],
        "factor": DOWNSAMPLE_FACTOR, "scale": DOWNSAMPLE_FACTOR,
        "out_size": DOWNSAMPLE_SIZE,
        "channel_order": [CLASS_NAMES[c] for c in CLASS_IDS],
        "class_ids": list(CLASS_IDS),
        "tie_policy": TIE_POLICY,
        "n_instances": sum(len(r["anns"]) for r in rows),
        "n_instances_per_class": {CLASS_NAMES[k]: per_class.get(k, 0)
                                  for k in CLASS_IDS},
        "n_tie_pixels_2048": int(cons["n_tie_pixels"]),
        "n_tie_pixels_512": int((((v512 == v512.max(0)).sum(0) > 1)
                                 & (v512.sum(0) > 0)).sum()),
        "intra_multiclass_px": int(cons["intra_multiclass_px"]),
        "support_max": int(cons["support2048"].max()),
        "note": "support=每像素覆盖标注者数(按全实例并集); votes=按类别标注者票数; "
                "class=argmax(votes)+1, 0=背景, 平票取通道序",
    }
    path = os.path.join(out_dir, stem + ".npz")
    np.savez_compressed(
        path,
        support2048=cons["support2048"], votes2048=cons["votes2048"],
        class2048=cons["class2048"],
        support512=s512, votes512=v512, class512=c512,
        file_name=np.array(fn),
        annotators=np.array(meta["annotators"]),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return {"npz": path, "file_name": fn, "M": len(rows),
            "n_instances": meta["n_instances"],
            "n_instances_per_class": meta["n_instances_per_class"],
            "support_max": meta["support_max"],
            "n_tie_pixels_2048": meta["n_tie_pixels_2048"],
            "intra_multiclass_px": meta["intra_multiclass_px"],
            "support_ge2_px": int((cons["support2048"] >= 2).sum())}


# -------------------------------------------------------------------- 构建

def build_all(ann_path: str = DEFAULT_ANN_PATH, labels_dir: str = DEFAULT_LABELS_DIR,
              limit: int | None = None) -> dict:
    """全量构建 (A)+(B)，写 index/manifest，返回统计 dict（也打印）。"""
    t0 = time.time()
    vocab, rows = load_rows(ann_path)
    by_file: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        by_file[r["file_name"]].append(r)
    files = sorted(by_file)
    if limit:
        files = files[:limit]
    img_dir = os.path.join(os.path.dirname(ann_path), "train_images")
    gt_dir = os.path.join(labels_dir, "gt_per_annotator")
    co_dir = os.path.join(labels_dir, "consensus")

    gt_rows_index: List[dict] = []
    co_index: List[dict] = []
    n_gt_inst = 0
    per_class = collections.Counter()
    per_pop = collections.Counter()
    tie_px_total = 0
    intra_px_total = 0
    sup2_px_total = 0
    n_empty_masks = 0
    t_ras = 0.0
    for k, fn in enumerate(files, 1):
        rs = by_file[fn]
        ta = time.time()
        info = _write_gt_npz(rs, gt_dir)
        gt_rows_index.extend(info["rows"])
        n_gt_inst += info["n_instances"]
        for r in rs:
            for a in r["anns"]:
                per_class[int(a["category_id"])] += 1
                if a.get("area", 0) <= 0:
                    n_empty_masks += 1
        cons = write_consensus_npz(rs, co_dir)
        t_ras += time.time() - ta   # (A)+(B) 本文件的光栅化+压缩耗时（单次计入）
        co_index.append({"file_name": fn, "npz": os.path.relpath(cons["npz"], labels_dir),
                         "M": cons["M"], "n_instances": cons["n_instances"],
                         "n_instances_per_class": cons["n_instances_per_class"],
                         "support_max": cons["support_max"]})
        tie_px_total += cons["n_tie_pixels_2048"]
        intra_px_total += cons["intra_multiclass_px"]
        sup2_px_total += cons["support_ge2_px"]
        per_pop[cons["M"]] += 1
        if k % 100 == 0 or k == len(files):
            print(f"  [{k}/{len(files)}] rows={len(gt_rows_index)} "
                  f"inst={n_gt_inst} 光栅化+压缩累计 {t_ras:.1f}s", flush=True)

    images_on_disk = 0
    if os.path.isdir(img_dir):
        names = set(os.listdir(img_dir))
        images_on_disk = sum(1 for fn in files if fn in names)

    manifest = {
        "ann_path": ann_path,
        "labels_dir": labels_dir,
        "vocab": {str(k): v for k, v in vocab.items()},
        "n_entries": len(rows),
        "n_files": len(files),
        "n_gt_rows": len(gt_rows_index),
        "n_gt_instances": n_gt_inst,
        "n_instances_per_class": {CLASS_NAMES[k]: per_class.get(k, 0)
                                  for k in CLASS_IDS},
        "observed_class_ids": sorted(per_class),
        "ambiguous_note": "categories 表含 Ambiguous(id=4) 但本批 8,199 实例中 "
                          "category_id=4 的实例数为 0；votes 通道 3 照常保留（全 0）",
        "entries_per_file_dist": {str(k): v for k, v in sorted(per_pop.items())},
        "n_annotators": len({r["annotator"] for r in rows}),
        "mask_shape": [HEIGHT, WIDTH],
        "downsample": {"factor": DOWNSAMPLE_FACTOR, "size": DOWNSAMPLE_SIZE,
                       "tie_policy": TIE_POLICY},
        "aggregate": {"tie_pixels_2048_total": tie_px_total,
                      "intra_multiclass_px_total": intra_px_total,
                      "support_ge2_px_total": sup2_px_total,
                      "nonpositive_area_anns": n_empty_masks},
        "images_on_disk": images_on_disk,
        "rasterize_compress_seconds": round(t_ras, 2),
        "total_seconds": round(time.time() - t0, 2),
    }
    os.makedirs(labels_dir, exist_ok=True)
    with open(os.path.join(labels_dir, "gt_index.json"), "w", encoding="utf-8") as f:
        json.dump({"mask_shape": [HEIGHT, WIDTH], "dtype": "uint8",
                   "rows": gt_rows_index}, f, ensure_ascii=False)
    with open(os.path.join(labels_dir, "consensus_index.json"), "w", encoding="utf-8") as f:
        json.dump({"files": co_index}, f, ensure_ascii=False)
    with open(os.path.join(labels_dir, "build_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "vocab"},
                     ensure_ascii=False, indent=1))
    return manifest


# -------------------------------------------------------------------- 校验

def verify_outputs(ann_path: str = DEFAULT_ANN_PATH,
                   labels_dir: str = DEFAULT_LABELS_DIR) -> dict:
    """构建后全量校验（真实重读所有 npz + 与 probe2 全量对拍）。"""
    from scipy import ndimage
    import annot_consensus_probe2 as probe2

    t0 = time.time()
    vocab, rows = load_rows(ann_path)
    pairs = {(r["annotator"], r["file_name"]) for r in rows}
    with open(os.path.join(labels_dir, "gt_index.json"), encoding="utf-8") as f:
        gt_index = json.load(f)["rows"]
    with open(os.path.join(labels_dir, "consensus_index.json"), encoding="utf-8") as f:
        co_index = {e["file_name"]: e for e in json.load(f)["files"]}

    checks: List[Tuple[str, bool, str]] = []
    checks.append(("GT 行数 = 标注者-图像对总数", len(gt_index) == len(pairs),
                   f"index={len(gt_index)} pairs={len(pairs)}"))
    idx_pairs = {(r["annotator"], r["file_name"]) for r in gt_index}
    checks.append(("GT index 行键与 JSON 完全一致", idx_pairs == pairs, ""))

    by_file: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        by_file[r["file_name"]].append(r)

    n_inst_checked = 0
    n_mask_arrays = 0
    n_empty_masks = 0
    cat_seen = set()
    for k, fn in enumerate(sorted(by_file), 1):
        stem = os.path.splitext(fn)[0]
        with np.load(os.path.join(labels_dir, "gt_per_annotator", stem + ".npz"),
                     allow_pickle=False) as z:
            anns_in_file = sorted(by_file[fn], key=lambda r: r["annotator"])
            assert list(z["annotators"]) == [r["annotator"] for r in anns_in_file]
            for i, r in enumerate(anns_in_file):
                masks = z[f"masks_{i:03d}"]
                n_mask_arrays += 1
                assert masks.shape == (len(r["anns"]), HEIGHT, WIDTH), fn
                assert masks.dtype == np.uint8
                assert set(np.unique(masks)).issubset({0, 1})
                cats = [int(c) for c in z[f"categories_{i:03d}"]]
                cat_seen.update(cats)
                assert cats == [int(a["category_id"]) for a in r["anns"]], fn
                n_empty_masks += int((masks.sum(axis=(1, 2)) == 0).sum())
                n_inst_checked += masks.shape[0]
        if k % 200 == 0:
            print(f"  [gt {k}/{len(by_file)}] inst={n_inst_checked}", flush=True)
    checks.append(("(A) 实例总数 = JSON 实例总数",
                   n_inst_checked == sum(len(r["anns"]) for r in rows),
                   f"checked={n_inst_checked}"))
    checks.append(("(A) 类别集合 ⊆ 4 类", cat_seen <= set(CLASS_IDS),
                   f"observed={sorted(cat_seen)}"))

    # (B) 全量对拍 probe2（多标注者图逐像素 equality）
    n_cmp = 0
    for fn, rs in sorted(by_file.items()):
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: r["annotator"])
        h, w = rs[0]["height"], rs[0]["width"]
        unions = []
        for r in rs:
            u = np.zeros((h, w), dtype=np.uint8)
            for a in r["anns"]:
                u |= probe2.poly_to_arr(a["segmentation"], h, w)
            unions.append(u)
        votes_probe2 = np.sum(np.stack(unions, 0), 0).astype(np.uint8)
        stem = os.path.splitext(fn)[0]
        with np.load(os.path.join(labels_dir, "consensus", stem + ".npz"),
                     allow_pickle=False) as z:
            sup = z["support2048"]
            assert np.array_equal(sup, votes_probe2), fn
            assert np.array_equal((sup >= 2).astype(np.uint8),
                                  (votes_probe2 >= 2).astype(np.uint8)), fn
            _, n_cc = ndimage.label((sup >= 2).astype(np.uint8))
            assert z["votes2048"].shape == (4, h, w)
            assert z["votes2048"][3].sum() == 0, "Ambiguous 通道应恒 0（本批数据）"
            anyv = z["votes2048"].max(0) > 0
            assert np.array_equal(anyv, sup > 0)
            # 512 守恒：块和总票数相等；support512×16 ≈ 块和（块均值定义）
            assert int(z["votes512"].sum()) == int(z["votes2048"].sum())
            s16 = z["support512"].astype(np.float32) * 16.0
            assert abs(s16.sum() - float(z["support2048"].sum())) < 1.0
            assert n_cc >= 0
        n_cmp += 1
        if n_cmp % 100 == 0:
            print(f"  [probe2 对拍 {n_cmp}/{sum(1 for v in by_file.values() if len(v) >= 2)}]",
                  flush=True)
    n_multi = sum(1 for v in by_file.values() if len(v) >= 2)
    checks.append(("(B) support 与 probe2 逐像素一致（全部多标注者图）",
                   n_cmp == n_multi, f"compared={n_cmp}/{n_multi}"))

    report = {"checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
              "all_passed": all(ok for _, ok, _ in checks),
              "gt_instances_checked": n_inst_checked,
              "empty_decoded_masks": n_empty_masks,
              "probe2_files_compared": n_cmp,
              "vocab": vocab,
              "observed_categories_in_masks": sorted(cat_seen),
              "seconds": round(time.time() - t0, 2)}
    for n, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n} {d}")
    return report


# --------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--ann", default=DEFAULT_ANN_PATH)
    ap.add_argument("--labels", default=DEFAULT_LABELS_DIR)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "build":
        build_all(args.ann, args.labels, args.limit)
    else:
        rep = verify_outputs(args.ann, args.labels)
        if not rep["all_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
