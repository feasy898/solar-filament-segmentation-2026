# -*- coding: utf-8 -*-
"""Solar · 共识伪标签 probe v2：修 v1 两个问题（2026-10-29）。

v1 发现：CC 投票共识实例大量合并（1176 CC vs 3006 GT 实例），且设计A对照
人群混杂（single@0 含 411 张单人图满分行）。v2 改：
- 所有对比按人群切片（m1/m2/m3 = 每图 1/2/3 位标注者），只在同人群内比。
- 新增【实例锚定共识】：跨标注者实例按 IoU>0 并查集成组（同一标注者内部
  不连边），支持度≥2 源的组才保留；组形状取 成员并/交/组内多数投票 三种。
  这绕开 CC 合并伪影，是合成实验 consensus.py majority_vote 的逐实例版。
- 诊断：实例级两两 IoU（贪心 1-1 匹配）、单标注者自身实例是否自相交/粘连
  （CC(自身并集) 数 vs 实例数）、每个 CC 投票共识块覆盖几个匹配组。
- 设计 A：GT=全部行 vs 预测∈{照抄单人i, CC共识, 实例锚定共识}，分人群池化官方 PQ。
- 设计 B（3人图 LOO）：GT=留出者 H，预测∈{单人, 双人并/交, 实例锚定(并/交/对内投票)}。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from pycocotools import mask as maskUtils
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
# 公开仓库版：官方口径 PQ 复刻件 solar_pq.py 随本仓库同目录发布（原内部
# 工作区以绝对路径指向 metrics_local，此处改为同目录导入，语义不变）。
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from solar_pq import Mask, _row_stats, iou  # noqa: E402

# 公开仓库版：数据在仓库外，用环境变量 SOLAR_ANN 指向官方训练标注 JSON
# （缺省回落到原内部工作区的仓库相对路径）。
ANN_PATH = os.environ.get(
    "SOLAR_ANN",
    os.path.join(_HERE, "MAGFiLO_1.0_Kaggle_2026", "train",
                 "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))


def poly_to_arr(seg, h, w) -> np.ndarray:
    rles = maskUtils.frPyObjects(seg, h, w)
    return maskUtils.decode(maskUtils.merge(rles)).astype(np.uint8)


def arr_to_mask(arr) -> Mask:
    rows, cols = np.nonzero(arr)
    return frozenset(zip(rows.tolist(), cols.tolist()))


def sets_to_arr(masks: List[Mask], h: int, w: int) -> np.ndarray:
    u = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m:
            idx = np.array(list(m), dtype=np.int64)
            u[idx[:, 0], idx[:, 1]] = 1
    return u


def cc_instances(votes: np.ndarray, min_area: int = 0) -> List[Mask]:
    lab, n = ndimage.label(votes)
    out = []
    if n:
        areas = np.bincount(lab.ravel())[1:]
        for k in np.nonzero(areas >= max(min_area, 1))[0] + 1:
            out.append(arr_to_mask(lab == k))
    return out


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def match_groups(inst_sets: List[List[Mask]], iou_thr: float = 0.0
                 ) -> List[List[Tuple[int, int]]]:
    """跨标注者实例并查集分组（同源不连边）。返回 [组] = [(源i,实例k),...]。"""
    nodes = [(i, k) for i, ms in enumerate(inst_sets) for k in range(len(ms))]
    idx = {nk: q for q, nk in enumerate(nodes)}
    dsu = DSU(len(nodes))
    for i in range(len(inst_sets)):
        for j in range(i + 1, len(inst_sets)):
            for k, mk in enumerate(inst_sets[i]):
                for l, ml in enumerate(inst_sets[j]):
                    if iou(mk, ml) > iou_thr:
                        dsu.union(idx[(i, k)], idx[(j, l)])
    groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for nk, q in idx.items():
        groups[dsu.find(q)].append(nk)
    return list(groups.values())


def group_shape(masks: List[Mask], h: int, w: int, mode: str) -> Mask:
    """组内融合：union / inter / vote2（成员≥3 时像素多数，否则并）。"""
    if mode == "union" or (mode == "vote2" and len(masks) < 3):
        out = set()
        for m in masks:
            out |= m
        return frozenset(out)
    if mode == "inter":
        out = None
        for m in masks:
            out = set(m) if out is None else (out & m)
        return frozenset(out or ())
    # vote2
    votes = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        votes += sets_to_arr([m], h, w)
    return arr_to_mask((votes >= 2).astype(np.uint8))


def pooled(stats) -> Dict[str, float]:
    s = np.array(stats, dtype=float)
    ti, tp, fp, fn = s[:, 0].sum(), s[:, 1].sum(), s[:, 2].sum(), s[:, 3].sum()
    den = tp + 0.5 * fp + 0.5 * fn
    return {"pq": ti / den if den > 0 else 0.0, "rows": int(len(stats)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def pct(xs):
    if not xs:
        return {}
    a = np.array(xs)
    return {f"p{p}": round(float(np.percentile(a, p)), 4)
            for p in (5, 25, 50, 75, 95)}


def main() -> None:
    with open(ANN_PATH) as f:
        coco = json.load(f)
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img: Dict[str, List[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    by_file: Dict[str, List[str]] = defaultdict(list)
    for im in coco["images"]:
        by_file[im["file_name"]].append(im["id"])

    A: Dict[Tuple[str, str], List] = defaultdict(list)   # (variant, pop) -> stats
    B: Dict[str, List] = defaultdict(list)
    inst_pair_ious: List[float] = []
    unmatched: List[Tuple[int, int]] = []                # (源i实例数-匹配数, 同j)
    intra_cc_vs_inst: List[Tuple[int, int]] = []         # (自身实例数, CC(自身并)数)
    cc_cover_groups: List[int] = []                      # 每个CC投票块覆盖的匹配组数
    group_sizes: Counter = Counter()

    for fn, img_ids in sorted(by_file.items()):
        rows = []
        for iid in img_ids:
            im = imgs[iid]
            h, w = im["height"], im["width"]
            arrs = [poly_to_arr(a["segmentation"], h, w) for a in anns_by_img[iid]]
            sets = [arr_to_mask(a) for a in arrs]
            rows.append((h, w, sets, arrs))
        M = len(rows)
        pop = {1: "m1", 2: "m2", 3: "m3"}[M]
        unions_np = []
        for h, w, sets, arrs in rows:
            u = np.zeros((h, w), dtype=np.uint8)
            for a in arrs:
                u |= a
            unions_np.append(u)

        # ---- 设计A：照抄单人 / CC共识 ----
        for i in range(M):
            for j in range(M):
                A[(f"copy_single_{i}", pop)].append(
                    _row_stats(rows[j][2], rows[i][2], 0.5))
        for s in range(1, M + 1):
            votes = np.sum(np.stack(unions_np, 0), 0)
            cons = cc_instances((votes >= s).astype(np.uint8))
            for j in range(M):
                A[(f"cc_consensus_s{s}", pop)].append(
                    _row_stats(rows[j][2], cons, 0.5))

        # ---- 实例锚定共识（源=全部M人）----
        inst_sets = [r[2] for r in rows]
        groups = match_groups(inst_sets)
        for g in groups:
            group_sizes[len({s for s, _ in g})] += 1
        for mode in ("union", "inter", "vote2"):
            for sup in (1, 2):
                preds = []
                for g in groups:
                    srcs = {s for s, _ in g}
                    if len(srcs) < sup:
                        continue
                    preds.append(group_shape([inst_sets[s][k] for s, k in g],
                                             rows[0][0], rows[0][1], mode))
                for j in range(M):
                    A[(f"ia_{mode}_sup{sup}", pop)].append(
                        _row_stats(rows[j][2], preds, 0.5))

        # ---- 诊断 ----
        if M >= 2:
            for i in range(M):
                for j in range(i + 1, M):
                    pairs = sorted(
                        ((iou(a, b), k, l) for k, a in enumerate(inst_sets[i])
                         for l, b in enumerate(inst_sets[j])), reverse=True)
                    used_i, used_j, n_match = set(), set(), 0
                    for v, k, l in pairs:
                        if v <= 0:
                            break
                        if k in used_i or l in used_j:
                            continue
                        used_i.add(k)
                        used_j.add(l)
                        n_match += 1
                        inst_pair_ious.append(v)
                    unmatched.append((len(inst_sets[i]) - n_match,
                                      len(inst_sets[j]) - n_match))
            for i in range(M):
                lab, n = ndimage.label(unions_np[i])
                intra_cc_vs_inst.append((len(inst_sets[i]), n))
        if M == 3:
            votes = np.sum(np.stack(unions_np, 0), 0)
            for cc in cc_instances((votes >= 2).astype(np.uint8)):
                n_cov = 0
                for g in groups:
                    gm = group_shape([inst_sets[s][k] for s, k in g],
                                     rows[0][0], rows[0][1], "union")
                    if iou(cc, gm) > 0:
                        n_cov += 1
                cc_cover_groups.append(n_cov)

        # ---- 设计B：3人图 LOO ----
        if M == 3:
            for hold in range(3):
                others = [i for i in range(3) if i != hold]
                gt = rows[hold][2]
                for o in others:
                    B["loo_copy_single"].append(_row_stats(gt, rows[o][2], 0.5))
                u2 = [unions_np[o] for o in others]
                for name, s in (("loo_cc_union2", 1), ("loo_cc_inter2", 2)):
                    B[name].append(_row_stats(
                        gt, cc_instances(
                            (np.sum(np.stack(u2, 0), 0) >= s).astype(np.uint8)), 0.5))
                loo_groups = [g for g in match_groups([inst_sets[o] for o in others])
                              if len({s for s, _ in g}) >= 2]
                for mode in ("union", "inter"):
                    preds = [group_shape([inst_sets[others[s]][k] for s, k in g],
                                         rows[0][0], rows[0][1], mode)
                             for g in loo_groups]
                    B[f"loo_ia_{mode}"].append(_row_stats(gt, preds, 0.5))

    # ---------------- 汇总 ----------------
    print("== 诊断：实例级分歧 ==")
    print(f"实例级匹配对 IoU (贪心1-1, n={len(inst_pair_ious)}): {pct(inst_pair_ious)}")
    frac = sum(v > 0.5 for v in inst_pair_ious) / max(1, len(inst_pair_ious))
    print(f"  其中 IoU>0.5 的比例 = {frac:.3f}")
    un = [a for a, b in unmatched]; un2 = [b for a, b in unmatched]
    print(f"  每图对 unmatched 实例数: 侧1 {pct([float(x) for x in un])} "
          f"侧2 {pct([float(x) for x in un2])}")
    same = sum(1 for a, b in intra_cc_vs_inst if a == b)
    bad = [(a, b) for a, b in intra_cc_vs_inst if a != b][:8]
    print(f"单标注者自身实例粘连: {same}/{len(intra_cc_vs_inst)} 行自身实例==CC数"
          f"（不等样例(实例,CC): {bad}）")
    print(f"CC(vote>=2) 块覆盖匹配组数分布: "
          f"{dict(sorted(Counter(cc_cover_groups).items()))}")
    print(f"匹配组按源数分布: {dict(sorted(group_sizes.items()))}")

    print("\n== 设计A（分人群，池化官方 PQ；copy_single_i=照抄第i标注者）==")
    keys = sorted(A)
    pops = ["m1", "m2", "m3"]
    for pop in pops:
        print(f"-- 人群 {pop} --")
        pop_keys = [k for k in keys if k[1] == pop]
        base = {}
        for k in pop_keys:
            v = k[0]
            r = pooled(A[k])
            print(f"  {v:18s} PQ={r['pq']:.4f} rows={r['rows']} TP={r['tp']} "
                  f"FP={r['fp']} FN={r['fn']}")
            if v.startswith("copy_single"):
                base[v] = r["pq"]
        singles = [x for k, x in base.items()]
        mean_s = sum(singles) / len(singles) if singles else 0
        for k in pop_keys:
            if k[0].startswith(("cc_", "ia_")):
                r = pooled(A[k])
                print(f"    → {k[0]:18s} vs 该人群copy_single均值({mean_s:.4f}): "
                      f"{r['pq'] - mean_s:+.4f}")

    print("\n== 设计B（151 张3人图 ×3折 LOO，池化官方 PQ）==")
    res = {k: pooled(v) for k, v in sorted(B.items())}
    for k, r in res.items():
        print(f"  {k:18s} PQ={r['pq']:.4f} rows={r['rows']} TP={r['tp']} "
              f"FP={r['fp']} FN={r['fn']}")
    s0 = res["loo_copy_single"]["pq"]
    for k in res:
        if k != "loo_copy_single":
            print(f"    → {k} vs copy_single({s0:.4f}): {res[k]['pq'] - s0:+.4f}")


if __name__ == "__main__":
    main()
