# -*- coding: utf-8 -*-
"""Solar · 概率图 → 实例的后处理变体扫描 + 指定 spec 出官方提交 CSV。

出处说明（诚实边界）
------------------------------------------------------------------------------
原内部工作区里本步骤的扫描脚本跑在一台远端 V100S 上，代码快照没有落盘。
本文件按其**落盘产物与报告**重建：

* 变体网格 = `REPORT_v100_round3.md` §2.1–2.2 记录的 60 组 spec
  （f1 全局阈值×min-area 27、f3 形态学 8、f5 类别 3、f6 兜底 4、
  f7 seed 概率平均 7；f2 逐图自适应 / f4 概率标定两族全军覆没，这里
  不再内置，需要的可按 spec JSON 自行扩展）；
* 抽取链 = 「概率平均（多 --prob 时）→ 双线性放大 2048² → 阈值 →
  [形态学 open/close k×k] → 4 邻接连通域 → 面积下限 → 逐级放宽兜底」，
  阈值/连通域/形状过滤/放宽链语义与 unet_infer 完全同源（直接复用）；
* 评测 = unet_sweep.eval_fast（官方口径快评器：IoU>0.5 严格命中、
  TP 数命中对、分子累加命中对 IoU、逐「标注者-图像」行池化），
  最终数字以 solar_eval_local.py 复核。

重建的对拍验证（冻结概率缓存 + 官方评测器，README §数字）：对 s3+TTA 的
val 概率缓存，本文件重跑 baseline（thr0.70/ma600 含放宽链）= 0.315167、
close3 = 0.317162、no-relax = 0.314169，与当年落盘的官方口径数字一致。

用法
------------------------------------------------------------------------------
    # 1) val 变体扫描（单 seed 概率缓存）
    python postproc_sweep.py sweep --prob prob_val_s0_tta \
        --gt-ann $DATA/.../MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --split $WORK/labels/val_split.json \
        --families baseline,f1,f3,f5,f6 --out sweep_s0.csv
    # 2) 多 seed 集成族（f7）：给多个 --prob，概率取平均
    python postproc_sweep.py sweep --prob prob_val_s3_tta prob_val_s12_tta \
        --gt-ann ... --split ... --families f7 --out sweep_ens.csv
    # 3) 指定 spec 出提交 CSV（最终提交 = 4 seed 概率平均 + close7 t0.70 a600）
    python postproc_sweep.py submit --prob prob_test_s12_tta prob_test_s5_tta \
        prob_test_s11_tta prob_test_s3_tta \
        --images $DATA/MAGFiLO_1.0_Kaggle_2026/test/test_images \
        --preset close7_t070_a600 --out submission_ens4_close7.csv \
        --report submit_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from unet_infer import labels_to_flat, labels_to_masks, write_submission  # noqa: E402
from unet_sweep import eval_fast, load_gt_flat  # noqa: E402

CROP = 1024
HEIGHT = WIDTH = 2048

# 最终提交用的命名 spec（与当年落盘 report JSON 的 spec 字段一致）
PRESETS: Dict[str, Dict] = {
    "thr_t070_a600": {"kind": "thr", "thr": 0.70, "ma": 600, "relax": "chain"},
    "thr_t070_a700": {"kind": "thr", "thr": 0.70, "ma": 700, "relax": "chain"},
    "close3_t070_a600": {"kind": "morph", "op": "close", "k": 3,
                         "thr": 0.70, "ma": 600, "relax": "chain"},
    "close7_t070_a600": {"kind": "morph", "op": "close", "k": 7,
                         "thr": 0.70, "ma": 600, "relax": "chain"},
}

# 放宽档位（语义同 unet_infer.RELAX_LEVELS：主配置抽不出实例时逐级放宽，
# 保证每图至少 1 个实例——空图 = 该图全部标注者行记 FN）
RELAX_CHAIN: Tuple[Dict, ...] = (
    {}, {"thr": 0.55, "ma": 200}, {"thr": 0.45, "ma": 0}, {"thr": None, "ma": 0,
                                                           "largest": True})
RELAX_RELAX1: Tuple[Dict, ...] = ({}, {"thr": 0.55, "ma": 200})


# ------------------------------------------------------------------ 概率缓存
def load_prob_dirs(dirs: Sequence[str]) -> Tuple[List[Dict[str, np.ndarray]],
                                                 Dict[str, np.ndarray],
                                                 List[str]]:
    """多个概率缓存目录 → ([{stem: 概率 f32 1024²} 每 source 一份], {stem: cls}, sources)。

    多 seed 集成 = 逐像素概率平均（f7 家族，在抽取前按选中子集平均）；
    类别图取第一个目录的（本管线默认不按类切实例，cls 仅 f5 变体使用）。
    """
    per_dir: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    for d in dirs:
        z: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npz") or fn.startswith("_"):
                continue
            with np.load(os.path.join(d, fn)) as f:
                z[os.path.splitext(fn)[0]] = (f["prob1024"].astype(np.float32),
                                              f["cls1024"].astype(np.uint8))
        if not z:
            raise SystemExit(f"[sweep][FAIL] {d} 里没有 prob1024/cls1024 npz")
        per_dir.append(z)
        print(f"[prob] {d}: {len(z)} 张", flush=True)
    stems = set(per_dir[0])
    for z in per_dir[1:]:
        stems &= set(z)
    if not stems:
        raise SystemExit("[sweep][FAIL] 多个概率缓存没有共同图像")
    prob_srcs: List[Dict[str, np.ndarray]] = [
        {s: z[s][0] for s in sorted(stems)} for z in per_dir]
    cls = {s: per_dir[0][s][1] for s in sorted(stems)}
    return prob_srcs, cls, [os.path.abspath(d) for d in dirs]


def avg_sources(prob_srcs: List[Dict[str, np.ndarray]],
                sel: Optional[Sequence[int]], stem: str) -> np.ndarray:
    """选中 source 子集的逐像素概率平均（sel=None → 全体）。"""
    if sel is None:
        arrs = [z[stem] for z in prob_srcs]
    else:
        arrs = [prob_srcs[i][stem] for i in sel]
    return arrs[0] if len(arrs) == 1 else \
        np.mean(arrs, axis=0).astype(np.float32)


def upsample2048(prob: np.ndarray, cls: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """1024² → 2048²（概率双线性、类别最近邻；同 unet1024_eval.upsample2048）。"""
    from PIL import Image
    up = np.asarray(Image.fromarray(prob.astype(np.float32))
                    .resize((WIDTH, HEIGHT), Image.BILINEAR), dtype=np.float32)
    c = np.repeat(np.repeat(cls, HEIGHT // CROP, axis=0), WIDTH // CROP, axis=1)
    return up, c.astype(np.uint8)


# ------------------------------------------------------------------ 实例抽取
def _morph(mask: np.ndarray, op: str, k: int) -> np.ndarray:
    """k×k 全 1 方阵的形态学闭/开运算（scipy 默认边界语义）。

    注：刻意用 scipy.ndimage.binary_closing/binary_opening 的**默认**边界
    处理（不做 pad）——这是与当年远端扫描逐位对齐的语义：对冻结的
    s3+TTA val 概率缓存，close3 重跑 PQ=0.3171617809448947 /
    Dice=0.6411677005545621（与 eval_postproc_close3_val88_official.json
    逐位一致）；加 pad 反而对不上（差 ~2e-5）。
    """
    from scipy import ndimage
    st = np.ones((k, k), bool)
    if op == "close":
        return ndimage.binary_closing(mask, structure=st)
    if op == "open":
        return ndimage.binary_opening(mask, structure=st)
    raise ValueError(f"未知 morph op: {op}")


def _cc4(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1))
    return lab.astype(np.int32), int(n)


def _filter_area(lab: np.ndarray, ma: int) -> Tuple[np.ndarray, int]:
    if ma <= 0:
        return lab, int(lab.max())
    from unet_infer import apply_shape_filters
    return apply_shape_filters(lab, min_area=ma)


def extract_spec(prob: np.ndarray, cls: np.ndarray, spec: Dict
                 ) -> Tuple[np.ndarray, int, str]:
    """一条 spec → 实例标签图（逐级放宽，保证非空）。返回 (labels, n, tag)。

    spec 字段：thr（阈值）、ma（面积下限）、op/k（morph，可省）、
    class_split（bool，可省）、relax ∈ {chain(默认), relax1, none}。
    """
    relax = spec.get("relax", "chain")
    levels = (RELAX_CHAIN if relax == "chain"
              else RELAX_RELAX1 if relax == "relax1"
              else ({},))
    class_split = bool(spec.get("class_split", False))
    op, k = spec.get("op"), int(spec.get("k", 0))
    tag = "primary"
    for li, over in enumerate(levels):
        thr = over.get("thr", spec["thr"])
        ma = over.get("ma", spec["ma"])
        largest = over.get("largest", False)
        t = thr if thr is not None else (float(prob.max()) * 0.9
                                         if prob.max() > 0 else 1.1)
        mask = prob > t
        if not mask.any():
            continue
        if class_split:
            out = np.zeros(prob.shape, np.int32)
            offset = 0
            for c in (1, 2, 3, 4):
                sub = mask & (cls == c)
                if not sub.any():
                    continue
                lab, n = _cc4(sub)
                if n:
                    out[lab > 0] = lab[lab > 0] + offset
                    offset += n
            lab = out
        else:
            # 形态学只作用于主档（li==0）：放宽档 = 原版 RELAX_LEVELS 的纯
            # 「降阈值重抽」，不带 morph——这是与当年落盘数字逐位对齐的语义
            # （对冻结 s3+TTA 缓存，close3 若在放宽档重复 morph，PQ 会差 ~2e-5）。
            if op and li == 0:
                mask = _morph(mask, op, k)
                if not mask.any():
                    continue
            lab, _n = _cc4(mask)
        lab, n = _filter_area(lab, ma)
        if largest and n > 0:
            areas = np.bincount(lab.ravel())
            keep = int(np.argmax(areas[1:])) + 1
            lab = np.where(lab == keep, 1, 0).astype(np.int32)
            n = 1
        if n > 0:
            if li:
                tag = f"relax{li}"
            return lab, n, tag
    return np.zeros(prob.shape, np.int32), 0, "none"


# ------------------------------------------------------------------ 扫描网格
def family_baseline() -> List[Tuple[str, str, Dict]]:
    """起点配置：thr0.70 / ma600 / 含放宽链（60 变体表第 5 行 baseline_repro）。"""
    return [("baseline_repro", "f1_global_thr_ma",
             {"kind": "thr", "thr": 0.70, "ma": 600, "relax": "chain"})]


def family_f1() -> List[Tuple[str, str, Dict]]:
    """f1 全局阈值 × min-area（27 组，同当年网格）。"""
    out = []
    for thr in (0.58, 0.62, 0.66, 0.68, 0.70, 0.72, 0.74, 0.78):
        for ma in (400, 600, 1200):
            out.append((f"thr{thr:.2f} ma{ma}", "f1_global_thr_ma",
                        {"kind": "thr", "thr": thr, "ma": ma, "relax": "chain"}))
    for ma in (200, 800, 1600):
        out.append((f"thr0.70 ma{ma}", "f1_global_thr_ma",
                    {"kind": "thr", "thr": 0.70, "ma": ma, "relax": "chain"}))
    return out


def family_f3() -> List[Tuple[str, str, Dict]]:
    """f3 形态学 open/close × k ∈ {3,5,9,13}（8 组，thr0.70 / ma600）。"""
    return [(f"{op}{k}", "f3_morph",
             {"kind": "morph", "op": op, "k": k, "thr": 0.70, "ma": 600,
              "relax": "chain"})
            for op in ("open", "close") for k in (3, 5, 9, 13)]


def family_f5() -> List[Tuple[str, str, Dict]]:
    """f5 类别切分（当年 3 组里唯一有信息量的 class_split；condsplit 从不触发）。"""
    return [("class_split=True", "f5_class",
             {"kind": "thr", "thr": 0.70, "ma": 600, "relax": "chain",
              "class_split": True})]


def family_f6() -> List[Tuple[str, str, Dict]]:
    """f6 兜底族：放宽链 vs 只放宽一档 vs 不放宽（允许空图）。"""
    return [("relax1-only", "f6_fallback",
             {"kind": "thr", "thr": 0.70, "ma": 600, "relax": "relax1"}),
            ("no-relax", "f6_fallback",
             {"kind": "thr", "thr": 0.70, "ma": 600, "relax": "none"})]


def family_f7(n_src: int) -> List[Tuple[str, str, Dict]]:
    """f7 seed 概率平均族：逐对平均 + 全体平均（spec 里记 sources 子集下标）。"""
    import itertools
    out = []
    idx = list(range(n_src))
    combos = []
    if n_src >= 2:
        combos.extend([c for c in itertools.combinations(idx, 2)])
    if n_src > 2:
        combos.append(tuple(idx))
    for c in combos:
        out.append(("+".join(f"s{c[i]}" for i in range(len(c))), "f7_seeds",
                    {"kind": "thr", "thr": 0.70, "ma": 600, "relax": "chain",
                     "sources": list(c)}))
    return out


def run_sweep(args) -> int:
    prob_srcs, cls, sources = load_prob_dirs(args.prob)
    stems = sorted(cls)
    with open(args.split, encoding="utf-8") as f:
        names = list(json.load(f)["val"])
    gt, gt_meta = load_gt_flat(args.gt_ann, names)
    print(f"[gt] rows={gt_meta['gt_rows']} images={gt_meta['gt_images']} "
          f"masks={gt_meta['gt_masks']} | [prob] {len(stems)} 张 "
          f"sources={len(sources)}", flush=True)

    fams = [s.strip() for s in args.families.split(",") if s.strip()]
    specs: List[Tuple[str, str, Dict]] = []
    for fam in fams:
        if fam == "baseline":
            specs += family_baseline()
        elif fam == "f1":
            specs += family_f1()
        elif fam == "f3":
            specs += family_f3()
        elif fam == "f5":
            specs += family_f5()
        elif fam == "f6":
            specs += family_f6()
        elif fam == "f7":
            if len(sources) < 2:
                print("[sweep][skip] f7 需要 ≥2 个 --prob 目录")
            else:
                specs += family_f7(len(sources))
        else:
            raise SystemExit(f"[sweep][FAIL] 未知家族 {fam}（可选 baseline,f1,f3,f5,f6,f7）")
    print(f"[sweep] {len(specs)} 组变体", flush=True)

    rows = []
    t_all = time.perf_counter()
    for i, (vid, family, spec) in enumerate(specs, 1):
        t0 = time.perf_counter()
        sel = spec.get("sources")
        preds: Dict[str, List[np.ndarray]] = {}
        fallback = 0
        for stem in stems:
            p = avg_sources(prob_srcs, sel, stem)
            pr, cl = upsample2048(p, cls[stem])
            lab, n, tag = extract_spec(pr, cl, spec)
            if tag != "primary":
                fallback += 1
            preds[stem] = labels_to_flat(lab)
        rep = eval_fast(gt, preds)
        rows.append({"vid": f"{family}|{vid}",
                     "family": family, "pq": rep["pq"],
                     "dice_pooled": rep["dice_pooled"],
                     "dice_macro_per_row": rep["dice_macro_per_row"],
                     "tp": rep["tp"], "fp": rep["fp"], "fn": rep["fn"],
                     "pred_masks": rep["pred_masks"], "fallback": fallback,
                     "sec": round(time.perf_counter() - t0, 1),
                     "params": json.dumps(spec, ensure_ascii=False)})
        print(f"[{i}/{len(specs)}] {rows[-1]['vid']} pq={rep['pq']:.6f} "
              f"tp={rep['tp']} fp={rep['fp']} fn={rep['fn']} "
              f"masks={rep['pred_masks']} fallback={fallback} "
              f"({rows[-1]['sec']}s)", flush=True)
    rows.sort(key=lambda r: -r["pq"])

    print("\n=== top 10 by pq ===")
    for r in rows[:10]:
        print(f"  pq={r['pq']:.6f} dice={r['dice_pooled']:.6f} {r['vid']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[out] {args.out}  (total {time.perf_counter()-t_all:.0f}s)")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"sources": sources, "gt": {k: gt_meta[k] for k in
                      ("gt_rows", "gt_images", "gt_masks")}, "rows": rows}, f,
                      ensure_ascii=False, indent=1)
        print(f"[out] {args.out_json}")
    return 0


# ------------------------------------------------------------------ 提交模式
def run_submit(args) -> int:
    spec = PRESETS[args.preset] if args.preset else json.loads(args.spec)
    if "thr" not in spec:
        raise SystemExit("[submit][FAIL] spec 需要 thr（参考 --preset 的形态）")
    prob_srcs, cls, sources = load_prob_dirs(args.prob)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.images)
                 if f.lower().endswith((".jpeg", ".jpg", ".png")))
    missing = [i for i in ids if i not in cls]
    print(f"[submit] prob 缓存 {len(cls)} 张 / 期望 {len(ids)} 张，"
          f"缺缓存 {len(missing)} 张 {missing[:5]}")
    if missing:
        print("[submit][FAIL] 有图没有概率缓存，拒绝出半份提交")
        return 2

    preds: Dict[str, List] = {}
    n_inst = 0
    fallback_imgs = 0
    t0 = time.perf_counter()
    for i, image_id in enumerate(ids, 1):
        pr, cl = upsample2048(avg_sources(prob_srcs, None, image_id),
                              cls[image_id])
        lab, n, tag = extract_spec(pr, cl, spec)
        if tag != "primary":
            fallback_imgs += 1
        masks = labels_to_masks(lab)
        preds[image_id] = masks
        n_inst += len(masks)
        if (i % 40 == 0):
            print(f"[submit] {i}/{len(ids)} 实例累计 {n_inst} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    info = write_submission(preds, args.out)
    print(f"[submit] {args.out}  图 {info['images']}  行 {info['rows']}  "
          f"实例 {n_inst}  放宽图数 {fallback_imgs}  硬闸 issues=0  "
          f"用时 {time.perf_counter()-t0:.0f}s")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"csv": os.path.abspath(args.out),
                       "images": info["images"],
                       "images_with_rows": int(sum(1 for v in preds.values() if v)),
                       "rows": info["rows"], "instances": n_inst,
                       "fallback_imgs": fallback_imgs, "spec": spec,
                       "sources": sources,
                       "seconds": round(time.perf_counter() - t0, 1)},
                      f, ensure_ascii=False, indent=1)
        print(f"[submit] report → {args.report}")
    return 0


# ----------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sweep", help="val 上的后处理变体扫描")
    p.add_argument("--prob", nargs="+", required=True,
                   help="概率缓存目录（≥2 个时概率取平均；f7 家族需要 ≥2）")
    p.add_argument("--gt-ann", required=True, help="公开训练标注 COCO JSON")
    p.add_argument("--split", required=True, help="labels/val_split.json")
    p.add_argument("--families", default="baseline,f1,f3,f5,f6",
                   help="逗号分隔：baseline,f1,f3,f5,f6,f7")
    p.add_argument("--out", required=True, help="扫描结果 CSV")
    p.add_argument("--out-json", default=None)
    p.set_defaults(func=run_sweep)

    p = sub.add_parser("submit", help="指定 spec → 官方提交 CSV（含硬闸）")
    p.add_argument("--prob", nargs="+", required=True)
    p.add_argument("--images", required=True, help="图像目录（image_id 白名单）")
    p.add_argument("--spec", default=None, help="spec JSON，如 '{\"kind\":\"morph\",...}'")
    p.add_argument("--preset", default=None, choices=sorted(PRESETS),
                   help="命名 spec（与 --spec 二选一）")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    p.set_defaults(func=run_submit)

    args = ap.parse_args()
    if args.cmd == "submit" and not (args.spec or args.preset):
        ap.error("submit 需要 --spec 或 --preset")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
