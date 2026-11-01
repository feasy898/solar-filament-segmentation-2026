# -*- coding: utf-8 -*-
"""Solar 暗条分割 · U-Net 推理（概率图 → 按类别取掩膜 → 实例抽取 → 官方 RLE CSV）。

管线（与提交契约对齐）
------------------------------------------------------------------------------
  图像 (2048² jpeg, L)
    ├─ 方案 frame：整幅 4× 双线性降采样到 512²（**与训练同分布**：训练输入就是
    │   `Image.open(p).convert("L").resize((512,512), BILINEAR)`）→ 模型 → 概率图
    │   512² → 双线性放大到 2048²；
    └─ 方案 tile：原生 2048² 上取 4×4=16 块 512² 滑窗（stride 512，无重叠），
        每块跑模型 → 概率块 512² → 各放大 4× 贴回 2048²（与旧 v3 的 16 块一致）。
  概率图 2048² (float32, [0,1]) + 类别图 2048² (uint8, 0=背景, 1..4=四类)
    → 按类别取掩膜 mask_c = (prob>thr) & (cls==c)（不跨类粘连）
    → 实例抽取（cc 连通域 / watershed 分水岭，见 extract_instances）
    → 形状过滤（面积 / 伸长率 / 主轴长度）
    → 每实例一个 COCO 压缩 RLE → 官方 CSV（filament_id,segmentation_rle）

实例抽取为什么影响 PQ
------------------------------------------------------------------------------
官方 PQ 数的是「预测-标注者」的**命中对**：一行 = (标注者, 图像)，GT 是该标注者
画的暗条，预测在该图内共享；命中判据 IoU>0.5（严格大于）。一根预测若同时罩住
2 根 GT 暗条，它与每根的 IoU 都 ≈0.5（不到阈），于是**一根都不算命中**——
合并成一个连通域的预测会同时制造 FN（GT 没被单独命中）和潜在的 FP。
反过来，一根暗条被拆成碎片时碎片 IoU 掉到 0.5 以下同样不命中，还额外造 FP。
所以实例抽取要在"不过度合并"和"不过度切碎"之间取平衡，这是 PQ 的第一杠杆。

用法
------------------------------------------------------------------------------
    # 1) 概率图缓存（GPU，一次）
    ./.venv-solar-gpu/bin/python unet_infer.py prob \
        --images MAGFiLO_1.0_Kaggle_2026/test/test_images \
        --out output/unet_test_prob_frame --scheme frame
    # 2) 缓存 → 实例 → 官方 CSV
    ./.venv-solar-gpu/bin/python unet_infer.py csv \
        --prob output/unet_test_prob_frame --out output/submission_unet_v1.csv \
        --thr 0.65 --method watershed --min-area 300
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 公开仓库版：脚本平铺在同一目录，rle.py / submit_writer.py 就在 HERE；
# 原内部工作区里的 "../../solar" 包路径 shim 在此不再需要。
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SIZE = 512          # 模型输入边长
HEIGHT = WIDTH = 2048
CLASS_NAMES = ("bg", "Left", "Right", "Unidentifiable", "Ambiguous")
Mask = frozenset    # frozenset[(row, col)]，与 solar/rle.py、solar_pq 一致


# --------------------------------------------------------------------- 模型
def load_model(ckpt_path: str, device: Optional[str] = None):
    """加载 unet_train.FilamentUNet 权重（base/结构取自 ckpt）。

    公开仓库注：本仓库的训练线是 unet_train_crop.py 的 FilamentUNet5
    （1024²），对应推理入口是 unet_infer1024.py；这里的 FilamentUNet 属
    早期 512² 线（unet_train.py 未随仓库发布），`prob` 子命令因此不可用，
    实例抽取 / 提交写出部分（csv 子命令）不受影响。
    """
    import torch
    try:
        from unet_train import FilamentUNet
    except ImportError as e:
        raise ImportError(
            "unet_infer.py 的 `prob` 子命令需要早期 512² 线的 unet_train.py"
            "（未随本仓库发布）。1024² 主线请用 unet_infer1024.py。") from e
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FilamentUNet(base=ck.get("base", 16)).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck, device


def _forward(model, device, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(N,512,512) float32 [0,1] → (prob N×512² float32, clsarg N×512² uint8)。

    类别编码：0=背景，1..4 = Left/Right/Unidentifiable/Ambiguous（标签约定）。
    """
    import torch
    x = torch.from_numpy(batch[:, None].astype(np.float32)).to(device)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            sup_logit, cls_logit = model(x)
        prob = torch.sigmoid(sup_logit.float())
        cls_prob = torch.softmax(cls_logit.float(), dim=1)
    p = prob[:, 0].cpu().numpy().astype(np.float32)
    c = cls_prob.argmax(axis=1).cpu().numpy().astype(np.uint8) + 1  # 1..4
    return p, c


# --------------------------------------------------------------------- 解码
def decode_frame(path: str) -> np.ndarray:
    """2048² jpeg → 512² uint8（与 unet_train._decode_one 逐字一致）。"""
    from PIL import Image
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def decode_native(path: str) -> np.ndarray:
    """2048² jpeg → 原生 2048² uint8。"""
    from PIL import Image
    img = Image.open(path).convert("L")
    if img.size != (WIDTH, HEIGHT):
        img = img.resize((WIDTH, HEIGHT), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def tile_grid(tile: int = SIZE, stride: int = SIZE) -> List[Tuple[int, int]]:
    """滑窗块左上角（2048 空间）。tile=stride=512 → 4×4=16 块，无重叠。"""
    return [(r, c) for r in range(0, HEIGHT - tile + 1, stride)
            for c in range(0, WIDTH - tile + 1, stride)]


# ------------------------------------------------------------- 概率图（缓存）
def prob_for_image(path: str, model, device, scheme: str = "frame",
                   batch: int = 8, tta: str = "none"
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """图像 → (prob512 N×512², cls512 N×512², origin N×4)。

    frame: N=1，origin=[-1,-1,-1,-1]（整幅降采样，非滑窗）。
    tile : N=16（4×4 块），origin=[r0,c0,0,0]（块在 2048 空间的左上角）。
    tta="hflip": 额外跑一次**水平翻转**输入，把概率图翻回来取平均（训练时用过
    hflip 增广，测试时用同样对称性做 TTA 可降方差）。
    """
    if scheme == "frame":
        img = decode_frame(path).astype(np.float32) / 255.0
        prob, cls = _forward(model, device, img[None])
        if tta in ("hflip", "hvflip"):
            pf, cf = _forward(model, device, img[None, :, ::-1].copy())
            prob = (prob + pf[:, :, ::-1]) / 2.0
            cls = np.where(prob >= pf[:, :, ::-1], cls, cf[:, :, ::-1])
        if tta in ("vflip", "hvflip"):
            pv, cv = _forward(model, device, img[None, ::-1, :].copy())
            prob = (prob + pv[:, ::-1, :]) / 2.0
            cls = np.where(prob >= pv[:, ::-1, :], cls, cv[:, ::-1, :])
        return prob, cls, np.array([[-1, -1, 0, 0]], np.int32)
    if scheme == "tile":
        nat = decode_native(path)
        grid = tile_grid()
        probs, clss, origins = [], [], []
        for i in range(0, len(grid), batch):
            chunk = grid[i:i + batch]
            arr = np.stack([nat[r:r + SIZE, c:c + SIZE].astype(np.float32) / 255.0
                            for r, c in chunk])
            p, cl = _forward(model, device, arr)
            if tta in ("hflip", "hvflip"):
                pf, cf = _forward(model, device, arr[:, :, ::-1].copy())
                p = (p + pf[:, :, ::-1]) / 2.0
                cl = np.where(p >= pf[:, :, ::-1], cl, cf[:, :, ::-1])
            if tta in ("vflip", "hvflip"):
                pv, cv = _forward(model, device, arr[:, ::-1, :].copy())
                p = (p + pv[:, ::-1, :]) / 2.0
                cl = np.where(p >= pv[:, ::-1, :], cl, cv[:, ::-1, :])
            probs.append(p)
            clss.append(cl)
            origins.extend([[r, c, 0, 0] for r, c in chunk])
        return (np.concatenate(probs), np.concatenate(clss),
                np.array(origins, np.int32))
    raise ValueError(f"未知 scheme: {scheme}")


def build_prob2048(prob512: np.ndarray, cls512: np.ndarray,
                   origin: np.ndarray, mode: str = "bilinear",
                   native_tiles: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """缓存块 → 2048² 概率图 + 类别图。

    frame：整块放大 4×。tile ：每块放大 4× 后贴回原位（原生 2048² 分辨率）。
    mode="bilinear"：概率双线性放大（边界平滑，IoU 更高但可能把两段粘连）；
    mode="nearest" ：概率按像素复制 4×（等价"在 512 上二值化再放大"，
                     与旧版导出口径一致，形状方块化但不会引入新粘连）。
    native_tiles=True：滑窗方案——每个 512² 块本身就是 2048² 原图的 1:1 裁块，
                     模型输出与原图像素对齐，**不再放大**，直接贴回原位。
    """
    import torch
    if cls512.ndim == 3:          # 缓存里带 batch 维 (N,512,512)
        cls512 = cls512[0]
    p3 = prob512 if prob512.ndim == 3 else prob512[None]
    if mode == "nearest":
        up = np.repeat(np.repeat(p3.astype(np.float32),
                                  HEIGHT // SIZE, axis=1), WIDTH // SIZE, axis=2)
    elif mode == "bilinear":
        t = torch.from_numpy(p3.astype(np.float32))
        up = torch.nn.functional.interpolate(t[:, None], size=(HEIGHT, WIDTH),
                                             mode="bilinear",
                                             align_corners=False)[:, 0].numpy()
    else:
        raise ValueError(f"未知 upsample mode: {mode}")
    if native_tiles:
        prob = np.zeros((HEIGHT, WIDTH), np.float32)
        cls = np.zeros((HEIGHT, WIDTH), np.uint8)
        for i, (r0, c0, _, _) in enumerate(origin):
            prob[r0:r0 + SIZE, c0:c0 + SIZE] = prob512[i]
            cls[r0:r0 + SIZE, c0:c0 + SIZE] = cls512[i]
        return prob, cls.astype(np.uint8)
    if origin.shape[0] == 1 and origin[0, 0] < 0:
        cls = np.repeat(np.repeat(cls512, HEIGHT // SIZE, axis=0),
                        WIDTH // SIZE, axis=1)
        return up[0].astype(np.float32), cls.astype(np.uint8)
    prob = np.zeros((HEIGHT, WIDTH), np.float32)
    cls = np.zeros((HEIGHT, WIDTH), np.uint8)
    for i, (r0, c0, _, _) in enumerate(origin):
        prob[r0:r0 + SIZE, c0:c0 + SIZE] = up[i]
        cls[r0:r0 + SIZE, c0:c0 + SIZE] = cls512[i]
    return prob, cls.astype(np.uint8)


def upsample_prob_and_class(prob512: np.ndarray, cls512: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """512² 概率/类别 → 2048²（概率双线性、类别 nearest）。"""
    import torch
    t = torch.from_numpy(prob512.astype(np.float32))
    up = torch.nn.functional.interpolate(t[None, None], size=(HEIGHT, WIDTH),
                                         mode="bilinear",
                                         align_corners=False)[0, 0].numpy()
    cls = np.repeat(np.repeat(cls512.astype(np.uint8), HEIGHT // SIZE, axis=0),
                    WIDTH // SIZE, axis=1)
    return up.astype(np.float32), cls


# ------------------------------------------------------------- 实例抽取
def _label_cc(mask: np.ndarray, connectivity: int = 4) -> Tuple[np.ndarray, int]:
    from scipy import ndimage
    st = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    lab, n = ndimage.label(mask, structure=st)
    return lab.astype(np.int32), int(n)


def _find_markers(prob_small: np.ndarray, thr: float, min_distance: int,
                  sigma: float, max_markers: int = 4000
                  ) -> Tuple[np.ndarray, int]:
    """概率图局部极大 → 分水岭标记（marker）。

    在**降采样尺度**（peak_scale）上做：先高斯平滑消除平台，再贪心取极大并按
    min_distance 抑制邻域，避免把一根暗条脊线上的平台切成一串碎片。
    """
    from scipy import ndimage
    sm = ndimage.gaussian_filter(prob_small, sigma=sigma) if sigma > 0 else prob_small
    work = sm.copy()
    markers = np.zeros(sm.shape, np.int32)
    k = 0
    H, W = sm.shape
    while k < max_markers:
        idx = int(np.argmax(work))
        v = float(work.flat[idx])
        if v <= thr:
            break
        k += 1
        markers.flat[idx] = k
        r, c = divmod(idx, W)
        work[max(0, r - min_distance):r + min_distance + 1,
             max(0, c - min_distance):c + min_distance + 1] = -1.0
    return markers, k


def _watershed_labels(mask: np.ndarray, priority: np.ndarray,
                      marker_labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """经典 priority-flood 分水岭（等价 skimage.segmentation.watershed）。

    priority 越小越先淹没（此处传 -prob：高概率区先被淹没，脊线留在低概率处）。
    """
    H, W = mask.shape
    labels = np.zeros((H, W), np.int32)
    flat_mask = mask.ravel()
    flat_lab = labels.ravel()
    flat_pri = priority.ravel()
    heap: List[Tuple[float, int, int]] = []
    mflat = marker_labels.ravel()
    for idx in np.nonzero(mflat)[0]:
        if not flat_mask[idx]:
            continue          # 标记落在本类掩膜外 → 不作为种子（避免跨类渗漏）
        lab = int(mflat[idx])
        flat_lab[idx] = lab
        heapq.heappush(heap, (float(flat_pri[idx]), int(idx), lab))
    while heap:
        pri, idx, lab = heapq.heappop(heap)
        r, c = divmod(idx, W)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            nidx = rr * W + cc
            if not flat_mask[nidx] or flat_lab[nidx]:
                continue
            flat_lab[nidx] = lab
            heapq.heappush(heap, (max(pri, float(flat_pri[nidx])), nidx, lab))
    return labels, int(labels.max())


def _downsample_prob(prob: np.ndarray, size: int = SIZE) -> np.ndarray:
    """2048² 概率图 → size² 块均值（找 marker 用）。"""
    if prob.shape[0] == size:
        return prob
    f = prob.shape[0] // size
    return prob[:size * f, :size * f].reshape(size, f, size, f).mean(axis=(1, 3))


def extract_instances(prob: np.ndarray, cls: np.ndarray, thr: Optional[float] = 0.65,
                      method: str = "cc", class_split: bool = True,
                      connectivity: int = 4, marker_min_distance: int = 12,
                      marker_sigma: float = 4.0, marker_thr: Optional[float] = None,
                      min_area: int = 0, max_area: int = 0, min_elo: float = 0.0,
                      min_len: float = 0.0, largest_only: bool = False
                      ) -> Tuple[np.ndarray, int]:
    """概率图 → 实例标签图（int32，0=背景）。

    method:
      cc        —— 阈值二值图上取连通域（4/8 邻接）。
      watershed —— 以概率局部极大为标记做分水岭，把**粘连**的暗条拆开。

    class_split: 逐类别独立抽取（Left/Right/Unidentifiable/Ambiguous 各自成块），
                 避免异类暗条在二值图上粘成一根。
    """
    if thr is None:      # 自适应阈值：贴着本图概率峰值取（保证至少一个实例）
        mx = float(prob.max())
        thr = (mx * 0.9) if mx > 0 else 1.1   # 全零图 → 阈值高于最大值 → 无实例
    mask = prob > thr
    if cls.shape != prob.shape:
        raise ValueError(f"类别图 {cls.shape} 与概率图 {prob.shape} 尺寸不一致")
    if not mask.any():
        return np.zeros(prob.shape, np.int32), 0

    if method == "watershed":
        small = _downsample_prob(prob, SIZE)
        m_thr = thr if marker_thr is None else marker_thr
        markers_s, _ = _find_markers(small, m_thr, marker_min_distance, marker_sigma)
        if markers_s.shape != prob.shape:
            f = prob.shape[0] // markers_s.shape[0]
            markers = np.repeat(np.repeat(markers_s, f, axis=0), f, axis=1)
        else:
            markers = markers_s
        priority = -prob
    else:
        markers = None
        priority = None

    classes = ([1, 2, 3, 4] if class_split else [0])
    out = np.zeros(prob.shape, np.int32)
    offset = 0
    for c in classes:
        sub = mask if c == 0 else (mask & (cls == c))
        if not sub.any():
            continue
        if method == "watershed":
            lab, _ = _watershed_labels(sub, priority, markers)
            # 标记落在别类/掩膜外的像素要清掉
            lab[~sub] = 0
        elif method == "cc":
            lab, _ = _label_cc(sub, connectivity)
        else:
            raise ValueError(f"未知 method: {method}")
        nlab = int(lab.max())
        if nlab:
            out[lab > 0] = lab[lab > 0] + offset
            offset += nlab
    out, n = apply_shape_filters(out, min_area=min_area, max_area=max_area,
                                  min_elo=min_elo, min_len=min_len)
    if largest_only and n > 0:
        st = label_stats(out)
        keep = int(np.argmax(st["area"])) + 1
        out = np.where(out == keep, 1, 0).astype(np.int32)
        n = 1
    return out, n


RELAX_LEVELS: Tuple[Dict, ...] = (
    {},                                          # 主配置
    {"thr": 0.55, "min_area": 200},              # 放宽阈值 + 面积
    {"thr": 0.45, "min_area": 0},                # 只留阈值
    {"thr": None, "min_area": 0, "largest_only": True},   # 贴峰值取最大块
)


def extract_instances_pipeline(prob: np.ndarray, cls: np.ndarray,
                               cfg: Dict, relax: Sequence[Dict] = RELAX_LEVELS
                               ) -> Tuple[np.ndarray, int, str]:
    """逐级放宽的实例抽取，保证**每张图至少一个实例**（空图 = 全行 FN）。

    主配置抽不出实例时（暗条太少 / 全被面积过滤掉）逐级放宽，并把用到哪一级
    记在返回的 tag 里，便于核对覆盖率与放宽比例。
    """
    for i, over in enumerate(relax):
        c = dict(cfg)
        c.update(over)
        lab, n = extract_instances(prob, cls, **c)
        if n > 0:
            tag = "primary" if i == 0 else f"relax{i}:{json.dumps(over, ensure_ascii=False)}"
            return lab, n, tag
    return np.zeros(prob.shape, np.int32), 0, "none"


def label_stats(lab: np.ndarray) -> Dict[str, np.ndarray]:
    """逐实例统计：面积 / 伸长率 / 主轴长度（协方差特征值，免 per-label 全图扫描）。"""
    n = int(lab.max())
    if n == 0:
        return {"area": np.zeros(0), "elo": np.zeros(0), "length": np.zeros(0)}
    ys, xs = np.nonzero(lab)
    labs = lab[ys, xs]
    cnt = np.bincount(labs, minlength=n + 1).astype(np.float64)
    sy = np.bincount(labs, weights=ys.astype(np.float64), minlength=n + 1)
    sx = np.bincount(labs, weights=xs.astype(np.float64), minlength=n + 1)
    syy = np.bincount(labs, weights=(ys.astype(np.float64) ** 2), minlength=n + 1)
    sxx = np.bincount(labs, weights=(xs.astype(np.float64) ** 2), minlength=n + 1)
    sxy = np.bincount(labs, weights=(ys.astype(np.float64) * xs.astype(np.float64)),
                      minlength=n + 1)
    area = cnt[1:n + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        my = sy[1:n + 1] / area
        mx = sx[1:n + 1] / area
        cyy = syy[1:n + 1] / area - my ** 2
        cxx = sxx[1:n + 1] / area - mx ** 2
        cxy = sxy[1:n + 1] / area - mx * my
    tr = cxx + cyy
    det = cxx * cyy - cxy ** 2
    disc = np.maximum(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + np.sqrt(disc)
    l2 = tr / 2.0 - np.sqrt(disc)
    elo = np.sqrt(np.maximum(l1, 1e-9) / np.maximum(l2, 1e-9))
    length = 4.0 * np.sqrt(np.maximum(l1, 0.0))     # 均匀分布主轴长度 ≈ 4σ
    return {"area": area, "elo": elo, "length": length}


def apply_shape_filters(lab: np.ndarray, min_area: int = 0, max_area: int = 0,
                        min_elo: float = 0.0, min_len: float = 0.0
                        ) -> Tuple[np.ndarray, int]:
    """按面积/伸长率/主轴长度剔除实例（PQ 里 FP 罚 0.5、FN 罚 0.5，滤错要两头亏）。"""
    n = int(lab.max())
    if n == 0:
        return lab, 0
    st = label_stats(lab)
    drop = np.zeros(n + 1, bool)
    if min_area > 0:
        drop[1:] |= st["area"] < min_area
    if max_area > 0:
        drop[1:] |= st["area"] > max_area
    if min_elo > 0:
        drop[1:] |= st["elo"] < min_elo
    if min_len > 0:
        drop[1:] |= st["length"] < min_len
    if not drop.any():
        return lab, n
    keep_ids = np.nonzero(~drop[1:])[0] + 1
    remap = np.zeros(n + 1, np.int32)
    remap[keep_ids] = np.arange(1, len(keep_ids) + 1, dtype=np.int32)
    out = remap[lab]
    return out, int(out.max())


# ------------------------------------------------------------------- 产物
def labels_to_flat(lab: np.ndarray) -> List[np.ndarray]:
    """实例标签图 → 升序 flat index 数组列表（c*2048+r，向量化，快评器用）。"""
    if lab is None or int(lab.max()) == 0:
        return []
    ys, xs = np.nonzero(lab)
    labs = lab[ys, xs]
    flats = (xs.astype(np.int64) * HEIGHT + ys.astype(np.int64))
    order = np.argsort(labs, kind="stable")
    labs, flats = labs[order], flats[order]
    n = int(lab.max())
    bounds = np.searchsorted(labs, np.arange(1, n + 2))
    out: List[np.ndarray] = []
    for k in range(n):
        a, b = int(bounds[k]), int(bounds[k + 1])
        if b > a:
            out.append(flats[a:b])
    return out


def labels_to_masks(lab: np.ndarray) -> List[Mask]:
    """实例标签图 → 掩膜列表（frozenset[(row,col)]）。"""
    if lab is None or int(lab.max()) == 0:
        return []
    ys, xs = np.nonzero(lab)
    labs = lab[ys, xs]
    order = np.argsort(labs, kind="stable")
    ys, xs, labs = ys[order], xs[order], labs[order]
    bounds = np.searchsorted(labs, np.arange(1, int(lab.max()) + 2))
    out: List[Mask] = []
    for k in range(int(lab.max())):
        a, b = bounds[k], bounds[k + 1]
        if b <= a:
            continue
        out.append(frozenset(zip(ys[a:b].tolist(), xs[a:b].tolist())))
    return out


def write_submission(preds: Dict[str, Sequence[Mask]], out_path: str) -> Dict:
    """掩膜字典 → 官方 CSV（走 solar/submit_writer.py 的 writer + 硬闸）。"""
    from submit_writer import validate_submission_csv, write_submission_csv
    text = write_submission_csv(preds)
    expect = list(preds.keys())
    issues = validate_submission_csv(text, expect_images=expect)
    if issues:
        raise RuntimeError(f"提交 CSV 硬闸不通过: {issues[:5]}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return {"csv": out_path, "images": len(preds),
            "rows": sum(len(v) for v in preds.values()), "issues": 0}


# ------------------------------------------------------------------- CLI
def _list_images(d: str) -> List[str]:
    exts = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
    names = sorted(n for n in os.listdir(d) if n.lower().endswith(exts))
    return [os.path.join(d, n) for n in names]


def cmd_prob(args) -> None:
    model, ck, device = load_model(args.ckpt)
    paths = _list_images(args.images)
    if args.names_file:
        with open(args.names_file, encoding="utf-8") as f:
            keep = {ln.strip() for ln in f if ln.strip()}
        paths = [p for p in paths if os.path.basename(p) in keep]
    if args.limit:
        paths = paths[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    for i, p in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(args.out, stem + ".npz")
        if os.path.exists(dst) and not args.overwrite:
            continue
        prob, cls, origin = prob_for_image(p, model, device, args.scheme,
                                            tta=args.tta)
        np.savez_compressed(dst, prob512=prob.astype(np.float16),
                            cls512=cls.astype(np.uint8), origin=origin,
                            scheme=np.array(args.scheme))
        if i % 20 == 0 or i == len(paths):
            print(f"[prob] {i}/{len(paths)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    meta = {"scheme": args.scheme, "ckpt": os.path.abspath(args.ckpt),
            "ckpt_epoch": ck.get("epoch"), "n_images": len(paths),
            "device": device}
    with open(os.path.join(args.out, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[prob] {len(paths)} 张 → {args.out} ({time.perf_counter()-t0:.0f}s)")


def _iter_prob_cache(prob_dir: str):
    for fn in sorted(os.listdir(prob_dir)):
        if not fn.endswith(".npz") or fn.startswith("_"):
            continue
        with np.load(os.path.join(prob_dir, fn)) as z:
            scheme = str(z["scheme"]) if "scheme" in z.files else "frame"
            yield (os.path.splitext(fn)[0],
                   z["prob512"].astype(np.float32),
                   z["cls512"].astype(np.uint8),
                   z["origin"].astype(np.int32),
                   scheme)


def cmd_csv(args) -> None:
    preds: Dict[str, List[Mask]] = {}
    stats = []
    t0 = time.perf_counter()
    n_img = 0
    for stem, prob512, cls512, origin, scheme in _iter_prob_cache(args.prob):
        prob, cls = build_prob2048(prob512, cls512, origin,
                                   mode=args.upsample,
                                   native_tiles=(scheme == "tile"))
        cfg = dict(thr=args.thr, method=args.method, class_split=args.class_split,
                   connectivity=args.connectivity,
                   marker_min_distance=args.marker_min_distance,
                   marker_sigma=args.marker_sigma, min_area=args.min_area,
                   max_area=args.max_area, min_elo=args.min_elo,
                   min_len=args.min_len)
        lab, n, tag = extract_instances_pipeline(prob, cls, cfg)
        masks = labels_to_masks(lab)
        preds[stem] = masks
        stats.append({"image": stem, "instances": n, "tag": tag,
                      "area_sum": int(sum(len(m) for m in masks))})
        n_img += 1
        if n_img % 20 == 0 or n_img == args.limit or True:
            if n_img % 20 == 0:
                print(f"[csv] {n_img} ({time.perf_counter()-t0:.0f}s)", flush=True)
    info = write_submission(preds, args.out)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    info["instances_total"] = int(sum(s["instances"] for s in stats))
    tags = {}
    for s_ in stats:
        tags[s_["tag"]] = tags.get(s_["tag"], 0) + 1
    info["extraction_tags"] = tags
    info["images_with_rows"] = int(sum(1 for v in preds.values() if v))
    print(json.dumps(info, ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prob = sub.add_parser("prob")
    p_prob.add_argument("--images", required=True)
    p_prob.add_argument("--out", required=True)
    p_prob.add_argument("--ckpt", default=os.path.join(HERE, "output", "unet_v1_best.pt"))
    p_prob.add_argument("--scheme", choices=["frame", "tile"], default="frame")
    p_prob.add_argument("--tta", choices=["none", "hflip", "vflip", "hvflip"], default="none")
    p_prob.add_argument("--names-file", default=None,
                        help="只处理名单里的图（一行一个 file_name）")
    p_prob.add_argument("--limit", type=int, default=0)
    p_prob.add_argument("--overwrite", action="store_true")

    p_csv = sub.add_parser("csv")
    p_csv.add_argument("--prob", required=True)
    p_csv.add_argument("--out", required=True)
    p_csv.add_argument("--thr", type=float, default=0.65)
    p_csv.add_argument("--upsample", choices=["bilinear", "nearest"], default="bilinear")
    p_csv.add_argument("--method", choices=["cc", "watershed"], default="cc")
    p_csv.add_argument("--class-split", action="store_true",
                       help="按类别分别抽取实例（默认关：val 实测类别图会把暗条切碎，"
                            "PQ 从 0.26 掉到 0.01 量级）")
    p_csv.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    p_csv.add_argument("--marker-min-distance", type=int, default=12)
    p_csv.add_argument("--marker-sigma", type=float, default=4.0)
    p_csv.add_argument("--min-area", type=int, default=0)
    p_csv.add_argument("--max-area", type=int, default=0)
    p_csv.add_argument("--min-elo", type=float, default=0.0)
    p_csv.add_argument("--min-len", type=float, default=0.0)
    p_csv.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "prob":
        cmd_prob(args)
    else:
        cmd_csv(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
