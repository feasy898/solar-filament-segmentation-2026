# -*- coding: utf-8 -*-
"""Solar Filament Segmentation 2026 — COCO 压缩 RLE 适配层（纯 python 实现）。

缺口出处：`poc/metrics/README.md` 差异点——官方自评 notebook（raw/samples-self-
evaluation-notebook.md In[3] rles_to_layers）与官方提交格式（raw/official.md
"Submission File"）均以 **COCO 压缩 RLE 字符串** 交换 2048×2048 掩膜：

    掩膜 → pycocotools mask_util.encode → counts 字符串 → CSV 列 segmentation_rle
    （"You only need to store RLE Counts; no need to store RLE Size. The Size
      is fixed for all images: 2048 X 2048 pixels."，不要引号）

而 P1 的 `metrics_local/solar_pq.py` 用像素集合 frozenset[(row,col)] 表达掩膜。
本模块补上两侧的桥：

    encode_counts(mask) / decode_counts(s)        —— 压缩 RLE 字符串编解码
    mask_to_rle(mask) / rle_to_mask(s, h, w)      —— 像素集合 ↔ RLE 字符串
    rle_csv_rows(preds) / parse_gt_ids(...)       —— 提交 writer / GT id 解析

编码算法逐行对齐 pycocotools `maskApi.c`（rleToString / rleFrString）：
- 行程按**列主序**（Fortran order，先下后右）扫描，counts 从 0 像素个数开始交替；
- 压缩：类 LEB128，每字符 6 位有效（5 位数据 + 1 位续位），ASCII 48..111；
- counts[i] (i>2) 存储为对 counts[i-2] 的**差分**（可为负，符号扩展）。

pycocotools 已装入本目录 .venv-pycoco（安装成功）；核心实现保持纯 python 零依赖，
对拍验证见 tests/test_solar_rle.py::test_pycocotools_crosscheck（环境无
pycocotools 时自动 skip，数学等价由黄金样例与往返测试保证）。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

Mask = frozenset  # frozenset[(int, int)] 像素坐标集合，与 solar_pq 一致
HEIGHT = WIDTH = 2048  # 官方固定尺寸（raw/official.md Submission File）

# ---------------------------------------------------------------- 压缩编解码

def rle_counts_to_string(counts: Sequence[int]) -> str:
    """counts（行程长度，首段为 0 的个数）→ COCO 压缩字符串。对齐 rleToString。"""
    out: List[str] = []
    for i, x in enumerate(counts):
        if i > 2:  # 第 3 段起存差分
            x = x - counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5  # python 对负数为算术右移，与 C long 语义一致
            # 逐字对齐 maskApi.c: more = (c&0x10) ? x!=-1 : x!=0
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def rle_string_to_counts(s: str) -> List[int]:
    """COCO 压缩字符串 → counts。对齐 rleFrString（含差分还原与符号扩展）。

    字符必须落在 48..111（LEB128+48 字符集），否则 ValueError——比 pycocotools
    的静默错值更安全，校验器可据此拦坏提交。
    """
    counts: List[int] = []
    p = 0
    n = len(s)
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            o = ord(s[p])
            if not 48 <= o <= 111:
                raise ValueError(f"非法 RLE 字符 {s[p]!r} (ord={o}) 于位 {p}")
            c = o - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)  # 负差分符号扩展
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


# ------------------------------------------------------------ 掩膜 ↔ counts

def mask_to_counts(mask: Mask, height: int = HEIGHT, width: int = WIDTH) -> List[int]:
    """像素集合 → 行程 counts（列主序扫描：先沿行方向走，再换列）。

    COCO 约定：从像素 (0,0) 起、**列优先**扫描（maskApi 把 H×W 数组按 Fortran
    序拉平），counts[0] 为第一段 0 的个数。空掩膜 → [height*width]。

    实现按行程算术（O(P log P)，P=前景像素数），不逐格扫描 H×W——2048×2048
    定尺下逐格扫描单次即 420 万次迭代，writer 批量编码不可用。
    """
    flats = sorted(c * height + r for r, c in mask)
    if not flats:
        return [height * width]
    counts: List[int] = [flats[0]]          # 首段 0 的个数
    start = prev = flats[0]
    for f in flats[1:]:
        if f == prev + 1:                    # 同一 1-行程延续
            prev = f
            continue
        counts.append(prev - start + 1)     # 1-行程长度
        counts.append(f - prev - 1)         # 其间 0-行程长度
        start = prev = f
    counts.append(prev - start + 1)         # 末段 1-行程
    if sum(counts) < height * width:         # 尾段 0（可省略则省略）
        counts.append(height * width - sum(counts))
    return counts


def counts_to_mask(counts: Sequence[int], height: int = HEIGHT,
                   width: int = WIDTH) -> Mask:
    """counts → 像素集合。

    pycocotools 语义（rleFrString 实测，2026-10-27）：官方样例串的 counts 之和
    可**小于** height*width——尾部零行程被省略，解码时余下像素按 0 补齐（官方
    Overview 页黄金串即此形态：和=3,362,558，解码出 177 像素小暗条）。本函数
    同样宽容：和 ≤ height*width 时尾部补 0；和 > height*width 才是坏数据。
    """
    total = sum(counts)
    if total > height * width:
        raise ValueError(f"counts 总和 {total} > {height}×{width}={height*width}")
    px: Set[Tuple[int, int]] = set()
    idx = 0
    val = 0
    for run in counts:
        if val:
            for j in range(idx, idx + run):
                px.add((j % height, j // height))  # 列主序拉平的逆
        idx += run
        val = 1 - val
    return frozenset(px)


# --------------------------------------------------------- 顶层便捷接口

def mask_to_rle(mask: Mask, height: int = HEIGHT, width: int = WIDTH) -> str:
    """像素集合 → COCO 压缩 RLE 字符串（官方 CSV 里 segmentation_rle 列的形态）。"""
    return rle_counts_to_string(mask_to_counts(mask, height, width))


def rle_to_mask(s: str, height: int = HEIGHT, width: int = WIDTH) -> Mask:
    """COCO 压缩 RLE 字符串 → 像素集合（送入 solar_pq 的形态）。"""
    return counts_to_mask(rle_string_to_counts(s), height, width)


def decode_rle_any(rle, height: int = HEIGHT, width: int = WIDTH) -> Mask:
    """适配 pycocotools 两种 RLE 形态：压缩字符串 或 未压缩 counts 列表。"""
    if isinstance(rle, str):
        return rle_to_mask(rle, height, width)
    return counts_to_mask(list(rle), height, width)


def mask_area(mask: Mask) -> int:
    return len(mask)


# ------------------------------------------------------- CSV / id 适配层

def parse_gt_ids(filament_ids: Iterable[str]) -> List[Tuple[str, str]]:
    """GT filament_id → [(annotator, image), ...]。

    官方 notebook 存在两种分隔符（solar_pq.py 歧义点 2）：get_overlap_df 用
    "_"（`filament_id.str.split("_", n=1).str[0]` 得"标注者-图像"），process_entry
    用 "-"（`split("-", maxsplit=1)[1]` 得图像名）。本函数两步还原：
    先按 "_" 切掉行尾序号，再按首个 "-" 切出标注者前缀。

    例："ann01-20150125172714Mh_3" → ("ann01", "20150125172714Mh")。
    旧数据若标注者与图像间也是 "_"（无 "-"），退化为 ("", 整段) 并保持可追溯。
    """
    pairs: List[Tuple[str, str]] = []
    for fid in filament_ids:
        head = fid.split("_", 1)[0]
        if "-" in head:
            ann, img = head.split("-", 1)
        else:
            ann, img = "", head
        pairs.append((ann, img))
    return pairs


def parse_pred_ids(filament_ids: Iterable[str]) -> List[str]:
    """预测 filament_id → [image_id, ...]（官方口径：按 "_" 切掉行尾序号）。"""
    return [fid.split("_", 1)[0] for fid in filament_ids]


def gt_entries_from_csv(rows: Dict[str, List[str]], height: int = HEIGHT,
                        width: int = WIDTH
                        ) -> List[Tuple[str, str, Mask]]:
    """官方 GT CSV（filament_id/segmentation_rle 两列）→ solar_pq 的 GTEntry 列表。

    rows 顺序给出两列：{"filament_id": [...], "segmentation_rle": [...]}。
    """
    ids = rows["filament_id"]
    rles = rows["segmentation_rle"]
    out: List[Tuple[str, str, Mask]] = []
    for (ann, img), rle in zip(parse_gt_ids(ids), rles):
        out.append((ann, img, decode_rle_any(rle, height, width)))
    return out


def pred_entries_from_csv(rows: Dict[str, List[str]], height: int = HEIGHT,
                          width: int = WIDTH
                          ) -> List[Tuple[str, Mask]]:
    """官方预测/提交 CSV → solar_pq 的 PredEntry 列表。"""
    ids = rows["filament_id"]
    rles = rows["segmentation_rle"]
    return [(img, decode_rle_any(rle, height, width))
            for img, rle in zip(parse_pred_ids(ids), rles)]


# 与 metrics_local.solar_pq 的一站式对拍入口
def pq_from_csv(gt_rows: Dict[str, List[str]], pred_rows: Dict[str, List[str]],
                height: int = HEIGHT, width: int = WIDTH) -> float:
    """GT CSV + 预测 CSV → 官方口径 PQ（经 RLE 解码 → solar_pq）。"""
    from solar_pq import solar_pq  # metrics_local 需在 sys.path（conftest 注入）
    gt = gt_entries_from_csv(gt_rows, height, width)
    pred = pred_entries_from_csv(pred_rows, height, width)
    return solar_pq(gt, pred)
