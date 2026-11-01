# Solar Filament Instance Segmentation — 公开仓库

Kaggle 比赛 **filament-segmentation-2026**（IEEE BigData Cup 2026 · 太阳暗条实例分割，
官方指标 Panoptic Quality）的**可复现方法仓库**。仓库里**不含任何比赛数据**
（官方条款：数据仅限非商业学术研究使用、不得再分发），但**含 4 个训练好的集成成员权重**
（`weights/`；官方要求权重入库并能在 notebook 里直接加载复现，无需重训）。数据请自行从
Kaggle 下载（§2）。**复现入口就是 `reproduce.ipynb`**：加载 `weights/`、对 test 180 张
推理、4 seed 概率平均 + 后处理、写出并通过校验的官方提交 CSV（§4 开头）。

英文技术报告有两个版本：`report.tex`（本仓库的完整技术说明，**要跑两遍 `pdflatex report.tex`**）
与 `paper/`（**提交用的 ACM sigconf 报告源与成品 PDF**，编译方式见 §8）。

---

## 1. 这个方法在做什么

**一段话**：比赛的训练标注是「多位标注者对同一张图各自画的多边形」，而测试图没有任何
公开标注。本方法把同一张图的所有标注者多边形光栅化到同一画布上**逐像素投票**，得到
「支持度图（该像素被几位标注者覆盖）+ 类别图（Left/Right 归属）」作为**共识伪标签**；
用它训练一个 1024² 裁剪上的 U-Net（base32/depth5，7,759,653 参数）预测暗条概率图；
再把概率图变成实例（阈值 → 形态学闭运算 → 4 邻接连通域 → 面积下限 → 逐级放宽兜底，
保证每图至少 1 个实例）。最终提交用 **4 个独立 seed 的概率平均 + 翻转 TTA** 再走同一套
实例抽取。这样做的依据是官方 PQ 口径本身：预测按图像共享、命中按 IoU>0.5 严格判定、
TP 数「命中对」——一根同时贴住多位标注者暗条的预测在每个标注者行上都同时贡献分子，
所以**逐图最优单预测就是共识形状**。

**流程图（文字版）**：

```
Kaggle 公开训练标注 (COCO JSON, 1154 条「标注者-图像」/ 8199 多边形 / 707 图 / 37 标注者)
        │
        ▼ build_pixel_labels.py
[共识伪标签] 同图所有标注者多边形 → 2048² 画布逐像素投票
        support2048 (被几位标注者覆盖) + votes2048 (4 类票数) + class2048 (argmax)
        + 512/1024 面积平均视图          ──labels/consensus/*.npz──┐
        ▼ make_val_split.py                                        │
[val 固定划分] 多标注者图按 (年代, 标注者数) 分层、seed 20261031    │
        完整数据 ⇒ val 106 / train 601（确定性，见 §7 口径说明）     │
        ▼ unet_train_crop.py                                       │
[U-Net 训练] 1024² crop，soft BCE(w=1+9t) + 0.5·CE，Adam 1e-3 cosine，
        50 轮 / bs 4 / fp16，单 seed ≈ 29 min（V100S-32GB）         │
        ▼ unet_infer1024.py                                        │
[推理] 2048² jpeg → 1024² 双线性（与训练同分布）→ 概率图 + 4 类图     │
        TTA：水平/垂直翻转平均；每图存 prob1024/cls1024 npz 缓存      │
        ▼ postproc_sweep.py                                        │
[实例抽取] 多 seed 概率平均 → 双线性放大 2048² → 阈值 θ →
        闭运算 k×k → 4 邻接连通域 → 面积下限 a → 逐级放宽兜底         │
        ▼ rle.py + submit_writer.py                                │
[提交 CSV] filament_id,segmentation_rle（COCO 压缩 RLE，2048² 定尺，
        无引号），写出即过硬闸校验                                   │
        ▼ solar_eval_local.py / solar_pq.py                        │
[本地评测] 官方口径 PQ：逐「标注者-图像」行池化、IoU>0.5 严格、
        TP 数命中对、分子累加命中对 IoU；另有 Dice 与三种宏平均对照
```

为什么这些步骤各自必要（全部有实测对照，见 §7 / report.tex §4）：
- **造共识标签**：单标注者的形状只是「一个人的画法」；像素投票后的共识形状才是逐图最优单预测（3 人图上实例级多数投票比照抄单人 +0.069 PQ，`annot_consensus_probe2.py` 实测）。
- **固定 val 划分**：训练监控、超参扫描、seed 比较必须共用同一份 val，否则数字不可比。
- **U-Net 学共识**：手写规则检测器把可调旋钮拧到头也只有线上 0.15→0.18；同 val 上换 U-Net 线 PQ 0.2201→0.3279。
- **实例抽取与兜底**：PQ 的 IoU>0.5 是断崖——两根暗条粘成一连通域＝双双不命中；一根拆碎同样不命中。闭运算把碎片缝回去（+0.002），面积下限滤碎片 FP；空图＝该图整行 FN，所以逐级放宽保证每图至少 1 个实例。
- **RLE + 校验**：官方提交契约是 `filament_id,segmentation_rle` 两列、COCO 压缩 RLE、无引号；写错格式＝白交。写出前硬闸 + 与 pycocotools 往返对拍。
- **官方口径评测**：所有本地数字必须和官方同一口径才可比较（本仓库 `solar_pq.py` 是官方自评 notebook 的逐行复刻，`solar_eval_local.py` 对其 Δ=0）。
- **4-seed 集成**：早期在 s2+s3 / 旧 4-seed 上集成确实输给最佳单成员，但最终用的
  s12+s5+s11+s3 在 val-88 上 0.3370 / 0.3449，**高于最佳单成员 0.3268**（见 §7）。

---

## 2. 数据从哪下（不打包，自行下载）

```
kaggle competitions download -c filament-segmentation-2026     # 整包 ≈716 MB / 888 文件
unzip filament-segmentation-2026.zip -d $DATA/MAGFiLO_1.0_Kaggle_2026
```

解包后本仓库用到的公开文件（路径相对 `$DATA`，逐个列出）：

| 路径 | 内容 | 规模 |
|---|---|---|
| `MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` | 官方训练标注（COCO 格式，polygon） | 48,616,700 B；1154 条目 / 8199 实例 / 707 图 / 37 标注者；类别 Left 2535 / Right 2590 / Unidentifiable 3074 / **Ambiguous 0** |
| `MAGFiLO_1.0_Kaggle_2026/train/train_images/` | 训练图（2048×2048, L 灰度 jpeg） | 707 张 |
| `MAGFiLO_1.0_Kaggle_2026/test/test_images/` | 测试图（同规格，无标注） | 180 张 |

（整包里没有 sample_submission.csv，提交契约只由 Overview 文本定义，固化在本仓库
`submit_writer.py` 的校验器里。）数据出处：MAGFiLO v1.0（Scientific Data 2024，
DOI 见 §9）的 Kaggle 2026 赛事抽取版。**本仓库不包含也不再分发以上任何文件。**

---

## 3. 依赖与版本

`pip install -r requirements.txt`（**锁版本**）。`reproduce.ipynb` 全流程实测通过的环境
（= requirements.txt 所锁版本，提交 CSV 与 val-88 PQ 均在此环境产出）：

| 包 | 锁定版本 | 说明 |
|---|---|---|
| Python | 3.11.6 | |
| torch | 2.5.1+cu124 | GPU 训练/推理；纯 CPU 也能跑通全流程（推理走 fp32，可能有毫级数值差异） |
| numpy | 2.4.6 | |
| scipy | 1.17.1 | 连通域 / 形态学 |
| pillow | 13.0.0 | jpeg 解码与双线性缩放 |
| pycocotools | 2.0.11 | 官方语义光栅化（评测）+ RLE 往返对拍 |
| nbconvert / ipykernel / nbformat | 7.17.2 / 7.4.0 / 5.11.1 | 执行 `reproduce.ipynb` |
| kaggle（可选） | — | 只在下载数据时用 |

实测硬件：notebook 全流程 Tesla V100S-PCIE-32GB；训练冒烟 RTX 3070 Laptop 8GB；
完整 50 轮单 seed 在 Tesla V100S-PCIE-32GB（峰值显存 18.99 GiB，**8GB 卡请把 bs 降到
1–2**，本仓库冒烟即 bs 1）。（历史备注：§4 的命令链此前另在 torch 2.15.0+cu132 /
numpy 2.5.4 / scipy 1.18.1 环境下实测过一轮，结论一致。）

---

## 4. 实测跑通的复现命令链

### 4.0 一键全流程：`reproduce.ipynb`（复现提交件的推荐入口）

仓库根目录执行（数据位置用环境变量 `SOLAR_DATA_ROOT` 指定，工作目录 `SOLAR_WORK_DIR`
缺省写到系统临时目录、不污染仓库；概率缓存可断点续跑）：

```bash
pip install -r requirements.txt
export SOLAR_DATA_ROOT=/path/to/data          # 其下有 MAGFiLO_1.0_Kaggle_2026/test/test_images 等
jupyter nbconvert --to notebook --execute reproduce.ipynb \
    --output repro_executed.ipynb --ExecutePreprocessor.timeout=3600
```

notebook 做的事 = 加载 `weights/unet_s{3,5,11,12}_best.pt` → test 180 张 1024² 推理
（hvflip TTA）→ s12+s5+s11+s3 概率平均 → `close7_t070_a600` 实例抽取 → 官方提交 CSV
（`submit_writer.validate_submission_csv` 硬闸）→（可选，需 `SOLAR_VAL_SPLIT` 指向冻结
val 划分）val-88 官方口径 PQ。实测（V100S-32GB，本次完整重跑）：见 §7「notebook 全流程
从权重复现提交件」一节。

> 下面的每条命令都在本机实跑过（2026-11-01，16 核 + RTX 3070 Laptop 8GB + 上表环境），
> 括号里是**本次实测**耗时与产物。`$DATA` = 数据根目录，`$WORK` = 工作目录（标签/权重/
> 概率缓存都写在仓库外）。`python` 指装好依赖的解释器。

```bash
# 0) 装依赖（本次未重装，使用满足 requirements.txt 的现有环境；新环境请先执行）
pip install -r requirements.txt
export DATA=/path/to/data WORK=/path/to/work && mkdir -p $WORK

# 1) 共识伪标签（全量 707 图；本次实测 908.6 s ≈ 15 min）
python build_pixel_labels.py build \
    --ann $DATA/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --labels $WORK/labels
#   产物：$WORK/labels/consensus/*.npz（707 个）、labels/gt_per_annotator/*.npz（707 个）、
#         labels/{consensus_index,gt_index,build_manifest}.json
#   关键产物校验：manifest 里 n_files=707 / n_gt_rows=1154 / n_gt_instances=8199 / 37 标注者

# 2) 标签自检（重读全部 npz + 与独立探针在 296 张多标注者图上逐像素对拍）
python build_pixel_labels.py verify \
    --ann $DATA/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --labels $WORK/labels
#   期望输出：5 条 [PASS]（GT 行数=1154；实例总数=8199；support 与探针 296/296 逐像素一致）
#   本次实测：5/5 PASS，4 分 31 秒（time 实测 real 4m30.7s）

# 3) 固定 val 划分（<1 s；完整数据 ⇒ val 106 / train 601，seed 20261031）
python make_val_split.py \
    --ann $DATA/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --images $DATA/MAGFiLO_1.0_Kaggle_2026/train/train_images \
    --out $WORK/labels/val_split.json
#   产物：$WORK/labels/val_split.json（本次实测 sha256=0dc83e49…，val 265 个 GT 行）

# 4) 训练（完整：50 轮 bs 4，单 seed ≈ 29–30 min on V100S-32GB；
#    本次 8GB 卡冒烟 = 2 轮 bs 1：缓存 88 s + 训练 178.6 s，峰值显存 2.44 GiB）
#    ⚠ 只想复现提交**不需要这一步**：仓库已带 4 个训练好的权重，直接跳到 §5 / §6。
python unet_train_crop.py --gpu 0 --crop 1024 --epochs 50 --bs 4 \
    --base 32 --depth 5 --seed 0 --tag s0 \
    --data-root $WORK --out $WORK/out
#   产物：$WORK/out/unet_s0_best.pt / unet_s0_last.pt / history_s0.json / dropped_s0.json
#   期望首行日志：[model] FilamentUNet5 base=32 depth=5 params=7759653 (7.76M)
#   ⚠ 冒烟只验证管线通：本次 2 轮 bs1 的模型概率还压不过阈值（val PQ≈0.001），不代表 50 轮水平
#   多 seed：对 4 个 --seed 各跑一遍（s3/s5/s11/s12 是当年最终集成用的 seed 号，
#   它们的权重已在 weights/ 里，见 §5）

# 5) 推理 → 概率缓存（val 用 --split 只推 val 名单；test 推全集）
python unet_infer1024.py --ckpt $WORK/out/unet_s0_best.pt \
    --images $DATA/MAGFiLO_1.0_Kaggle_2026/train/train_images \
    --split $WORK/labels/val_split.json --out $WORK/prob_val_s0 --tta hflip
python unet_infer1024.py --ckpt $WORK/out/unet_s0_best.pt \
    --images $DATA/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --out $WORK/prob_test_s0 --tta hflip
#   产物：$WORK/prob_*/<图名>.npz（prob1024 float16 + cls1024 uint8）+ _meta.json
#   本次实测：106 张 21 s、180 张 37 s（RTX 3070 Laptop，hflip TTA）
#   想用仓库自带权重，把 --ckpt 换成 weights/unet_s{3,5,11,12}_best.pt 即可（§5）

# 6) 实例抽取/后处理变体扫描（在 val 上挑配置）
python postproc_sweep.py sweep --prob $WORK/prob_val_s0 \
    --gt-ann $DATA/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --split $WORK/labels/val_split.json \
    --families baseline,f1,f3,f5,f6,f7 --out $WORK/sweep_s0.csv --out-json $WORK/sweep_s0.json
#   产物：CSV/JSON 各一份，按官方口径 PQ 降序（f7 族需 ≥2 个 --prob 目录做概率平均）
#   本次实测：上面这组 families 共 **39 个变体**（f1 基线族 28 + f3 形态族 8 + f5 类别族 1
#     + f6 放宽族 2），单进程 CPU、16 核机器干净实跑 **604 s**
#   ⚠ 口径：另一次只跑 **11 个变体** 的子集（f1 基线 1 + f3 形态族 8 + f6 放宽族 2）是 **173 s**
#     （落盘工件 11 行、sum(sec)=172.5）。两个数字对应**不同变体集**，不要混用。
#   ⚠ 机器有并发负载时复跑同一组 39 个变体：sum(sec)=1075.1 s、wall clock 19:04.8 s。
#     所以「39 变体 ≈ 10 分钟」是干净机器下的数，被别的活占着时要按 1.8× 预留。

# 7) 出提交 CSV（最终配方 = 4 seed 概率平均 + close7 / θ0.70 / 面积下限 600；
#    单 seed 时给一个 --prob 即可）
python postproc_sweep.py submit \
    --prob $WORK/prob_test_s3 $WORK/prob_test_s5 $WORK/prob_test_s11 $WORK/prob_test_s12 \
    --images $DATA/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --preset close7_t070_a600 --out $WORK/submission_ens4_close7.csv \
    --report $WORK/submit_report.json
#   产物：提交 CSV + report JSON（行数/覆盖/放宽图数/spec/sources）
#   写出即过 submit_writer.validate_submission_csv 硬闸；180 张图有缺缓存会直接拒绝出半份提交

# 8) 提交前校验（独立于写出的一遍；<1 s）
python - <<'EOF'
from submit_writer import validate_submission_csv
text = open("submission_ens4_close7.csv", encoding="utf-8").read()
ids = {ln.split(",")[0].rsplit("_", 1)[0] for ln in text.splitlines()[1:] if ln.strip()}
print("issues =", validate_submission_csv(text, expect_images=sorted(ids)))
EOF
#   期望输出：issues = 0 []（表头/两列/id 唯一/RLE 可解码/非空掩膜/无引号）

# 9) 官方口径本地评测（val；CSV 或 npz 目录皆可）
python solar_eval_local.py \
    --gt-ann $DATA/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --split $WORK/labels/val_split.json --preds $WORK/preds_val.csv --pred-format csv
#   期望输出：pq / dice_pooled / tp / fp / fn + 三种宏平均对照（JSON 可 --out 落盘）
#   配套：把某套概率缓存按第 9 步同款命令 --images <val图目录> 出成 val 的 CSV 即可评测

# 10) 仓库卫生闸（无数据文件 / 无超限文件 / 隐藏目录；weights/ 白名单权重放行；生成清单）
python check_repo.py --manifest MANIFEST.md
#   期望输出："passed": true, "data_files": [], "n_violations": 0
#   （weights/ 下 4 个白名单 .pt 单文件上限放宽到 64 MiB，其余位置仍禁 .pt）
```

**本次复现产出的两份提交 CSV**（落在仓库外的复现工作目录 `$WORK` 里；提交 CSV 属衍生物、
按卫生闸不入库，所以此处只登记文件名与可核对的行数/覆盖）：

| 文件 | 概率缓存来源 | 行数 | 覆盖 | 硬闸（本次实测） |
|---|---|---|---|---|
| `submission_smoke_close7.csv` | 冒烟模型（2 轮 bs 1）概率缓存，spec = close7 / θ0.70 / 面积下限 600 / 放宽链 | **180** | 180/180 图有行（每图 1 个兜底实例，fallback 180） | `issues = []` |
| `submission_ens2_t065_a800.csv` | 真实模型 2-seed 概率平均缓存（s0+s1，hvflip TTA），spec = θ0.65 / 面积下限 800 / 放宽链 | **1012** | 180/180 图有行（fallback 12） | `issues = []` |

> 口径提醒：这两份是**本次复现**（从零造标签 → 冒烟训练 → 推理 → 出提交）的产物，
> 不是历史提交件；历史 9 次提交件的行数见 §7 公榜表。提交 CSV 不入库是仓库卫生闸的
> 硬性要求（`.csv` 在禁传扩展名里），故此处以「文件名 + 行数 + 覆盖 + 硬闸结论」登记。

**重构脚本的对拍验证**（本仓库 `unet_infer1024.py`/`postproc_sweep.py` 是按落盘契约
重建的，见 §8 注）：对当年冻结的 s3+TTA val 概率缓存重跑，`close3` = **0.3171617809448947**
（与 `eval_postproc_close3_val88_official.json` 逐位一致），baseline/no-relax/relax1-only
= 0.31516699298930984 / 0.31416930530119935 / 0.31516699298930984、close5/9/13、open3/5
全部与当年扫描表逐位一致；仅 open9/open13（当年即判有害弃用的尾部两档）差 ~0.001–0.002
（大开运算在近空掩膜上的边界语义，不影响任何结论）。`solar_eval_local.py` 对当年
close3 预测 CSV 复测 = 同一数字逐位复现；`unet_sweep.py --crosscheck` 与其完全一致；
提交 CSV 与 pycocotools 解码往返 20/20 一致。

---

## 5. 仓库自带的权重（免重训复现）

官方 Final Submission 要求原文（`official.md:67,69`）：仓库里必须有 `requirements.txt`、
必须有一个覆盖全流程的 jupyter notebook，且「**For ML/DL models, the model's weights must
be stored in the repo, and be loaded in the notebook such that the results can be reproduced
without needing to re-train the model.**」；主办方在讨论区补充（`faq.md:81`）：
「Otherwise, the reproducibility requirement is not satisfied, and therefore, such submissions
will not be considered for the award.」

所以本仓库的口径是 **含权重、不含数据**：`weights/` 下是最终提交配方（4 seed 概率平均 +
hvflip TTA（4 变体：原图+h+v+h∘v）+ close7）的 **4 个集成成员权重**，notebook 直接加载，
**复现提交不需要重训**。

| 文件 | seed | 大小 | sha256（前 12 位） | 训练配置 | 硬件 | 单 seed 训练耗时 |
|---|---|---|---|---|---|---|
| `weights/unet_s3_best.pt` | 3 | 31,063,818 B | `e163f8c79013` | 1024² crop · base32/depth5 · 7,759,653 参数 · 50 轮 · bs 4 · Adam lr 1e-3 cosine · soft BCE(w=1+9t) + 0.5·CE · fp16 | Tesla V100S-PCIE-32GB | 29 min/seed（提交描述） |
| `weights/unet_s5_best.pt` | 5 | 31,063,498 B | `73abd93dbded` | 同上 | 同上 | 1747.2 s = 29.1 min |
| `weights/unet_s11_best.pt` | 11 | 31,062,590 B | `e07eb7ecc7cd` | 同上 | 同上 | 1809.8 s = 30.2 min |
| `weights/unet_s12_best.pt` | 12 | 31,063,870 B | `283176de3b23` | 同上 | 同上 | 1791.3 s = 29.9 min |

- seed ↔ 文件的对应关系来自各次训练的 `args.seed` / `args.tag`（s5/s11/s12 的训练记录
  `history_s5.json` / `history_s11.json` / `history_s12.json` 里的 `args` 字段）；
  s3 的配置（base=32 / depth=5 / 7.76M 参数 / 50 轮 / 29 min / 1×V100S）取自它那次提交的
  官方描述文本（ref 57847381）。
- 推理时这 4 个权重按 §4 第 5 步逐个跑（`--ckpt weights/unet_s{3,5,11,12}_best.pt`，
  `--tta hvflip`），再在第 7 步用 `--prob` 一起喂给 `postproc_sweep.py submit` 做概率平均
  （或直接走 §4.0 的 notebook）。
- 权重是本团队自己训练出来的**模型产物**，不含任何比赛数据，因此不受「数据不得再分发」
  条款约束；`.gitignore` 因此**故意不忽略** `*.pt`（见该文件末尾的显式放行）。
- 想自己重训（例如换 seed 或改分辨率）见 §4 第 4 步：单 seed 约 29 分钟 / 50 轮 / V100S-32GB。

---

## 6. reproduce.ipynb：一条命令跑通

### 6.1 怎么跑

```bash
pip install -r requirements.txt
pip install jupyter nbconvert        # notebook 额外依赖（requirements.txt 里已锁 nbconvert/ipykernel|nbformat 版本）

# 方式 A：命令行整本执行（执行结果会写回同目录的 .ipynb）
jupyter nbconvert --to notebook --execute reproduce.ipynb

# 方式 B：交互式打开（逐 cell 跑）
jupyter notebook
```

notebook 走的就是 §4 的同一条流水线，只是把 CLI 调用换成了函数调用：
加载 `weights/` 的 4 个成员权重 → 在 180 张测试图上推理（hflip TTA）→ 概率平均 →
close7 / θ0.70 / 面积下限 600 / 逐级放宽兜底 → 写出提交 CSV 并过硬闸；
再用 `solar_eval_local.py` 在 val 上做官方口径复测。

### 6.2 数据从哪下（notebook 实际读到的路径）

```bash
kaggle competitions download -c filament-segmentation-2026     # 整包 ≈716 MB / 888 文件
unzip filament-segmentation-2026.zip -d $DATA/MAGFiLO_1.0_Kaggle_2026
```

解包后 notebook 会读到的路径（相对 `$DATA`，逐个列出）：

| 路径 | 在 notebook 里的用途 | 规模 |
|---|---|---|
| `MAGFiLO_1.0_Kaggle_2026/test/test_images/` | 推理输入 → 提交 CSV 的 180 张图 | 180 张（2048² L 灰度 jpeg） |
| `MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` | 官方训练标注：共识伪标签的原料 + 本地评测的 GT | 48,616,700 B；1154 条目 / 8199 实例 / 707 图 / 37 标注者 |
| `MAGFiLO_1.0_Kaggle_2026/train/train_images/` | 训练图：造共识伪标签 + 复现训练链路 | 707 张 |

（下载与解包命令见 §2；表格是 §2 的同一份清单，这里按 notebook 的使用顺序重排。）

### 6.3 预期产出与预期数字

| 产出 | 预期值 | 出处 |
|---|---|---|
| 提交 CSV 的覆盖 | **180/180 图有行** | 已提交的 ref 57855587 同配方同覆盖 |
| 提交 CSV 的行数 | **1096 行** | `submission_ens4_close7_t070_a600_full180_report.json`：`rows=1096, images=180, images_with_rows=180` |
| 提交 CSV 的 spec | close7 / θ0.70 / 面积下限 600 / 逐级放宽兜底 | 同上：`spec = {kind:morph, op:close, k:7, thr:0.7, ma:600}` |
| 本地 val-88 PQ（官方口径池化） | **0.3449** | `ens_specs_val88.csv` 第 2 行 `f3\|close7 thr0.70 ma600 E4` → `pq=0.34489537422485717`（同表 `dice_pooled=0.6547094149962307`，TP 779 / FP 632 / FN 735） |
| 同一 4-seed、不做闭运算的 val-88 PQ | **0.3370** | `ens_specs_val88.csv` 第 20 行 `f1\|thr0.70 ma600 E4` → `pq=0.3369696638244073` |

> ⚠ **口径提醒**：0.3449 是**冻结 val-88** 名单上的数字（88 图 / 220 个「标注者-图像」行 /
> 1514 根 GT），而那份名单**不在公开数据里**——用公开数据从头跑划分得到的是 val-106
> （完整数据下的确定性划分，见 §4 第 3 步），绝对数会有位移（§7 口径声明第 1 条）。

---

## 7. 实测数字（本地 val 与公榜）

**口径声明（先读这个）**：
1. 历史 val 数字测在**冻结 val-88**（88 图 / 220 个「标注者-图像」行 / 1514 根 GT）上；
   那份名单是当年「本地只有 558/707 张图」时刻同一划分脚本的产物，不在公开数据里。
   读者按 §4 全流程跑出来的是 **val-106**（完整数据下的确定性划分）——绝对数会略有
   位移，同口径相对比较可迁移。
2. 本地 PQ/Dice **不可当线上分预估**：官方 API 元数据 `evaluationAlgorithm=
   panoptic_quality` 与页面 "mean Dice" 互相矛盾，榜单快照也无分项列。
3. val-88 同时承担了早停、超参扫描、seed 选择——所有数字都是「选完之后」的极大值；
   分半交叉测得选择乐观偏差 **+0.002 ~ +0.015 PQ**。

> 本节表格里的 `output/…` 指上游工作区的工件目录（**不随本仓库发布**）；每行都给出
> 文件名与字段名，便于内部逐条回溯。

**本地 val-88（官方池化口径 PQ；同一份 GT 行、同一个评测实现）**：

| 系统 | PQ | Dice（行池化） | TP/FP/FN | 出处 |
|---|---|---|---|---|
| v3 手写检测器 | 0.220085 | 0.486084 | 455/698/1059 | `output/eval_v3_val_88.json` |
| 512² U-Net（base16/depth4） | 0.269595 | 0.621649 | 603/636/911 | `output/eval_unet_v1_val88.json` |
| 1024² s3 + hvflip TTA（θ.70/a600） | 0.315167 | 0.640948 | 704/637/810 | `output/eval_unet1024_s3tta_val88.json` |
| └ + 闭运算 k=3（60 变体表冠军） | 0.317162 | 0.641168 | 708/628/806 | `output/eval_postproc_close3_val88_official.json` |
| 最佳单 seed s12（θ.70/a700） | **0.326789**（含放宽链 0.327912） | 0.643550 | 714/630/800 | `output/u1024_sweep_s12_tta.json`（放宽链 0.327912 来自 `output/eval_unet1024_s12tta_val88.json`） |
| **4-seed 集成 s12+s5+s11+s3（θ.70/a600）** | **0.336970** | 0.651222 | 758/654/756 | `output/ens_specs_val88.csv` 第 20 行 |
| **└ + 闭运算 k=7（最终提交配方）** | **0.344895** | 0.654709 | 779/632/735 | `output/ens_specs_val88.csv` 第 2 行 |

**关键消融（同批概率图、同一评测器）**：hflip TTA **+0.0203**；闭运算 **+0.002**、开运算有害
（open13 → 0.102）；按类别切实例 **−0.0224**、几何分水岭 **−0.2019**；放大到 base48
（17.5M 参数）**反而更差**（0.2096 vs 同族 0.2966）。6/15 个 seed 训练发散（fp16 NaN）。

**多 seed 概率平均：早期结论已被本仓库自己的工件推翻。** 早期在 **s2+s3**
（`u1024_sweep_ens23.json`，best_pq = 0.298458）与**旧 4-seed 组合 s0+s1+s2+s3**
（`u1024_sweep_ens4.json`，best_pq = 0.264421，对当时的最佳成员 s3 = 0.314169）上，
概率平均确实**低于最佳单成员**（**−0.0157 / −0.0497**）。
但**最终用于提交的 s12+s5+s11+s3 四 seed 集成**在 val-88 上是
**0.3370**（`ens_specs_val88.csv` 第 20 行 `f1|thr0.70 ma600 E4`，无闭算）/
**0.3449**（第 2 行 `f3|close7 thr0.70 ma600 E4`，+close7），
**高于最佳单成员 s12 的 0.3268**（`u1024_sweep_s12_tta.json`，best_pq = 0.326789；
含放宽链 0.327912）。所以「多 seed 平均赢不了最佳成员」只对那两组早期组合成立，
**不是本方法的普遍结论**；最终提交件用的就是赢过最佳成员的那一组。

**公榜（Kaggle Public LB）——9 次提交全部完成，U-Net 线也真实提交过并计分**：

| 提交（ref） | 提交文件 | 覆盖 | 行数 | 公开分 | 榜位 |
|---|---|---|---|---|---|
| 57758939 | `submission_baseline_threshold.csv`（v1 固定阈值基线） | 126/180 图 | — | **0.15** | #1139/1253 |
| 57765705 | `submission_consensus_v2.csv`（v2 共识检测器） | 126/180 图 | — | **0.16** | #1140/1260 |
| 57768453 | `submission_consensus_v2_full180.csv`（v2 补齐全覆盖） | 174/180 图有行 | 845 | **0.17** | #1127/1261 |
| 57769686 | `submission_consensus_v3_full180.csv`（v3 多年代重调） | 174/180 图有行 | 874 | **0.18** | #1117/1261 |
| 57843907 | `submission_unet_v1_hflip_t065_a800_full180.csv`（512² U-Net + hflip TTA） | 180/180 图 | 940 | **0.21** | — |
| 57847381 | `submission_unet1024_s3tta_t070_a600_full180.csv`（1024² s3 + TTA） | 180/180 图 | 1066 | **0.27** | — |
| 57847390 | `submission_unet1024_ens23_t065_a800_full180.csv`（s2+s3 集成，对冲件） | 180/180 图 | 1063 | **0.25** | — |
| 57855587 | `submission_ens4_close7_t070_a600_full180.csv`（4-seed 集成 + close7，**最终件**） | 180/180 图 | 1096 | **0.29** | — |
| 57861772 | `submission_s12_s11_s3_close7_t070_a600_full180.csv`（3-seed 集成：s12+s11+s3） | 180/180 图 | 1095 | **0.29** | — |

出处：`output/kaggle_submissions_20261101.txt`（kaggle CLI 捕获，8 条
`SubmissionStatus.COMPLETE`，逐条含 ref / 文件名 / 时间 / publicScore）；第二份独立记录
`账号/Kaggle-提交台账.md` 第 65 / 67 / 68 / 69 / 82 / 83 / 84 / 85 行逐条登记同一批提交
（ref、文件、md5、描述、公开分、状态），两份记录完全一致。
榜位只对前 4 次提交有记录（台账同批记录），后 4 次 U-Net 线提交台账未留榜位快照，故留空。

> **公榜分的用法（重要）**：这 9 个公开分是官方榜单给出的数，而官方口径在 **PQ 与 Dice
> 之间混用**（API 元数据 `evaluationAlgorithm=panoptic_quality` 与页面 "mean Dice"
> 互相矛盾，见上面口径声明第 2 条）。因此**公榜分只用于给这 8 次提交排序**
> （U-Net 线 0.21→0.29 明显好于手写线 0.15→0.18），**不能当作线上分的泛化估计**；
> 要看可迁移的绝对数，请看上面本地表（同口径、同一份 GT、同一评测器）。
> **第 9 次提交（ref 57861772）是一次有意做的对照**：在 21 个已训 seed 上做贪心前向选择，
> 得到的 3-seed 组合（s12+s11+s3）在本仓库的 val-88 上 PQ 0.3463，比线上 4-seed 版的 0.3449 高
> **+0.0014**；但按图像聚类的配对 bootstrap 给出的 95% 置信区间是 [−0.0059, +0.0092]
> （P(更好)≈0.64），且这个划分本身的选参乐观偏差就有约 0.015。**把它提交上去就是为了检验这个
> marginal 增益能不能活到公榜——结果是 0.29，与 4-seed 版完全相同**，即开发集上的 +0.0014 确实
> 是选择噪声、不构成可兑现的提升。我们因此保留 4-seed 版作为最终候选（成员更多、平均化更稳），
> 并把这次「本地更高、线上不动」当作公开数字里的一个诚实注脚。


当时榜单锚点：0.70→#1、0.39→#164、0.30→#893、0.20→#1096、0.15→#1139、0.10→#1167。

**本次（建仓库时）从零复现实测**：707 图全量造标签 908.6 s、自检 5/5 PASS；划分
val-106/train-601（sha256 `0dc83e49…`）；2 轮冒烟训练 178.6 s（params=7,759,653
与当年一致）；106+180 张推理 21 s + 37 s；两份提交 CSV 硬闸 issues=0（冒烟模型 180 行
兜底实例、真实模型集成缓存 1012 行/180 图，见 §4 的登记表）；冻结 close3 预测经本仓库评测器
= 0.3171617809448947 逐位复现。

### 7.1 notebook 全流程从仓库权重复现提交件（2026-11-01 实测）

执行 `reproduce.ipynb`（§4.0；数据与冻结 val 划分用环境变量指入，概率缓存与 CSV 全部
新算、不与当年共享任何中间文件），在 Tesla V100S-32GB + §3 锁版本环境：

| 环节 | 实测 |
|---|---|
| test 180 张 × 4 seed 推理（hvflip=4 变体 TTA） | 216 s（fp16 autocast，float16 缓存） |
| 4 seed 平均 + close7/θ0.70/ma600 + 兜底 → 提交 CSV | 69 s，1096 实例、兜底图 5 张、硬闸 issues=0 |
| **提交 CSV vs 当年落盘提交件** | **同型号 GPU + 同 torch 版本下逐字节一致**（`cmp`/`diff` 0 差异；1097 行含表头；md5 `453859f070da5545afe823484857691d` 两侧相同） |
| **异构硬件上的预期差异** | 在**不同型号 GPU / 不同 torch 版本**上复现时**不应期待字节一致**：实测 RTX 3070 + torch 2.15 相对原 V100S + torch 2.5.1 的 fp16 kernel 存在 1 ULP 级差异（概率图 max\|Δ\|=0.000488），落到提交件上是 **1096 行中 161 行**的掩膜边界像素差异（单实例像素 IoU 中位 0.9994，对称差合计占预测像素 0.0106%），行数、每图实例数、覆盖与硬闸结论均不变。委员会换硬件复现时请以「结构与覆盖一致、像素差在 1e-4 量级」为验收标准 |
| val-88（可选单元，冻结 val 划分） | PQ **0.344895**、Dice 0.654709、TP 779 / FP 632 / FN 735、预测实例 554 —— 与上表 `ens_specs_val88.csv` 第 2 行（0.34489537422485717）逐位同源 |
| 耗时 | test 管线冷跑 ≈ 5 min；含 val-88 的完整重跑 ≈ 5 min（热缓存） |

口径说明：**hvflip TTA = 原图 + 水平 + 垂直 + （水平∘垂直）共 4 次前向等权平均**。
仓库早期快照缺失期间 `unet_infer1024.py` 曾把 hvflip 重建成 3 变体（缺对角），导致
推理概率对不上冻结缓存（|Δ|max≈0.17、thr0.70 像素 IoU≈0.91、提交件只能到「中位实例
IoU 0.969 / 行数 1104 vs 1096」的近似复现）；找回当年原始脚本对拍后已改回 4 变体口径，
现在对冻结概率缓存**逐位复现（|Δ|=0，float16 存储）**，提交件字节级复原。这段弯路
本身也是「为什么委员会必须拿到权重+notebook 而不只是代码」的实证。

---

## 8. 目录里各文件的作用

| 文件 | 作用 | 为什么需要 |
|---|---|---|
| `build_pixel_labels.py` | 多标注者多边形 → 逐像素投票的共识伪标签（support/votes/class，2048²+降采样视图）+ 按标注者分行的评测 GT + `verify` 自检 | 管线源头：没有共识标签就没有训练素材 |
| `make_val_split.py` | 固定 val 划分（多标注者图分层抽样，seed 20261031）| 训练/扫描/比较共用的同一把尺子 |
| `unet_train_crop.py` | 1024² crop 的 U-Net 训练（base32/depth5，soft BCE+0.5CE，fp16）| 学出概率图的模型；也向推理脚本提供 `FilamentUNet5` |
| `unet_infer1024.py` | checkpoint → 逐图 prob1024/cls1024 概率缓存（含翻转 TTA）| 推理入口；缓存格式 = 下游扫描/提交的契约 |
| `postproc_sweep.py` | 概率图 → 实例的后处理变体扫描（阈值/形态学/类别/放宽/seed 平均各族）+ 指定 spec 出提交 CSV | 实例抽取质量是 PQ 第一杠杆；submit 模式即最终提交生成器 |
| `unet_infer.py` | 实例抽取原语库（连通域/分水岭/形状过滤/放宽链 `RELAX_LEVELS`/labels↔masks/提交写出）+ 早期 512² 推理入口 | 上游抽取语义的唯一权威实现，被 sweep/submit 复用 |
| `unet_sweep.py` | 快评器（`eval_fast`/`load_gt_flat`）+ 512² 网格扫描 + `--crosscheck` | 扫描海量配置时的快口径；与官方口径逐位对拍过 |
| `solar_eval_local.py` | 官方口径评测台：池化 PQ（复刻语义）+ 3 种宏平均 + Dice 四口径；CSV/npz 预测皆可 | 最终数字一律以它复核 |
| `solar_pq.py` | 官方自评 notebook 的 PQ 逐行复刻（最小实现） | 口径的「法源」，`annot_consensus_probe2.py` 依赖它 |
| `annot_consensus_probe2.py` | 多标注者共识探针：标注者间实例分歧、照抄单人 vs 共识 vs 实例锚定共识、3 人图 LOO | 方法核心假设（共识形状更优）的证据 |
| `rle.py` | COCO 压缩 RLE 纯 Python 编解码 + GT/预测 id 解析（对齐 pycocotools，交叉对拍过）| 官方 CSV 的 segmentation_rle 列就靠它 |
| `submit_writer.py` | 官方提交 CSV writer + 提交前硬闸校验器 | 格式错＝白交；校验器独立于 writer 可复核 |
| `check_repo.py` | 仓库卫生闸（禁数据扩展名/体积上限/隐藏目录；`weights/` 下 4 个白名单 `.pt` 放行、单文件上限放宽到 64 MiB），`--manifest` 出清单 | 保证「仓库不含数据、但含权重」可持续检查 |
| `requirements.txt` | 依赖与实测版本 | 第三方照着装 |
| `report.tex` | 英文技术报告源文件（article 类，**`pdflatex report.tex` 要跑两遍**：第一遍解析文献与交叉引用，第二遍才 0 未解析；实测两遍均为 6 页） | 方法、实验与口径的完整引用版 |
| `paper/main.tex` | **提交用的 ACM sigconf 报告源**（4 页技术报告，与官方 Overleaf 模板同款 `acmart`） | 赛事要求的正式报告源 |
| `paper/preamble.tex` | `main.tex` 引入的宏包与宏定义（与正文分离，便于单独改排版） | 正文与排版解耦 |
| `paper/main.bib` | `main.tex` 的 BibTeX 文献库（3 条：比赛 / 官方自评 notebook / MAGFiLO 数据集） | 参考文献可追溯 |
| `paper/main.pdf` | 提交用的成品 PDF（4 页 letter；本仓库这版与 `paper/` 源文件同源） | 直接交给组委会的成品 |
| `weights/unet_s{3,5,11,12}_best.pt` | 4 个集成成员权重（各 31,063,xxx B，sha256 见 §5） | 官方要求权重入库并在 notebook 里直接加载，复现提交不需重训 |
| `reproduce.ipynb` | 覆盖全流程的 notebook：加载权重 → 推理 → 出提交 CSV → 本地评测 | 官方要求仓库内须有 notebook 说明整个 pipeline |
| `MANIFEST.md` | 由 `check_repo.py` 生成的文件清单与检查结论 | 发布时可核对的目录快照 |
| `.gitignore` | 禁入比赛数据与派生产物；**显式放行 `weights/*.pt`** | 推 GitHub 前保证「不含数据、但含权重」 |

**论文源怎么编译**（两套，别混）：

```bash
# A) 仓库技术报告 report.tex（article 类，自带 thebibliography，不需要 bibtex）
pdflatex report.tex        # 第 1 遍：3 处 citation + 4 处 reference 未解析
pdflatex report.tex        # 第 2 遍：0 处未解析，6 页（实测）

# B) 提交用 ACM 报告 paper/（sigconf，引用走 BibTeX）
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
#   实测：4 页，0 处未解析 citation / reference，PDF 内 0 个 "??"
```

> **两个文件的重建说明（诚实边界）**：原内部工作区里 `unet_infer1024.py` 与
> `postproc_sweep.py` 跑在远端 V100S，代码快照没有落盘。本仓库这两份是按其**落盘
> 契约**（`unet1024_eval.load_prob1024` 的缓存格式、当年扫描表的 60 组 spec、
> `REPORT_v100_round3.md` 的网格记录）重建的，并用冻结概率缓存对拍验证
> （§4 末段：决策相关变体全部逐位一致）。两份文件头部都写了这件事。
> 同理，`make_val_split.py` 加了 `--images`、`annot_consensus_probe2.py` 的标注路径
> 改为 `SOLAR_ANN` 环境变量、`solar_eval_local.py` 把 `polygon_to_mask` 内联——
> 均为「数据在仓库外」的适配，语义不变。

---

## 9. 引用

- **数据**：A. Ahmadzadeh, R. Adhyapak, K. Chaurasiya, et al. *A dataset of manually
  annotated filaments from H-alpha observations.* **Scientific Data** 11, 1031 (2024).
  [DOI 10.1038/s41597-024-03876-y](https://doi.org/10.1038/s41597-024-03876-y)（MAGFiLO v1.0；
  [PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11437148)）。本仓库所用为
  MAGFiLO 1.0 Kaggle 2026 赛事包。
- **比赛**：IEEE BigData Cup 2026 —
  [Kaggle filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)。
- **官方 PQ 口径**：主办方自评 notebook —
  [azimahmadzadeh/self-evaluation-notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
  （本仓库 `solar_pq.py` 的逐行复刻对象）。
- **指标**：A. Kirillov, K. He, R. Girshick, J. Deng, P. Dollár. *Panoptic Segmentation.*
  CVPR 2019（Panoptic Quality）。
- **RLE 语义**：[pycocotools COCO API](https://github.com/ppwwyyxx/cocoapi)。

---

## 10. 许可与使用边界

代码按仓库现状提供（无附加许可文件，如需复用请与作者确认）。**比赛数据不在本仓库、也不得
经本仓库再分发**：MAGFiLO/比赛数据仅限非商业学术研究使用，请读者自行从 Kaggle 接受条款
后下载。**仓库含 4 个训练好的模型权重**（`weights/`，本团队自己训练出的衍生产物；官方
明文要求权重入库并能在 notebook 里直接加载复现，见 §5）——权重不含任何比赛数据。
想自己重训见 §4 第 4 步（单 seed 约 29 分钟 / 50 轮 / V100S-32GB，8GB 卡请把 bs 降到 1–2）。
