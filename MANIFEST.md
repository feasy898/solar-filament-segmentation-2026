# 仓库清单（由 check_repo.py 自动生成）

- 清单自身（`MANIFEST.md`）不在下表内，重跑 `python check_repo.py --manifest MANIFEST.md` 可复核
- 文件数：**26**（不含清单自身）
- 总字节数：**124771433**（代码、文档与白名单模型权重）
- 最大文件：`weights/unet_s12_best.pt` 31063870 B
- 数据文件列表：**空**（检查扩展名 npz/npy/jpeg/jpg/png/tif/tiff/pth/ckpt/json/csv 等，0 个命中；任何位置含 .pt 亦禁）
- 模型权重白名单：**4** 个（`weights/` 顶层、文件名逐一匹配 WEIGHTS_WHITELIST，单文件 ≤67108864 B，其余位置的 .pt 一律违规）
- >1048576 B 非白名单文件数：**0**
- 检查结论：**PASS**

| 文件 | 字节 | sha256（前 12 位） |
|---|---|---|
| `.gitignore` | 1858 | `bd994b3e592c` |
| `README.md` | 39097 | `d738af231a67` |
| `annot_consensus_probe2.py` | 12941 | `cc48b8f1394d` |
| `build_pixel_labels.py` | 26207 | `0077ad0c826b` |
| `check_repo.py` | 7766 | `e0e5edaf4cf7` |
| `make_val_split.py` | 5904 | `90540e46815b` |
| `paper/main.bib` | 1024 | `6f383805d133` |
| `paper/main.pdf` | 237958 | `e43b35d9c714` |
| `paper/main.tex` | 18788 | `ac51244ad0bc` |
| `paper/preamble.tex` | 2103 | `55bc1756ac9a` |
| `postproc_sweep.py` | 20213 | `20b1783b0ef0` |
| `report.tex` | 19086 | `f4020cbffd36` |
| `reproduce.ipynb` | 18038 | `91a26db84029` |
| `requirements.txt` | 992 | `6653ff809ba7` |
| `rle.py` | 9669 | `9b1ecf38c305` |
| `solar_eval_local.py` | 16663 | `5d786ccfafcc` |
| `solar_pq.py` | 5174 | `523aa40574b9` |
| `submit_writer.py` | 4299 | `9483275f1096` |
| `unet_infer.py` | 29442 | `a828ef41f2f3` |
| `unet_infer1024.py` | 9267 | `73fd9b896f77` |
| `unet_sweep.py` | 12328 | `3573c4804836` |
| `unet_train_crop.py` | 18840 | `e2d54c04fce1` |
| `weights/unet_s11_best.pt` | 31062590 | `e07eb7ecc7cd` |
| `weights/unet_s12_best.pt` | 31063870 | `283176de3b23` |
| `weights/unet_s3_best.pt` | 31063818 | `e163f8c79013` |
| `weights/unet_s5_best.pt` | 31063498 | `73abd93dbded` |
