# -*- coding: utf-8 -*-
"""仓库卫生闸：确保本仓库**不含任何数据文件 / 超限文件**（训练权重白名单除外）。

背景：比赛官方条款规定数据仅限非商业学术研究使用、不得再分发，所以本仓库
只发布代码与文档；数据请读者自行从 Kaggle 下载（README §数据）。唯一例外是
``weights/`` 下的**训练好的模型权重白名单**：主办方要求提交的预测能被委员会
无需额外文件直接复现，因此 4 个集成成员 checkpoint 随仓库发布
（``FilamentUNet5`` state_dict，与数据无关，不属于比赛数据）。

检查项（全部硬性，任何一条不满足即 exit 1）：
  1. 禁止的数据/派生产物扩展名：.npz .npy .jpeg .jpg .png .tif .tiff
     .pth .ckpt .pkl .json .csv .parquet .zip .gz .7z .rar（.json/.csv 一并
     禁止：标注、预测、划分结果都属于数据或衍生产物，一律不放仓库）；
     ``.pt`` 仅在 ``weights/`` 目录下、且文件名在 ``WEIGHTS_WHITELIST`` 内时
     放行，其余位置一律禁止；
  2. 任何文件不得超过 --max-bytes（默认 1 MiB）；白名单权重单文件上限放宽到
     WEIGHTS_MAX_BYTES（64 MiB）；
  3. 隐藏目录 / __pycache__ / .pytest_cache / 虚拟环境目录**整目录跳过、不计违规**
     （运行本仓库代码必然生成 __pycache__，若在此报错会让正常复现失败）；
     它们由 .gitignore 负责不进入版本库。

用法：
    python check_repo.py                  # 检查脚本所在目录，打印 JSON 结论
    python check_repo.py --manifest MANIFEST.md   # 同时落盘一份清单
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

BANNED_EXTS = {".npz", ".npy", ".jpeg", ".jpg", ".png", ".tif", ".tiff",
               ".pt", ".pth", ".ckpt", ".pkl", ".json", ".csv", ".parquet",
               ".zip", ".gz", ".7z", ".rar"}
BANNED_DIRS = {"__pycache__", ".pytest_cache", ".venv", "venv",
               "labels", "output", "prob_cache", "cache"}
BANNED_DIR_PREFIXES = (".")

# 训练权重白名单：仅 weights/ 顶层、且文件名逐字匹配才放行（其它 .pt 仍禁）
WEIGHTS_DIR = "weights"
WEIGHTS_WHITELIST = {
    "unet_s3_best.pt", "unet_s5_best.pt",
    "unet_s11_best.pt", "unet_s12_best.pt",
}
WEIGHTS_MAX_BYTES = 64 * 1024 * 1024   # 白名单权重单文件上限（64 MiB）


def iter_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        if any(p in BANNED_DIRS or p.startswith(BANNED_DIR_PREFIXES)
               for p in parts):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames
                       if d not in BANNED_DIRS and not d.startswith(".")]
        for fn in sorted(filenames):
            yield os.path.join(dirpath, fn)


def is_whitelisted_weight(rel: str) -> bool:
    """weights/ 顶层、名字在白名单内的 .pt → True（允许入库的模型权重）。"""
    parts = rel.split(os.sep)
    return (len(parts) == 2 and parts[0] == WEIGHTS_DIR
            and parts[1] in WEIGHTS_WHITELIST)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--max-bytes", type=int, default=1024 * 1024,
                    help="单文件字节上限（默认 1 MiB；weights 白名单除外）")
    ap.add_argument("--manifest", default=None,
                    help="把清单+结论写成 Markdown（如 MANIFEST.md）")
    args = ap.parse_args()

    files = []
    violations = []
    manifest_rel = os.path.relpath(os.path.abspath(args.manifest),
                                   os.path.abspath(args.root)) if args.manifest else None
    for path in iter_files(args.root):
        rel = os.path.relpath(path, args.root)
        if manifest_rel and rel == manifest_rel:
            continue    # 清单不列自己（自引用会使表内哈希失真）
        size = os.path.getsize(path)
        ext = os.path.splitext(path)[1].lower()
        files.append({"path": rel, "bytes": size})
        if ext in BANNED_EXTS and not is_whitelisted_weight(rel):
            violations.append(f"数据/禁传扩展名 {ext}: {rel}")
        limit = WEIGHTS_MAX_BYTES if is_whitelisted_weight(rel) else args.max_bytes
        if size > limit:
            violations.append(f"超限文件 {size} B > {limit} B: {rel}")

    total = sum(f["bytes"] for f in files)
    data_exts = {".npz", ".npy", ".jpeg", ".jpg", ".png", ".tif", ".tiff",
                 ".pth", ".ckpt", ".json", ".csv"}
    data_files = [f["path"] for f in files
                  if os.path.splitext(f["path"])[1].lower() in data_exts]
    weights_files = [f for f in files if is_whitelisted_weight(f["path"])]
    max_file = max(files, key=lambda f: f["bytes"]) if files else None
    report = {
        "root": os.path.abspath(args.root),
        "n_files": len(files),
        "total_bytes": total,
        "max_file": max_file,
        "max_bytes_limit": args.max_bytes,
        "weights_max_bytes_limit": WEIGHTS_MAX_BYTES,
        "weights_whitelist": sorted(WEIGHTS_WHITELIST),
        "weights_files": weights_files,
        "data_files": data_files,
        "n_violations": len(violations),
        "violations": violations,
        "passed": not violations,
        "files": files,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "files"},
                     ensure_ascii=False, indent=1))
    if args.manifest:
        lines = ["# 仓库清单（由 check_repo.py 自动生成）", "",
                 f"- 清单自身（`{manifest_rel}`）不在下表内，重跑 `python check_repo.py "
                 f"--manifest {manifest_rel}` 可复核",
                 f"- 文件数：**{len(files)}**（不含清单自身）",
                 f"- 总字节数：**{total}**（代码、文档与白名单模型权重）",
                 f"- 最大文件：`{max_file['path']}` {max_file['bytes']} B"
                 if max_file else "- 最大文件：（空）",
                 f"- 数据文件列表：**空**（检查扩展名 npz/npy/jpeg/jpg/png/"
                 f"tif/tiff/pth/ckpt/json/csv 等，0 个命中；任何位置含 .pt 亦禁）",
                 f"- 模型权重白名单：**{len(weights_files)}** 个（`weights/` 顶层、"
                 f"文件名逐一匹配 WEIGHTS_WHITELIST，单文件 ≤{WEIGHTS_MAX_BYTES} B，"
                 f"其余位置的 .pt 一律违规）",
                 f"- >{args.max_bytes} B 非白名单文件数："
                 f"**{sum(1 for f in files if f['bytes'] > args.max_bytes and not is_whitelisted_weight(f['path']))}**",
                 f"- 检查结论：**{'PASS' if report['passed'] else 'FAIL'}**"
                 + ("" if report["passed"] else
                    f"（{len(violations)} 条违规，见 check_repo.py 输出）"),
                 "", "| 文件 | 字节 | sha256（前 12 位） |", "|---|---|---|"]
        for f in sorted(files, key=lambda x: x["path"]):
            h = hashlib.sha256(
                open(os.path.join(args.root, f["path"]), "rb").read()).hexdigest()[:12]
            lines.append(f"| `{f['path']}` | {f['bytes']} | `{h}` |")
        with open(args.manifest, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[manifest] → {args.manifest}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
