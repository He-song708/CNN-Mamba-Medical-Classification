#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按比例（默认 train 0.8、val 0.1、test 0.1）分层拆分数据集。
原始目录结构：
    dataset/
        classA/
        classB/
        ...
拆分后在 dataset 同级目录生成：
    train/  val/  test/
        classA/ | classB/ | ...
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser("Split dataset into train/val/test subsets")
    p.add_argument("--data_dir", type=str, default="dataset",
                   help="原始数据根目录（包含类别子文件夹）")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--test_ratio",  type=float, default=None,
                   help="测试集比例；若留空则自动 = 1 - train - val")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--move", action="store_true", help="移动文件而非复制")
    return p.parse_args()


def safe_split_counts(n_total, train_r, val_r, test_r):
    """
    根据比例返回 (n_train, n_val, n_test)，保证：
    - 每类样本数 ≥3 时，val/test 至少 1
    - 总和等于 n_total
    """
    n_train = round(n_total * train_r)
    n_val   = round(n_total * val_r)
    n_test  = n_total - n_train - n_val  # 余下全部放 test

    # 保底：若数据量足够但某子集为 0，则从 train 借 1 张
    if n_total >= 3:
        if n_val == 0:
            n_val += 1
            n_train -= 1
        if n_test == 0:
            n_test += 1
            n_train -= 1
    return n_train, n_val, n_test


def split_one_class(src_cls_dir: Path, dst_root: Path,
                    ratios, move: bool):
    imgs = sorted([p for p in src_cls_dir.iterdir() if p.is_file()])
    n_total = len(imgs)
    if n_total == 0:
        print(f"[WARN] {src_cls_dir} 为空，已跳过")
        return 0, 0, 0

    random.shuffle(imgs)
    n_train, n_val, n_test = safe_split_counts(
        n_total, *ratios
    )
    parts = {
        "train": imgs[:n_train],
        "val":   imgs[n_train: n_train + n_val],
        "test":  imgs[n_train + n_val:]
    }

    for subset, files in parts.items():
        dst_dir = dst_root / subset / src_cls_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            tgt = dst_dir / f.name
            if move:
                shutil.move(str(f), tgt)
            else:
                shutil.copy2(f, tgt)

    return n_train, n_val, n_test


def main():
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"未找到目录：{data_dir}")

    if args.test_ratio is None:
        args.test_ratio = 1.0 - args.train_ratio - args.val_ratio

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train+val+test 比例之和必须为 1")

    out_root = data_dir.parent
    stat = defaultdict(lambda: [0, 0, 0])  # class → [train,val,test]

    for cls_dir in data_dir.iterdir():
        if cls_dir.is_dir():
            nt, nv, nte = split_one_class(
                cls_dir, out_root,
                (args.train_ratio, args.val_ratio, args.test_ratio),
                args.move
            )
            stat[cls_dir.name] = [nt, nv, nte]
            print(f"{cls_dir.name}: train={nt}, val={nv}, test={nte}")

    # 打印汇总
    print("\n===== 完成！数据集概览 =====")
    print(f"{'Class':<20} {'Train':>6} {'Val':>6} {'Test':>6}")
    for cls, (nt, nv, nte) in stat.items():
        print(f"{cls:<20} {nt:6d} {nv:6d} {nte:6d}")
    print("\n输出目录：", out_root)


if __name__ == "__main__":
    main()