# -*- coding: utf-8 -*-
"""
patch_drop_point.py
读取已存在的结果文件（没有 Drop point），根据 Heading/Speed/Drop time 推算投放点并写回。
插入位置：紧邻 "Explosion point = (...)" 之前。
支持：Q2Results_*.txt 与 Q2_All_Summary.txt

使用：
  python patch_drop_point.py
  或指定目录：
  python patch_drop_point.py --dir ./results
"""

import os
import re
import math
import argparse
from glob import glob

FY1_INIT = (17800.0, 0.0, 1800.0)  # 题面给定

# 正则：严格按你文件的行格式
RE_H = re.compile(r"^Heading \(rad\)\s*=\s*([\-0-9.eE]+)")
RE_V = re.compile(r"^Speed \(m/s\)\s*=\s*([\-0-9.eE]+)")
RE_D = re.compile(r"^Drop time \(s\)\s*=\s*([\-0-9.eE]+)")
RE_EXP = re.compile(r"^Explosion point\s*=\s*\(")
RE_DROP_ALREADY = re.compile(r"^Drop point\s*=\s*\(")

def compute_drop_point(heading, speed, drop):
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    x = FY1_INIT[0] + vx * drop
    y = FY1_INIT[1] + vy * drop
    z = FY1_INIT[2]
    return (x, y, z)

def patch_one_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 如果已有 Drop point 行，直接跳过
    if any(RE_DROP_ALREADY.search(ln) for ln in lines):
        print(f"[SKIP] already has Drop point: {os.path.basename(path)}")
        return False

    # 解析所需参数
    heading = speed = drop = None
    for ln in lines:
        m = RE_H.search(ln)
        if m: heading = float(m.group(1)); continue
        m = RE_V.search(ln)
        if m: speed = float(m.group(1)); continue
        m = RE_D.search(ln)
        if m: drop = float(m.group(1)); continue

    if heading is None or speed is None or drop is None:
        print(f"[WARN] missing fields (H/V/D): {os.path.basename(path)}")
        return False

    dp = compute_drop_point(heading, speed, drop)
    new_line = f"Drop point = ({dp[0]:.6f}, {dp[1]:.6f}, {dp[2]:.6f})"

    # 找到 Explosion point 行，把新行插在其前
    out_lines = []
    inserted = False
    for ln in lines:
        if (not inserted) and RE_EXP.search(ln):
            out_lines.append(new_line)
            inserted = True
        out_lines.append(ln)

    # 如果找不到 Explosion point，就附加到“Best Solution”段尾部（兜底）
    if not inserted:
        out_lines.append(new_line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"[OK] patched: {os.path.basename(path)}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="结果文件所在目录（默认当前）")
    args = ap.parse_args()

    # 匹配你这 6 个结果文件
    candidates = []
    candidates += glob(os.path.join(args.dir, "Q2Results_*.txt"))
    candidates += glob(os.path.join(args.dir, "Q2_All_Summary.txt"))

    if not candidates:
        print("[Info] no files found.")
        return

    ok = 0
    for f in candidates:
        if patch_one_file(f):
            ok += 1
    print(f"[Done] patched {ok}/{len(candidates)} files.")

if __name__ == "__main__":
    main()
