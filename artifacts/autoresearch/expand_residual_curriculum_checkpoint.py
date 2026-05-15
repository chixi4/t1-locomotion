#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--yaw-radius", type=float, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path("envs", f"{args.task}.yaml").read_text(encoding="utf-8"))
    lin_levels = int(cfg["commands"]["lin_vel_levels"])
    yaw_levels = int(cfg["commands"]["ang_vel_levels"])
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    curriculum = data["curriculum"].clone().float()

    lx = torch.arange(-lin_levels, lin_levels + 1, dtype=torch.float).view(-1, 1, 1)
    ly = torch.arange(-lin_levels, lin_levels + 1, dtype=torch.float).view(1, -1, 1)
    yaw = torch.arange(-yaw_levels, yaw_levels + 1, dtype=torch.float).view(1, 1, -1)
    seeded = ((lx.square() + ly.square()).sqrt() <= float(args.radius)) & (yaw.abs() <= float(args.yaw_radius))
    if tuple(seeded.shape) != tuple(curriculum.shape):
        raise ValueError(f"curriculum shape {tuple(curriculum.shape)} does not match task grid {tuple(seeded.shape)}")
    before = int(torch.count_nonzero(curriculum > 0.5))
    curriculum[seeded] = 1.0
    center = (lin_levels, lin_levels, yaw_levels)
    curriculum[center] = 1.0
    after = int(torch.count_nonzero(curriculum > 0.5))
    data["curriculum"] = curriculum

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(f"expanded curriculum {before} -> {after}; saved {out}")


if __name__ == "__main__":
    main()
