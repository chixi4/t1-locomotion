"""Mirror a Booster T1 motion csv across the sagittal plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from booster_train.motion_tools.t1 import mirror_t1_csv_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror a Booster T1 23DoF csv motion.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the source csv motion.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the mirrored csv motion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion = np.loadtxt(args.input_file, delimiter=",", dtype=np.float32)
    mirrored = mirror_t1_csv_motion(motion)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, mirrored, delimiter=",")
    print(f"[INFO] Mirrored motion saved to {output_path}")


if __name__ == "__main__":
    main()
