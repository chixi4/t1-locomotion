"""Convert Booster T1 motions exported by GMR 27DoF model to Booster T1 23DoF CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from booster_train.motion_tools.t1 import convert_gmr_t1_motion_27dof_to_t1_23dof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GMR Booster T1 27DoF csv to Booster T1 23DoF csv.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the 27DoF csv file exported by GMR.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the converted 23DoF csv file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion = np.loadtxt(args.input_file, delimiter=",", dtype=np.float32)
    converted = convert_gmr_t1_motion_27dof_to_t1_23dof(motion)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, converted, delimiter=",")
    print(f"[INFO] Converted motion saved to {output_path}")


if __name__ == "__main__":
    main()
