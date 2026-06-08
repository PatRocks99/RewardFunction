from __future__ import annotations

import os
from pathlib import Path

import rosbag_to_reward_datasets as processor


DATA_ROOT = Path(os.environ.get("F110_DATA_ROOT", "/mnt/p/Car/NewCar"))
INPUT_ROOTS = [
    Path(os.environ.get("F110_NEW_TRACK_ROOT", str(DATA_ROOT / "NewTrackData"))),
    Path(os.environ.get("F110_SECOND_TRACK_ROOT", str(DATA_ROOT / "secondTrack"))),
]
OUTPUT_ROOT = Path(os.environ.get("F110_OUTPUT_ROOT", str(DATA_ROOT)))
LIDAR_BEAM_COUNTS = (108, 54, 27)


def main() -> None:
    processor.BAG_ROOTS = INPUT_ROOTS
    processor.OUTPUT_LIDAR_BEAMS = LIDAR_BEAM_COUNTS
    processor.configure_output_dirs(OUTPUT_ROOT)
    saved = processor.run_processing()
    print(f"Finished combined run; saved {len(saved)} file(s)", flush=True)


if __name__ == "__main__":
    main()
