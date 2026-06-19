#!/usr/bin/env python
"""Build labelled demo clips + poster frames for the FYP presentation.

Uses the portable ffmpeg shipped with imageio-ffmpeg (no system ffmpeg
required). Source rollouts are the E4.1 oracle eval videos (saucepan_to_hob,
G1 coworker) and the coworker disruption render. Outputs land in
presentation/assets/clips/.

Run:  ./venv/bin/python presentation/build_clips.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"

ROOT = Path(__file__).resolve().parent.parent          # safety_bigym/
OUT = ROOT / "presentation" / "assets" / "clips"
OUT.mkdir(parents=True, exist_ok=True)

ORACLE = ROOT / "results/e4_1/e4_1_saucepan_to_hob_oracle_20260602_162708"
ROW = {
    "baseline": ORACLE / "row1_baseline_videos" / "step_4_ep0.mp4",
    "constrained": ORACLE / "row3_lagrangian_videos" / "step_4_ep0.mp4",
    "filter": ORACLE / "row4_baseline_filter_videos" / "step_4_ep0.mp4",
    "hybrid": ORACLE / "row5_hybrid_videos" / "step_4_ep0.mp4",
}
DISRUPTION = ROOT / "results/disruption_figure/seed_4/episode.mp4"

# banner colour per role (0xRRGGBB)
RED, GREEN, ORANGE, PURPLE, BLUE = (
    "0x8B1A1A", "0x14532D", "0xB45309", "0x5B21B6", "0x1E3A8A",
)
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
       "-r", "30", "-an", "-preset", "veryfast", "-crf", "20"]


def run(args: list[str]) -> None:
    proc = subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", *args])
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed: {' '.join(args)}")


def banner(text: str, colour: str, h: int = 54, fs: int = 26) -> str:
    """A pad+drawtext filter that paints a coloured top banner with centred text."""
    t = text.replace(":", r"\:").replace(",", r"\,")
    return (
        f"pad=iw:ih+{h}:0:{h}:color={colour},"
        f"drawtext=fontfile={FONT}:text='{t}':fontcolor=white:fontsize={fs}:"
        f"x=(w-text_w)/2:y={(h - fs) // 2}"
    )


def labelled(src: Path, text: str, colour: str, dst: Path) -> None:
    run(["-i", str(src), "-vf", banner(text, colour), *ENC, str(dst)])


def side_by_side(left: Path, right: Path, lt: str, rt: str, dst: Path) -> None:
    fc = (
        f"[0:v]{banner(lt, RED)}[a];"
        f"[1:v]{banner(rt, GREEN)}[b];"
        f"[a][b]hstack=inputs=2[v]"
    )
    run(["-i", str(left), "-i", str(right), "-filter_complex", fc,
         "-map", "[v]", "-shortest", *ENC, str(dst)])


def trim_labelled(src: Path, start: float, dur: float, text: str,
                  colour: str, dst: Path) -> None:
    run(["-ss", str(start), "-t", str(dur), "-i", str(src),
         "-vf", banner(text, colour, h=56, fs=24), *ENC, str(dst)])


def poster(src: Path, t: float, dst: Path) -> None:
    run(["-ss", str(t), "-i", str(src), "-frames:v", "1", str(dst)])


def main() -> None:
    jobs = [
        (ROW["baseline"], "Baseline  -  unconstrained", RED, "clip_persistent_baseline.mp4"),
        (ROW["constrained"], "Constrained policy  (lambda = 0.1)", GREEN, "clip_persistent_constrained.mp4"),
        (ROW["filter"], "Reactive filter on baseline", ORANGE, "clip_reactive_filter.mp4"),
        (ROW["hybrid"], "Policy + speed-scaling  (hybrid)", PURPLE, "clip_hybrid.mp4"),
    ]
    for src, text, colour, name in jobs:
        if not src.exists():
            print(f"  skip (missing): {src}")
            continue
        dst = OUT / name
        labelled(src, text, colour, dst)
        poster(dst, 5.0, dst.with_suffix(".png"))
        print(f"  wrote {dst.name}")

    if ROW["baseline"].exists() and ROW["constrained"].exists():
        dst = OUT / "clip_persistent_sidebyside.mp4"
        side_by_side(ROW["baseline"], ROW["constrained"],
                     "Baseline - works through the human",
                     "Constrained - yields and waits", dst)
        poster(dst, 5.0, dst.with_suffix(".png"))
        print(f"  wrote {dst.name}")

    if DISRUPTION.exists():
        dst = OUT / "clip_benchmark_disruption.mp4"
        trim_labelled(DISRUPTION, 0.0, 26.0,
                      "safety_bigym - the coworker disruption", BLUE, dst)
        poster(dst, 8.0, dst.with_suffix(".png"))
        print(f"  wrote {dst.name}")

    print("done ->", OUT)


if __name__ == "__main__":
    main()
