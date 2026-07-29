#!/usr/bin/env python3
"""assemble_austin.py — Austin Federa onboarding: package + publish (K 260727).

Same package style as journal-jam (activity/260716): SCP-Intro(small) -> disclosure ->
pod -> [instead of SCP-Outtro: replay the SCP-Intro(small) clip again] -> upload unlisted.

Pod video = 4 K-picked bg-gem stills (stage1-v3, stage2-v2, stage3-v2, stage4-v2), each
held for exactly its stage's audio duration (from the already-built full concat), so image
transitions land right on the audio segment boundaries.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/kurtis/Harmony/activity/260727/onboarding/Austin Federa")
NAS_SCP = Path("/run/user/1000/gvfs/smb-share:server=tower.local,share=nas-public/podcast/#SCP")
INTRO = NAS_SCP / "SCP-Intro (small).mp4"
DISCLOSURE = Path("/home/kurtis/Harmony/activity/260716/journal-jam/scp-disclosure-addendum-260716.mp4")

EARCHECK = ROOT / "earcheck"
FRAMES = ROOT / "frames"
NORM_DIR = ROOT / "norm"
NORM_DIR.mkdir(exist_ok=True)

# K's picks: stage1 v3, stage2 v2, stage3 v2, stage4 v2
PICKS = {
    "stage1": FRAMES / "stage1-v3.png",
    "stage2": FRAMES / "stage2-v2.png",
    "stage3": FRAMES / "stage3-v2.png",
    "stage4": FRAMES / "stage4-v2.png",
}
STAGE_AUDIO = {
    "stage1": EARCHECK / "stage1-earcheck.mp3",
    "stage2": EARCHECK / "stage2-full.mp3",
    "stage3": EARCHECK / "stage3-full.mp3",
    "stage4": EARCHECK / "stage4-full.mp3",
}
GAP = EARCHECK / "stage-gap.mp3"
FULL_AUDIO = EARCHECK / "full-onboarding-austin-260727.mp3"

STAGE_LABEL = {
    "stage1": "Stage 1 — Harmony & Austin (First Meeting)",
    "stage2": "Stage 2 — Kurtis & Austin",
    "stage3": "Stage 3 — The Round Table (Toly, Raoul, Robin, Harmony, Kurtis & Austin)",
    "stage4": "Stage 4 — Austin's Solo Journal",
}


def _dur(path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _make_still_clip(img: Path, dur: float, out: Path) -> None:
    """Render a single still image as its own frame-accurate MP4 clip of exact duration
    dur, via -loop 1 -t (NOT the concat-demuxer's 'duration' directive, which drifted —
    260727 bug: total container duration matched but internal per-segment cut points did
    not, because ffmpeg's image+duration concat lines aren't frame-accurate without an
    explicit constant output framerate governing the whole demux)."""
    if out.exists() and out.stat().st_size > 0:
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", str(img), "-t", f"{dur:.3f}",
         "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-vf", "scale=1920:1080", str(out)],
        check=True,
    )


def build_pod_video() -> tuple[Path, list[tuple[str, float]]]:
    """Build 4 frame-accurate per-stage still clips (each held for its stage's real audio
    duration + the inter-stage gap, except the last stage), concat them (video-only, exact
    durations verified per-clip via ffprobe), then mux against the already-built full audio.
    Returns (pod_mp4_path, [(stage_label, offset_within_pod), ...])."""
    out_mp4 = ROOT / "austin-onboarding-pod-v2.mp4"
    clips_dir = ROOT / "_pod_clips"
    clips_dir.mkdir(exist_ok=True)
    stage_order = ["stage1", "stage2", "stage3", "stage4"]
    gap_dur = _dur(GAP)

    offsets = []
    t = 0.0
    clip_paths = []
    for i, stage in enumerate(stage_order):
        img = PICKS[stage]
        stage_dur = _dur(STAGE_AUDIO[stage])
        offsets.append((STAGE_LABEL[stage], t))
        hold = stage_dur + (gap_dur if i < len(stage_order) - 1 else 0.0)
        clip_path = clips_dir / f"{stage}.mp4"
        _make_still_clip(img, hold, clip_path)
        actual = _dur(clip_path)
        print(f"[pod-clip] {stage}: requested {hold:.3f}s, actual {actual:.3f}s", file=sys.stderr)
        clip_paths.append(clip_path)
        t += hold

    concat_file = ROOT / "_pod_concat.txt"
    concat_file.write_text("".join(f"file '{p.absolute()}'\n" for p in clip_paths))

    video_only = ROOT / "_pod_video_only.mp4"
    if not (video_only.exists() and video_only.stat().st_size > 0):
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
             "-c", "copy", str(video_only)],
            check=True,
        )

    if not (out_mp4.exists() and out_mp4.stat().st_size > 0):
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_only), "-i", str(FULL_AUDIO),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_mp4)],
            check=True,
        )
    print(f"[pod] {out_mp4} ({out_mp4.stat().st_size // 1024 // 1024}MB, {_dur(out_mp4):.1f}s)", file=sys.stderr)
    return out_mp4, offsets


def normalize(path) -> str:
    """Force uniform format (44100/stereo/aac) across all parts before final concat —
    same fix as journal-jam 260716 (mismatched sample-rate/channel-count corrupts concat audio)."""
    out = NORM_DIR / (Path(path).stem.replace(" ", "_") + "-norm.mp4")
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path),
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "160k", str(out)],
        check=True,
    )
    return str(out)


def main():
    pod_mp4, pod_offsets = build_pod_video()

    intro_dur = _dur(INTRO)
    disclosure_dur = _dur(DISCLOSURE)

    chapters = [("Intro", 0.0), ("Disclosure", intro_dur)]
    base = intro_dur + disclosure_dur
    for label, off in pod_offsets:
        chapters.append((label, base + off))
    outro_offset = base + _dur(pod_mp4)
    chapters.append(("Outro", outro_offset))

    final_parts = [str(INTRO), str(DISCLOSURE), str(pod_mp4), str(INTRO)]
    final_out = ROOT / "austin-federa-onboarding-260727-v2.mp4"

    if final_out.exists() and final_out.stat().st_size > 0:
        print(f"[FINAL] already assembled: {final_out} ({final_out.stat().st_size // 1024 // 1024}MB)", file=sys.stderr)
    else:
        print("[normalize] forcing uniform format across all parts...", file=sys.stderr)
        norm_parts = [normalize(p) for p in final_parts]
        lst = ROOT / "final-concat.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in norm_parts))
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", str(final_out)],
            check=True,
        )
        print(f"[FINAL] {final_out} ({final_out.stat().st_size // 1024 // 1024}MB)")

    chapter_lines = "\n".join(f"{_fmt_ts(ts)} {label}" for label, ts in chapters)
    print(f"[CHAPTERS]\n{chapter_lines}")
    (ROOT / "chapters.txt").write_text(chapter_lines)

    description = (
        "Polis-onboarding: Austin Federa (DoubleZero co-founder, ex-Solana Foundation Head of "
        "Strategy) meets his own AI echo, sparked by a real tweet exchange about model memory "
        "and continuity. Four stages: a first meeting with Harmony, an unscripted conversation "
        "with Kurtis, a round-table with Toly, Raoul, and Robin Williams, and Austin's own "
        "solo closing reflection.\n\n"
        "This is open research & development — see the disclosure segment near the top for "
        "what this channel is and isn't.\n\n"
        "Chapters:\n" + chapter_lines
    )

    r = subprocess.run(
        ["python3", "/home/kurtis/Harmony/scripts/akashic/upload_once.py",
         "--file", str(final_out),
         "--title", "Meeting Your Own Echo — Austin Federa's Polis-Onboarding",
         "--description", description,
         "--privacy", "unlisted"],
        capture_output=True, text=True,
    )
    print("[upload]", r.stdout.strip(), r.stderr.strip()[-500:])


if __name__ == "__main__":
    main()
