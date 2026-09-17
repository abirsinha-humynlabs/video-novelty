#!/usr/bin/env bash
# Cut the demo clips used by the README walkthrough and eval/pairs.yaml.
#
# Usage:
#   SRC=/path/to/left_rectified.mp4 SRC2=/path/to/front_left_rectified.mp4 \
#     bash scripts/prepare_eval_clips.sh
#
# SRC   should be a long recording of ONE repetitive task in ONE place.
# SRC2  should be a recording of DIFFERENT work in a DIFFERENT place, ideally
#       from the same camera rig -- that is the hard negative, and it is the
#       only clip here that tells you whether your scores mean anything.
#
# Everything lands in data/clips/.
set -euo pipefail

SRC="${SRC:-}"
SRC2="${SRC2:-}"
OUT="${OUT:-data/clips}"
CRF="${CRF:-23}"

if [[ -z "$SRC" ]]; then
  echo "set SRC=/path/to/a/long/video.mp4 (and ideally SRC2=/path/to/a/different/site.mp4)" >&2
  exit 2
fi
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 2; }
mkdir -p "$OUT"

cut () {  # cut <input> <start> <dur> <name>
  echo "  -> $4"
  ffmpeg -hide_banner -loglevel error -y -ss "$2" -i "$1" -t "$3" -an \
         -c:v libx264 -preset veryfast -crf "$CRF" -pix_fmt yuv420p "$OUT/$4"
}

echo "cutting from $SRC"
# Two adjacent minutes of the same work: the "I split one recording in half"
# case. These SHOULD come out near-identical on both axes.
cut "$SRC"  40 60 clipA_040-100.mp4
cut "$SRC" 100 60 clipB_100-160.mp4
# Much later in the same recording: same place, possibly a different phase of
# the work. The interesting middle case.
cut "$SRC" 400 60 clipC_v5_400-460.mp4

if [[ -n "$SRC2" ]]; then
  echo "cutting the hard negative from $SRC2"
  cut "$SRC2" 40 60 clipD_front_040-100.mp4
else
  echo "WARNING: no SRC2 given, so there is no negative pair. Every score you" >&2
  echo "         get from this index will be uncalibrated and unfalsifiable." >&2
fi

echo
echo "done. next:"
echo "  novelty index $OUT --index .novelty --window 30 --hop 15"
echo "  novelty calibrate --index .novelty --labels eval/pairs.yaml"
echo "  novelty report --index .novelty"
