#!/usr/bin/env bash
set -e

FPS=30
INPUT_DIR="out/teacher_frames"
OUTPUT_DIR="out/videos/teacher"
OUTPUT="$OUTPUT_DIR/teacher_$(date +%H%M%S).mp4"

mkdir -p "$OUTPUT_DIR"
echo "Creating video from $INPUT_DIR at ${FPS}fps..."

ffmpeg -y \
  -framerate $FPS \
  -pattern_type glob \
  -i "$INPUT_DIR/*.png" \
  -c:v libx264 \
  -preset slow \
  -crf 16 \
  -pix_fmt yuv420p \
  "$OUTPUT"

echo "Done → $OUTPUT"
