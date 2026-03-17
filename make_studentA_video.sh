#!/usr/bin/env bash
set -e

FPS=30
INPUT_DIR="out/stageA/frames"
OUTPUT_DIR="out/videos/stageA"
OUTPUT="$OUTPUT_DIR/stageA_$(date +%H%M%S).mp4"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Input directory not found: $INPUT_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Creating video from $INPUT_DIR at ${FPS}fps..."

ffmpeg -y \
  -framerate "$FPS" \
  -pattern_type glob \
  -i "$INPUT_DIR/*.png" \
  -c:v libx264 \
  -preset slow \
  -crf 16 \
  -pix_fmt yuv420p \
  "$OUTPUT"

echo "Done → $OUTPUT"
