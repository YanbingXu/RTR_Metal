#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SHADER_SRC="$PROJECT_ROOT/Sources/RTRMetalEngine/Shaders/MetalRayTracing.metal"
OUTPUT_METALLIB="$PROJECT_ROOT/Sources/RTRMetalEngine/Shaders/default.metallib"
TEMP_AIR="$PROJECT_ROOT/.build/MetalRayTracing.air"
SANDBOX_HOME="$PROJECT_ROOT/.metal_home"

if [[ ! -f "$SHADER_SRC" ]]; then
  echo "Shader source not found: $SHADER_SRC" >&2
  exit 1
fi

mkdir -p "$(dirname "$TEMP_AIR")"
mkdir -p "$SANDBOX_HOME/.cache/clang/ModuleCache"

export HOME="$SANDBOX_HOME"

xcrun metal -std=metal3.1 -c "$SHADER_SRC" -o "$TEMP_AIR"
xcrun metallib "$TEMP_AIR" -o "$OUTPUT_METALLIB"
rm -f "$TEMP_AIR"

echo "Generated $OUTPUT_METALLIB"
