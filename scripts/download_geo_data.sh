#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
mkdir -p "$RAW_DIR"

fetch_geo() {
  local accession="$1"
  local out_file="$RAW_DIR/${accession}.tar"
  echo "Downloading ${accession} to ${out_file}"
  curl -L "https://www.ncbi.nlm.nih.gov/geo/download/?acc=${accession}&format=file" -o "$out_file"
}

fetch_geo GSE115189
fetch_geo GSE96583

echo "Done. Inspect archives in $RAW_DIR before extraction."
