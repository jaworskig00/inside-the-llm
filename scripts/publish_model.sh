#!/usr/bin/env bash
#
# Publishes the exported weights as a GitHub Release asset.
#
# The app downloads model.onnx from a release rather than from the repository: an 89 MB
# binary has no business in git history, and GitHub's release CDN answers with
# `access-control-allow-origin: *`, so a Web Worker on github.io can fetch it.
#
#   ./scripts/publish_model.sh [tag] [path/to/model.onnx]
#
# Re-running with the same tag replaces the asset in place, so the URL baked into the app
# (src/worker/client.ts) keeps working.
set -euo pipefail

TAG="${1:-model-v1}"
MODEL="${2:-.devmodel/model.onnx}"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

if [ ! -f "$MODEL" ]; then
  echo "no weights at $MODEL — run: python scripts/export_model.py --out .devmodel" >&2
  exit 1
fi

size=$(du -h "$MODEL" | cut -f1)
echo "publishing $MODEL ($size) to $REPO @ $TAG"

if ! gh release view "$TAG" >/dev/null 2>&1; then
  gh release create "$TAG" \
    --title "DistilGPT-2 ONNX (attentions, hidden states, logit lens)" \
    --notes "DistilGPT-2 exported with \`scripts/export_model.py\` and dynamically quantized to int8.

The graph returns the model's internals as first-class outputs: per-layer/head attention
weights, residual-stream norms, teacher-forced token log-probabilities and a top-k logit
lens read-out per layer. See DECISIONS.md for why these are produced inside the graph.

Consumed by the app at runtime; override the location with \`VITE_MODEL_URL\`."
fi

gh release upload "$TAG" "$MODEL" --clobber
echo
echo "asset URL:"
echo "  https://github.com/$REPO/releases/download/$TAG/$(basename "$MODEL")"
echo
echo "Check CORS before trusting it:"
echo "  curl -sIL -H 'Origin: https://example.github.io' <asset URL> | grep -i access-control"
