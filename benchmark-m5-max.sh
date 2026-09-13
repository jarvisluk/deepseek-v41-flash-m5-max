#!/bin/zsh
set -euo pipefail

project_dir=${0:A:h}
model_dir="$project_dir/models/DeepSeek-V4.1-Flash"
prompt="Explain why SSD streaming works for a sparse mixture-of-experts model."
timestamp=$(date +%Y%m%d-%H%M%S)

[[ -f "$model_dir/model-00048-of-00048.safetensors" ]] || {
  print -u2 "模型尚未下载完成。"
  exit 1
}

mkdir -p "$project_dir/runs"
cd "$project_dir"

print "运行上游 16GB 基线档……"
caffeinate -i .venv/bin/dsv41-mac "$model_dir" \
  --prompt "$prompt" --max-tokens 3 \
  --profile low-memory --dense-cache-gib 6 \
  --reuse-allocator --compile-dequant \
  | tee "runs/benchmark-low-memory-$timestamp.jsonl"

print "运行 M5 Max 48GB 自动档……"
caffeinate -i .venv/bin/dsv41-mac "$model_dir" \
  --prompt "$prompt" --max-tokens 3 \
  --profile auto --reuse-allocator --compile-dequant \
  | tee "runs/benchmark-auto-$timestamp.jsonl"

print "基准完成；比较两个日志中的 first_token_seconds、decode_tokens_per_second 和 read_seconds。"
