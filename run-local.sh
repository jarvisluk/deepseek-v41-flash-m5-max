#!/bin/zsh
set -euo pipefail

project_dir=${0:A:h}
model_dir="$project_dir/models/DeepSeek-V4.1-Flash"
prompt=${1:-"用通俗的中文解释纳维-斯托克斯方程。"}
max_tokens=${2:-8}

if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  print -u2 "Python 环境不存在，请先在项目目录运行：uv venv --python 3.12 .venv"
  exit 1
fi

if [[ ! -f "$model_dir/model.safetensors.index.json" ]]; then
  print -u2 "模型尚未下载完成。可执行 ./download-model.sh 断点续传。"
  exit 1
fi

mkdir -p "$project_dir/runs"
timestamp=$(date +%Y%m%d-%H%M%S)
log_file="$project_dir/runs/run-$timestamp.jsonl"

print "日志：$log_file"
print "提示词：$prompt"
print "最大生成：$max_tokens tokens"

cd "$project_dir"
caffeinate -i .venv/bin/python -u -m deepseek_v41_mlx.native_stream \
  "$model_dir" \
  --prompt "$prompt" \
  --max-tokens "$max_tokens" \
  --profile auto \
  --reuse-allocator \
  --compile-dequant \
  | tee "$log_file"

print "\n模型回答："
.venv/bin/python scripts/extract_answer.py "$log_file"
