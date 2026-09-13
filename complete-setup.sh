#!/bin/zsh
set -euo pipefail

project_dir=${0:A:h}
model_dir="$project_dir/models/DeepSeek-V4.1-Flash"
status_file="$project_dir/runs/setup-status.txt"

mkdir -p "$project_dir/runs"

if pgrep -f '[.]venv/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash' >/dev/null; then
  print "WAITING_FOR_DOWNLOAD $(date -Iseconds)" > "$status_file"
  while pgrep -f '[.]venv/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash' >/dev/null; do
    sleep 30
  done
fi

print "DOWNLOADING_OR_RESUMING $(date -Iseconds)" > "$status_file"
"$project_dir/download-model.sh"

shard_count=$(find "$model_dir" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')
if [[ "$shard_count" != 48 ]]; then
  print "FAILED expected 48 shards, found $shard_count $(date -Iseconds)" > "$status_file"
  exit 1
fi

for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
  if [[ ! -s "$model_dir/$required" ]]; then
    print "FAILED missing $required $(date -Iseconds)" > "$status_file"
    exit 1
  fi
done

print "VERIFYING $(date -Iseconds)" > "$status_file"
"$project_dir/verify-model.sh"

print "SMOKE_TEST $(date -Iseconds)" > "$status_file"
"$project_dir/run-local.sh" "请只回答：本地部署成功" 1

print "COMPLETE $(date -Iseconds)" > "$status_file"
