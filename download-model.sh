#!/bin/zsh
set -euo pipefail

project_dir=${0:A:h}
model_dir="$project_dir/models/DeepSeek-V4.1-Flash"
mirror="https://hf-mirror.com"
repo="deepseek-ai/DeepSeek-V4.1-Flash"
revision="dba1be0a40aa45a94ad051997016db3960a90277"
manifest="$model_dir/.direct-download-manifest.json"
official_checksums="$project_dir/checksums/DeepSeek-V4.1-Flash-dba1be0.sha256"
parallel_downloads=${DIRECT_DOWNLOAD_JOBS:-6}

mkdir -p "$model_dir"
if [[ "$parallel_downloads" != <-> ]] || (( parallel_downloads < 1 || parallel_downloads > 16 )); then
  print -u2 "DIRECT_DOWNLOAD_JOBS 必须是 1 到 16 的整数"
  exit 2
fi

print "获取直连文件清单……"
curl --noproxy '*' -fL --retry 10 --retry-all-errors \
  --connect-timeout 15 --max-time 120 -sS \
  "$mirror/api/models/$repo/revision/$revision?blobs=true" \
  -o "$manifest.tmp"
mv "$manifest.tmp" "$manifest"

download_one() {
  local filename=$1
  local expected_size=$2
  local expected_sha=$3
  local target="$model_dir/$filename"
  local partial="$target.part"
  local actual_size

  mkdir -p "${target:h}"
  if [[ -f "$target" ]]; then
    actual_size=$(stat -f %z "$target")
    if [[ "$actual_size" == "$expected_size" ]]; then
      print "已有：$filename"
      return 0
    fi
    mv "$target" "$partial"
  fi

  print "下载：$filename"
  curl --noproxy '*' -fL --retry 30 --retry-all-errors --retry-delay 3 \
    --connect-timeout 20 --speed-time 120 --speed-limit 1024 \
    -C - -sS \
    "$mirror/$repo/resolve/$revision/$filename" \
    -o "$partial"

  actual_size=$(stat -f %z "$partial")
  if [[ "$actual_size" != "$expected_size" ]]; then
    print -u2 "大小不符：$filename，期望 $expected_size，实际 $actual_size"
    return 1
  fi

  if [[ -n "$expected_sha" ]]; then
    local actual_sha
    actual_sha=$(shasum -a 256 "$partial" | awk '{print $1}')
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      print -u2 "SHA-256 不符：$filename"
      return 1
    fi
  fi

  mv "$partial" "$target"
  print "完成：$filename"
}

pids=()
failed=0
while IFS=$'\t' read -r filename expected_size expected_sha; do
  if [[ "$filename" == *.safetensors ]]; then
    expected_sha=$(awk -v name="$filename" '$2 == name {print $1}' "$official_checksums")
    [[ -n "$expected_sha" ]] || { print -u2 "官方清单缺少：$filename"; exit 1; }
  fi
  download_one "$filename" "$expected_size" "$expected_sha" &
  pids+=($!)
  if (( ${#pids[@]} >= parallel_downloads )); then
    for pid in $pids; do
      wait "$pid" || failed=1
    done
    pids=()
    (( failed == 0 )) || exit 1
  fi
done < <(jq -r '.siblings[]
  | select((.rfilename | endswith(".safetensors"))
      or (.rfilename | endswith(".json"))
      or .rfilename == "LICENSE"
      or .rfilename == "README.md")
  | [.rfilename, (.size | tostring), (.lfs.sha256 // "")]
  | @tsv' "$manifest")

for pid in $pids; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || exit 1

print "直连下载完成。"
