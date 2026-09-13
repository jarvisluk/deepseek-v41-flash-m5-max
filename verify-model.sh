#!/bin/zsh
set -euo pipefail

project_dir=${0:A:h}
model_dir="$project_dir/models/DeepSeek-V4.1-Flash"
manifest="$model_dir/.direct-download-manifest.json"
official_checksums="$project_dir/checksums/DeepSeek-V4.1-Flash-dba1be0.sha256"

[[ -s "$manifest" ]] || { print -u2 "缺少直连下载清单"; exit 1; }

while IFS=$'\t' read -r filename expected_size expected_sha; do
  if [[ "$filename" == *.safetensors ]]; then
    expected_sha=$(awk -v name="$filename" '$2 == name {print $1}' "$official_checksums")
    [[ -n "$expected_sha" ]] || { print -u2 "官方清单缺少：$filename"; exit 1; }
  fi
  target="$model_dir/$filename"
  [[ -f "$target" ]] || { print -u2 "缺少：$filename"; exit 1; }
  actual_size=$(stat -f %z "$target")
  [[ "$actual_size" == "$expected_size" ]] || {
    print -u2 "大小不符：$filename"; exit 1
  }
  if [[ -n "$expected_sha" ]]; then
    actual_sha=$(shasum -a 256 "$target" | awk '{print $1}')
    [[ "$actual_sha" == "$expected_sha" ]] || {
      print -u2 "SHA-256 不符：$filename"; exit 1
    }
  fi
  print "通过：$filename"
done < <(jq -r '.siblings[]
  | select((.rfilename | endswith(".safetensors"))
      or (.rfilename | endswith(".json"))
      or .rfilename == "LICENSE"
      or .rfilename == "README.md")
  | [.rfilename, (.size | tostring), (.lfs.sha256 // "")]
  | @tsv' "$manifest")

print "全部文件校验通过。"
