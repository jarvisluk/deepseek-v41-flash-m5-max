# DeepSeek V4.1 Flash Mac SSD Streaming

这是一个面向 Apple Silicon 的实验性本地推理库。它不把完整的 DeepSeek V4.1
Flash 权重装入统一内存，而是从 SSD 按需读取当前层、路由专家和 Engram 行。

本分支在上游 16GB M1 复现运行器的基础上增加：

- 自动检测 Apple 芯片和物理内存；
- M5 Max 48GB 安全档及其他内存档位；
- 可安装的 Python 包和 `dsv41-mac` CLI；
- 强制绕过系统代理的直连分片下载器；
- 文件大小与 SHA-256 校验；
- 不把模型权重、运行日志或本机虚拟环境打进 Git/Python 包。

## 环境

- Apple Silicon Mac；
- macOS 15 或更高版本；
- Python 3.12；
- `curl` 与 `jq`（下载和清单解析）；
- 原始模型需要约 510GB，建议至少预留 600GB；
- 这是文本、贪心解码、研究用途的运行器，不支持视觉、DSpark 或生产服务。

## 安装

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
```

查看自动选择的硬件档位：

```sh
.venv/bin/dsv41-mac-info
```

## 下载与校验

```sh
./download-model.sh
./verify-model.sh
```

下载器使用 `curl --noproxy '*'`，不会读取 macOS/Clash 的 HTTP 代理设置。元数据
入口为第三方 `hf-mirror.com`，大文件由其返回的 CDN 地址提供。权重 SHA-256
固化自 Hugging Face 官方固定 revision，不取信于镜像返回的校验值。需要完全信任
官方入口的使用者应自行替换下载方式。

模型 revision 固定为：

```text
dba1be0a40aa45a94ad051997016db3960a90277
```

## 运行

最简单的方式：

```sh
./run-local.sh "用中文解释纳维-斯托克斯方程" 8
```

或使用安装后的 CLI：

```sh
.venv/bin/dsv41-mac models/DeepSeek-V4.1-Flash \
  --prompt "用中文解释纳维-斯托克斯方程" \
  --max-tokens 8 \
  --profile auto \
  --reuse-allocator \
  --compile-dequant
```

M5 Max 48GB 的初始安全配置为：

| 参数 | 数值 |
|---|---:|
| MLX 内存上限 | 28GiB |
| Dense 权重缓存 | 16GiB |
| MLX 分配缓存 | 512MiB |

这些数值是保守的容量配置，不代表已经证明最快。请用下面的基准脚本在具体机器上
比较上游低内存配置和自动配置。

```sh
./benchmark-m5-max.sh
```

在一台 48GB M5 Max 上的首次 3-token 顺序测试结果：低内存档首 token 为 47.98 秒、
后续为 0.149 token/s；自动档首 token 为 22.68 秒、后续为 0.276 token/s，MLX
峰值分别约为 6.62GiB 和 14.39GiB。由于自动档是第二个运行，可能受 macOS 文件页
缓存影响；该结果用于本机配置验证，不应视为严格的跨机器性能结论。

## 其他机器

内置档位包括 `low-memory`、`apple-32gb`、`apple-48gb`、`m5-max-48gb`、
`apple-64gb` 和 `apple-128gb`。也可以覆盖参数：

```sh
.venv/bin/dsv41-mac models/DeepSeek-V4.1-Flash \
  --profile m5-max-48gb \
  --memory-limit-gib 24 \
  --dense-cache-gib 12 \
  --mlx-cache-mib 512
```

运行器会拒绝占用超过物理内存 80% 的配置，并要求 dense cache 在 MLX 上限内至少
留下 2GiB 临时空间。

## 分享

```sh
uv build
```

可以分享 `dist/` 中的 wheel 或源码包。wheel 适合已经取得模型文件的人直接安装；
完整的下载和校验脚本位于源码包中。模型权重受 DeepSeek 原始许可约束，且体积巨大，
默认不会包含在构建产物中。请同时保留本仓库的 `LICENSE` 与 `NOTICE`。
