# 本机运行说明

本目录使用原始 DeepSeek V4.1 Flash FP4/FP8 权重，通过 SSD 按需读取专家权重。

## 权重清理

2026-10-11 清理本地 48 个权重分片及 16 个未完成下载文件，合计约 525.45 GB。
模型配置、分词器和校验清单保留在本地；源码、下载和运行脚本保存在 Git 私有仓库。
模型目录、权重和下载临时文件均由 `.gitignore` 排除。再次运行模型前，请先执行下方下载命令。

## 下载

```sh
./download-model.sh
```

下载约 510 GB，可重复执行并断点续传。下载命令使用 `curl --noproxy '*'`
强制绕过系统代理，通过 `hf-mirror.com` 获取元数据，并从其返回的 Hugging Face
Xet CDN 地址直连下载。每个新分片会核对文件大小和 SHA-256。
完成后应有 48 个 `model-*.safetensors` 分片。

当前自动配置任务的状态可查看：

```sh
cat runs/setup-status.txt
tail -f runs/setup-background.log
```

## 短测试

```sh
./run-local.sh "用通俗的中文解释纳维-斯托克斯方程。" 8
```

第二个参数是最大生成 token 数。首次测试建议保持为 1 到 8；该实验运行器速度较慢。
输出日志保存在 `runs/`，结束后脚本会自动提取最终回答。

当前启动配置自动识别硬件。M5 Max 48GB 初始使用 28 GiB MLX 内存上限、16 GiB
dense cache、复用分配器并编译权重解码；实际性能需在完整模型下载后通过基准确认。
