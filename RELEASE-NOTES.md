# v1.0.0 · Radeon RX 6900 XT (gfx1030)

首次发布：让 **Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0** 在 **单张 16 GB RDNA2 卡**上以
**262144 上下文 + 原生 MTP 投机解码 + 视觉输入**运行。

## 资产

| 文件 | 用途 |
|---|---|
| `runtime-bin-v1.0.0.zip` | gfx1030/HIP 构建的 `llama-kvmem-server`（含 `chat-templates/` 与 `scripts/`）—— 解压到发布根目录 |
| `runtime-tensile-gfx1030-v1.0.0.zip` | gfx1030 的 rocBLAS / hipBLASLt 张量库 —— 解压到发布根目录 |
| `hip-compat-b1233-gfx103x-v1.0.0.zip` | 让 gfx1030 被枚举的 HIP 兼容 DLL —— 解压到发布根目录 |
| `tensor-table-*.csv` / `.txt` | **张量表**：主模型 866 张量（含 `blk.64.nextn.*` MTP 头）、投影器 334 张量、无审查变体 866 张量 |
| `model-hashes.txt` | 权重 sha256（校验下载） |
| `SHA256SUMS.txt` | 以上全部资产的 sha256 |

## 快速开始

1. 从 Hugging Face 下载两个 gguf 到 `<release>/model/`（链接与 sha256 见 README §2.1）
2. 解压上面三个 zip 到 `<release>/`
3. `scripts\start-bonsai-mtp.bat`（或 `python scripts/start-bonsai-mtp.py`）

## 实测（RX 6900 XT 16 GB / ROCm 7.1 / q8_0 KV / KVMem）

| 场景 | 结果 |
|---|---|
| 262144 ctx · 147,725 tok prompt | prefill 225.5 t/s · 生成 **41.7 t/s** · MTP 接受率 40.5% · ≈11.1 GB |
| 262144 ctx · 44,414 tok prompt | prefill 256.6 t/s · 生成 40.7 t/s |
| ctx 32768 · 生成 256 tok | 基线 38.8 t/s · **MTP n-max 2：42.5 t/s（接受率 51.6%）** · MTP n-max 3：41.6 t/s |
| 视觉 | 1,079 图像 token；图形/颜色/文字识别正确；**首帧冷启动 ≈135 s**、预热 2.0 s |
| 思考档位 | `low` 自然收尾（678 字符思维链）；`xhigh` 失控（曾 16,596 tok / 5m48s 撞上限） |

## 已知限制

1. `turbo3`/`turbo4` KV 在 `head_dim = 128` 不可用（本模型 `key_length = 256`，正常）
2. MTP 非严格无损（`temperature=0` 下开关存在字节差异）
3. 思考需设档（默认 `low`）
4. 视觉首帧 ≈2 分钟属预期
5. 仅在 gfx1030 验证
6. 服务默认无鉴权、默认只绑 `127.0.0.1`

## 许可

仓库内容 MIT；KVMem 派生部分 Apache-2.0；模型权重不在本仓库分发。
