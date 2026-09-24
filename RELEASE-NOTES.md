# v1.0.0 · Radeon RX 6900 XT (gfx1030)

> **2026-09-24 更新**（同 tag 资产已刷新）：`--kvmem-budget` 默认 36864→98304（长上下文 pp 提速）、
> gated_delta_net rows 状态上 ROCm、llama-bench 回归修复。官方 Q8_0 实测 pp512=607.5 t/s、
> pp4096=591.4 t/s、tg128=45.7 t/s、视觉识别 32 s 正确。详见 `docs/PERFORMANCE.md`。

首次发布：让 **Ternary-Bonsai-2-27B-PQ2_0-MTP** 系列在 **单张 16 GB RDNA2 卡**上以
**262144 上下文 + 原生 MTP 投机解码 + 视觉输入**运行。**两个权重包都支持**：

- 官方 `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`（ProCreations）—— 含官方视觉投影器
- **无审查 `Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf`（BoldingBuilds）** —— 本包不自带投影器，但可复用官方 mmproj（两包张量逐项相同）

两者张量集合**逐项相同**（866 张量 · `block_count = 65` · MTP 头在 `blk.64.nextn.*`），
因此同一运行时与同一套参数通用；实测速度差在噪声内（见下）。

## 资产

| 文件 | 用途 |
|---|---|
| `runtime-bin-v1.0.0.zip` | gfx1030/HIP 构建的 `llama-kvmem-server`（含 `chat-templates/` 与 3 个启动脚本）—— 解压到发布根目录 |
| `runtime-tensile-gfx1030-v1.0.0.zip` | gfx1030 的 rocBLAS / hipBLASLt 张量库 —— 解压到发布根目录 |
| `hip-compat-b1233-gfx103x-v1.0.0.zip` | 让 gfx1030 被枚举的 HIP 兼容 DLL —— 解压到发布根目录 |
| `tensor-table-*.csv` / `.txt` | **张量表**：官方包 866 张量 · **无审查包 866 张量** · 投影器 334 张量 |
| `model-hashes.txt` | 三个权重文件的 sha256（校验下载） |
| `SHA256SUMS.txt` | 以上全部资产的 sha256 |

## 快速开始

1. 从 Hugging Face 下载所需 gguf 到 `<release>/model/`（链接与 sha256 见 README §2.1）
2. 解压上面三个 zip 到 `<release>/`
3. 按包选启动脚本：

```bat
scripts\start-bonsai-mtp.bat            :: 官方包 · 262144 ctx
scripts\start-bonsai-abliterated.bat    :: 无审查包（视觉复用官方 mmproj）
```

## 实测（RX 6900 XT 16 GB / ROCm 7.1 / q8_0 KV / KVMem）

| 场景 | 官方 PQ2_0-MTP-Q8_0 | 无审查 Abliterated-PQ2_0-MTP |
|---|---|---|
| 262144 ctx · 147,725 tok prompt | pp 225.5 · gen **41.7 t/s** · 接受率 40.5% | pp 225.5 · gen **41.7 t/s** · 40.5% |
| 65536 ctx · 44,414 tok prompt | pp 256.2 · gen 43.5 t/s | pp 256.6 · gen 40.7 t/s |
| 32768 ctx · 生成 256 tok | **MTP n-max 2：42.5 t/s（51.6%）** · MTP3：41.6（40.9%）· 基线 38.8 | 同口径（该组即在本包上测） |
| 召回 | 32K/64K **6/6** · 147K 6/6 | 32K/64K **6/6** · 147K 5/6 |
| 视觉 | ✅ 1,079 图像 token；识别正确；首帧冷启动 ≈135 s、预热 2.0 s | ✅ 复用官方 mmproj（实测 32 s 冷启动、识别正确） |
| 思考档位 | `low` 自然收尾（678 字符思维链）；`xhigh` 失控（16,596 tok / 5m48s 撞上限） | 同 |

## 已知限制

1. `turbo3`/`turbo4` KV 在 `head_dim = 128` 不可用（本模型 `key_length = 256`，正常）
2. MTP 非严格无损（`temperature=0` 下开关存在字节差异）
3. 思考需设档（默认 `low`）
4. 视觉首帧 ≈2 分钟属预期；**无审查包不自带投影器，但复用官方 mmproj**
5. 仅在 gfx1030 验证
6. 服务默认无鉴权、默认只绑 `127.0.0.1`

## 许可

仓库内容 MIT；KVMem 派生部分 Apache-2.0；**两个权重包均不在本仓库分发**，其许可以上游 HF 页面为准。
