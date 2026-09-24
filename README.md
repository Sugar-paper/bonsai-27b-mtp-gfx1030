# 27B 塞进 16GB 显卡 · 256K 上下文 · Qwen3.8 系无审查三值模型 · RX 6900 XT 实测

> **Ternary-Bonsai-2-27B**（Qwen3.8 系架构三值量化，PQ2_0/Q8_0，2.13 bpw）在 **16 GB 的 AMD RX 6900 XT
> （gfx1030 / RDNA2）** 上以 **262144 上下文 + MTP 投机解码 + 视觉输入** 运行，官方与**无审查**两个版本
> 均实测通过。融合源码补丁 + 预编译运行时 + 张量表 + 全套启动脚本。
>
> A **fusion** (patch series + prebuilt runtime) that runs a **Qwen3.8-class ternary 27B** model
> (uncensored + vision) at **256K context** on a single 16 GB RDNA2 card. Both weight packs verified.

> **2026-09-24 更新**：长上下文 pp 提速（`--kvmem-budget` 默认 36864 → 98304，52K prompt 顶端边际
> pp **+37%**）、gated_delta_net rows 状态上 ROCm、llama-bench 回归修复；官方 Q8_0 实测 pp512=607.5 t/s、
> 视觉识别 32 s 正确。详见 [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) §0/§0.1/§4。

---

## 目录

| 内容 | 位置 |
|---|---|
| 快速开始 | 本文 §2 |
| 前置条件与硬件适配 | 本文 §3 · [`docs/HARDWARE-GFX1030.md`](docs/HARDWARE-GFX1030.md) |
| 融合原理与流程 | [`docs/FUSION.md`](docs/FUSION.md) |
| 性能测试结果 | [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) |
| Release 资产（含张量表） | [`docs/PACKAGING.md`](docs/PACKAGING.md) |
| 上游与本仓库的关系 | 本文 §1 · [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) |
| 融合补丁（24 个） | [`patches/`](patches/) · [`patches/SERIES.md`](patches/SERIES.md) |

---

## 1. 这是什么

一个三层上游 + 本地融合的工程：

```
ggml-org/llama.cpp                       ← 推理引擎（MIT）
   └── PrismML-Eng/llama.cpp @ prism     ← 三值权重类型 PQ2_0 / PTQ1_0 + dflash/MTP（MIT）
          └── 本仓库的 patches/           ← 融合层（本仓库新增，MIT）
                 ├── iamwavecut/llama-cpp-turboquant  ← turbo3/turbo4 KV cache 类型（MIT）
                 ├── kvmem/kvmem-llama.cpp            ← 分层 KV 内存适配器（Apache-2.0）
                 └── lemonade-sdk/llamacpp-rocm b1233  ← gfx103X HIP 兼容运行时（MIT）
```

**为什么需要融合**：三值权重类型只存在于 PrismML 的 fork，turbo KV 类型只存在于另一个 fork，
而这张卡（gfx1030）既不被系统 ROCm 7.1 枚举、16 GB 显存也装不下 256K 的 KV cache。
三个问题各自有解，但必须合成一棵树才能同时成立 —— 这就是 `patches/` 里的 24 个提交。

| 上游 | 链接 | 用途 |
|---|---|---|
| llama.cpp | https://github.com/ggml-org/llama.cpp | 引擎本体 |
| PrismML fork（`prism` 分支） | https://github.com/PrismML-Eng/llama.cpp | 三值权重类型、dflash/MTP |
| TurboQuant 融合参考 | https://github.com/iamwavecut/llama-cpp-turboquant | turbo3/turbo4 KV 类型的来源 |
| KVMem | https://github.com/kvmem/kvmem-llama.cpp | 分层 KV（256K 的关键） |
| gfx103X HIP 兼容运行时 | https://github.com/lemonade-sdk/llamacpp-rocm | 让 ROCm 在 gfx1030 上枚举设备 |

**基准提交**：本补丁序列切在 `PrismML-Eng/llama.cpp` 的 `9a9394a` 上（详见 `patches/SERIES.md`）。

---

## 2. 快速开始

### 2.1 下载模型（本仓库不包含权重）

| 文件 | 来源 | 大小 | sha256 |
|---|---|---:|---|
| `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` | https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP | 7,657,489,728 B | `3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4` |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP | 629,246,976 B | `6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903` |
| **`Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf`**（**无审查版**） | https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF | 7,657,489,696 B | `7aa43b9a42f5bebc170d54f45657a7ccad841dd1f5b94434b107945740b02e86` |
| （可选）原始三值权重/投影器 | https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf | — | 见该仓库 |

> 权重版权归上游发布者所有，本仓库**只发布融合源码、预编译运行时与张量表**，不再分发权重。
> 国内网络可把 `huggingface.co` 换成 `hf-mirror.com`。

### 2.1.1 两个版本的区别（同时提供、均已实测可运行）

| | `PQ2_0-MTP-Q8_0`（ProCreations） | `Abliterated-PQ2_0-MTP`（BoldingBuilds） |
|---|---|---|
| 定位 | 官方 MTP 包 | **无审查 / 拒答已移除（abliterated）** |
| 张量结构 | 866 张量 · `block_count = 65` · MTP 头 `blk.64.nextxn.*` | **完全相同**（866 张量 · 65 blocks） |
| 视觉投影器 | ✅ 官方 mmproj | ✅ **可复用官方 mmproj**（两包张量逐项相同；本包不自带） |
| 启动脚本 | `scripts\start-bonsai-mtp.bat` · `-vision.bat` | `scripts\start-bonsai-abliterated.bat` |
| 别名 | `bonsai-27b-mtp` / `bonsai-27b-mtp-vision` | `bonsai-27b-abliterated-mtp` |
| 实测（262144 ctx / 147,725 tok prompt） | pp 225.5 · gen 41.7 t/s · 接受率 40.5% | pp **225.5** · gen **41.7 t/s** · 接受率 **40.5%** |
| 实测（65536 ctx / 44,414 tok prompt） | pp 256.2 · gen 43.5 t/s | pp **256.6** · gen 40.7 t/s |
| 召回（32K/64K 上下文埋码） | 6/6 | **6/6** |
| 召回（147K 上下文） | 6/6 | 5/6（147K 下丢 1 个埋点） |

> **两个包的张量集合逐项相同**（866 张量、零 shape/type 差异，唯一 KV 元数据差异是 `general.name`），
> 所以同一套运行时/参数对两者通用；差异只在**权重本身**（消融 vs 官方）。视觉投影器共用官方 mmproj（两包张量逐项相同，可复用）。
> 两个版本的张量表都在本仓库 Release 里（`tensor-table-*.csv`），可自行 diff 验证。

用 Python 启动器跑无审查包：

```bash
python scripts/start-bonsai-mtp.py --model model/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf \
                                   --alias bonsai-27b-abliterated-mtp
```

### 2.2 取运行时

从本仓库的 **Releases** 下载并解压到同一目录：

```
<release>/
├─ runtime/            ← runtime-bin zip（llama-kvmem-server.exe + dlls）
├─ runtime/rocblas|hipblaslt  ← tensile zip（张量库，必需）
├─ hip-compat/         ← hip-compat zip（b1233 gfx103X HIP DLL）
├─ chat-templates/     ← 本仓库自带
├─ scripts/            ← 本仓库自带
└─ model/              ← 把 §2.1 的两个 gguf 放这里
```

### 2.3 启动

```bat
:: Windows，命令行里执行
cd <release>\scripts
start-bonsai-mtp.bat                :: 262144 ctx + MTP + KVMem，思考默认 on/low
start-bonsai-mtp-vision.bat         :: 同上 + 视觉投影器
start-bonsai-abliterated.bat        :: 无审查包（纯文本）
start-bonsai-mtp.bat --check        :: 只校验前置条件与最终命令，不启动

:: 无审查包开视觉：复用官方 mmproj（两包张量逐项相同）
set MMPROJ=<release>\model\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
start-bonsai-abliterated.bat
```

或跨平台用 Python：

```bash
python scripts/start-bonsai-mtp.py --check          # 自检
python scripts/start-bonsai-mtp.py                  # 默认 256K, MTP n-max 3
python scripts/start-bonsai-mtp.py --vision         # 开启视觉
python scripts/start-bonsai-mtp.py --ctx 65536 --effort xhigh
```

服务是 **OpenAI 兼容** 的：

```bash
curl http://127.0.0.1:11234/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "bonsai-27b-mtp",
  "messages": [{"role":"user","content":"用一句话说明 KV cache 量化的作用"}],
  "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": true}
}'
```

> 思考档位：本模型模板只接受 `reasoning_effort ∈ {xhigh, medium, low}`。
> OpenAI 生态常见的 `high` 会被模板拒绝（HTTP 400），因此运行时自带兼容模板
> `chat-templates/abl-27b-reasoning-compat.jinja`（`high/max → xhigh`，`minimal/none → low`）。
> 思维链计入输出 token，`max_tokens` 要给够。

---

## 3. 前置条件与硬件适配（要点）

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | **AMD Radeon RX 6900 XT（gfx1030 / RDNA2 / 16 GB）** | 本方案只在这一张卡上验证；其它 gfx 需自行重编译 |
| 驱动 | AMD Adrenalin（Windows）/ amdgpu（Linux），支持 ROCm 7.1 | — |
| ROCm | **7.1**，含 `clang/clang++`（HIP 编译链） | 运行时需要 `%ROCM_PATH%\bin` 在 PATH 中 |
| HIP 兼容层 | `hip-compat/`（b1233 gfx103X）**必须排在最前** | 系统 HIP 运行时在这张卡上**不枚举设备**；不带它服务器会在加载前退出 |
| 张量库 | `runtime/rocblas/library` + `runtime/hipblaslt/library` | 必须与 gfx1030 匹配；缺失时报错发生在 `rocblas_initialize`，不会给 HTTP 状态码 |
| 显存 | 16 GB | 262144 ctx 的 KV cache 约 17 GB ⇒ **必须开 KVMem**；纯 KV 方案在 131072 就会 OOM |
| 内存 | ≥ 32 GB（KVMem 会把 KV 分层缓存到主机内存） | 实测常驻约 3–8 GB |
| Python（可选） | 3.10+ | 仅 `scripts/start-bonsai-mtp.py` 需要，零第三方依赖 |

编译细节、VRAM 预算推导、常见报错对照表见 [`docs/HARDWARE-GFX1030.md`](docs/HARDWARE-GFX1030.md)。

---

## 4. 性能测试结果（实测，本机单卡）

环境：RX 6900 XT 16 GB / gfx1030 / ROCm 7.1 / Windows ·
`--kv-dtype q8_0` · `--kvmem-budget 98304 --kvmem-gen-reserve 16384 --kvmem-method retrieval`（98304 为 2026-09-24 起的新默认；早期 36864 数据见 PERFORMANCE §0） ·
`--spec-type draft-mtp` · temperature 0 · 完整数值与原始日志见 [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)。

### 4.1 长上下文（ctx 262144）

| prompt 长度 | pp | 生成 | MTP 接受率 | 显存 |
|---:|---:|---:|---:|---:|
| **199,950 tok（≈256K 上下文）** | **242.97 t/s**（budget 98304） | 33.9 t/s（70 tok，思考计入） | — | ~11.5 GB |
| 147,725 tok | 225.5 t/s（旧 budget 36864） | 41.7 t/s（1024 tok 持续） | 40.5% | ~11.1 GB |
| 44,414 tok | 256.6 t/s（旧 budget） | 40.7 t/s | 40.5% | ~11.3 GB |

> 2026-09-24 起默认 `--kvmem-budget 98304`：**256K 上下文已实测可用**（~200K token 长 prompt
> 全量 prefill + 生成，答案正确、无 OOM），且 pp 比旧 budget 的 147K（225.5）还高 **+7.7%**。

### 4.2 MTP 投机解码的收益（ctx 32768，1620 tok prompt，生成 256 tok）

| 配置 | 生成 | MTP 接受率 |
|---|---:|---:|
| 基线（不开 MTP） | 38.8 t/s | — |
| MTP n-max 3 + KVMem | 41.6 t/s | 40.9% |
| **MTP n-max 2 + KVMem** | **42.5 t/s** | **51.6%** |

**+8.4%**（n-max 2 对基线）。这个数字**低于**模型作者报告的 +37%，原因是收益高度依赖内容类型：
作者的中位数取自 reasoning 类 prompt（接受率 0.83），而本轮是技术说明类文本（接受率 0.4–0.5）。
**接受率随生成长度衰减**（同一份权重实测）：49 token 短回答 **94.9%** → 256 token 持续生成 **40.9–51.6%**。

### 4.3 视觉（`--mmproj` + `--image-min-tokens 1024`）

| 指标 | 实测 |
|---|---|
| 图像 token 数 | 1,079（不加该开关时为 315） |
| 识别正确性 | 图形/颜色/图内文字全部正确读出 |
| **首帧延迟（冷）** | **≈135 s** —— 投影器在 CPU 侧预热 + kernel 编译，**属正常现象** |
| 预热后延迟 | **2.0 s** |

### 4.4 思考档位（模板默认 `xhigh` 会失控）

| 档位 | 输出 token | 思维链长度 | 结果 |
|---|---:|---:|---|
| `xhigh` | 撞上限 3000 | 5,898 字符 | ❌ 不收尾（真实场景出现过 16,596 tok / 5m48s 撞上限） |
| `medium` | 撞上限 3000 | 3,158 字符 | ❌ |
| **`low`（默认）** | **2,084** | **678 字符** | ✅ 自然结束 |

### 4.5 加载与运维

| 项 | 实测 |
|---|---|
| 服务就绪（纯文本） | 8–10 s |
| 服务就绪（带视觉投影器） | 10 s |
| KVMem 常驻池 | 53,248 cells（budget 36864 MB） |
| 内存占用 | 模型权重常驻显存约 7.7 GB + 分层 KV |

---

## 5. 已知限制（请先读）

1. **turbo3/turbo4 KV 在 `head_dim = 128` 的模型上不可用**（本机实测 PPL 异常）；
   本机两个模型都是 `key_length = 256`（qwen35 架构），在该几何下 1–2% PPL 代价、正常可用。
2. **MTP 不是严格无损**：`temperature=0` 下开关 MTP 会产生字节级差异。
3. **思考需自行设档**：默认 `low`；`xhigh` 会长时间思考直到撞 `max_tokens`。
4. **视觉首帧慢**（≈135 s）为预期行为。
5. 本方案固定 `-ngl 999`（全部层上卡）；长上下文下**不要降 `-ngl`**，KV 填满后性能会塌。
6. 仅验证 `gfx1030`。其它 RDNA/RDNA2 卡需按 `docs/HARDWARE-GFX1030.md` 重新编译与重测。
7. 服务默认**无鉴权**（`HOSTBIND` 默认 `127.0.0.1`；改 `0.0.0.0` 前请自行加访问控制）。

---

## 6. Release 资产

| 资产 | 内容 |
|---|---|
| `runtime-bin-<ver>.zip` | `llama-kvmem-server.exe` + `llama-kvmem-cli.exe` + llama/ggml DLL（gfx1030 构建） |
| `runtime-tensile-gfx1030-<ver>.zip` | `rocblas` / `hipblaslt` 的 gfx1030 张量库（必需；来自 ROCm 发行包） |
| `hip-compat-b1233-gfx103x-<ver>.zip` | 7 个 HIP 兼容 DLL（让 gfx1030 被枚举） |
| **`tensor-table-*.csv` / `.txt`** | **张量表**：本模型 866 张量（含 `blk.64.nextn.*` MTP 头）、投影器 334 张量、无审查版 866 张量 |
| `model-hashes.txt` | 权重 sha256（下载校验用） |
| `SHA256SUMS.txt` | 全部资产的 sha256 |

张量表字段与用法见 [`docs/PACKAGING.md`](docs/PACKAGING.md)。

---

## 7. 许可证与隐私

- 本仓库新增的融合补丁与脚本：**MIT**（见 [`LICENSE`](LICENSE)）
- 上游组件：llama.cpp / PrismML fork / TurboQuant fork / HIP 兼容运行时 = **MIT**；
  KVMem 适配层 = **Apache-2.0**（保留其署名与 NOTICE）—— 汇总见 [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)
- **模型权重不在本仓库分发**，其许可以上游 Hugging Face 仓库页面为准
- 本仓库**不包含**任何作者机器的路径、IP、主机名、邮箱等个人信息
