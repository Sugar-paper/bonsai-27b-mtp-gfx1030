# 性能测试结果（实测）

**测试平台**：AMD Radeon RX 6900 XT 16 GB · gfx1030 / RDNA2 · ROCm 7.1 · Windows
**模型**：`Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`（+ 官方 mmproj）
**运行时**：本仓库 `patches/` 构建出的 `llama-kvmem-server`
**固定条件**：`-ngl 999` · `--kv-dtype q8_0` · `np 1` · `temperature 0` ·
`--kvmem-budget 36864 --kvmem-gen-reserve 16384 --kvmem-method retrieval --kvmem-mtp-state snapshots` ·
每次测量前 GPU 降温

> 数字均为**单次实测**给出的中位/持续值；`±1.5%` 属机内run-to-run 噪声。

---

## 1. 长上下文（ctx 262144，MTP n-max 3）

| prompt tokens | prefill (pp) | 持续生成 | MTP 接受率 | 峰值显存 |
|---:|---:|---:|---:|---:|
| 147,725 | 225.5 t/s | 41.7 t/s（1024 tok 连续生成） | 40.5% | ≈11.1 GB |
| 44,414 | 256.6 t/s | 40.7 t/s（1024 tok） | 40.5% | ≈11.3 GB |

- KVMem 全程生效：常驻池 53,248 cells，长 prompt 期间按块 offload 到主机内存（`need_offload` 触发）
- 生成阶段无抖动，说明分层 KV 的取回路径没有落在关键路径上

## 2. 投机解码（MTP）的收益与代价

**ctx 32768 · prompt 1,620 tok · 生成 256 tok · 温度 0**

| 配置 | 生成 | MTP 接受率 | 备注 |
|---|---:|---:|---|
| 基线（`--spec-type` 省略） | 38.8 t/s | — | 对照 |
| MTP n-max 3 + KVMem | 41.6 t/s | 40.9% | 深投机、接受率低 |
| **MTP n-max 2 + KVMem** | **42.5 t/s** | **51.6%** | 本机推荐 |
| MTP n-max 2（不开 KVMem） | 42.7 t/s | 51.6% | KVMem 在中短上下文下**无增益** |

**增益 = +8.4%（n-max 2）**。低于模型作者报告的 +37%，原因见 §3。

**MTP 不是严格无损**：`temperature=0` 下开关 MTP 的输出存在字节级差异。

## 3. 为什么增益比作者低（接受率依赖内容类型）

作者给出的 94.0 t/s 是**五类 prompt 的中位数**，其中 reasoning 类接受率 **0.833**；
本机这一轮用的是技术说明类文本，接受率 0.4–0.5。同一份权重、同一套内核，**内容类型决定收益**。

**接受率随生成长度衰减**（实测）：

| 场景 | 生成长度 | 接受率 |
|---|---:|---:|
| 短回答（召回式） | 49 tok | **94.9%** |
| 短回答，n-max 2 | 49 tok | **97.1%** |
| 持续生成 | 256 tok | 40.9%（n-max 3） |
| 持续生成 | 256 tok | 51.6%（n-max 2） |

⇒ 引用 MTP 速度**必须带生成长度**，否则短突发值会被高估。

## 4. 视觉（`--mmproj` + `--image-min-tokens 1024`）

| 指标 | 实测 |
|---|---|
| 图像 token 数 | 1,079（不加 `--image-min-tokens` 时 315） |
| 识别正确性 | 形状（圆/方/三角）、颜色、图内文字全部正确 |
| **首帧（冷）** | **≈135 s** |
| 预热后 | 2.0 s |
| 文本任务影响 | 无（投影器在 CPU 侧，文本生成不受影响） |

模型自带加载器在一开始就警告：Qwen-VL 类投影器建议 ≥1024 图像 token 以保证接地精度 ——
因此默认打开 `--image-min-tokens 1024`（可用 `IMGMINTOK=0` 关掉换取 3 倍便宜的图像预填充）。

## 5. 思考档位与输出长度

| `reasoning_effort` | 输出 token | 思维链长度 | 是否自然结束 |
|---|---:|---:|---|
| `xhigh`（模板默认） | 撞上限 | 5,898 字符 | ❌（真实使用中出现过 16,596 tok / 5m48s 撞上限） |
| `medium` | 撞上限 | 3,158 字符 | ❌ |
| **`low`（本方案默认）** | 2,084 | 678 字符 | ✅ |

另测：`reasoning_budget_tokens=512/256` 能**截断思考**（思维链 5,898 → 893/598 字符），
但被强制打断后正文会明显变长、仍可能撞上限 —— 因此本方案用 `effort=low` 而不是预算截断。

## 6. 加载与资源

| 项 | 实测 |
|---|---|
| 服务就绪（纯文本） | 8–10 s |
| 服务就绪（含视觉投影器） | 10 s |
| 权重常驻显存 | ≈7.7 GB |
| KVMem 常驻池 | 53,248 cells（budget 36864 MB） |
| 生成期总显存 | 11.1–11.3 GB（长上下文） |

## 7. 复现方式

```bash
# 服务端（脚本已内置全部环境）
scripts/start-bonsai-mtp.bat                    # 262144 ctx · MTP 3 · KVMem

# 长上下文测量：构造长 prompt 后取服务端日志里的两行
#   prompt eval time =  ... ms / <N> tokens ( <pp> tokens per second)
#   eval time        =  ... ms / <M> tokens ( <gen> tokens per second)
# MTP 接受率：日志中的 KVMEM_TRACE spec_stats n_gen=... n_accept=... accept_pct=...

# 短上下文对照（本仓库不含 bench harness，可自行用 llama-bench 或 openai 客户端循环）
```

**测量注意**：

1. 先让 GPU 降温（实测高温显著掉速）；
2. 首次图像请求会额外花掉 ≈2 分钟，测速时不要算进去；
3. 同一 prompt/生成长度/温度才可比；跨内容类型的 t/s 不可直接比。
