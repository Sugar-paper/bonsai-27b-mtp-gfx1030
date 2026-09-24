# 融合原理与流程（FUSION）

本文件说明这套融合**做了什么、为什么必须这么做**，以及如何从上游复现同一棵树。

---

## 1. 问题：三个独立能力，一棵树才能同时成立

| 需求 | 只在哪里 | 缺了会怎样 |
|---|---|---|
| 三值权重类型 `PQ2_0 = 142` / `PTQ1_0 = 143` | PrismML 的 `prism` fork | 模型直接加载失败：`tensor 'output.weight' has invalid ggml type 142` |
| `turbo3_0 = 43` / `turbo4_0 = 44` KV cache 类型 | TurboQuant fork（以及本融合层） | 只能用 f16/q8_0 KV，长上下文显存爆炸 |
| 分层 KV 内存（KVMem） | KVMem 项目（Apache-2.0） | 262144 ctx 需要 ≈17 GB KV，16 GB 卡**装不下** |
| gfx1030 设备枚举 | gfx103X HIP 兼容运行时 | 系统 ROCm 不枚举这张卡，任何 llama 二进制都在加载前退出 |

三个能力分属三棵不同的树。融合层 = 把后三者移植到第一棵树上，并按 gfx1030 的几何做适配。

---

## 2. 融合层做了什么（`patches/` 的 24 个提交）

### 2.1 TurboQuant KV 类型移植与可用化

- 移植 `turbo3_0` / `turbo4_0` 的**块布局、WHT 变换内核、FA vec 实例、CMake 登记**；
- 补上让类型**真正可用**的集成层：CLI 可选项、FA dispatch、convert 表、set-rows、图重排、CPU 路径、
  `quantize_chunk`、`ggml_turbo_wht` 声明、llama-bench 名称、以及测试；
- `gguf-py/gguf/constants.py` 里补齐 `TURBO3_0 = 43` / `TURBO4_0 = 44`（此前只存在于 `ggml.h`）。

> **注意**：turbo 类型的成本是真实的 —— 端到端相对 f16 有 4–10% 的吞吐代价，换来的是长上下文
> 显存开销。这是选择，不是免费午餐。

### 2.2 修掉一个真实的 KV 缺陷（本融合层引入并自行修复）

移植过程中 `ggml-cuda/dequantize.cuh :: dequantize_q8_0()` 丢了一行 `v.y *= d;`，
导致 q8_0 行内**奇数元素**用原始 int8 幅值而不是块缩放值，任何在 GPU 上反量化 q8_0 的路径
都产出半错数据。表现极具误导性：只有 batch ≥ 3 的 FA 用例、以及 KV 长度非 64 倍数的用例失败。

- 定位：`test-backend-ops -o FLASH_ATTN_EXT` 2629/2936 → 修复后 2936/2936（`FA_ALL_QUANTS=OFF`）、
  3949/3949（`=ON`）
- 交叉验证：`llama-perplexity` 上 q8_0×q8_0 从 **92.26 → 6.94**
- 结论：此前把问题归咎于编译选项、再归咎于上游 fork，**两次结论都是错的**，已在补丁信息中更正

### 2.3 KVMem 分层 KV 融合

- 把 KVMem 侧的分层 KV 适配器（`src/adapter/llama-memory-kvmem*.cpp`、`llama-kvmem-stagein.cu`、
  `tools/kvmem/*`，Apache-2.0）接入本树，并补 HIP/ROCm 兼容层与 CMake 接线；
- 修正 `GGML_OP_COUNT`（103，不是 102）等接线细节；
- **关键结论：必须用 `llama-kvmem-server`，不能用 `llama-server`**。
  适配器以 `g_kvmem_params.enabled` 为门控，而它只由 `llama_kvmem_set_params()` 写入，
  llama.cpp 没有任何内置工具调用它 —— `--kvmem*` 参数是由 `tools/kvmem/llama-kvmem-server.cpp`
  里一个手写的解析器处理的。因此 `llama-server` 会把适配器链接进来，但**永远不激活**。

### 2.4 思考预算（reasoning budget）

`common/reasoning-budget.*` 与采样链路的改动，让思考可以被预算约束。
本模型模板只接受 `reasoning_effort ∈ {xhigh, medium, low}`，OpenAI 生态的 `high` 会触发模板
`raise_exception`（HTTP 400，且 minja 的报错文本是截断的，看起来像乱码）。
运行时自带兼容模板把 `high/max → xhigh`、`minimal → low`，避免客户端被拒。

### 2.5 MTP（投机解码）与 Hadamard 变换

本模型是 `qwen35` 架构、`block_count = 65`、MTP 头在 `blk.64.nextn.*`。

> **两个权重包都适用**：官方 `PQ2_0-MTP-Q8_0`（ProCreations）与**无审查 `Abliterated-PQ2_0-MTP`**
> （BoldingBuilds）张量集合逐项相同（866 张量、零 shape/type 差异，唯一 KV 差异是 `general.name`），
> 因此本融合层与同一套启动参数对两者通用；差别只在权重行为与**投影器是否随包提供**
> （无审查包不自带投影器，但可复用官方 mmproj——两包张量逐项相同）。两个包的张量表都在 Release 里，可自行 diff 验证。

MTP 头的 token-embedding 查表必须施加与主干相同的**逆 Hadamard 变换**，否则
`token_embd.weight` 在被旋转过的基里被读取，图校验会直接拒绝 MTP 上下文。

- 本融合层最早是**移植社区补丁**实现这一点；
- 上游随后**原生落地**了同一修复：
  - `518ad10` `qwen35: apply the Hadamard inverse to the MTP token-embedding lookup`
    （作者 **zhaoyilun**，经 **PR #205** 合入，父提交恰为本融合层的基准 `9a9394a`）
  - `288859a` dflash 借入嵌入的 Hadamard（作者 **usmaneth**，**PR #210**）
- 因此**本融合层已经移除了自己的重复实现**，改为跟随上游。当前 `patches/` 不再包含 MTP
  Hadamard 的私有改动。

---

## 3. 融合流程（复现同一棵树）

```bash
# 1) 取上游并切到基准提交
git clone --filter=blob:limit=204800 https://github.com/PrismML-Eng/llama.cpp.git llm
cd llm && git fetch --no-tags origin prism
git checkout -B fusion-local 9a9394a895b96003ca842a6041cb28ac49a108f7

# 2) 应用本仓库的补丁序列（24 个）
git am --3way /path/to/patches/*.patch     # 或用 patches/apply-fusion.sh
```

### 编译（gfx1030 / Windows）

```bat
cmake -S . -B build-hip -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DAMDGPU_TARGETS=gfx1030 ^
  -DGGML_NATIVE=OFF ^
  -DCMAKE_C_COMPILER="<rocm>\bin\clang.exe" ^
  -DCMAKE_CXX_COMPILER="<rocm>\bin\clang++.exe" ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="%CD%"

ninja -C build-hip -j 16 llama-server llama-kvmem-server llama-kvmem-cli
```

**不要 `ninja all`**：KVMem 的 `CMakeLists.txt` 无条件添加 5 个测试可执行文件，
其中部分在 Windows 上编不过（依赖 `setenv`）。构建命名目标。

### 验证门（本方案的验收标准）

1. `llama-server --list-devices` 必须列出 `AMD Radeon RX 6900 XT`
2. `llama-kvmem-server` 启动日志出现
   `llama-kvmem-server listening on http://<host>:<port> ... kvmem=1 method=retrieval n_ctx=262144 spec=draft-mtp`
3. 一次 chat 请求返回正确内容；
4. 长 prompt（≥ 44K tok）能完整 prefill 并持续生成（KVMem offload 生效）；
5. 视觉：图形/颜色/图内文字识别正确。

---

## 4. 与上游的关系

| 上游 | 本仓库如何处理 |
|---|---|
| llama.cpp / PrismML fork / TurboQuant fork | 不复制不镜像，只发**补丁**；使用者自行 clone 上游再 `git am` |
| KVMem 适配层（Apache-2.0） | 随补丁分发，保留署名；见 `THIRD-PARTY-NOTICES.md` |
| 模型权重 | **不分发**；README §2.1 提供上游链接与 sha256 |
