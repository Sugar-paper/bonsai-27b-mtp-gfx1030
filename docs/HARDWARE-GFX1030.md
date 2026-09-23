# 硬件适配：Radeon RX 6900 XT（gfx1030 / RDNA2 / 16 GB）

本方案**只在这一张卡上验证过**。本文件给出该硬件上必须满足的前置条件、显存推导、编译参数
与常见报错对照表。换卡需要重新编译并重测。

---

## 1. 为什么这张卡需要专门适配

1. **设备枚举**：在这台机器上，系统 ROCm 7.1 的 HIP 运行时**不枚举** Radeon RX 6900 XT
   （`gfx1030`）。任何 llama 二进制启动后都会在加载模型前退出。
   解法是把一份 gfx103X 的 HIP 兼容运行时（b1233）**放在 PATH 最前面**，让它的
   `amdhip64_7.dll` / `rocm_kpack.dll` 等遮蔽系统版本。
   本仓库的启动脚本都做这件事；手工启动时必须自己做。
2. **显存**：16 GB 要装下 27B 三值权重（≈7.7 GB）+ 262144 ctx 的 KV。
3. **RDNA2 的 kernel 覆盖**：`-DGPU_TARGETS=gfx1030` 必须显式给出，且张量库（rocBLAS /
   hipBLASLt 的 tensile）必须与 gfx1030 匹配，否则大型 matmul 会失败。

---

## 2. 依赖

| 组件 | 版本/来源 | 用途 | 缺失症状 |
|---|---|---|---|
| AMD 驱动 | Adrenalin（Win）/ amdgpu（Linux），支持 ROCm 7.1 | 设备 | `no HIP devices found` |
| ROCm | **7.1**（含 `clang.exe` / `clang++.exe`） | HIP 编译与运行时 | 编译失败 / `amdhip64` 加载失败 |
| HIP 兼容运行时 | `lemonade-sdk/llamacpp-rocm` **b1233 gfx103X**（7 个 DLL，见 Release 资产） | gfx1030 枚举 | 见 §4 报错 1 |
| 张量库 | `rocblas` + `hipblaslt`（按 gfx1030 构建，随 Release `runtime-tensile` zip） | matmul | 见 §4 报错 2 |
| 编译器环境（Win） | Visual Studio BuildTools（提供 `LIB` / `INCLUDE`）+ Ninja | 编译 | 找不到 `windows.h` 等 |
| Python | 3.10+（可选，仅启动脚本） | 启动 | — |

> Windows 上若 `LIB`/`INCLUDE` 不由 `vcvars64` 注入（某些环境里 `reg.exe` 被安全策略拦截，
> `vcvars64.bat` 会失败），请手工设置这两个变量指向 MSVC 与 Windows SDK 的目录。

---

## 3. 显存预算（为什么 KVMem 是必需的）

| 项 | 估算 |
|---|---|
| 三值 27B 权重（PQ2_0 + Q8_0 混合） | ≈ 7.7 GB |
| KV cache（`q8_0`，`key_length = 256`，`head_count_kv = 4`，65 层，其中约 3/4 为线性/SSM 层） | 262144 ctx ≈ **17 GB** ⇒ **超过 16 GB** |
| 结论 | 262144 ctx **必须** 走 KVMem 分层缓存；不开 KVMem 时 ctx 131072 就已装不下 |

KVMem 工作方式（本方案参数）：

```
--kvmem-budget 36864      常驻池预算（MB）
--kvmem-gen-reserve 16384 生成阶段预留
--kvmem-method retrieval  检索式取回
--kvmem-mtp-state snapshots
```

实测常驻池 **53,248 cells**（block_tokens = 128），长 prompt 时按块 offload 到主机内存；
147,725 token prompt 下显存占用约 **11.1 GB**，44,414 token 下约 **11.3 GB**。

---

## 4. 常见报错对照表

| 报错 | 根因 | 处理 |
|---|---|---|
| `error while loading shared libraries: api-ms-win-crt-convert-l1-1-0.dll` | 没带 HIP/ROCm 环境（缺 `PATH` / `HIP_PATH` / compat HIP） | 用本仓库启动脚本；手工启动时确保 `hip-compat` 与 `%ROCM_PATH%\bin` 在 PATH 中 |
| 启动即退出，无 HTTP 状态码 | `rocblas`/`hipblaslt` 找不到或与 gfx1030 不匹配 | 解压 `runtime-tensile` zip 到 `runtime/`；确认 `runtime/rocblas/library` 存在 |
| `[ERROR] HIP device preflight failed - gfx1030 was not enumerated` | compat HIP 未前置（或它被杀软删/隔离） | 把 `hip-compat/` 放 PATH **最前**；确认 `amdhip64_7.dll` 在 |
| 加载阶段直接退出（ctx 131072 及以上） | KV 显存超预算 | 开 KVMem（默认）；或降低 `--ctx` |
| `error: unable to execute command: program not executable`（**编译期**） | 杀毒软件把 `%ROCM_PATH%\bin\lld-link.exe` 删了（本机实测遇到过：目录里只剩 `ld.lld.exe`） | LLD 是**多调用二进制**，把 `ld.lld.exe` 复制成 `lld-link.exe` 与 `lld.exe` 放到**自己的目录**，把该目录排到 PATH 最前（`cmake -fuse-ld=lld-link` 只认裸名字，会按名字查 PATH） |
| 长 prompt 中途进程消失 | 多为被外部终止（会话/job 对象回收），**不是** KVMem 缺陷 | 用前台终端或任务计划托管服务；查看 Windows 事件日志有没有 WER 记录 |
| 首次发图卡住约 2 分钟 | 视觉投影器在 CPU 侧预热 + kernel 编译 | 属预期；预热后约 2 s |

---

## 5. 编译参数（本方案使用）

```
-G Ninja
-DCMAKE_BUILD_TYPE=Release
-DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DAMDGPU_TARGETS=gfx1030
-DGGML_NATIVE=OFF
-DGGML_CUDA_FA_ALL_QUANTS=ON          # 编译混合 KV 组合（含 turbo 实例）
-DLLAMA_CURL=OFF
-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
-DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
-DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT=<source root>
```

说明：

- `GGML_CUDA_FA_ALL_QUANTS=ON/OFF` **不影响正确性**（两种配置下 KV 矩阵都在噪声范围内），
  `ON` 会编译更多混合组合、链接更慢；本方案用 `ON`。
- `GGML_HIP_MMQ_MFMA=ON`、`GGML_HIP_NO_VMM=ON` 为本机实测可用的组合。
- 构建目标只取 `llama-server` / `llama-kvmem-server` / `llama-kvmem-cli`，不要 `ninja all`。

---

## 6. 运行前置条件清单（逐条可勾）

- [ ] 驱动已装，`rocminfo` 能看到 GPU（或至少 HIP 能加载）
- [ ] `hip-compat/` 解压完成，且**排在 PATH 最前**
- [ ] `runtime/` 解压完成，`llama-kvmem-server.exe` 存在
- [ ] `runtime/rocblas/library` 与 `runtime/hipblaslt/library` 存在
- [ ] `model/` 下两个 gguf 的 sha256 与 `assets/model-hashes.txt` 一致
- [ ] 空闲显存 ≥ 12 GB（跑 256K ctx 时）
- [ ] 跑基准测试前先让 GPU 降温到 50 ℃ 以下（本机实测高温会显著影响吞吐）
