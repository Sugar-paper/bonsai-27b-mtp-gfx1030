# 第三方组件与许可（THIRD-PARTY NOTICES）

本仓库（`qwen3.8-27b-6900xt-40tps`）新增的内容 —— 融合补丁、启动脚本、文档、张量表生成工具 ——
以 **MIT** 发布（见 `LICENSE`）。下面是它所依赖或派生的第三方组件及其许可。

---

## 1. 上游源码组件

| 组件 | 仓库 | 许可 | 本仓库如何使用 |
|---|---|---|---|
| llama.cpp | https://github.com/ggml-org/llama.cpp | **MIT** | 引擎本体；补丁的最终落点 |
| PrismML fork of llama.cpp | https://github.com/PrismML-Eng/llama.cpp（分支 `prism`） | **MIT** | 三值权重类型 `PQ2_0` / `PTQ1_0`、dflash/MTP；本补丁序列的**基准** |
| llama-cpp-turboquant | https://github.com/iamwavecut/llama-cpp-turboquant | **MIT** | `turbo3_0` / `turbo4_0` KV cache 类型的移植来源 |
| kvmem-llama.cpp | https://github.com/kvmem/kvmem-llama.cpp | **Apache-2.0**（其 README：`KVMem-qw3 source is Apache-2.0; this port should be treated the same`） | 分层 KV 内存适配层（`src/adapter/llama-memory-kvmem*`、`tools/kvmem/*`） |
| lemonade-sdk/llamacpp-rocm | https://github.com/lemonade-sdk/llamacpp-rocm（**b1233 gfx103X**） | **MIT** | `hip-compat` zip 中的 HIP 兼容 DLL（让 gfx1030 被枚举） |
| AMD ROCm 7.1 | https://github.com/ROCm/ROCm | MIT / BSD 系列 | `runtime-tensile` zip 中的 rocBLAS / hipBLASLt 张量库，以及编译工具链 |

### Apache-2.0 说明（KVMem）

随补丁分发的 KVMem 适配层代码保留其原始署名与版权声明。
Apache-2.0 要求：

- 保留版权、专利、商标与归属声明；
- 修改过的文件需标注已修改；
- 若分发二进制，需在随附材料中包含许可文本与 NOTICE（如上游提供）。

本仓库对 KVMem 代码的修改：接入 PrismML 三值树的构建系统（CMake 接线）、
补 HIP/ROCm 兼容层、修正 `GGML_OP_COUNT` 断言值、以及让 `--kvmem*` 参数在
`llama-kvmem-server` 内可用（`llama-server` 不具备这条通路）。

---

## 2. 模型权重（本仓库不分发）

| 模型文件 | 上游仓库 | 许可 |
|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` | https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP | **以该仓库页面声明为准** |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 同上 | 同上 |
| 原始三值权重与投影器 | https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf | 以该仓库页面为准 |

本仓库只分发**元数据**（张量表 = 张量名/类型/形状/偏移的清单）与校验用 sha256，**不含权重数据**。

---

## 3. 商标

AMD、Radeon、ROCm 是 Advanced Micro Devices, Inc. 的商标。
本仓库与 AMD 无隶属关系，未获其背书。所有其他名称可能是各自所有者的商标。
