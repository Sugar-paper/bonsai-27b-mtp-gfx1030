# 融合补丁序列（24 个补丁）

## 基准

| 项 | 值 |
|---|---|
| 上游 | `https://github.com/PrismML-Eng/llama.cpp`（分支 `prism`） |
| **基准提交** | **`9a9394a895b96003ca842a6041cb28ac49a108f7`** |
| 补丁数 | 24 |
| 改动规模 | 195 个文件 |

补丁用 `git format-patch --zero-commit` 导出：`index` 行不含真实 blob 哈希，
`From:` 为中立身份（原始开发者的身份信息已移除）。因此**请用 `git am --3way` 应用**，
不要逐条手工 patch。

## 应用

```bash
# 推荐：一键脚本（clone + checkout 基准 + git am）
patches/apply-fusion.sh <workdir>

# 手工
git clone --filter=blob:limit=204800 https://github.com/PrismML-Eng/llama.cpp.git llm
cd llm && git fetch --no-tags origin prism
git checkout -B fusion-local 9a9394a895b96003ca842a6041cb28ac49a108f7
git am --3way /path/to/patches/*.patch
```

Windows：

```powershell
pwsh patches/apply-fusion.ps1 -WorkDir D:\src\llm
```

## 补丁分组

### A. TurboQuant KV 类型：移植与可用化（0001–0003）

| # | 提交 |
|---|---|
| 0001 | `feat(ggml): port TurboQuant turbo3/turbo4 KV + WHT kernels` |
| 0002 | `style(fusion): drop 2362 blank lines the port inserted` |
| 0003 | `feat(fusion): complete the TurboQuant fusion, the cache types were unusable` |

移植块布局与 WHT 内核，并补齐让它**真正可选可用**的集成层（CLI、FA dispatch、convert 表、
set-rows、图重排、CPU 路径、`quantize_chunk`、`llama-bench` 命名、测试）。

### B. 缺陷定位、更正与范围（0004–0009、0011、0013–0017）

| # | 提交 |
|---|---|
| 0004 | `docs(fusion): give this worktree its first ledger` |
| 0005 | `docs(fusion): run the ctest gate, and prove the head_dim claim with perplexity` |
| 0006 | `docs(fusion): CORRECTION - two independent KV defects, not one head_dim story` |
| 0007 | `docs(fusion): localise defect A to the V side and rule out seven causes` |
| 0008 | `docs(fusion): defect A is two things, and one has a demonstrated fix` |
| 0009 | `docs(fusion): scope is Bonsai-2-27B, and the stack is green for it` |
| 0011 | `docs(fusion): the KV defects were one line we dropped, and they are withdrawn` |
| 0013 | `docs(fusion): close the gguf-py gap in the ledger, keep mmvq-tq.cu as a deliberate gap` |
| 0014 | `docs(fusion): record why neither fusion tree can replace the other` |
| 0015 | `docs(fusion): the model directory now matches the layout LM Studio indexes` |
| 0016 | `docs(fusion): this branch is 437 commits behind upstream master, and that is not ours to fix` |
| 0017 | `docs(fusion): the four Bonsai packs, measured properly` |

这一组是**排障记录**：把"KV 异常"从一个 head_dim 猜想，收敛到我们自己在移植中丢的一行代码。
过程包含两次被推翻的结论（先怪编译选项、再怪上游 fork）——保留在补丁信息里，是有意为之。

### C. 修复与常量（0010、0012）

| # | 提交 |
|---|---|
| 0010 | `fix(cuda): restore the scale on the second q8_0 element, dropped by our port` |
| 0012 | `feat(gguf-py): give the TurboQuant cache types their entries back` |

`0010` 修复 `dequantize_q8_0()` 里丢失的 `v.y *= d;`；`0012` 在 `gguf-py` 常量表里补上
`TURBO3_0 = 43` / `TURBO4_0 = 44`。

### D. KVMem 分层 KV 融合（0018–0024）

| # | 提交 |
|---|---|
| 0018 | `chore: checkpoint working tree before kvmem fusion` |
| 0019 | `feat(kvmem): fuse KVMem tiered KV memory adapter into PrismML turboquant fusion` |
| 0020 | `feat(kvmem): HIP/ROCm compat layer and CMake wiring; adapter compiles clean` |
| 0021 | `fix(ggml): GGML_OP_COUNT is 103, not 102` |
| 0022 | `feat(kvmem): wire llama-kvmem-server into the tree, links clean on gfx1030` |
| 0023 | `docs(kvmem): turbo3/turbo4 is a kernel port, not a flag change` |
| 0024 | `feat(kvmem): turbo3/turbo4 KV cache types, full support` |

关键结论写在这一组里：**只有 `llama-kvmem-server` 能开启 KVMem**；
`llama-server` 会把适配器链接进来但永不激活（门控 `g_kvmem_params.enabled`
只由 `llama_kvmem_set_params()` 写入，而它只在 `tools/kvmem/llama-kvmem-server.cpp`
的手写参数解析里被调用）。

## 关于 MTP / Hadamard

本模型（`qwen35`，`block_count = 65`）的 MTP 头需要与主干相同的逆 Hadamard 变换。
本融合层最早以**移植社区补丁**的方式实现；上游随后**原生落地**了同一修复
（`518ad10`，作者 zhaoyilun，PR #205；`288859a`，作者 usmaneth，PR #210）。
因此当前序列**不再包含** MTP Hadamard 的私有改动 —— 跟随上游即可。

## 许可

补丁中的 KVMem 适配层代码为 **Apache-2.0**，其余为 **MIT**。详见 `THIRD-PARTY-NOTICES.md`。
