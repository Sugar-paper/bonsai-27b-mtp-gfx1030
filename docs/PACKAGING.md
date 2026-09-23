# Release 资产与张量表（PACKAGING）

## 1. 资产清单

| 资产 | 内容 | 必需 |
|---|---|---|
| `runtime-bin-<ver>.zip` | `llama-kvmem-server.exe`、`llama-kvmem-cli.exe`、`llama-server.exe` + `llama.dll`、`ggml*.dll`（gfx1030/HIP 构建） | ✅ |
| `runtime-tensile-gfx1030-<ver>.zip` | `rocblas/` 与 `hipblaslt/` 的 gfx1030 张量库（解压到 `runtime/` 下） | ✅ |
| `hip-compat-b1233-gfx103x-<ver>.zip` | b1233 gfx103X HIP 兼容 DLL（解压到 `hip-compat/`） | ✅ |
| `tensor-table-Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.csv` / `.txt` | **张量表**：主模型 866 个张量（含 MTP 头） | 参考 |
| `tensor-table-mmproj-Q8_0.csv` | 视觉投影器 334 个张量 | 参考 |
| `tensor-table-Abliterated-PQ2_0-MTP.csv` | 无审查版（同为 866 张量，可供对照） | 参考 |
| `model-hashes.txt` | 三个权重文件的 sha256 与字节数 | ✅ 校验用 |
| `SHA256SUMS.txt` | 以上所有资产自身的 sha256 | ✅ |

**不随 Release 分发**：模型权重（见 README §2.1 的上游链接）、上游源码（用 `patches/` 打到你自己 clone 的上游上）。

## 2. 张量表字段

CSV 表头：

```
name,type,type_id,n_dims,dims,offset,bytes
```

| 字段 | 含义 |
|---|---|
| `name` | 张量名，例如 `blk.0.attn_q.weight`、`blk.64.nextn.eh_proj.weight` |
| `type` / `type_id` | ggml 量化类型名与编号（本模型含 `PQ2_0 = 142`、`Q8_0 = 8`、`F32 = 0`） |
| `n_dims` / `dims` | 维度数与形状（`x` 连接） |
| `offset` | 张量在 GGUF 数据段内的字节偏移 |
| `bytes` | 按块大小推算的字节数（仅对已知块布局的类型给出，否则留空） |

`.txt` 版本是人读格式（对齐的 name / type / shape / offset）。

## 3. 这张表能验证什么

1. **张量总数与结构**：866 个张量；`blocks = 65`（主干 0–63 + 一层 MTP）
2. **MTP 头存在**：搜索 `blk.64.nextn.`，应出现
   `eh_proj`、`enorm`、`hnorm`、`shared_head_norm` 等张量
   （判定规则：`block_count = 65` ⇒ 有 MTP；`64` ⇒ 没有）
3. **三值权重类型**：大量张量为 `type_id = 142`（`PQ2_0`），部分为 `Q8_0`
4. **形状自检**：`token_embd.weight` / `output.weight` 的 `dims` 可核对词表与隐层宽度

> 表是**从权重文件直接解析 GGUF 头**生成的（不依赖任何第三方库），
> 生成工具随仓库提供：`tools/dump-gguf-tensors.py`。
> 用法：`python tools/dump-gguf-tensors.py <model.gguf> out.csv`

## 4. 校验下载

```bash
# 权重
sha256sum -c model-hashes.txt          # Linux/macOS
certutil -hashfile <file> SHA256       # Windows（逐文件）

# 资产
sha256sum -c SHA256SUMS.txt
```

## 5. 版本与构建溯源

运行时由 `patches/` 的补丁序列构建，基准提交见 `patches/SERIES.md`。
构建参数：`-DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_KVMEM=ON`
（完整列表见 `docs/HARDWARE-GFX1030.md` §5）。

每个 zip 内附 `BUILD-INFO.txt`：构建参数、目标 GPU、编译器版本、以及该 zip 内文件的 sha256 清单。
