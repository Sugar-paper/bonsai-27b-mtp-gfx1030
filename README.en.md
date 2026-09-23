# Ternary-Bonsai-2-27B — censored & uncensored versions, side by side — on a single Radeon RX 6900 XT (gfx1030)

A **fusion** (patch series + prebuilt runtime + measured numbers) that makes a ternary 27B
MTP model usable on **one 16 GB RDNA2 card** with **262144 context, native MTP speculative
decoding and image input**.

Two weight packs are published upstream and both run on this stack:

- **stock / censored**: `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0` (ProCreations) + official vision projector
- **uncensored**: `Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP` (BoldingBuilds), text-only (no projector ships)

They are tensor-for-tensor identical (866 tensors, 65 blocks), so the same runtime and the
same flags drive both.

> 中文主文档：[`README.md`](README.md) · 融合原理：[`docs/FUSION.md`](docs/FUSION.md)

---

## What problem this solves

Four capabilities live in four different trees, and this GPU needs all of them at once:

| Capability | Lives in | Without it |
|---|---|---|
| Ternary weight types `PQ2_0 = 142` / `PTQ1_0 = 143` | PrismML `prism` fork | model refuses to load (`invalid ggml type 142`) |
| `turbo3_0 = 43` / `turbo4_0 = 44` KV cache types | TurboQuant fork | only f16/q8_0 KV, no long context |
| Tiered KV memory (KVMem) | kvmem project (Apache-2.0) | 262144 ctx needs ≈17 GB KV — does not fit in 16 GB |
| gfx1030 device enumeration | gfx103X HIP compatibility runtime | system ROCm does not enumerate this card |

`patches/` is the layer that puts the latter three onto the first one (24 commits, 195 files).

## Upstreams

- engine: https://github.com/ggml-org/llama.cpp (MIT)
- ternary types + MTP: https://github.com/PrismML-Eng/llama.cpp `prism` (MIT) — **base commit `9a9394a`**
- turbo KV types: https://github.com/iamwavecut/llama-cpp-turboquant (MIT)
- tiered KV: https://github.com/kvmem/kvmem-llama.cpp (Apache-2.0)
- gfx103X HIP compat: https://github.com/lemonade-sdk/llamacpp-rocm (MIT)

## Models (not redistributed here)

Two weight packs are covered; both are `qwen35`, 866 tensors, `block_count = 65` (with the MTP
head at `blk.64.nextn.*`), so the same runtime and the same flags work for both:

| File | Source | sha256 | Vision |
|---|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` | https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP | `3cb3f005…131dd4` | ✅ official mmproj |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | same repo | `6807ede6…631903` | projector |
| **`Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf`** (**uncensored**) | https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF | `7aa43b9a…b02e86` | ❌ none ships |
| original ternary weights | https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf | see that repo | — |

Full hashes: `assets/model-hashes.txt`. Measured on the uncensored pack: 147,725-token prompt →
prefill 225.5 t/s, generation **41.7 t/s**, MTP accept 40.5%; 44,414-token prompt → prefill
256.6 t/s, generation 40.7 t/s, recall 6/6; at 32768 ctx: baseline 38.8, MTP n-max 3 41.6, MTP
n-max 2 **42.5 t/s**. The two packs are tensor-for-tensor identical, so speeds match within noise.

## Quick start

1. Download the `.gguf` files you want into `<release>/model/`
2. Extract the three Release zips into `<release>/` (`runtime/`, `runtime/rocblas|hipblaslt`, `hip-compat/`)
3. Run one of:

```bat
scripts\start-bonsai-mtp.bat            :: stock pack, 262144 ctx, MTP 3, KVMem, thinking on/low
scripts\start-bonsai-mtp-vision.bat     :: stock pack + vision tower
scripts\start-bonsai-abliterated.bat    :: UNCENSORED pack (text only)
scripts\start-bonsai-mtp.bat --check    :: validate only
```

```bash
python scripts/start-bonsai-mtp.py --check
python scripts/start-bonsai-mtp.py --vision --effort medium
```

OpenAI-compatible endpoint: `POST http://127.0.0.1:11234/v1/chat/completions`.

## Measured results (RX 6900 XT, 16 GB, ROCm 7.1)

| Scenario | Result |
|---|---|
| 262144 ctx, 147,725-token prompt | prefill 225.5 t/s, generation **41.7 t/s**, MTP accept 40.5%, ≈11.1 GB VRAM |
| 262144 ctx, 44,414-token prompt | prefill 256.6 t/s, generation 40.7 t/s |
| ctx 32768, 256 tokens generated | baseline 38.8 t/s · **MTP n-max 2: 42.5 t/s (accept 51.6%)** · MTP n-max 3: 41.6 t/s |
| MTP acceptance vs length | 94.9% at 49 tokens → 40.9–51.6% at 256 tokens |
| Vision (mmproj, `--image-min-tokens 1024`) | 1,079 image tokens; correct shapes/colours/text; **cold first image ≈135 s**, warm 2.0 s |
| Thinking effort | `low` finishes naturally (678 chars of CoT); `xhigh` runs away (hit the cap at 16,596 tokens / 5m48s) |
| Server ready | 8–10 s (text), 10 s (with projector) |

MTP gain here is **+8.4%**, below the model author's +37%: the benefit is content-dependent
(their median came from reasoning prompts, acceptance 0.83). Details in `docs/PERFORMANCE.md`.

## Requirements (short form)

Radeon RX 6900 XT (gfx1030) · ROCm **7.1** · gfx103X HIP compatibility DLLs **first on PATH** ·
matching gfx1030 `rocblas`/`hipblaslt` tensor libs · 16 GB VRAM (262144 ctx **requires KVMem**) ·
≥32 GB system RAM for the tiered KV cache.

Common failures and fixes: `docs/HARDWARE-GFX1030.md` §4.

## Known limitations

1. `turbo3`/`turbo4` KV is broken at `head_dim = 128`; both models here are `key_length = 256` (fine there).
2. MTP is not bit-exact: outputs differ at `temperature=0` with MTP on/off.
3. Thinking must be tuned (`low` by default) or it will run to the output cap.
4. Vision cold start is ≈2 minutes — expected, not a hang.
5. Verified on gfx1030 only. Other RDNA cards need a rebuild and re-measurement.
6. The server ships **without authentication** and binds `127.0.0.1` by default.

## License

Repository material (patches, scripts, docs, tools): **MIT**. KVMem-derived adapter code carried
in the patch series: **Apache-2.0**. Model weights: not distributed; see their upstream pages.
See `THIRD-PARTY-NOTICES.md`.
