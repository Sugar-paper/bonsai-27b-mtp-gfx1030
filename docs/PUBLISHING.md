# 发布流程（GitHub / 自建 Gitea）

本目录已经是可直接推送的 git 仓库骨架。以下命令中的 `<owner>` / `<repo>` 请替换成你自己的名称。

---

## 1. 建议的仓库信息

| 项 | 建议值 |
|---|---|
| 仓库名 | `qwen3.8-27b-6900xt-40tps`（原名 `bonsai-27b-mtp-gfx1030`，旧链接自动重定向） |
| 描述 | `Fusion patches + prebuilt runtime + tensor tables: run a ternary 27B MTP model at 256K on a single 16 GB Radeon RX 6900 XT (gfx1030)` |
| 话题（Topics） | `llama-cpp` `rocm` `gfx1030` `rdna2` `kv-cache` `speculative-decoding` `ternary` `huggingface` |
| Release 标签 | `v1.0.0` |
| 许可证 | MIT（仓库内容）+ Apache-2.0（KVMem 派生部分，已在 NOTICES 说明） |

---

## 2. 建仓并推送

```bash
cd <this-dir>

# 用中立身份提交（避免个人信息进入 git objects）
git -c user.name="bonsai-27b-fusion" \
    -c user.email="noreply@users.noreply.github.com" \
    commit -m "release: fusion for Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0 on gfx1030"

# 持久化到本仓库配置，后续提交也保持中立
git config user.name  "bonsai-27b-fusion"
git config user.email "noreply@users.noreply.github.com"

# 建远程并推送（GitHub：先在网页上新建空仓库）
git remote add origin https://github.com/<owner>/<repo>.git
git branch -M main
git push -u origin main
```

> **不要**在 remote URL 里写 token（会留在 `.git/config`）。用凭据管理器或一次性
> `git -c credential.helper=...` 方式，或在网页上创建仓库后使用 SSH key。

### 自建 Gitea（可选，局域网更快）

```bash
git remote add gitea http://<your-gitea-host>:3000/<owner>/<repo>.git
git push -u gitea main
```

---

## 3. 创建 Release 并上传资产

仓库里**不含** zip（`assets/*.zip` 已在 `.gitignore` 中），它们作为 Release 资产上传。

```bash
gh release create v1.0.0 \
  --title "v1.0.0 · gfx1030 / RX 6900 XT" \
  --notes-file RELEASE-NOTES.md \
  assets/runtime-bin-v1.0.0.zip \
  assets/runtime-tensile-gfx1030-v1.0.0.zip \
  assets/hip-compat-b1233-gfx103x-v1.0.0.zip \
  assets/tensor-table-Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.csv \
  assets/tensor-table-Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.txt \
  assets/tensor-table-mmproj-Q8_0.csv \
  assets/tensor-table-Abliterated-PQ2_0-MTP.csv \
  assets/model-hashes.txt \
  assets/SHA256SUMS.txt
```

没有 `gh` 时，用网页 **Releases → Draft a new release** 逐个上传同一组文件即可。

### 资产体积参考

| 资产 | 大小 |
|---|---:|
| `runtime-bin-v1.0.0.zip` | ≈29 MiB |
| `hip-compat-b1233-gfx103x-v1.0.0.zip` | ≈85 MiB |
| `runtime-tensile-gfx1030-v1.0.0.zip` | ≈184 MiB |
| 张量表 3 份 + 哈希 + SHA256SUMS | < 1 MiB |

> 合计约 300 MB。若上传链路较慢（例如国际出口受限），可先推 Gitea 并把
> GitHub Release 作为镜像；或把 tensile / hip-compat 两个 zip 换成"自行从 ROCm 发行包与
> `lemonade-sdk/llamacpp-rocm` b1233 获取"的说明，只发 `runtime-bin`（29 MiB）。

---

## 4. 发布前自检清单

- [ ] `privacy scan`：仓库内没有任何个人路径 / IP / 邮箱 / 主机名
- [ ] `git log --format='%an <%ae>'` 只出现中立身份
- [ ] 三个 zip 解压到同一目录后，`scripts/start-bonsai-mtp.py --check` 通过
- [ ] 真实启动一次并成功对话（验证过 `17*23=391`）
- [ ] `.gitignore` 生效：`git status` 中不出现 `assets/*.zip`
- [ ] README 里的 sha256 与实际资产一致（`sha256sum -c assets/SHA256SUMS.txt`）
