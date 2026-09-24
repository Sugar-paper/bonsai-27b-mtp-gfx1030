# 聊天模板评估：compat vs Qwen-Sharp（2026-09-24 实测）

## 背景

「思考废话」问题（thinking 模式输出冗长 CoT）的候选方案：**Qwen-Sharp-Chat-Templates**
（peculiar-ragdoll，v22.5.0，Apache-2.0）—— drop-in jinja 模板，覆盖 Qwen3.5/3.6/3.8，
宣称砍思考 token、SWE-bench 修复任务中位耗时 54.6→20.0 分钟（2.7 倍快）。

本仓库现行模板：`abl-27b-reasoning-compat.jinja`（compat），effort 只认 low/medium/xhigh。

## A/B 实测（官方 Q8_0 包，llama-kvmem-server，MTP×2，budget 98304，temperature 0）

三个任务（9.11vs9.9 陷阱题 / 苹果数学 / 回文函数），thinking ON：

| 模板 + effort | 9.11 思考 | 回文思考 | 合计思考 | 合计正文 | 正确率 | 三题 wall |
|---|---:|---:|---:|---:|---|---:|
| **compat @ low（现行）** | **289 ch** | **226 ch** | **648 ch** | 126 ch | 3/3 | **8.4 s** |
| Sharp @ low | 1190 ch | 676 ch | 2013 ch | 131 ch | 3/3 | 16.3 s |
| Sharp @ minimal | 1190 ch | 676 ch | 2366 ch¹ | 139 ch | 5/5² | — |
| Sharp @ none（无 CLI 覆盖） | 0 ch | 0 ch | **0 ch** | 165 ch | 2/3³ | — |

¹ minimal/low 在本模型上行为相同；首个 9.11 题固定 1190 ch，会话内复测同题仅 267 ch
  （历史里有先前答案）。²「锄禾日当午」下一句答错（判定脚本太宽松误标 OK）。
³ 9.11 陷阱题答错 —— 无思考模式的经典失败。

## 关键发现

1. **Sharp 在本栈上思考更多、不是更少**：@low/@minimal 的思考量是 compat @low 的 **3.1 倍**，
   wall 时间 ~2 倍，正确率无提升。Sharp 卡片自己也声明「未针对单模型调优，结果因模型而异」
   —— 在 Bonsai 27B 上不适配。compat 模板的 low 档 steer 本身就比 Sharp 的 low 更激进。
2. **effort kwargs 的传递链路（重要 gotcha）**：`--chat-template-kwargs` 里的
   `reasoning_effort` 会被 CLI 参数 `--reasoning-effort` **覆盖**（两者同时传时 kwargs 失效，
   none/minimal/low 全部输出相同的 2013 ch 就是证据）。要让 kwargs 生效，**必须不传
   `--reasoning-effort`**。且 OpenAI 风格的顶层 `reasoning_effort` 请求字段也会被服务器吞掉
   （Sharp 卡片同款警告）。
3. **effort=none 可以完全关闭思考**（reasoning=0），但代价是陷阱题答错 —— 与「关闭思考
   损失推理质量」的已知规律一致。
4. Sharp 的强项在 **agentic/coding 长任务**（其 2.7 倍数据来自 SWE-bench 修复任务、medium
   effort），不在简单 QA 的废话削减。本轮 3 个简单任务无法体现该优势。
5. **MTP 提示**：Sharp 卡片记录了「答案中途冒 `<think>` 标签」的未复现问题（MTP 是头号嫌疑）。
   本轮 A/B 的全部输出未观察到该现象，但重度 MTP 用户应留意。

## 结论

- **compat 模板保持默认**。Sharp 在「思考废话」这一目标上不敌现行的 low 档 steer。
- `qwen-sharp-v22.5.0.jinja` 已放入模型目录与 `chat-templates/`，作为**可选替代**保留：
  切换方式 = 启动参数换 `--chat-template-file <sharp.jinja>`，effort 只能走
  `--chat-template-kwargs {"reasoning_effort":"..."}`（不要同时传 `--reasoning-effort`）。
- 若未来在 agentic/coding 长任务上需要 Sharp，按上表方法重测后再切。
