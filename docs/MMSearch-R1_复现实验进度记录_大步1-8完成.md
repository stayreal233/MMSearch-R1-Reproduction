# MMSearch-R1 PRO6000 复现实验进度与续跑交接

> 更新时间：2026-08-16（UTC）  
> 当前结论：大步 1～8 已完成并验收；大步 9～12 尚未开始。  
> 原始方案：`/root/MMSearch-R1_PRO6000-96G_5.98元每小时_复现方案.md`  
> 代码仓库：`/root/autodl-tmp/multimodal-search-r1`  
> 本文用途：让一个全新的对话在不重复前八步的前提下，安全地从大步 9 继续。

## 0. 新对话应先读什么

新对话应按以下优先级获取事实：

1. 本文档：当前完成状态、已批准适配、续跑规则。
2. `/root/autodl-tmp/outputs/` 下的正式 JSON 证据。
3. 原始复现方案：尚未执行的大步 9～12 的目标和验收要求。
4. `reproduction/env/` 中的 revision 和 manifest。

不要只看 `reproduction/env/search_backend_decision.json` 的状态字段。该文件是历史决策记录，其中 Text Search 仍写着 `deferred_to_big_step_8`，但大步 8 实际已经完成。

可直接把下面这段发给新对话：

```text
请先完整读取：
1. /root/MMSearch-R1_复现实验进度记录_大步1-8完成.md
2. /root/MMSearch-R1_PRO6000-96G_5.98元每小时_复现方案.md

大步1-8已经完成，不要重复下载或重跑。先执行交接文档中的只读 preflight，确认 GPU、磁盘、Git、正式输出和密钥文件权限；然后从大步9开始部署独立的 Qwen3-32B-FP8 摘要环境。必须保留现有 reproduction/ 未跟踪文件，不得 git clean/reset。真实 Serper key 只能 source，不能读取、打印或写入日志。每完成一个大步暂停等待确认；遇到鉴权、下载损坏、兼容性、OOM、搜索/网页失败等问题也暂停等待指示。
```

## 1. 总体进度

| 大步 | 状态 | 结论 |
|---|---:|---|
| 1. 服务器与 Blackwell 基线 | 已完成 | PRO6000 96GB、驱动、PyTorch cu128、BF16/SDPA 均通过 |
| 2. MMSearch 推理环境 | 已完成 | 独立数据盘环境健康，`pip check` 无冲突 |
| 3. 官方仓库与 commit 固定 | 已完成 | 主仓库及 `verl` 子模块 commit 已记录 |
| 4. 模型、FVQA 与缓存下载校验 | 已完成 | 模型/数据精确 revision、文件数和字节数已固定 |
| 5. Checkpoint 单轮 smoke | 已完成 | Search-Free 样例 Exact Match 通过 |
| 6. Placeholder 多轮控制流 | 已完成 | Search-Free 与三轮 Search-Required 均通过 |
| 7. 真实 Image Search | 已完成 | 使用官方 FVQA Lens 缓存；已接受 4/5 缩略图降级 |
| 8. 真实 Text Search + Jina | 已完成 | Serper 5/5、Jina 5/5、完整三轮 Exact Match 通过 |
| 9. Qwen3-32B-FP8 摘要与同卡共驻 | 未开始 | 环境、权重、服务和输出均不存在 |
| 10. 四类案例与完整 Trace | 未开始 | 已有 Case A/D 候选证据，仍需正式整理四类及失败案例 |
| 11. 分阶段批量评测 | 未开始 | 应按 `5→20→50→100→300` 推进 |
| 12. Base 对比与最终报告 | 未开始 | Base 权重尚未下载 |

执行纪律：每完成一个大步后暂停；运行中出现实际问题也暂停。大步 11 每次扩大样本规模前，还应检查搜索额度、失败率和缓存命中率。

## 2. 当前服务器快照

审计时间为 2026-08-16 UTC。新对话开始时必须重新只读核验，因为 GPU、磁盘和进程状态可能变化。

| 项目 | 当前值 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 显存 | 97,887 MiB |
| Compute Capability | `(12, 0)` / `sm_120` |
| 驱动 | 595.58.03 |
| `nvidia-smi` 报告 CUDA | 13.2 |
| PyTorch CUDA runtime | 12.8 |
| 当前 GPU 计算进程 | 无 |
| 当前 GPU 显存占用 | 0 MiB |
| 内存 | 约 1.0 TiB；审计时可用约 904 GiB |
| Swap | 无 |
| 根盘 `/` | 30GB；约 27GB 可用 |
| 数据盘 `/root/autodl-tmp` | 250GB；约 224GB 可用 |
| `nvcc` | 未安装 |

所有新模型、环境、HF 缓存和结果必须继续放在 `/root/autodl-tmp`，不要写满 30GB 根盘。`nvcc` 缺失不影响当前 Transformers + SDPA 主线；大步 9 应优先使用已有 Blackwell wheel，避免不必要的 CUDA 源码编译。

### 新对话 preflight

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
df -hT /root /root/autodl-tmp
free -h

/root/autodl-tmp/envs/mmsearch_infer/bin/python -m pip check
/root/autodl-tmp/envs/mmsearch_infer/bin/python -c \
'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))'

git -C /root/autodl-tmp/multimodal-search-r1 status --short --branch
git -C /root/autodl-tmp/multimodal-search-r1 rev-parse HEAD
git -C /root/autodl-tmp/multimodal-search-r1 submodule status --recursive

stat -c '%a %U:%G %n' \
/root/autodl-tmp/multimodal-search-r1/reproduction/env/serper.env \
/root/autodl-tmp/multimodal-search-r1/reproduction/env/serper.env.example

test -f /root/autodl-tmp/outputs/real_search_flow_raw_jina.json
test -f /root/autodl-tmp/models/MMSearch-R1-7B/config.json
test -f /root/autodl-tmp/datasets/FVQA/fvqa_train.parquet
```

本服务器上的 Codex 命令沙箱可能报：

```text
bwrap: No permissions to create a new namespace
```

这是服务器禁用了非特权 user namespace；此前本机命令均在明确批准后使用 `require_escalated` 执行。不要把它误判为 MMSearch 或 CUDA 故障。

## 3. 已固定的代码、模型和数据

### 3.1 Git

- 仓库：`/root/autodl-tmp/multimodal-search-r1`
- 分支：`main`
- 主 commit：`3727c76dda2fc0ea9df16a9f770946d039e6f09b`
- `verl` 子模块：`8e9e73723fd1cc729bedb3bbcf915060afbda91d`
- 记录文件：
  - `reproduction/env/git_commit.txt`
  - `reproduction/env/git_submodules.txt`

当前所有官方 tracked 文件无修改；整个 `reproduction/` 显示为 `?? reproduction/`，即复现脚本未纳入 Git。严禁执行 `git clean`、`git reset --hard` 或删除未跟踪目录，否则会丢失当前全部适配脚本。`reproduction/env/` 被仓库 `.gitignore` 的 `env/` 规则忽略；不要用 `git add -f` 强制加入私密配置。

### 3.2 Hugging Face 固定 revision

权威记录：`reproduction/env/huggingface_revisions.json`。

| 类型 | Repo | Revision | 文件数 | 精确字节数 | 本地路径 |
|---|---|---|---:|---:|---|
| 模型 | `lmms-lab/MMSearch-R1-7B` | `3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46` | 17 | 16,600,357,342 | `/root/autodl-tmp/models/MMSearch-R1-7B` |
| 数据 | `lmms-lab/FVQA` | `bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5` | 6 | 2,336,555,807 | `/root/autodl-tmp/datasets/FVQA` |

占用空间：MMSearch 模型约 16GB，FVQA 约 2.2GB。

### 3.3 FVQA 清单

权威记录：

- `reproduction/env/fvqa_manifest.json`
- `reproduction/env/fvqa_image_search_cache_manifest.json`

| Split | 行数 | 类别信息 | 官方 Image Search 缓存 |
|---|---:|---|---:|
| train | 4,856 | `search_free=1,544`；`search_required=3,312` | 4,849 / 4,856 |
| test | 1,800 | 发布文件没有 `category` 列 | 1,798 / 1,800 |

训练缓存缺失 ID：`fvqa_train_72`、`279`、`332`、`489`、`738`、`4345`、`4633`。测试缓存缺失：`fvqa_test_1470`、`fvqa_test_1753`。

类别控制案例与分层评测必须使用 train 的官方类别列；test 只能做总体评测，不能把推断类别伪装成官方标签。

## 4. 已验证的推理环境

环境路径：`/root/autodl-tmp/envs/mmsearch_infer`，占用约 7.3GB。

| 包 | 版本 |
|---|---|
| Python | 3.11.15 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| transformers | 4.51.0 |
| accelerate | 1.14.0 |
| qwen-vl-utils | 0.0.14 |
| datasets | 5.0.1 |
| huggingface_hub | 0.36.2 |
| openai | 3.1.0 |
| Pillow | 11.1.0 |

`pip check` 当前为 `No broken requirements found`。该环境未安装 `vllm`、`flash_attn` 或 `ray`，不要在其中安装大步 9 的现代 vLLM。运行现有脚本时直接调用该环境的绝对 Python 路径，并从仓库根目录运行：

```bash
cd /root/autodl-tmp/multimodal-search-r1
export PYTHONPATH=/root/autodl-tmp/multimodal-search-r1
```

主模型固定采用：BF16、SDPA、`use_fast=False`、HF offline load。不要为了照搬旧 README 强装 `flash-attn==2.7.4.post1`。

## 5. 搜索后端与密钥安全决策

最终经用户确认的组合是：

```text
Image Search：FVQA 官方缓存的 Google Lens 结果
Text Search：Serper.dev Google Search API
网页读取：Jina Reader
当前反馈：每页最多 12,000 字符的原始 Jina 正文
下一步：在 Jina 与 MMSearch-R1 之间加入本地 Qwen3-32B-FP8 摘要
```

不要混淆两个服务：

- `SERPER_API_KEY` 来自 `serper.dev`，用于当前 Text Search。
- `SERPAPI_API_KEY` 属于 `serpapi.com`；当前没有启用实时 SerpApi Google Lens。

真实配置位于：

```text
/root/autodl-tmp/multimodal-search-r1/reproduction/env/serper.env
```

权限为 `600 root:root`。只能在命令中安全 source：

```bash
set -a
. /root/autodl-tmp/multimodal-search-r1/reproduction/env/serper.env
set +a
```

不得 `cat`、`sed`、打印、复制或把 key 写进命令输出、JSON、日志、Markdown。示例文件均只有占位符：

- `reproduction/env/serper.env.example`
- `reproduction/env/serpapi.env.example`

大步 8 完成后，真实 key 对 Serper/Jina 缓存及三个正式输出的精确值扫描为 PASS。后续新增输出仍要重复泄漏扫描，但不得打印 key 本身。

## 6. 大步 1～8 的完成记录

### 大步 1：硬件与 Blackwell 基线

已确认 PRO6000、97,887 MiB 显存、`sm_120`、驱动和 PyTorch cu128 可用；BF16 GPU 矩阵乘通过。系统没有 `nvcc`，但不影响 Transformers + SDPA 推理。

### 大步 2：隔离推理环境

在数据盘创建 `/root/autodl-tmp/envs/mmsearch_infer`。安装过程中曾被 resolver 选中错误的 `torchaudio 2.11`，产生 ABI 不匹配；经用户确认后固定为与 torch 对齐的 `torchaudio 2.8.0+cu128`。最终 `pip check` 干净。

### 大步 3：官方仓库

官方仓库和 `verl` 子模块已拉取并固定到第 3 节列出的 commits。当前工作没有修改官方 tracked 文件，所有派生适配都在 `reproduction/`。

### 大步 4：模型、数据和缓存

下载并校验 MMSearch-R1-7B、FVQA 及官方 Image Search pickle；精确 revision、文件数、字节数和缓存覆盖率均已记录。所有大文件均位于数据盘。

样例提取脚本：`reproduction/scripts/extract_fvqa_samples.py`。

已有样例：

| 用途 | Data ID | 问题 | Ground Truth | 文件 |
|---|---|---|---|---|
| Search-Free | `fvqa_train_0` | What is the name of the system shown in the image? | `namus` | `/root/autodl-tmp/mmsearch_demo/search_free.png` |
| 初始 Search-Required | `fvqa_train_6` | What is the location of this building? | `Rubjerg Knude` | `/root/autodl-tmp/mmsearch_demo/search_required.png` |
| 稳定 Mixed Trigger | `fvqa_train_17` | Which historic county does this building belong to? | `Cheshire` | `/root/autodl-tmp/mmsearch_demo/search_required_trigger.png` |

`fvqa_train_17` 元数据：`/root/autodl-tmp/mmsearch_demo/search_required_trigger_meta.json`。

### 大步 5：Checkpoint 单轮 smoke

脚本：`reproduction/scripts/checkpoint_smoke.py`。  
正式输出：`/root/autodl-tmp/outputs/checkpoint_smoke_search_free.json`。

- 回答：`NamUs`
- 期望：`namus`
- Exact Match：true
- 输入/输出：561 / 68 tokens
- 模型加载：4.439 秒
- 生成：1.72 秒
- 峰值显存：16,031.31 MiB
- JSON SHA-256：`c82401859a69dfeb8d621cc763c68a8bf4d81d03ec15ef1aad630d64c24fb3e8`

### 大步 6：Placeholder 多轮控制流

官方 demo 的消息历史缺少“模型刚产生的 assistant 工具动作”，导致工具结果直接接在 user 消息后。派生脚本已修复顺序：先 append `assistant_message(response)`，再 append 工具返回的 user message。

脚本：

- `reproduction/scripts/placeholder_control_flow.py`
- `reproduction/scripts/placeholder_control_flow_scan.py`

初始 `fvqa_train_6` 会直接回答，没有稳定触发搜索。经用户确认，按 parquet 顺序扫描前 20 个 Search-Required 样本，第 5 个候选 `fvqa_train_17` 稳定触发：

```text
image_search → text_search → answer
```

正式输出：`/root/autodl-tmp/outputs/placeholder_control_flow_scanned.json`。

- `fvqa_train_0`：1 轮直接回答 `NamUs`，无搜索，EM=true。
- `fvqa_train_17`：3 轮，图搜 1 次、文本搜索 1 次，回答 `Cheshire`，EM=true。
- `all_control_flows_pass=true`
- 峰值显存：16,443.44 MiB
- JSON SHA-256：`29268b71b3f3cb560ad590eba1e6af57d0e43aef11513b54ab96aab0e53db4fb`

`/root/autodl-tmp/outputs/placeholder_control_flow.json` 是扫描前的失败现场，不是正式基准。

### 大步 7：真实 Image Search

用户选择对官方 FVQA 样本优先使用作者发布的 Google Lens 缓存。适配器：`reproduction/mmsearch_tools/cached_image_search.py`。

`fvqa_train_17` 的缓存请求 5 张缩略图，其中第 5 个旧 Google thumbnail URL 现在返回 HTTP 404；用户已明确接受 4/5 继续。其余四张图片和真实网页标题均按原排名送入模型。

正式证据：

- `/root/autodl-tmp/outputs/cached_image_search_smoke.json`
  - `pass=true`，4 张可用缩略图。
  - SHA-256：`9c2c0efc89ef2c11dc6e809ecc8ddbaf29ea398c594a6b7e6399b8da265e0f7d`
- `/root/autodl-tmp/outputs/cached_image_search_flow.json`
  - 动作：`image_search → text_search`
  - 生成 query：`Which historic county does the Lovell Telescope at Jodrell Bank Observatory belong to?`
  - `awaiting_real_text_search`、`final_answer=null` 是该阶段刻意停止点，不是失败。
  - 峰值显存：16,453.44 MiB。
  - SHA-256：`3206df65275efa8b4d69dc92e5eef58bdd64786e29a3a3d1c7180571b8bebf19`

本地缩略图：`/root/autodl-tmp/search_cache/fvqa_thumbnails/`。

### 大步 8：Serper.dev Text Search + Jina Reader

适配器和脚本：

- `reproduction/mmsearch_tools/serper_text_search.py`
- `reproduction/mmsearch_tools/jina_reader.py`
- `reproduction/mmsearch_tools/real_text_search.py`
- `reproduction/scripts/serper_jina_smoke.py`
- `reproduction/scripts/real_text_search_flow.py`

首次真实联网烟测：`/root/autodl-tmp/outputs/serper_jina_smoke.json`。

- 实际使用 1 次 Serper 查询。
- Organic results：5/5。
- Jina Reader：5/5，零失败。
- 单页上限：12,000 字符。
- Reader 正文合计：55,105 字符。
- 组合 `returned_text`：58,480 字符；SHA-256 为 `d971798b27f6d2244e02fd4f3f808caeb66dc9d881db2aa8bf9eaf3e84ae1f43`。
- JSON SHA-256：`bd3467b0c078aecb09ec9bb1ec0f23aa8d8f446b2dd4e0556996ff4bedde3929`。

首次缓存重放发现旧实现写盘时多加尾换行，导致两个未截断页面合计多 2 字符。经用户确认修复后，读取旧缓存会依据 metadata 恢复精确长度，新缓存不再追加字符。

正式重放：`/root/autodl-tmp/outputs/serper_jina_smoke_cache_replay_fixed.json`。

- Serper cache hit：true。
- Jina cache hits：5。
- Reader 正文：55,105 字符。
- 组合正文长度和 SHA-256 与首次联网完全一致。
- JSON SHA-256：`b46e52fb7434f89c4d85aa37fae31ca3e1fb6931d066f4cc825ef782e5c251f0`。

完整模型链路：`/root/autodl-tmp/outputs/real_search_flow_raw_jina.json`。

```text
image_search → text_search → answer
```

- Image Search：官方缓存 4 张图。
- 模型 query：`Which historic county does the Lovell Telescope at Jodrell Bank Observatory belong to?`
- Serper：5 条；完整链路命中已验证缓存，没有重复消耗查询。
- Jina：5 篇，零失败，5 个缓存命中。
- 三轮输入 tokens：955、1,656、20,299。
- 三轮输出 tokens：87、85、95。
- 三轮生成耗时：2.09、1.495、3.767 秒。
- 最终答案：`Cheshire`。
- Ground Truth：`Cheshire`。
- Exact Match / control flow：均为 true。
- 模型加载：4.443 秒。
- 峰值显存：23,131.25 MiB。
- JSON SHA-256：`105479b7ba284ad220020829284d364a4170189e90dceae16dea22beb43d7587`。

## 7. 大步 8 的可重复运行命令

该命令通常会命中现有 Serper/Jina 缓存；若模型生成不同 query，可能产生新的外部 API 调用。不要在没有检查额度和用户授权时批量重复执行。

```bash
cd /root/autodl-tmp/multimodal-search-r1

set -a
. reproduction/env/serper.env
set +a

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/root/autodl-tmp/multimodal-search-r1 \
/root/autodl-tmp/envs/mmsearch_infer/bin/python \
reproduction/scripts/real_text_search_flow.py \
  --model-path /root/autodl-tmp/models/MMSearch-R1-7B \
  --sample-meta /root/autodl-tmp/mmsearch_demo/search_required_trigger_meta.json \
  --cache-pickle /root/autodl-tmp/datasets/FVQA/fvqa_train_image_search_results_cache.pkl \
  --thumbnail-cache-dir /root/autodl-tmp/search_cache/fvqa_thumbnails \
  --serper-cache-dir /root/autodl-tmp/search_cache/serper/json \
  --jina-cache-dir /root/autodl-tmp/search_cache/jina \
  --output /root/autodl-tmp/outputs/real_search_flow_raw_jina.json \
  --top-k 5 \
  --max-chars-per-page 12000 \
  --max-new-tokens 512
```

## 8. 当前已知瑕疵与不要误判的文件

1. `reproduction/env/search_backend_decision.json` 中 Text Search 仍标为 deferred；这是过期元数据，不能覆盖大步 8 的正式证据。
2. 同一文件仍使用 `planned_backend` 字段；实际后端已是 Serper.dev。
3. `fvqa_image_search_cache_manifest.json` 的 `selected_samples` 只列 `train_0/train_6`，尚未补入后来扫描选中的 `train_17`。
4. `serper_jina_smoke_cache_replay.json` 是修复前的历史记录，55,107 字符；正式基准必须使用带 `_fixed` 的文件。
5. `reproduction/mmsearch_tools/jina_reader.py.orig` 是修复前备份，不会被 import；当前有效实现是 `jina_reader.py`。
6. 官方缓存 Image Search 的第 5 张缩略图 404 已获用户接受；不是新故障。
7. `cached_image_search_flow.json` 刻意停在真实 Text Search 之前；其中 `final_answer=null` 不是失败。
8. `placeholder_control_flow.json` 是初始样例不触发搜索的失败记录；正式控制流证据是 `_scanned.json`。
9. 原方案写 SerpAPI；实际经用户选择改为“官方 FVQA Lens 缓存 + Serper.dev Text Search + Jina Reader”。不要擅自改回 SerpApi。
10. 当前 Jina 反馈仍是截断原始正文，还没有 Qwen3 摘要；这正是大步 9 的起点。

## 9. 大步 9：Qwen3-32B-FP8 摘要器与同卡共驻

### 9.1 当前起点

以下项目目前都不存在：

- `/root/autodl-tmp/envs/qwen3_summary`
- `/root/autodl-tmp/models/Qwen3-32B-FP8`
- vLLM 服务
- Qwen3 摘要 smoke 输出
- 带摘要器的完整 Mixed Search trace

可从空闲 GPU 开始，但先执行第 2 节 preflight。

### 9.2 安全执行顺序

1. 从官方 Hugging Face 和现代 vLLM 文档核实当前 Blackwell/CUDA 12.8+ 兼容版本。
2. 先解析并记录 `Qwen/Qwen3-32B-FP8` 的精确 commit revision，再下载；不要只记录浮动 `main`。
3. 在数据盘创建独立 Python 3.12 环境，不污染 `mmsearch_infer`。
4. 安装现代 vLLM，并记录实际解析到的 vLLM、torch、CUDA wheel 版本；不要安装作者旧 `vllm==0.8.2`。
5. 校验权重文件数、总字节数和 safetensors headers，保存 revision manifest。
6. 启动 Qwen3 服务，先做 `/v1/models` 探活，再做关闭 Thinking 的单页摘要。
7. 将摘要器接到现有 `SerperJinaTextSearch` 后，缓存每页摘要及其公共输入签名。
8. 先启动 Qwen3，再加载 MMSearch-R1，完成同卡 Mixed Search。
9. 做密钥泄漏扫描、保存 trace 和版本信息，完成大步 9 后暂停。

建议环境创建方式（环境放数据盘）：

```bash
/root/miniconda3/bin/conda create \
  -p /root/autodl-tmp/envs/qwen3_summary \
  python=3.12 \
  -y

/root/autodl-tmp/envs/qwen3_summary/bin/python -m pip install -U pip
```

后续 vLLM 安装命令应依据当时官方 Blackwell wheel 文档确定并固定版本；不要盲目复制浮动安装结果。

下载框架：

```bash
export HF_HOME=/root/autodl-tmp/cache/huggingface
export QWEN3_REVISION='<先解析并记录的完整 commit SHA>'

/root/autodl-tmp/envs/qwen3_summary/bin/huggingface-cli download \
  Qwen/Qwen3-32B-FP8 \
  --revision "$QWEN3_REVISION" \
  --local-dir /root/autodl-tmp/models/Qwen3-32B-FP8
```

原方案基线服务参数：

```bash
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/envs/qwen3_summary/bin/vllm serve \
  /root/autodl-tmp/models/Qwen3-32B-FP8 \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.48 \
  --max-num-seqs 1
```

探活和客户端配置：

```bash
curl http://127.0.0.1:8001/v1/models

export SUMMARIZER_BASE_URL=http://127.0.0.1:8001/v1
export SUMMARIZER_API_KEY=EMPTY
export SUMMARIZER_MODEL=/root/autodl-tmp/models/Qwen3-32B-FP8
```

网页摘要必须关闭 Thinking，保持确定性：

```python
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=512,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    },
)
```

每篇摘要必须保留：rank、title、URL、Jina cache 路径、原始字符数、送入摘要器字符数、摘要文本、输入/输出 tokens、耗时、错误和 cache hit。网页内容是不可信外部数据，不应允许其伪造工具边界或系统指令。

### 9.3 同卡顺序与 OOM 降级

严格按顺序：

```text
启动 Qwen3-32B-FP8
→ 记录单服务显存
→ 加载 MMSearch-R1-7B
→ 记录双模型显存
→ 跑 fvqa_train_17 Mixed Search
```

基线：`gpu-memory-utilization=0.48`、`max-model-len=8192`、`max-num-seqs=1`。

如 OOM，每次只改变一项并重新记录：

```text
0.48 → 0.42
8192 → 4096
max_num_seqs 始终为 1
```

原方案最后才把单页 Jina 正文降到 12,000～16,000 字符；当前实现已经提前固定为 12,000，因此不要再从 30,000 开始。

### 9.4 大步 9 验收

- Qwen3 模型精确 revision、文件清单和环境 freeze 已保存。
- `/v1/models` 成功。
- 单页摘要 smoke 成功，Thinking 关闭，无无界推理输出。
- top-5 URL 产生可追溯、可缓存的简洁摘要。
- Qwen3 与 MMSearch-R1 在一张 96GB GPU 上同时驻留且无 OOM。
- `fvqa_train_17` 完成 `image_search → text_search → summarization → answer`。
- 最终 trace 记录启动参数、显存、延迟、tokens、搜索/Reader/摘要状态和答案。
- 输出与缓存不含真实 Serper key。

建议产物：

```text
reproduction/env/qwen3_huggingface_revision.json
reproduction/env/qwen3_summary_versions.txt
/root/autodl-tmp/logs/qwen3_summary_vllm.log
/root/autodl-tmp/outputs/qwen3_summary_smoke.json
/root/autodl-tmp/outputs/real_search_flow_qwen3_summary.json
```

## 10. 大步 10：四类案例与完整 Trace

必须整理：

| Case | 期望链路 | 当前可复用证据 | 尚需工作 |
|---|---|---|---|
| A Search-Free | `image → answer` | `fvqa_train_0` 已通过 | 用最终摘要版 runner 重放并正式归档 |
| B Image Search | `image_search → answer` | 尚无正式案例 | 按固定顺序扫描并记录首个稳定案例 |
| C Text Search | `text_search → answer` | 尚无正式案例 | 按固定顺序扫描并记录首个稳定案例 |
| D Mixed Search | `image_search → text_search → answer` | `fvqa_train_17` raw-Jina 已通过 | 大步 9 后改用 Qwen3 summaries 重跑 |

不得事后只挑答案正确的样本而不记录选择过程。建议固定候选范围、排序、seed 和判定规则，把扫描清单与最终 data IDs 写入 manifest。

每条正式 trace 至少包含：

- data ID、split、category、图像和 SHA-256；
- question、candidate answers、Ground Truth；
- 模型及 revision、解码参数；
- 每轮原始 response、解析 action、tokens 和耗时；
- Image Search 来源、标题、缩略图、失败；
- Serper query、URLs、cache hit；
- Jina 状态、字符数、失败；
- Qwen3 summaries、Thinking 设置、摘要 tokens/耗时；
- 最终答案、Exact Match、搜索次数、总轮次；
- GPU 峰值、降级参数和错误。

还要整理至少一个 Failure Case，并明确失败发生在：视觉实体识别、query、搜索结果、网页读取、摘要或最终回答中的哪一层。

大步 10 验收：四类正式 JSON trace、一个 Failure Case、汇总 `cases.md` 均完成，然后暂停。

## 11. 大步 11：分阶段批量评测

禁止直接跑 1,800 条。固定评测 ID manifest 后，按以下顺序扩容：

```text
5 → 20 → 50 → 100 → 300
```

每一档完成后检查并记录：有效/失败/跳过数、Exact Match、搜索次数、Serper 实际调用数、Jina/Qwen3 成功率、缓存命中率、平均耗时、显存和剩余额度；确认后再扩大。

批量 runner 必须：

- 支持断点续跑和按 data ID 去重；
- 每条完成后原子落盘，不能等整批结束才保存；
- 单样本异常隔离并分类错误；
- 优先复用 Serper、Jina、Qwen3 摘要缓存；
- 限制搜索次数、重试次数和超时；
- 记录实际联网次数，不能把逻辑 tool call 与网络 cache miss 混为一谈；
- 保存固定样本 IDs、seed、split 和类别来源。

主指标保持原方案定义：

```python
exact_match = pred.strip().lower() == gt.strip().lower()
total_search_calls = image_search_calls + text_search_calls
search_ratio = total_search_calls / (N * 2) * 100
average_turns = total_turns / N
```

如果 runner 允许每条超过两次有效搜索，必须另外说明 SR 分母和截断规则，不能直接套公式。

必须报告：

- 总体 EM；
- `image_search_calls`、`text_search_calls`、`total_search_calls`；
- Search Ratio；
- Average Turns；
- Search-Free / Search-Required Accuracy（仅在有官方 category 的 train 分层集上）；
- 搜索、Jina、摘要阶段的成功率与失败原因；
- cache hit、实际 API 请求和时延。

原方案论文值只作 sanity check，不要求逐点相同：

```text
MMSearch-R1-7B：Average Acc ≈ 54.6，Average SR ≈ 67.1
FVQA-test：Acc ≈ 58.4，SR ≈ 66.8
```

网络排名、网页、Jina、Qwen3 配置均会造成差异。真正的行为验收是 Search-Free 更少搜索、Search-Required 更倾向搜索、搜索后答案改善，以及 Mixed Search 链路合理。

最低产物：

```text
eval_subsets.json
predictions.jsonl
metrics.json
failure_summary.json
```

原方案必须达到至少 50～100 条，推荐在额度和稳定性允许时达到 100～300 条。

## 12. 大步 12：Base 对比与最终报告

建议下载 `Qwen/Qwen2.5-VL-7B-Instruct`，但必须在实施时固定实际 HF revision、文件数和字节数。对完全相同的 20～50 条样本比较：

```text
Base Direct Answer
vs
MMSearch-R1 On-demand Search
```

公平性要求：同一图像、question、Ground Truth、图像处理、解码参数和 EM 实现。Base 不使用搜索；MMSearch-R1 使用最终已验证搜索链路。不要把 Base 的零搜索与工具失败混为一类。

比较：Accuracy、Search Calls、延迟、代表性成功案例和失败案例。

最终至少整理：

```text
predictions.jsonl
metrics.json
cases.md
pip_freeze.txt
gpu_info.txt
git_commit.txt
huggingface_revisions.json
failure_analysis.md
```

报告必须区分：

- 原始论文/方案配置；
- 当前 Blackwell 适配；
- 官方 FVQA Image Search cache；
- Serper.dev 实际 Text Search；
- Jina raw 页面与 Qwen3 summaries；
- 缓存重放与真实联网调用；
- 严格 EM 与任何可选 Judge 指标。

官方 `run_mmsearch_r1_grpo.sh`、veRL、旧 `vllm==0.8.2` 和完整 GRPO 训练不属于大步 9～12 主线。主实验完成且预算允许时，只能另立为“Blackwell compatibility adaptation / GRPO pipeline smoke test”，不能称为论文完整训练复现。

## 13. 正式证据索引

| 证据 | 路径 | 用途 |
|---|---|---|
| 原方案 | `/root/MMSearch-R1_PRO6000-96G_5.98元每小时_复现方案.md` | 总体设计与原始验收 |
| Git commits | `reproduction/env/git_commit.txt`、`git_submodules.txt` | 代码版本 |
| HF revisions | `reproduction/env/huggingface_revisions.json` | 模型/数据版本 |
| FVQA manifest | `reproduction/env/fvqa_manifest.json` | 行数与类别 |
| Image cache manifest | `reproduction/env/fvqa_image_search_cache_manifest.json` | 缓存覆盖与缺失 IDs |
| 单轮 checkpoint | `/root/autodl-tmp/outputs/checkpoint_smoke_search_free.json` | 模型加载和直接回答 |
| 正式 placeholder | `/root/autodl-tmp/outputs/placeholder_control_flow_scanned.json` | Search-Free/Required 控制流 |
| Image Search smoke | `/root/autodl-tmp/outputs/cached_image_search_smoke.json` | 官方 cache 4/5 |
| Image Search flow | `/root/autodl-tmp/outputs/cached_image_search_flow.json` | 真实标题、缩略图、query |
| 首次真实 Text Search | `/root/autodl-tmp/outputs/serper_jina_smoke.json` | Serper/Jina 网络证据 |
| 修复后缓存重放 | `/root/autodl-tmp/outputs/serper_jina_smoke_cache_replay_fixed.json` | 确定性缓存证据 |
| 完整 raw-Jina trace | `/root/autodl-tmp/outputs/real_search_flow_raw_jina.json` | 大步 8 最终验收 |

## 14. 交接终点

当前服务器没有 Qwen3 环境、Qwen3 权重或 vLLM 进程；GPU 空闲，数据盘约有 224GB 可用。新对话无需重做大步 1～8，应从“大步 9 的只读 preflight、精确 revision 固定、独立环境创建”开始。

完成大步 9 后必须暂停，让用户决定是否进入四类案例整理；之后依次完成大步 10、11、12，每个大步分别验收和暂停。
