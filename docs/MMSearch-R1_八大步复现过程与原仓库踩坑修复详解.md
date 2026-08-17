# MMSearch-R1 八大步复现过程与原仓库踩坑修复详解

> 记录时间：2026-08-16（UTC）  
> 服务器：NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB  
> 原方案：`/root/MMSearch-R1_PRO6000-96G_5.98元每小时_复现方案.md`  
> 当前交接：`/root/MMSearch-R1_复现实验进度记录_大步1-8完成.md`  
> 仓库：`/root/autodl-tmp/multimodal-search-r1`  
> 本文目标：逐步还原大步 1～8 是怎样复现的，并解释原仓库、环境、数据和外部服务中的坑如何被发现、定位、修复和验收。

## 0. 先说明“复现”到底复现了什么

这八步完成的是单卡 Blackwell 上的开源权重系统复现：

```text
MMSearch-R1-7B checkpoint
→ 按需判断是否搜索
→ 真实 Image Search 证据
→ 真实 Text Search
→ Jina Reader 网页正文
→ 多轮回答
```

最终正式链路是：

```text
image_search → text_search → answer
```

最终样例回答 `Cheshire`，与 Ground Truth 严格匹配；三轮控制流、真实 Serper、Jina 5/5、缓存确定性和密钥泄漏检查均通过。

这八步没有声称完成完整 GRPO 训练。原因不是漏做，而是原仓库训练脚本依赖旧 `vllm==0.8.2`、旧 FlashAttention、8 GPU 和 veRL 训练栈；当前主目标是先在单张 PRO6000 上可靠复现发布 checkpoint、工具协议和真实搜索系统。训练兼容改造属于主实验完成后的可选项。

本文中的命令依据当前脚本参数、Conda history、正式 JSON 和固定 revision 重建。没有读取 shell history，以免历史命令中出现密钥；因此本文优先给出“当前可重复执行的等价命令”。

## 1. 最重要的结论：官方仓库不是开箱即用的真实搜索系统

官方 README 已经明确写明，使用者需要自行实现 search tool pipeline。仓库里的两个工具是 placeholder：

- [image_search.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/utils/tools/image_search.py:5) 固定生成 3 张 64×64 纯色图片和假标题。
- [text_search.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/utils/tools/text_search.py:1) 固定返回 3 条 `Summary of webpage content...`。
- 两者不管输入内容是什么都默认 `success=True`。

所以：

```text
直接运行官方 demo
≠ 复现真实 Image Search
≠ 复现真实 Text Search
≠ 复现论文中的网页摘要链路
```

本次真正的工程工作集中在四件事：

1. 另建适配 Blackwell 的 checkpoint 推理环境，绕开作者的旧训练栈。
2. 在 `reproduction/` 派生 runner 中修复多轮消息历史和真实标题传递。
3. 在 `reproduction/` 中把 placeholder 图搜换成官方 FVQA Google Lens cache。
4. 在 `reproduction/` 中实现带安全缓存的 Serper.dev + Jina Reader Text Search。

> 重要边界：官方 tracked 文件没有被修改。`inference_torch_demo.py` 仍保留原消息历史 bug，`utils/tools/image_search.py` 与 `text_search.py` 仍是 placeholder；本文中的“修复”均指未跟踪的 `reproduction/` runner/adapter 对这些问题的绕开或替代。这样既保留上游原貌，也能逐项审计适配代码。

## 2. 坑点总览

| ID | 坑点 | 性质 | 当前处理 |
|---|---|---|---|
| P01 | 根盘只有 30GB，模型/环境会写满根盘 | 服务器 | 全部大文件迁到 `/root/autodl-tmp` |
| P02 | README 的 Python 3.10 + vLLM 0.8.2 +旧 flash-attn 不适配 `sm_120` | 原仓库环境 | 主线改用 PyTorch 2.8 cu128 + BF16 + SDPA |
| P03 | README 没有完整 inference requirements/lock | 原仓库依赖 | 建独立环境并固定核心版本 |
| P04 | resolver 安装 torchaudio 2.11，与 torch 2.8 ABI 冲突 | 实际故障 | 三件套固定为 2.8/0.23/2.8 cu128 |
| P05 | 主项目无 setup/pyproject，也无传统 `__init__.py` | 原仓库打包 | 仓库根目录运行并显式 `PYTHONPATH` |
| P06 | prompt 路径是相对路径 | 原 demo | 固定 `cd` 到仓库根目录 |
| P07 | 自带 2k InfoSeek parquet 只是 134 字节 LFS 指针 | 原仓库数据 | 不当正式数据，使用固定 revision FVQA |
| P08 | 浮动 HF `main` 无法严格复现 | 资产管理 | 固定 model/data commit、文件数和字节数 |
| P09 | FVQA test 实际没有 `category` | 数据发布 | train 做分层，test 只做总体 |
| P10 | FVQA `image_urls` train/test 全为空 | 数据/rollout | 按 `data_id` 使用官方 Lens cache |
| P11 | 官方 Image cache 有缺 ID、结果不足 5、类型混合 | 数据缓存 | manifest、PIL/URL 双支持、逐条失败隔离 |
| P12 | demo 的 `torch_dtype="auto"` 会读取 config 的 float32 | 原 demo | 显式 BF16，峰值从设计上受控 |
| P13 | demo 未固定 attention/processor/offline 行为 | 原 demo | SDPA、`use_fast=False`、local only |
| P14 | data URI 写死 PNG，真实字节可能是 JPEG | 原 demo | 样例转 PNG；PIL 结果按真实 MIME 编码 |
| P15 | 官方 search tools 全是假实现 | 原仓库功能缺口 | 大步 6 只验协议；大步 7/8 换真实后端 |
| P16 | demo 丢弃真实网页标题，改喂 `Webpage Title N` | 原 demo 实质 bug | 从 `tool_stat["titles"]` 取真实标题 |
| P17 | Text Search 前遗漏 assistant 工具动作 | 原 demo 最大 bug | 严格保持 user/assistant/user 交替历史 |
| P18 | demo 是脆弱分支，不是状态机；regex/trace 不完整 | 原 demo | 通用 action parser、轮次/工具上限、JSON trace |
| P19 | `search_required` 样例仍可能直接回答 | 模型行为 | 固定顺序扫描，保留失败现场，选首个自然触发 |
| P20 | SerpApi 与 Serper.dev 是不同服务 | 外部服务 | 不混用 key；Lens cache 做图搜，Serper 做文搜 |
| P21 | 真实 key 曾放在 0644 `.example` | 安全 | 整体迁移到 0600 私有 env，模板恢复占位符 |
| P22 | 实时 Lens 要公网图片，而 FVQA 只有嵌入 bytes/空 URL | 数据/API | 使用作者按 data ID 发布的 Lens 结果 |
| P23 | 官方缓存的旧 thumbnail URL 已 404 | 外部时效性 | 用户接受 4/5，保存四图，透明记录失败 |
| P24 | 官方没有真实 Text Search、超时、缓存或长度控制 | 原仓库功能缺口 | 实现 Serper + Jina + 12k/page + 状态记录 |
| P25 | 自建 Jina cache 首次重放多 2 字符 | 派生实现故障 | 精确写盘 + metadata 兼容旧缓存 |
| P26 | 官方 GRPO 脚本硬编码 8 GPU、W&B、30 epochs | 原训练脚本 | 不作为单卡 checkpoint 主线 |
| P27 | 默认 YAML 有 `use_remove_padding: Falseva` | 原训练配置 | 已识别；训练路径尚未改 tracked 文件 |
| P28 | 正式 rollout 靠固定 sentinel 判断工具类型 | 原 rollout | 当前 Transformers runner可用；训练接入前仍需修 |
| P29 | rollout 用固定英文字符串反切原问题 | 原 rollout | 当前直接保存结构化 question；训练路径待修 |
| P30 | rollout 硬编码 Qwen token IDs | 原 rollout | 当前 checkpoint适用；模型迁移前待修 |
| P31 | raw Jina 第三轮达 20,299 tokens，超官方训练长度 | 系统限制 | 大步 8 只验链路；大步 9 必须摘要 |
| P32 | Codex 沙箱 bwrap 被服务器内核禁用 | 基础设施 | 经批准用 escalated 命令；不是 CUDA 故障 |
| P33 | `reproduction/` 未被 Git 跟踪，部分 manifest 状态过期 | 交接风险 | 禁止 `git clean`；正式 JSON 优先 |

下文按真实执行顺序详细说明。

## 3. 大步 1：硬件、磁盘和 Blackwell 基线

### 3.1 目标

先回答四个问题：

1. GPU 是否真是 96GB Blackwell？
2. 驱动能否支持 CUDA 12.8 wheel？
3. 根盘是否足够？
4. 是否应该照搬 README 的旧 vLLM/flash-attn？

### 3.2 实际检查

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv
df -h / /root/autodl-tmp
free -h
```

结果：

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition。
- 显存：97,887 MiB。
- Compute Capability：`(12, 0)`，即 `sm_120`。
- 驱动：595.58.03；`nvidia-smi` 报告 CUDA 13.2。
- 根盘仅 30GB；数据盘 250GB。
- 内存约 1 TiB；无 swap。
- 系统没有 `nvcc`。

驱动报告 CUDA 13.2 与 PyTorch wheel 的 cu128 不冲突：前者是驱动上限，后者是应用自带 runtime；新驱动可以运行较旧的 12.8 runtime。

### 3.3 坑 P01：根盘容量不够

MMSearch 权重约 16GB，推理环境约 7.3GB，后续 Qwen3 和 vLLM 更大。若默认把 Conda、pip、HF cache 和模型都写根盘，很容易在下载中途耗尽 30GB 根盘，留下不完整文件。

处理：一开始统一创建数据盘目录：

```bash
mkdir -p \
  /root/autodl-tmp/cache/conda \
  /root/autodl-tmp/cache/pip \
  /root/autodl-tmp/cache/huggingface \
  /root/autodl-tmp/envs \
  /root/autodl-tmp/models \
  /root/autodl-tmp/datasets \
  /root/autodl-tmp/outputs \
  /root/autodl-tmp/search_cache

export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export HF_HOME=/root/autodl-tmp/cache/huggingface
export CONDA_PKGS_DIRS=/root/autodl-tmp/cache/conda/pkgs
```

### 3.4 坑 P02：README 旧训练栈不能直接照搬

官方 [README.md](/root/autodl-tmp/multimodal-search-r1/README.md:35) 建议：

```text
Python 3.10
vllm==0.8.2
transformers==4.51.0
flash-attn==2.7.4.post1
```

这套组合服务于作者原训练环境，不是 Blackwell `sm_120` 的稳妥起点。当前又没有 `nvcc`，强行编译旧 FlashAttention 很可能在 CUDA arch、Triton 或扩展编译阶段失败。

处理不是“等报错再说”，而是预防性拆分目标：

- 大步 1～8 只建 Transformers checkpoint inference 环境。
- Torch 使用官方 cu128 wheel。
- attention 明确选 PyTorch SDPA。
- 不安装旧 vLLM、FlashAttention、Ray 或 veRL 训练依赖。
- 训练兼容实验延后，不让它阻塞真实搜索复现。

### 3.5 基线验收

在环境建好后执行：

```bash
/root/autodl-tmp/envs/mmsearch_infer/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list()
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(y.dtype, y.shape)
PY
```

确认 torch 2.8.0+cu128、CUDA available、`sm_120` 和 BF16 matmul 均通过。

## 4. 大步 2：创建隔离推理环境

### 4.1 真实创建命令

Conda history 记录的原始命令是：

```bash
export CONDA_PKGS_DIRS=/root/autodl-tmp/cache/conda/pkgs

/root/miniconda3/bin/conda create \
  --prefix /root/autodl-tmp/envs/mmsearch_infer \
  python=3.11 \
  pip \
  -y
```

随后用该环境的绝对 Python 安装 Blackwell wheel：

```bash
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip

/root/autodl-tmp/envs/mmsearch_infer/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 \
  torchvision==0.23.0+cu128 \
  torchaudio==2.8.0+cu128
```

再安装推理和数据依赖：

```bash
/root/autodl-tmp/envs/mmsearch_infer/bin/python -m pip install \
  transformers==4.51.0 \
  accelerate==1.14.0 \
  qwen-vl-utils==0.0.14 \
  pillow==11.1.0 \
  datasets==5.0.1 \
  huggingface-hub==0.36.2 \
  openai==3.1.0 \
  pandas==3.0.5 \
  pyarrow==25.0.1 \
  requests==2.34.2 \
  safetensors==0.8.0
```

实际环境中的 NumPy 为 2.4.6。

### 4.2 坑 P03：README 不是完整的 inference requirements

README 只明确安装 veRL、vLLM、Transformers 和 FlashAttention，但官方 demo 还直接 import：

- `qwen_vl_utils`
- Pillow/NumPy
- Requests
- 使用 `device_map="auto"` 时需要 Accelerate

仓库根目录没有完整的 inference requirements lock。若只装 README 列出的包，单独运行 Torch demo 可能先遇到缺包，而不是模型问题。

修复：为推理主线独立列出并验证所需依赖；训练依赖不混进来。

### 4.3 坑 P04：torchaudio 2.11 ABI 冲突

初次 resolver 选择了 torchaudio 2.11，而 torch 是 2.8.0+cu128，导入时产生二进制 ABI 不匹配。根因是 PyTorch 三件套没有作为一个 release 原子固定。

经用户确认后的修复原则：

```text
torch 2.8.0+cu128
torchvision 0.23.0+cu128
torchaudio 2.8.0+cu128
```

若单独修复现有环境，可使用同一 cu128 index，并避免依赖 resolver 再次升级 torch：

```bash
/root/autodl-tmp/envs/mmsearch_infer/bin/python -m pip install \
  --force-reinstall \
  --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  torchaudio==2.8.0+cu128
```

验收：

```bash
/root/autodl-tmp/envs/mmsearch_infer/bin/python -m pip check
/root/autodl-tmp/envs/mmsearch_infer/bin/python -c \
'import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)'
```

最终 `pip check` 为 `No broken requirements found`。

## 5. 大步 3：拉取仓库并固定代码

### 5.1 Clone

```bash
git clone --recurse-submodules \
  https://github.com/EvolvingLMMs-Lab/multimodal-search-r1.git \
  /root/autodl-tmp/multimodal-search-r1
```

Clone 后必须显式 checkout 本次已验证的主仓库 commit，再让子模块回到 superproject 记录的精确位置；不能只记录当时碰巧拿到的浮动 `main`：

```bash
git -C /root/autodl-tmp/multimodal-search-r1 \
  checkout 3727c76dda2fc0ea9df16a9f770946d039e6f09b

git -C /root/autodl-tmp/multimodal-search-r1 \
  submodule update --init --recursive

test "$(git -C /root/autodl-tmp/multimodal-search-r1 rev-parse HEAD)" = \
  "3727c76dda2fc0ea9df16a9f770946d039e6f09b"

test "$(git -C /root/autodl-tmp/multimodal-search-r1/verl rev-parse HEAD)" = \
  "8e9e73723fd1cc729bedb3bbcf915060afbda91d"

git -C /root/autodl-tmp/multimodal-search-r1 rev-parse HEAD
git -C /root/autodl-tmp/multimodal-search-r1 submodule status --recursive
```

本次固定：

- 主仓库：`3727c76dda2fc0ea9df16a9f770946d039e6f09b`
- `verl`：`8e9e73723fd1cc729bedb3bbcf915060afbda91d`

记录文件：`reproduction/env/git_commit.txt` 和 `git_submodules.txt`。

### 5.2 坑 P05/P06：主项目不能安装，路径又依赖 CWD

仓库根目录没有 `setup.py`、`pyproject.toml` 或 `setup.cfg`；`mmsearch_r1/` 也没有传统 `__init__.py`。README 只做 `pip install -e ./verl`，没有安装主项目。

实际运行派生脚本时曾出现：

```text
ModuleNotFoundError: No module named 'mmsearch_r1'
```

同时，官方 demo 在 [inference_torch_demo.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/scripts/inference_torch_demo.py:102) 用相对路径读取 pickle prompt；即使 import 解决，换一个工作目录也会找不到 prompt。

统一修复：

```bash
cd /root/autodl-tmp/multimodal-search-r1
export PYTHONPATH=/root/autodl-tmp/multimodal-search-r1
```

后续所有命令都遵循这两项，不依赖偶然的 `sys.path`。

### 5.3 坑 P07：仓库自带 2k parquet 是 LFS 指针

`mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet` 实际只有 134 字节，内容以：

```text
version https://git-lfs.github.com/spec/v1
```

开头。PyArrow 会报：

```text
ArrowInvalid: Parquet magic bytes not found
```

原因是 `--recurse-submodules` 只处理子模块，不保证主仓库 Git LFS 大对象已经 materialize。仓库的 `mini_data.pq` 是可读的 5 行样例，但只能参考 schema。

本次不把 LFS 指针当损坏的正式数据继续折腾，而是使用作者在 Hugging Face 发布、带官方 image cache 的 FVQA 固定 revision。

### 5.4 为什么不直接跑 `run_mmsearch_r1_grpo.sh`

官方脚本 [run_mmsearch_r1_grpo.sh](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/scripts/run_mmsearch_r1_grpo.sh:3) 存在以下前提：

- `cd multimodal-search-r1;`：README 已让用户进入仓库，再运行会 cd 失败；因为没有 `set -e`，脚本仍继续。
- 依赖外部 `$TRAIN_DATA_PATH`、`$VAL_DATA_PATH`，没有前置校验。
- 默认模型是 Base `Qwen/Qwen2.5-VL-7B-Instruct`，不是发布 checkpoint。
- 强制 W&B。
- `trainer.n_gpus_per_node=8`。
- 默认训练 30 epochs。

这不是单卡 checkpoint smoke 命令。八步主线因此使用 Transformers runner，而不是伪装成“原脚本已在单卡训练通过”。

## 6. 大步 4：下载、固定并验证模型/FVQA

### 6.1 坑 P08：不能只下载浮动 main

同一个 HF repo 的 `main` 可能变化。正确顺序是先查询 commit SHA，再把 SHA 写进下载命令和 manifest。

本次固定：

```text
lmms-lab/MMSearch-R1-7B
revision = 3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46

lmms-lab/FVQA
revision = bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5
```

可重复下载代码：

```bash
export HF_HOME=/root/autodl-tmp/cache/huggingface

/root/autodl-tmp/envs/mmsearch_infer/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lmms-lab/MMSearch-R1-7B",
    revision="3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46",
    local_dir="/root/autodl-tmp/models/MMSearch-R1-7B",
)

snapshot_download(
    repo_id="lmms-lab/FVQA",
    repo_type="dataset",
    revision="bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5",
    local_dir="/root/autodl-tmp/datasets/FVQA",
)
PY
```

结果：

| 资产 | 文件数 | 精确字节数 | 本地路径 |
|---|---:|---:|---|
| MMSearch-R1-7B | 17 | 16,600,357,342 | `/root/autodl-tmp/models/MMSearch-R1-7B` |
| FVQA | 6 | 2,336,555,807 | `/root/autodl-tmp/datasets/FVQA` |

模型有 4 个 safetensors shards。用 `safe_open` 遍历全部 tensor headers，729 个 tensor 均为 BF16；没有 `.incomplete` 文件。记录见 `reproduction/env/huggingface_revisions.json`。

下载后不是看目录“像是有文件”就结束，而是运行下面的完整性检查。公共文件计数必须排除 Hugging Face 在 `local_dir/.cache` 中留下的 metadata 和 lock 文件：

```bash
/root/autodl-tmp/envs/mmsearch_infer/bin/python - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open

assets = [
    (
        Path("/root/autodl-tmp/models/MMSearch-R1-7B"),
        17,
        16_600_357_342,
        "3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46",
    ),
    (
        Path("/root/autodl-tmp/datasets/FVQA"),
        6,
        2_336_555_807,
        "bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5",
    ),
]

for root, expected_count, expected_bytes, expected_revision in assets:
    public_files = [
        path for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    ]
    actual_bytes = sum(path.stat().st_size for path in public_files)
    incomplete = list(root.rglob("*.incomplete"))
    metadata = list((root / ".cache/huggingface/download").rglob("*.metadata"))
    revisions = {
        path.read_text(encoding="utf-8").splitlines()[0]
        for path in metadata
    }

    print(root, len(public_files), actual_bytes, revisions)
    assert len(public_files) == expected_count
    assert actual_bytes == expected_bytes
    assert not incomplete
    assert metadata
    assert revisions == {expected_revision}

model_root = Path("/root/autodl-tmp/models/MMSearch-R1-7B")
index = json.loads(
    (model_root / "model.safetensors.index.json").read_text(encoding="utf-8")
)
weight_map = index["weight_map"]
shards = sorted(set(weight_map.values()))
header_keys = set()
dtypes = set()

for shard in shards:
    with safe_open(model_root / shard, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        header_keys.update(keys)
        dtypes.update(str(handle.get_slice(key).get_dtype()) for key in keys)

print("shards:", shards)
print("index tensors:", len(weight_map))
print("header tensors:", len(header_keys))
print("dtypes:", dtypes)
assert len(shards) == 4
assert len(weight_map) == 729
assert header_keys == set(weight_map)
assert dtypes == {"BF16"}
PY
```

这组断言把 revision、文件数、精确字节数、不完整下载、shard 数、索引键与 header dtype 连成同一条证据链；任一资产漂移都会非零退出。

### 6.2 坑 P12 的伏笔：config 写 float32，但权重实际 BF16

本地 `config.json` 顶层 `torch_dtype` 是 `float32`，而 729 个 safetensors tensor 实际都是 BF16。官方 demo 的 `torch_dtype="auto"` 会依赖 config，可能把 BF16 权重扩成 FP32，显存接近翻倍。

因此后续所有派生脚本都显式：

```python
torch_dtype=torch.bfloat16
```

不是随意量化，也不是牺牲精度，而是与实际权重 dtype 对齐。

### 6.3 检查 FVQA schema

Train 4,856 行，包含：`prompt`、嵌入图像 bytes、`reward_model`、`data_id`、`category`。Test 1,800 行，但没有 `category`。

### 6.4 坑 P09：test 没有 category

若按数据卡或直觉直接在 test 读取 `category`，会列不存在。处理：

- train 用于 Search-Free/Search-Required 控制样例及分层评测。
- test 只用于总体评测。
- 不能自行推断类别后冒充官方标签。

记录：`reproduction/env/fvqa_manifest.json`。

### 6.5 坑 P10/P22：image_urls 全为空

实际统计：

```text
train image_urls null = 4,856 / 4,856
test  image_urls null = 1,800 / 1,800
```

而正式 rollout 在 [vllm_rollout_spmd.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/workers/multimodal/rollout/vllm_rollout_spmd.py:228) 会直接把 `image_urls` 交给图搜。真实 Google Lens 后端通常要求公网 HTTP(S) URL，本地路径和 null 都不能直接用。

处理：对 FVQA 不走实时上传，而是按 data ID 使用作者发布的 Google Lens cache。

### 6.6 坑 P11：官方缓存不能假设“每条固定 5 个 URL”

只读统计：

- Train 覆盖 4,849 / 4,856，缺 7 条。
- Test 覆盖 1,798 / 1,800，缺 2 条。
- Train 只有 4,730 条恰好 5 个结果，其余有 1～4 个。
- Test 只有 1,747 条恰好 5 个结果，其余有 1～4 个。
- 缩略图来源既可能是 URL 字符串，也可能是 pickle 内的 PIL JPEG/PNG 对象。

缺失 ID 必须显式保留，后续批量评测才能确定性跳过或报错：

```text
train:
fvqa_train_72, fvqa_train_279, fvqa_train_332,
fvqa_train_489, fvqa_train_738, fvqa_train_4345,
fvqa_train_4633

test:
fvqa_test_1470, fvqa_test_1753
```

处理：固定官方建议的 Pillow 11.1.0；manifest 记录缺 ID；适配器同时支持 URL 和 PIL；结果数量取 title/source/top_k 的最小值；逐条隔离错误。

只对固定、可信的官方 revision 使用 `pickle.load`。任意来源 pickle 可以执行恶意代码，不能照搬。

### 6.7 确定性提取样例

```bash
cd /root/autodl-tmp/multimodal-search-r1
PYTHONPATH=/root/autodl-tmp/multimodal-search-r1 \
/root/autodl-tmp/envs/mmsearch_infer/bin/python \
reproduction/scripts/extract_fvqa_samples.py \
  --parquet /root/autodl-tmp/datasets/FVQA/fvqa_train.parquet \
  --output-dir /root/autodl-tmp/mmsearch_demo \
  --dataset-revision bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5
```

选择规则是 parquet 顺序中每类第一条，不人工挑答案：

- `fvqa_train_0`：Search-Free，GT `namus`。
- `fvqa_train_6`：Search-Required，GT `Rubjerg Knude`。

原始嵌入 bytes 统一转 RGB PNG，并记录原始 bytes SHA-256、宽高、source row 和 revision。

## 7. 大步 5：Checkpoint 单轮 smoke

### 7.1 为什么先做最小闭环

在加入多轮和网络之前，先隔离验证四层：模型 shard、processor、图片输入、输出格式。如果这一步不过，后续任何“搜索失败”都无法定位。

### 7.2 官方三轮 prompt 与消息协议

八步 runner 没有自行发明动作格式，而是从固定 Git commit 中加载三份官方 prompt：

- `mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl`
- `mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl`
- `mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl`

三份文件都通过 `pickle.load` 读取，所以只应加载已经固定、可信的仓库 commit；它们又是相对路径，因此运行前必须 `cd` 到仓库根目录。

实际协议如下：

1. 首轮 prompt 允许 Image Search、Text Search 或最终 Answer。其末尾原有字面 `<image>`；派生 runner 先执行 `.replace("<image>", "").strip()`，再用 Qwen-VL 的结构化 image content item 注入真实图片，避免同一张图出现两个视觉占位符。
2. Image Search 返回后追加 `after_image_search_prompt_qwenvl.pkl`；这一轮只允许继续 Text Search 或直接 Answer。
3. Text Search 返回后追加 `after_text_search_prompt_qwenvl.pkl`；这一轮只允许 Answer，证据仍不足时输出官方 warning：`Unable to answer due to lack of relevant information`。
4. 每次工具返回都作为新的 user message，正文包在 `<information>...</information>` 中，并同时附回原问题和对应的 after-search prompt。
5. 每个 assistant action 都要求先给 `<reason>...</reason>`；最终答案必须位于 `<answer>...</answer>`。

这套 prompt、结构化图片、assistant action 和 user tool result 的组合，才是后续三轮控制流的完整输入协议。

### 7.3 官方 demo 的加载坑

官方 [inference_torch_demo.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/scripts/inference_torch_demo.py:28) 使用：

```python
torch_dtype="auto"
device_map="auto"
AutoProcessor.from_pretrained(model_path)
```

它没有固定 attention、offline 或 processor 类型，也只打印文本、不保存机器可验收结果。

派生 `checkpoint_smoke.py` 固定：

```text
torch.manual_seed(0)
torch.bfloat16
attn_implementation="sdpa"
local_files_only=True
low_cpu_mem_usage=True
use_fast=False
from_pretrained() 默认返回 eval 状态
torch.inference_mode()
do_sample=False
max_new_tokens=512
```

### 7.4 坑 P14：图片 MIME

官方 demo 无论原字节真实格式，都写成 `data:image/png`；搜索返回即使按 JPEG 编码，也仍标成 PNG。派生流程先把选定样例规范化成真正的 PNG；对 PIL 搜索结果则按实际格式生成 MIME，避免依赖 decoder 宽容行为。

### 7.5 复跑命令

```bash
cd /root/autodl-tmp/multimodal-search-r1

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/root/autodl-tmp/multimodal-search-r1 \
/root/autodl-tmp/envs/mmsearch_infer/bin/python \
reproduction/scripts/checkpoint_smoke.py \
  --model-path /root/autodl-tmp/models/MMSearch-R1-7B \
  --image /root/autodl-tmp/mmsearch_demo/search_free.png \
  --question 'What is the name of the system shown in the image?' \
  --expected-answer namus \
  --max-new-tokens 512 \
  --output /root/autodl-tmp/outputs/checkpoint_smoke_search_free.json
```

脚本要求同时存在 `<reason>` 和 `<answer>`，解析 answer 后做 lower/strip/空白归一 EM，并记录 tokens、耗时、显存、GPU、Torch 和 CUDA。

### 7.6 验收

- 回答：`NamUs`；GT：`namus`；EM=true。
- 561 input / 68 output tokens。
- 加载 4.439 秒；生成 1.72 秒。
- 峰值显存 16,031.31 MiB。
- SDPA、参数 BF16、GPU `sm_120`。
- 正式输出：`/root/autodl-tmp/outputs/checkpoint_smoke_search_free.json`。
- JSON SHA-256：`c82401859a69dfeb8d621cc763c68a8bf4d81d03ec15ef1aad630d64c24fb3e8`。

## 8. 大步 6：Placeholder 多轮控制流

### 8.1 这一步只测协议，不测搜索真实性

官方 placeholder 是假结果，但仍可用于回答一个重要问题：模型能否正确输出工具 tag，并在最多三轮内完成 action state transition？

### 8.2 坑 P17：官方 Torch demo 丢失 assistant 工具动作

官方 image-first 分支会写入第一轮 assistant，但第二轮产生 `<text_search>` 后，[inference_torch_demo.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/scripts/inference_torch_demo.py:166) 直接 append 文本工具的 user 结果，漏掉第二轮 assistant。

首轮直接 Text Search 分支在 [inference_torch_demo.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/scripts/inference_torch_demo.py:180) 也直接 append user 结果，漏掉第一轮 assistant。

错误历史变成：

```text
user(original)
user(tool result)
```

或：

```text
user(original)
assistant(image_search)
user(image result)
user(text result)
```

模型不知道自己刚刚为什么收到工具结果。

修复后的不变量是：

```text
user(original)
assistant(tool action)
user(tool result)
assistant(next action)
user(next tool result)
assistant(answer)
```

派生脚本在任何工具调用前都先执行：

```python
messages.append(assistant_message(response))
```

再 append tool-result user message。注意：这是官方 Torch demo 的 bug；正式 vLLM rollout 会先把生成 token 写回上下文，不能笼统说整个仓库都漏历史。

### 8.3 坑 P16：真实标题被官方 demo 覆盖

官方 demo 明知是假标题，仍在 [inference_torch_demo.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/scripts/inference_torch_demo.py:141) 写：

```python
img_tool_returned_web_title_list = [
    f"Webpage Title {i+1}" ...
]
```

即使替换真实后端，模型也看不到网页标题。修复接口：

```python
titles = tool_stat.get(
    "titles",
    [f"Webpage Title {i+1}" for i in range(len(images))],
)
```

真实适配器必须返回 `titles`；smoke 验收明确拒绝所有 `Webpage Title N`。

### 8.4 坑 P18：分支 demo 不是可审计状态机

原 demo 只有嵌套 if/elif：

- 不统一解析 answer/warning/invalid。
- Text Search regex 没有 DOTALL。
- 多行 query 或尾空白可能匹配失败。
- 没有显式、可配置、统一执行的轮次/工具计数器；路径上限只被硬编码在不可扩展的嵌套分支里。
- `else: pass` 会静默结束。
- 不保存 trace、EM 或退出状态。

派生 `placeholder_control_flow.py` 增加：

- `classify_response()`：strip + DOTALL + 明确 action。
- `max_rounds=3`。
- Image/Text Search 各最多 1 次。
- 每轮 response、action、tokens、耗时、工具状态。
- terminal status、EM、control-flow pass。

### 8.5 初次运行与行为失败

```bash
cd /root/autodl-tmp/multimodal-search-r1

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/root/autodl-tmp/multimodal-search-r1 \
/root/autodl-tmp/envs/mmsearch_infer/bin/python \
reproduction/scripts/placeholder_control_flow.py \
  --model-path /root/autodl-tmp/models/MMSearch-R1-7B \
  --samples-meta /root/autodl-tmp/mmsearch_demo/meta.json \
  --output /root/autodl-tmp/outputs/placeholder_control_flow.json \
  --max-new-tokens 512 \
  --max-rounds 3
```

Search-Free 正常；但 `fvqa_train_6` 虽标为 `search_required`，模型 1 轮直接答出 `Rubjerg Knude`，EM=true、零搜索，导致 control-flow pass=false，脚本按设计退出 2。

### 8.6 坑 P19：类别标签不等于硬策略约束

这不是 checkpoint 坏了。`search_required` 表示题目通常需要外部知识，不保证当前 greedy 解码一定请求搜索。模型可能记住答案或从图像直接识别。

经用户确认后采用可审计选择规则，而不是手工挑一个答对的样本：

```text
按 parquet 原顺序
→ 只看 search_required
→ 最多前 20 个
→ 选首轮第一个自然产生 image_search/text_search 的样本
→ 保存之前所有候选动作
```

### 8.7 扫描命令

```bash
cd /root/autodl-tmp/multimodal-search-r1

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/root/autodl-tmp/multimodal-search-r1 \
/root/autodl-tmp/envs/mmsearch_infer/bin/python \
reproduction/scripts/placeholder_control_flow_scan.py \
  --model-path /root/autodl-tmp/models/MMSearch-R1-7B \
  --parquet /root/autodl-tmp/datasets/FVQA/fvqa_train.parquet \
  --samples-meta /root/autodl-tmp/mmsearch_demo/meta.json \
  --selected-image /root/autodl-tmp/mmsearch_demo/search_required_trigger.png \
  --selected-meta /root/autodl-tmp/mmsearch_demo/search_required_trigger_meta.json \
  --output /root/autodl-tmp/outputs/placeholder_control_flow_scanned.json \
  --scan-limit 20 \
  --max-new-tokens 512 \
  --max-rounds 3
```

候选 1 `train_6`、2 `train_9`、3 `train_15`、4 `train_16` 均直接 answer；第 5 个 `fvqa_train_17` 首轮 image_search。

### 8.8 验收

- `train_0`：`answer`，1 轮，无搜索，`NamUs`，EM=true。
- `train_17`：`image_search → text_search → answer`，3 轮，各工具一次，`Cheshire`，EM=true。
- 每轮都有 `<reason>`。
- `all_control_flows_pass=true`。
- 峰值显存 16,443.44 MiB。
- 正式 JSON：`placeholder_control_flow_scanned.json`，SHA-256 `29268b71b3f3cb560ad590eba1e6af57d0e43aef11513b54ab96aab0e53db4fb`。

`placeholder_control_flow.json` 是失败现场，必须保留，但不能当最终证据。

## 9. 大步 7：真实 Image Search

### 9.1 后端选择

原论文/README 描述 SerpApi Google Lens；用户希望先用官方 cache，再试 API。后来确认现有 key 来自 `serper.dev`，不是 `serpapi.com`。

### 9.2 坑 P20：SerpApi 与 Serper.dev 不能混用

```text
SerpApi：serpapi.com，变量 SERPAPI_API_KEY，可提供 Google Lens
Serper：serper.dev，变量 SERPER_API_KEY，本次用于 Google Text Search
```

把一种 key 发给另一家既会鉴权失败，也会造成不必要的密钥暴露。经用户选择后的最终组合：

```text
Image Search = FVQA 官方 Google Lens cache
Text Search  = Serper.dev
Reader       = Jina
```

`serpapi_lens.py` 和 `serpapi_lens_smoke.py` 只是未采用草稿，从未用当前 Serper key 调用。

### 9.3 实现官方 cache adapter

`FVQACachedImageSearch`：

1. 只加载精确固定 revision 的官方 pickle。
2. 读取 `tool_returned_web_title_list` 与 `tool_returned_images_urls`。
3. 同时支持 URL 字符串和 pickle 中的 PIL 对象。
4. 最多 5 线程；连接/读取 timeout；单图 10 MiB 上限。
5. 下载后转 RGB PNG，文件名包含 data ID、rank、URL hash。
6. 并发完成后重新按原 index 排序。
7. 每条失败单独记录，绝不使用假图补位。
8. 把真实 title 放入 `tool_stat` 并真正送给模型。

### 9.4 Smoke 命令

```bash
cd /root/autodl-tmp/multimodal-search-r1

PYTHONPATH=/root/autodl-tmp/multimodal-search-r1 \
/root/autodl-tmp/envs/mmsearch_infer/bin/python \
reproduction/scripts/cached_image_search_smoke.py \
  --cache-pickle /root/autodl-tmp/datasets/FVQA/fvqa_train_image_search_results_cache.pkl \
  --thumbnail-cache-dir /root/autodl-tmp/search_cache/fvqa_thumbnails \
  --data-id fvqa_train_17 \
  --output /root/autodl-tmp/outputs/cached_image_search_smoke.json
```

### 9.5 坑 P23：官方缓存的 URL 会腐烂

`fvqa_train_17` 缓存有 5 条，但第 5 条 `Claudio Mele | Flickr` 对应旧 Google thumbnail 现在返回 HTTP 404。

当时按任务纪律暂停；用户明确接受 4/5。修复策略不是删除证据或伪造第五张：

- 保留 404 URL、title、rank 和异常。
- 使用前 4 张成功图片。
- 下载成功图片存本地，未来不再依赖它们的临时 URL。
- 不用后排结果错误补第 5 位。

Smoke：requested=5、num_images=4、1 failure、pass=true。JSON SHA-256：`9c2c0efc89ef2c11dc6e809ecc8ddbaf29ea398c594a6b7e6399b8da265e0f7d`。

注意：第 5 条目前没有负缓存，重跑时可能再次 404；这是已批准的历史降级，不应重复误判为新问题。

### 9.6 模型集成

```bash
cd /root/autodl-tmp/multimodal-search-r1

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/root/autodl-tmp/multimodal-search-r1 \
/root/autodl-tmp/envs/mmsearch_infer/bin/python \
reproduction/scripts/cached_image_search_flow.py \
  --model-path /root/autodl-tmp/models/MMSearch-R1-7B \
  --sample-meta /root/autodl-tmp/mmsearch_demo/search_required_trigger_meta.json \
  --cache-pickle /root/autodl-tmp/datasets/FVQA/fvqa_train_image_search_results_cache.pkl \
  --thumbnail-cache-dir /root/autodl-tmp/search_cache/fvqa_thumbnails \
  --output /root/autodl-tmp/outputs/cached_image_search_flow.json \
  --max-new-tokens 512
```

动作：

```text
round 1: image_search
round 2: text_search
```

生成 query：

```text
Which historic county does the Lovell Telescope at Jodrell Bank Observatory belong to?
```

本步刻意停在 `awaiting_real_text_search`。所以 `final_answer=null`、`exact_match=null` 不是失败。`control_flow_pass=true`，峰值 16,453.44 MiB。

## 10. 大步 8：真实 Text Search + Jina + 完整三轮

### 10.1 坑 P21：key 被放在错误示例文件且权限过宽

历史上真实 `SERPER_API_KEY` 位于命名错误的 `serpapi.env.example`，权限 0644。示例文件容易被分享或提交，且 0644 对其他本机用户可读。

安全处理原则：

- 不读取、不打印 key 内容。
- 整体移动到 `reproduction/env/serper.env`。
- `chmod 600`。
- 重建两个只有占位符的 0644 example。

历史操作等价于下面两行，但当前状态已经完成，禁止再次执行覆盖：

```bash
# 仅说明历史动作，不要在当前状态重放
mv reproduction/env/serpapi.env.example reproduction/env/serper.env
chmod 600 reproduction/env/serper.env
```

运行时只允许：

```bash
set -a
. reproduction/env/serper.env
set +a
```

不得 `cat`、不得 `set -x`、不得 echo key 或 `SERPER_API_KEY` 的变量值；输出不含秘密的 PASS/FAIL 状态字符串不受此限制。

### 10.2 实现 SerperTextSearch

`reproduction/mmsearch_tools/serper_text_search.py` 的关键约束：

- POST `https://google.serper.dev/search`。
- key 只进入 `X-API-KEY` header。
- 公开 body 固定为 `q/gl/hl/num`。
- query 先规范化空白。
- cache key 只哈希公开参数，不含 header/key。
- 响应递归删除字段名规范化后等于 `api_key`、`authorization`、`account_email`、`account_id`、`token` 的敏感字段。
- HTTP 错误只记录状态/异常类型，不回显 header 或响应正文。
- 只接受合法 HTTP(S) organic URL。
- 无可用 organic 直接失败，不制造假结果。

### 10.3 实现 JinaReader

`reproduction/mmsearch_tools/jina_reader.py`：

- 去 URL fragment 并校验 HTTP(S)。
- 请求 `https://r.jina.ai/<target-url>`。
- 以 URL SHA-256 缓存 `.md` 与 metadata `.json`。
- 最多 5 线程；60 秒；单响应 5 MiB。
- 每页最多 12,000 字符。
- 并发完成后恢复 Serper rank。
- 单页失败进入 failures；全部失败才硬退出。
- 记录 full/returned chars、truncated、cache hit、路径和状态。

官方仓库只描述 SerpAPI + Jina + Qwen3，没有发布这些超时、缓存、错误或长度策略；这是实际补齐的系统组件。

### 10.4 先做交互式 mock 检查

在真正消耗查询前，曾用 mock Serper/Jina response 交互式检查 5 条解析、并发 Reader、12k 截断、缓存写入和测试 secret 不落盘。但这次检查没有保存独立脚本、日志或 JSON，因此不能作为正式可复核的 PASS 证据；大步 8 的正式证据从下一节 `serper_jina_smoke.json` 开始。

### 10.5 首次联网 smoke

```bash
cd /root/autodl-tmp/multimodal-search-r1
set -a
. reproduction/env/serper.env
set +a

PYTHONPATH=/root/autodl-tmp/multimodal-search-r1 \
/root/autodl-tmp/envs/mmsearch_infer/bin/python \
reproduction/scripts/serper_jina_smoke.py \
  --query 'Which historic county does the Lovell Telescope at Jodrell Bank Observatory belong to?' \
  --serper-cache-dir /root/autodl-tmp/search_cache/serper/json \
  --jina-cache-dir /root/autodl-tmp/search_cache/jina \
  --output /root/autodl-tmp/outputs/serper_jina_smoke.json \
  --top-k 5 \
  --max-chars-per-page 12000
```

实际只消耗 1 次 Serper query：

- Serper 5/5，cache miss。
- Jina 5/5，零失败。
- 五页 full/returned chars：85,815/12,000；24,982/12,000；11,937/11,937；7,168/7,168；90,803/12,000。
- Reader 合计 55,105 字符。
- 组合正文 58,480 字符。
- 组合 SHA-256：`d971798b27f6d2244e02fd4f3f808caeb66dc9d881db2aa8bf9eaf3e84ae1f43`。

### 10.6 坑 P25：缓存重放多 2 字符

第一次缓存实现：

```python
text = response.text.replace("\x00", "").strip()
content_path.write_text(text + "\n", encoding="utf-8")
return text
```

在线内存返回没有尾换行，但磁盘多一个；metadata 又记录未加换行的长度。重放时直接 `read_text()`，5 页都多 1 个 full char。三页被截到 12,000，额外字符落在截断区外；两页未截断，各多 1，故合计：

```text
55,105 → 55,107
58,480 → 58,482
```

这是我们派生适配器的 bug，不是官方代码。按用户要求暂停，确认后修复：

1. 新缓存精确写 `text`，不添加字符。
2. 读取旧缓存时读取 metadata 的 `characters`。
3. 仅当文件恰好 `expected+1` 且尾部为换行时裁掉这一字符。
4. 其他长度不一致直接报错，避免掩盖损坏。

修复后 Serper cache hit、Jina hits=5；长度重新为 55,105/58,480，正文 SHA 与首次联网完全一致。

正式重放：`serper_jina_smoke_cache_replay_fixed.json`。未修复的 `serper_jina_smoke_cache_replay.json` 只保留为故障证据。`jina_reader.py.orig` 是修复前备份，不会被 import。

### 10.7 完整三轮

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

本次完整流复用已验缓存，没有再次消费 Serper：

```text
round 1: image_search
round 2: text_search
round 3: answer
```

每次工具前都先 append assistant action，再 append user tool result。

验收：

- 输入 tokens：955 / 1,656 / 20,299。
- 输出 tokens：87 / 85 / 95。
- 生成耗时：2.09 / 1.495 / 3.767 秒。
- 最终 `Cheshire` = GT `Cheshire`。
- `exact_match=true`、`trace.control_flow_pass=true`、`pass=true`。
- 峰值显存 23,131.25 MiB。
- 正式 JSON SHA-256：`105479b7ba284ad220020829284d364a4170189e90dceae16dea22beb43d7587`。

### 10.8 密钥泄漏验收

只做精确匹配，绝不输出 key：

```bash
if [ -z "${SERPER_API_KEY:-}" ]; then
  echo 'SECRET_SCAN=BLOCKED_MISSING_SERPER_API_KEY'
  exit 2
fi

if rg -F -q -- "$SERPER_API_KEY" \
  /root/autodl-tmp/search_cache/serper \
  /root/autodl-tmp/search_cache/jina \
  /root/autodl-tmp/outputs/serper_jina_smoke.json \
  /root/autodl-tmp/outputs/serper_jina_smoke_cache_replay_fixed.json \
  /root/autodl-tmp/outputs/real_search_flow_raw_jina.json; then
  echo 'SECRET_LEAK=FAIL'
else
  echo 'SECRET_LEAK=PASS'
fi
```

实际结果 PASS。

## 11. 原仓库训练/rollout 中已识别但尚未修的坑

这些不在大步 1～8 Transformers 主线中执行。必须诚实写成“已识别、尚未完成 Blackwell 训练适配”。

### 11.1 默认 YAML typo

[ppo_trainer.yaml](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/trainer/multimodal/config/ppo_trainer.yaml:26)：

```yaml
use_remove_padding: Falseva
```

YAML 会把它当字符串而不是 bool。官方 run 脚本显式覆盖为 True，所以该脚本路径暂时绕过，但默认配置本身有问题。

### 11.2 工具类型靠字符串哨兵

正式 rollout 在 [vllm_rollout_spmd.py](/root/autodl-tmp/multimodal-search-r1/mmsearch_r1/workers/multimodal/rollout/vllm_rollout_spmd.py:352) 只检查：

```text
[Text Search Results]
[Image Search Results]
```

当前 Serper adapter 返回 `[Serper Text Search Results]`，派生 Transformers runner直接知道 tool type，所以大步 8 可用；但不能原样塞进官方 GRPO worker。训练接入前必须：

- 把前缀改为官方精确 sentinel；或
- 更稳妥地显式传递 tool type，不靠字符串猜。

### 11.3 原问题靠固定英文字符串反切

rollout 使用：

```python
all_context.split("Here is the image and the question:\n ")[1]
           .split("assistant")[0]
```

prompt 文本、空格、模板或问题内容稍变就可能 IndexError/截错。当前 runner 始终保存结构化 `question`，没有这个问题；训练路径仍需改。

### 11.4 token IDs 硬编码

rollout 硬编码 151664、`<|im_end|>` 151645、`<|image_pad|>` 151655。当前 Qwen2.5-VL tokenizer 通常对应，但换 tokenizer/model 时不安全，应从 tokenizer 动态解析。

### 11.5 raw Jina 与官方长度配置冲突

本次第三轮输入 20,299 tokens；官方 run 的 `max_prompt_length=4096`、`response_length_total=8192`。若直接把 raw Jina 塞入 rollout，会截断或超长。

这就是大步 9 加 Qwen3 summarizer 的必要性：不是只为更漂亮，而是为了把真实网页证据压进正式 rollout 的预算。

## 12. 不是仓库 bug，但复现时必须知道的基础设施坑

### 12.1 bwrap user namespace

当前服务器禁用非特权 user namespace，Codex 默认文件/命令沙箱可能报：

```text
bwrap: No permissions to create a new namespace
```

这不是 Python、MMSearch、PyTorch 或 CUDA 故障。经用户批准后使用 `require_escalated` 执行本机命令。Jina 修复时补丁工具受此限制，最终用标准 patch 应用并用 `py_compile`、缓存重放和 SHA 验证。

### 12.2 reproduction 未跟踪

所有派生脚本都在未跟踪的 `reproduction/`，而且 `env/` 被 `.gitignore` 忽略。不要运行：

```text
git clean
git reset --hard
```

`jina_reader.py.orig` 是修复前备份；`search_backend_decision.json` 的 Text Search 状态已过期；正式 JSON 比历史 manifest 更权威。

### 12.3 外部网页是不可信输入

本次正式 5 页中没有出现伪造 `</information>` 或 Qwen 控制 token，扫描通过；但批量评测不能假设所有网页都安全。大步 9 摘要 prompt 和后续 runner 应隔离网页伪指令、工具 tag 和控制 token。

## 13. 八步最终证据链

| 阶段 | 正式文件 | 关键结论 |
|---|---|---|
| Checkpoint | `/root/autodl-tmp/outputs/checkpoint_smoke_search_free.json` | NamUs，EM=true，BF16/SDPA |
| Placeholder | `/root/autodl-tmp/outputs/placeholder_control_flow_scanned.json` | Free 1轮；Required 3轮 |
| Image smoke | `/root/autodl-tmp/outputs/cached_image_search_smoke.json` | 官方 Lens cache 4/5，真实标题 |
| Image flow | `/root/autodl-tmp/outputs/cached_image_search_flow.json` | 生成稳定 Text Search query |
| Text network | `/root/autodl-tmp/outputs/serper_jina_smoke.json` | Serper 5/5；Jina 5/5 |
| Cache replay | `/root/autodl-tmp/outputs/serper_jina_smoke_cache_replay_fixed.json` | 长度/SHA 与在线一致 |
| Full flow | `/root/autodl-tmp/outputs/real_search_flow_raw_jina.json` | 三轮 Cheshire，EM=true |

容易误判的历史文件：

- `placeholder_control_flow.json`：样例直接答，控制流失败现场。
- `serper_jina_smoke_cache_replay.json`：尾换行修复前记录。
- `cached_image_search_flow.json` 的 `final_answer=null`：阶段性设计，不是失败。
- `serpapi_lens*.py`：另一供应商的未采用草稿。

## 14. 这次复现最值得保留的方法

### 14.1 一层一层放大，而不是一次跑全系统

```text
硬件
→ Python/CUDA
→ 模型单轮
→ placeholder 控制流
→ 真实 Image Search
→ 真实 Text Search
→ 缓存重放
→ 完整三轮
```

这样每个故障只有少数可能根因。

### 14.2 把失败现场保留下来

`train_6` 不搜索、第五张 thumbnail 404、Jina +2 字符都没有被覆盖或删除。正式文件与失败文件分开，才能解释为什么修复是必要且有效的。

### 14.3 对外部服务先 smoke，再模型集成

Serper/Jina 先用固定 query 单独跑；确认 5/5、缓存和密钥安全后，才加载 7B 模型。这样鉴权问题不会与 GPU/模型问题混在一起。

### 14.4 复现不等于盲目照抄版本

对 H100 旧栈逐字照抄到 Blackwell 不是严格，而是制造新的不可比变量。真正需要固定的是：代码 commit、模型/data revision、运行 dtype、attention、prompt、解码参数、样本 ID、工具返回和验收结果。

## 15. 八步完成后的准确起点

当前 raw-Jina 系统已经证明真实搜索链路可用，但第三轮输入达到 20,299 tokens。下一大步应在不改变已验证 Serper/Jina 后端的前提下加入独立 Qwen3-32B-FP8 summarizer，再验证与 MMSearch-R1-7B 同卡共驻。

后续大步 9～12 的具体交接见：

`/root/MMSearch-R1_复现实验进度记录_大步1-8完成.md`
