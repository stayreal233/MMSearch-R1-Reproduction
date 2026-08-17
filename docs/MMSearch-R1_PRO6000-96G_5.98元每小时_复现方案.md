# MMSearch-R1 详细复现方案：RTX PRO 6000 Blackwell 96GB / 5.98 元每小时版

> 项目：**MMSearch-R1: Incentivizing LMMs to Search**  
> 官方仓库：`https://github.com/EvolvingLMMs-Lab/multimodal-search-r1`  
> 官方模型：`lmms-lab/MMSearch-R1-7B`  
> 官方数据：`lmms-lab/FVQA`  
> 当前资源：AutoDL `PRO6000-96G × 1`，约 **5.98 元/GPU·小时**，总预算约 **300 元**。

---

## 1. 结论：这张卡可以，而且很适合“开源权重系统复现”

如果 AutoDL 页面中的 `PRO6000-96G` 对应 NVIDIA RTX PRO 6000 Blackwell 96GB，那么它完全可以完成本项目的主体复现：

- 加载并运行官方 `MMSearch-R1-7B` checkpoint；
- 读取 FVQA；
- 跑 Search-Free / Search-Required 样例；
- 跑作者的多轮工具调用逻辑；
- 将 placeholder Image/Text Search 替换成真实搜索；
- 使用 SerpAPI + Jina Reader；
- 本地部署 `Qwen3-32B-FP8` 作为网页摘要器；
- 做 20 / 50 / 100 / 300 条 FVQA 批量评测；
- 做 MMSearch-R1 与 Qwen2.5-VL-7B Base 的简单对比。

但需要特别注意：

> **RTX PRO 6000 是 Blackwell GPU，环境不能继续照搬 A800/H800 版的 PyTorch 2.6 + CUDA 12.4。**

对 Blackwell `sm_120`，优先使用：

```text
CUDA >= 12.8
PyTorch >= 2.7，并使用 cu128 或更高版本
```

因此你截图当前选中的：

```text
PyTorch 2.5.1
Python 3.12
CUDA 12.4
```

**不建议。**

创建 PRO6000 实例时更推荐：

```text
PyTorch 2.8.0
Python 3.12
CUDA 12.8
```

---

## 2. 与论文原始环境的关系

MMSearch-R1 当前官方 README 仍固定：

```text
Python 3.10
vllm==0.8.2
transformers==4.51.0
flash-attn==2.7.4.post1
```

这套软件栈对应作者发布时的训练环境。论文正式 GRPO 训练则使用了 32×H100。

对于 PRO6000 Blackwell：

- **官方 checkpoint 的 Torch inference：可以稳定复现；**
- **真实搜索系统：可以稳定复现；**
- **FVQA 评测：可以稳定复现；**
- **作者 `vllm==0.8.2 + flash-attn==2.7.4` 原样运行：不建议作为主线。**

原因不是显存，而是 Blackwell `sm_120` 对 CUDA、PyTorch、Triton、vLLM 和 CUDA extension 的版本要求更高。

所以本方案把目标定义为：

```text
必须完成：
官方权重 + 多轮搜索 + 真实工具 + FVQA + 批量评测

可选完成：
将作者旧 veRL/vLLM 训练栈适配到 Blackwell
```

---

## 3. 预算

GPU 单价：

```text
5.98 元 / 小时
```

300 元理论最大 GPU 时间：

```text
300 / 5.98 ≈ 50.17 小时
```

不建议全部耗完。推荐：

| GPU 时间 | GPU 费用 |
|---:|---:|
| 32 h | 191.36 元 |
| 36 h | 215.28 元 |
| 40 h | 239.20 元 |
| 45 h | 269.10 元 |

**主方案建议 32～36 小时，最多先开到 40 小时。**

截图里的磁盘日费用只有约 1.88 元/天量级，所以主要成本仍然是 GPU。

推荐预算：

```text
主复现：36 小时 ≈ 215.28 元
系统盘：约 3～5 元
剩余：约 80 元
```

剩余资金用于重跑、扩大样本量或做兼容性实验。

---

## 4. AutoDL 最终配置

截图中的硬件：

```text
GPU：PRO6000-96G × 1
CPU：20～25 核
内存：100～120GB
系统盘：扩容约 200GB
价格：5.98 元/h
```

都足够。

### 镜像一定改为

```text
PyTorch 2.8.0
Python 3.12
CUDA 12.8
```

不要选：

```text
PyTorch 2.5.1
CUDA 12.4
```

PRO6000 是 Blackwell，CUDA 12.8 更适合作为原生运行基础。

---

## 5. 开机后先验证 GPU

```bash
nvidia-smi
df -h
free -h
nvcc -V
```

再运行：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("arch:", torch.cuda.get_arch_list())
PY
```

重点确认：

```text
CUDA >= 12.8
Compute Capability ≈ (12, 0)
```

如果出现：

```text
sm_120 is not compatible with the current PyTorch installation
```

说明当前 PyTorch wheel 不是 cu128+。

---

## 6. 建议拆成两个主环境

### 环境 A：MMSearch 推理

```text
mmsearch_infer
```

用于：

- MMSearch-R1-7B；
- Transformers；
- Qwen2.5-VL processor；
- Torch inference demo；
- FVQA；
- 搜索 agent。

### 环境 B：Qwen3 摘要服务

```text
qwen3_summary
```

用于：

- Qwen3-32B-FP8；
- 现代 vLLM；
- OpenAI-compatible HTTP API。

不要把作者旧 vLLM 和 Qwen3 的现代 vLLM 混在一个环境。

---

## 7. 创建 MMSearch 推理环境

```bash
conda create -n mmsearch_infer python=3.11 -y
conda activate mmsearch_infer
```

安装 Blackwell 适配的 PyTorch：

```bash
pip install torch==2.8.0   torchvision   torchaudio   --index-url https://download.pytorch.org/whl/cu128
```

再安装：

```bash
pip install   transformers==4.51.0   accelerate   qwen-vl-utils   pillow==11.1.0   requests   datasets   huggingface_hub   openai   pandas   pyarrow
```

### 第一阶段不要装

```text
vllm==0.8.2
flash-attn==2.7.4.post1
```

先使用 PyTorch SDPA，避免 Blackwell CUDA extension 兼容问题。

---

## 8. Clone 项目

```bash
cd /root/autodl-tmp

git clone --recurse-submodules   https://github.com/EvolvingLMMs-Lab/multimodal-search-r1.git

cd multimodal-search-r1
git submodule status
git rev-parse HEAD
```

保存 commit：

```bash
mkdir -p reproduction/env
git rev-parse HEAD > reproduction/env/git_commit.txt
```

---

## 9. 缓存与磁盘目录

```bash
mkdir -p /root/autodl-tmp/cache/huggingface
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/datasets
mkdir -p /root/autodl-tmp/outputs
mkdir -p /root/autodl-tmp/search_cache

export HF_HOME=/root/autodl-tmp/cache/huggingface
```

建议最终可用空间至少：

```text
180～200GB
```

用于：

```text
MMSearch-R1-7B
Qwen3-32B-FP8
Qwen2.5-VL-7B-Instruct（可选）
FVQA
搜索缓存
Conda / pip / HF cache
结果文件
```

---

## 10. 下载官方 MMSearch-R1-7B

```bash
huggingface-cli download   lmms-lab/MMSearch-R1-7B   --local-dir /root/autodl-tmp/models/MMSearch-R1-7B
```

检查：

```bash
du -sh /root/autodl-tmp/models/MMSearch-R1-7B
```

---

## 11. 下载 FVQA

```bash
huggingface-cli download   lmms-lab/FVQA   --repo-type dataset   --local-dir /root/autodl-tmp/datasets/FVQA
```

FVQA 的关键分类是：

```text
search_free
search_required
```

论文 RL 训练集大约：

```text
3400 search-required
1600 search-free
```

最终展示一定要同时包含两类。

---

## 12. PRO6000 上优先使用 SDPA

在：

```text
mmsearch_r1/scripts/inference_torch_demo.py
```

如果模型加载使用：

```python
attn_implementation="flash_attention_2"
```

改为：

```python
attn_implementation="sdpa"
```

推荐加载形式：

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
```

这是为了先避开作者旧 `flash-attn==2.7.4.post1` 在 `sm_120` 上可能出现的编译问题。

---

## 13. 第一阶段：只验证 checkpoint

```bash
conda activate mmsearch_infer

python mmsearch_r1/scripts/inference_torch_demo.py   --model_path /root/autodl-tmp/models/MMSearch-R1-7B   --image /path/to/image.png   --question "What is shown in this image?"
```

验收：

```text
[PASS] model load
[PASS] image processor
[PASS] generation
[PASS] <reason>...</reason>
[PASS] <answer>...</answer>
```

只要这一步成功，就说明 PRO6000 能承担主体复现。

---

## 14. 第二阶段：先用官方 placeholder 跑三轮

官方仓库目前的：

```text
mmsearch_r1/utils/tools/image_search.py
mmsearch_r1/utils/tools/text_search.py
```

是 placeholder。

先不改，用 search-required 样例验证：

```text
Round 1
MMSearch-R1
→ image_search / text_search action

Round 2
tool result
→ MMSearch-R1

Round 3
→ final answer
```

此阶段只验证 agent 控制流，不评估答案质量。

---

## 15. 提取 FVQA 两类样例

```python
from datasets import load_dataset
from pathlib import Path
import json

ds = load_dataset("lmms-lab/FVQA", split="train")
out = Path("/root/autodl-tmp/mmsearch_demo")
out.mkdir(parents=True, exist_ok=True)

samples = {}

for row in ds:
    cat = row["category"]
    if cat not in samples:
        samples[cat] = row
    if "search_free" in samples and "search_required" in samples:
        break

meta = {}

for cat, row in samples.items():
    image = row["images"][0]
    image_path = out / f"{cat}.png"
    image.save(image_path)

    meta[cat] = {
        "data_id": row["data_id"],
        "image": str(image_path),
        "question": row["prompt"][0]["content"],
        "reward_model": row["reward_model"],
    }

with open(out / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(json.dumps(meta, ensure_ascii=False, indent=2))
```

---

## 16. 真实 Image Search

论文流程：

```text
Image
→ SerpAPI
→ top-5 visual webpages
→ thumbnail + webpage title
```

推荐：

```text
官方 FVQA 样例：
优先使用作者发布的 image-search cache

自定义图片：
使用 SerpAPI Google Lens
```

设置：

```bash
export SERPAPI_API_KEY="YOUR_KEY"
```

### 必修复一个 demo 问题

官方 demo 当前有类似：

```python
img_tool_returned_web_title_list = [
    f"Webpage Title {i+1}"
    for i in range(len(tool_returned_images))
]
```

应改为：

```python
img_tool_returned_web_title_list = tool_stat.get(
    "titles",
    [
        f"Webpage Title {i+1}"
        for i in range(len(tool_returned_images))
    ],
)
```

否则真实网页 title 没有真正送回模型。

---

## 17. 真实 Text Search

论文正式流程：

```text
Model-generated Query
→ SerpAPI
→ top URLs
→ Jina Reader
→ Qwen3-32B
→ top-5 concise summaries
→ MMSearch-R1
```

建议分两步。

### Step A：先不部署 Qwen3

先做到：

```text
SerpAPI
→ URL
→ Jina Reader
→ 截断网页正文
→ MMSearch-R1
```

确认网络链路完全正常。

### Step B：再加入 Qwen3 summarizer

这样出现错误时容易定位。

---

## 18. 本地 Qwen3-32B-FP8

PRO6000 96GB 很适合：

```text
MMSearch-R1-7B BF16
+
Qwen3-32B-FP8
```

共同使用一张 GPU。

原因：

```text
MMSearch-R1-7B BF16 权重 ≈ 16GB
Qwen3-32B BF16 权重 ≈ 65GB
```

BF16 共驻理论权重已约 81GB，留给 KV Cache 和运行时的空间不够安全。

因此推荐：

```text
Qwen3-32B-FP8
```

---

## 19. 创建 Qwen3 环境

```bash
conda create -n qwen3_summary python=3.12 -y
conda activate qwen3_summary

pip install -U pip
pip install -U vllm
pip install openai huggingface_hub
```

这里故意**不使用 `vllm==0.8.2`**。

现代 vLLM 对 Blackwell / CUDA 12.8+ 的支持更合适。

下载：

```bash
huggingface-cli download   Qwen/Qwen3-32B-FP8   --local-dir /root/autodl-tmp/models/Qwen3-32B-FP8
```

---

## 20. 启动本地 summarizer

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /root/autodl-tmp/models/Qwen3-32B-FP8   --host 127.0.0.1   --port 8001   --max-model-len 8192   --gpu-memory-utilization 0.48   --max-num-seqs 1
```

测试：

```bash
curl http://127.0.0.1:8001/v1/models
```

另一个终端：

```bash
export SUMMARIZER_BASE_URL=http://127.0.0.1:8001/v1
export SUMMARIZER_API_KEY=EMPTY
export SUMMARIZER_MODEL=/root/autodl-tmp/models/Qwen3-32B-FP8
```

---

## 21. Qwen3 关闭 Thinking

网页摘要不需要长推理。

```python
resp = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    temperature=0,
    max_tokens=512,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)
```

这样可以显著减少 summarizer 延迟和 token 开销。

---

## 22. 96GB 同卡共驻策略

启动顺序：

```text
1. 先启动 Qwen3-32B-FP8
2. nvidia-smi 看显存
3. 再启动 MMSearch-R1
4. 跑一条 Mixed Search
```

建议先把 Qwen vLLM 限制在：

```text
gpu_memory_utilization = 0.48
max_model_len = 8192
max_num_seqs = 1
```

如果 OOM：

```text
0.48 → 0.42
8192 → 4096
Jina 网页 30000 chars → 12000～16000 chars
```

PRO6000 有 96GB，比 80GB H800 更容易实现同卡共驻。

---

## 23. 四类必须展示的案例

### Case A：Search-Free

```text
Image + Question
→ MMSearch-R1
→ 直接 Answer
```

说明模型知道时不搜索。

### Case B：Image Search

```text
Image
→ Image Search
→ 识别视觉实体
→ Answer
```

### Case C：Text Search

```text
已识别实体
→ Text Search
→ 获取外部事实
→ Answer
```

### Case D：Mixed Search

```text
Image
→ Image Search
→ identify entity
→ Text Search
→ factual information
→ Answer
```

Case D 最适合录屏和 PPT。

---

## 24. 保存完整 Trace

```json
{
  "data_id": "xxx",
  "category": "search_required",
  "question": "...",
  "ground_truth": "...",
  "rounds": [
    {
      "round": 1,
      "action": "image_search",
      "response": "..."
    },
    {
      "round": 2,
      "action": "text_search",
      "query": "...",
      "response": "..."
    },
    {
      "round": 3,
      "action": "answer",
      "response": "..."
    }
  ],
  "final_answer": "...",
  "image_search_calls": 1,
  "text_search_calls": 1,
  "exact_match": true
}
```

---

## 25. 批量评测

不要直接跑 1800 条。

按照：

```text
5
→ 20
→ 50
→ 100
→ 300
```

逐步增加。

真实搜索的瓶颈经常是：

```text
SerpAPI
Jina
网页失效
rate limit
Qwen summarization
```

而不是 PRO6000 算力。

---

## 26. 指标

### Exact Match

```python
pred.strip().lower() == gt.strip().lower()
```

### Search Calls

```text
image_search_calls
text_search_calls
total_search_calls
```

### Search Ratio

若每条最多 2 次有效搜索：

```text
SR = total_search_calls / (N × 2) × 100%
```

### Average Turns

```text
total_turns / N
```

### 分类别 Accuracy

```text
Search-Free Accuracy
Search-Required Accuracy
```

---

## 27. 论文结果作为 sanity check

论文报告：

```text
MMSearch-R1-7B
Average Acc ≈ 54.6
Average SR  ≈ 67.1

FVQA-test
Acc ≈ 58.4
SR  ≈ 66.8
```

你的结果不应要求逐点相同，因为：

```text
搜索引擎排名会变化
网页会变化
Jina 输出会变化
Qwen3 FP8 与论文服务端设置不完全相同
Judge 模型版本也会变化
```

验收重点应该是：

```text
Search-Free 明显更少搜索
Search-Required 更容易调用搜索
搜索后答案质量提高
Mixed Search 能形成合理多轮行为
```

---

## 28. Base Model 对比

可再下载：

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

对同一批 20～50 条：

```text
Base Direct Answer
vs
MMSearch-R1 On-demand Search
```

比较：

```text
Accuracy
Search Calls
成功案例
失败案例
```

这会让复现报告更完整。

---

## 29. 关于官方 `run_mmsearch_r1_grpo.sh`

官方脚本使用：

```text
veRL
vllm==0.8.2
旧 CUDA/PyTorch 栈
```

在 PRO6000 Blackwell 上不要把它设为第一阶段目标。

主实验全部完成后，如果还有预算，可单独建立：

```text
mmsearch_train_test
```

尝试将：

```text
PyTorch
vLLM
veRL
rollout worker
```

迁移到 Blackwell 新版本。

这一部分应写成：

```text
Blackwell compatibility adaptation
GRPO pipeline smoke test
```

而不是“论文完整训练复现”。

---

## 30. 推荐 32～36 小时执行时间表

### 0～3 h：Blackwell 环境

```text
CUDA 12.8
PyTorch 2.8 cu128
GPU capability
Clone
```

### 3～6 h：模型和数据

```text
MMSearch-R1-7B
FVQA
image cache
```

同时下载 Qwen3-FP8。

### 6～9 h：Checkpoint + placeholder

```text
Torch inference
SDPA
search-free
search-required
3-turn fake chain
```

### 9～13 h：真实 Image Search

```text
FVQA cache
SerpAPI Google Lens
real title
thumbnail
```

### 13～17 h：真实 Text Search

```text
SerpAPI
Jina Reader
raw webpage feedback
```

### 17～21 h：Qwen3-FP8

```text
vLLM
summary service
single-GPU coexistence
Mixed Search
```

### 21～25 h：典型案例

```text
Search-Free
Image Search
Text Search
Mixed Search
Failure Case
```

### 25～30 h：批量评测

```text
20
→ 50
→ 100
→ 条件允许再 200～300
```

### 30～34 h：Base 对比

```text
Qwen2.5-VL-7B
vs
MMSearch-R1
```

### 34～36 h：整理结果

```text
predictions.jsonl
metrics.json
cases.md
pip_freeze.txt
gpu_info.txt
git_commit.txt
```

36 小时后先关机。

---

## 31. 成本

```text
36 × 5.98 = 215.28 元
```

加磁盘费用后仍明显低于 300 元。

剩余约 80 元，相当于约：

```text
13 个 PRO6000 GPU 小时
```

用于：

```text
重跑
扩大评测
Base 对比
可选 GRPO 兼容实验
```

---

## 32. 常见 Blackwell 报错

### `no kernel image is available for execution on the device`

通常是：

```text
PyTorch wheel CUDA < 12.8
```

换：

```text
PyTorch 2.7+ / cu128+
```

### `sm_120 is not defined`

通常是：

```text
nvcc
Triton
FlashAttention
```

版本过旧。

主线直接使用：

```text
SDPA
```

### `flash-attn==2.7.4.post1` 编译失败

不要为了完全照抄 README 在 Blackwell 上耗数小时。

先不装它。

### Qwen vLLM OOM

```text
gpu_memory_utilization 0.48 → 0.42
max_model_len 8192 → 4096
max_num_seqs = 1
```

---

## 33. 最终验收清单

### 必须完成

- [ ] PRO6000 被识别为 Blackwell GPU；
- [ ] PyTorch CUDA >= 12.8；
- [ ] MMSearch-R1-7B 成功加载；
- [ ] FVQA 成功读取；
- [ ] Search-Free 成功；
- [ ] Search-Required 成功；
- [ ] placeholder 多轮链路成功；
- [ ] 真实 Image Search 成功；
- [ ] 真实 Text Search 成功；
- [ ] Jina Reader 成功；
- [ ] Qwen3-32B-FP8 summarizer 成功；
- [ ] 至少 4 类案例；
- [ ] 至少 50～100 条批量评测；
- [ ] Exact Match；
- [ ] Search Ratio；
- [ ] 完整 Trace。

### 推荐完成

- [ ] 100～300 条 FVQA；
- [ ] Base Model 对比；
- [ ] Search-Free / Search-Required 分开统计；
- [ ] 失败案例分析；
- [ ] FVQA 官方 Image Search Cache；
- [ ] LLM-as-Judge。

### 可选

- [ ] Blackwell 版 veRL/vLLM 兼容适配；
- [ ] mini GRPO smoke test。

---

## 34. PRO6000 vs H800：怎么选

如果：

```text
H800-80GB ≈ 8.88 元/h
PRO6000-96GB ≈ 5.98 元/h
```

而你的目标是：

```text
利用开源权重
完整跑通 MMSearch-R1
加入真实搜索
做 FVQA 展示与评测
预算 300 元
```

**PRO6000-96GB 更有性价比。**

优势：

```text
96GB 显存
价格更低
300 元能买约 50 GPU 小时
Blackwell FP8 能力适合 Qwen3-32B-FP8
单卡共驻余量更大
```

代价：

```text
作者旧 vLLM 0.8.2 / CUDA 12.4 软件栈不能原封不动照搬
```

因此：

> **“开源权重系统复现”优先 PRO6000；“原作者旧训练环境尽量少改”则 H800/A800 更省心。**

---

## 35. 你现在创建实例时应该选什么

最终：

```text
GPU：
PRO6000-96G × 1

价格：
5.98 元/h

CPU：
20～25 核

RAM：
100～120GB

系统盘：
约 200GB

基础镜像：
PyTorch 2.8.0
Python 3.12
CUDA 12.8
```

**不要使用当前截图中的 PyTorch 2.5.1 + CUDA 12.4。**

---

## 36. 参考

- MMSearch-R1：`https://github.com/EvolvingLMMs-Lab/multimodal-search-r1`
- Model：`https://huggingface.co/lmms-lab/MMSearch-R1-7B`
- FVQA：`https://huggingface.co/datasets/lmms-lab/FVQA`
- NVIDIA RTX PRO 6000 Blackwell：`https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/`
- CUDA GPU Compute Capability：`https://developer.nvidia.com/cuda/gpus`
- PyTorch：`https://pytorch.org/get-started/`
- vLLM GPU Installation：`https://docs.vllm.ai/en/stable/getting_started/installation/gpu/`
- Qwen3-32B-FP8：`https://huggingface.co/Qwen/Qwen3-32B-FP8`
- SerpAPI：`https://serpapi.com/`
- Jina Reader：`https://jina.ai/reader/`
