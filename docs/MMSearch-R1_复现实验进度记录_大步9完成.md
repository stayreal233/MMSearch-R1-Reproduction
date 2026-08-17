# MMSearch-R1 复现实验进度记录：大步 9 完成

> 更新时间：2026-08-16（UTC）  
> 当前结论：大步 1～9 已完成并验收；大步 10～12 尚未开始。  
> 前置记录：`/root/MMSearch-R1_复现实验进度记录_大步1-8完成.md`  
> 原始方案：`/root/MMSearch-R1_PRO6000-96G_5.98元每小时_复现方案.md`  
> 代码仓库：`/root/autodl-tmp/multimodal-search-r1`

## 1. 结论与边界

大步 9 已完成：Qwen3-32B-FP8 独立环境、固定 revision 权重、vLLM 摘要服务、关闭 Thinking 的摘要 smoke、top-5 Jina 摘要缓存，以及 Qwen3 与 MMSearch-R1-7B 同卡 Mixed Search 均已通过严格验收。

这不代表总复现完成：大步 10 的四类正式案例、大步 11 的分阶段批量评测、大步 12 的 Base 对比与最终报告仍未执行。当前成果属于开源权重推理与系统链路复现，不是论文完整训练复现。

## 2. 大步 9 验收总表

| 验收项 | 状态 | 证据摘要 |
|---|---:|---|
| Qwen3 独立环境 | 通过 | Python 3.12.3、vLLM 0.27.1、torch 2.13.0+cu130、CUDA runtime 13.0 |
| 精确模型 revision | 通过 | `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df` |
| 权重完整性 | 通过 | 17 文件、34,338,579,454 字节、7 个 safetensors 分片 SHA-256 全匹配 |
| 本机服务探活 | 通过 | `127.0.0.1:8001/v1/models` 返回固定本地模型 |
| Thinking 关闭 | 通过 | 服务模板与请求双重关闭；smoke/5篇摘要均验证为 false |
| 单页摘要 smoke | 通过 | rank-1 缓存页，摘要 384 字符，首次生成 cache miss，仅本机 Qwen3 网络 |
| top-5 摘要 | 通过 | Serper 5/5、Jina 5/5、Qwen3 summaries 5/5、failures=[] |
| 同卡共驻 | 通过 | Qwen3 服务与 MMSearch-R1 同一 GPU；MMSearch 参数全部在 `cuda:0` |
| Mixed Search | 通过 | `fvqa_train_17`，答案 `Cheshire`，Ground Truth `Cheshire`，EM=true |
| 密钥与正文防泄漏 | 通过 | 11 文件、0 个精确凭据命中；正式 JSON/cache 无 Jina 原文正文 |
| 严格终验 | 通过 | schema v2 completion manifest，5 cache 签名与服务进程身份均验证 |

## 3. 固定环境与模型

### 3.1 服务器

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition
- 总显存：97,887 MiB
- Compute Capability：`(12, 0)` / `sm_120`
- 驱动：595.58.03
- Qwen3 服务验收时显存占用：约 47,365 MiB
- Qwen3 服务 PID：136998（PID 是易变运行态，续跑前必须重新核验）

### 3.2 Qwen3 环境

- 环境：`/root/autodl-tmp/envs/qwen3_summary`
- Python：3.12.3
- vLLM：0.27.1
- torch：2.13.0+cu130
- `torch.version.cuda`：13.0
- transformers：5.15.0
- `pip check`：通过

### 3.3 模型

- Hugging Face repo：`Qwen/Qwen3-32B-FP8`
- 精确 revision：`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`
- 本地目录：`/root/autodl-tmp/models/Qwen3-32B-FP8`
- 文件数：17
- 总字节数：34,338,579,454
- safetensors：7 个分片，逐片 SHA-256 与固定 revision 的 LFS 元数据一致
- 量化：FP8 e4m3，block size `[128,128]`，dynamic activation

## 4. 最终服务合同

```text
host=127.0.0.1
port=8001
max_model_len=8192
gpu_memory_utilization=0.48
max_num_seqs=1
linear_backend=cutlass
VLLM_USE_DEEP_GEMM=0
VLLM_USE_FLASHINFER_SAMPLER=0
default enable_thinking=false
summary temperature=0
summary max_tokens=512
summary max_input_chars=12000
```

服务只监听 loopback，不使用远端 Qwen API。关闭 FlashInfer sampler 只影响 top-k/top-p 采样实现，不关闭 FlashAttention，也不改变 FP8 权重或 CUTLASS linear backend。本实验摘要固定 `temperature=0`、`seed=0`。

最终启动 readiness：84.047 秒。启动日志明确出现：

```text
Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod
FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.
```

## 5. 两个兼容性问题及批准后的修复

### 5.1 DeepGEMM SM120 scale-layout 错误

第一次服务启动在 7 个权重分片加载完成后的 FP8 scale 后处理阶段失败：

```text
Unknown SF transformation
```

这不是 OOM，也不是权重损坏。该问题与 vLLM 官方 RTX PRO 6000 / SM120 问题一致。经用户确认后采用最小兼容路径：

```text
VLLM_USE_DEEP_GEMM=0
--linear-backend cutlass
```

修复后日志实际选择 `CutlassFp8BlockScaledMMKernel`，32.04 GiB 权重成功加载。

### 5.2 FlashInfer sampler 的本地 CUDA 12.8 检测错误

CUTLASS 启动随后在 sampling warmup 首次触发 FlashInfer JIT 时失败：

```text
SM 12.x requires CUDA >= 12.9
FlashInfer requires GPUs with sm75 or higher
```

第二条是误导性错误：Torch wheel 是 cu130，但 FlashInfer 优先探测本地 CUDA 工具链；本机链接仍指向 CUDA 12.8，SM120 架构规范化失败后留下空架构集合。经用户确认后增加：

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

vLLM 因此使用 native sampler。没有升级 FlashInfer、没有替换 CUDA、没有更换 vLLM，也没有修改模型权重。

## 6. 摘要 smoke

- 输出：`/root/autodl-tmp/outputs/qwen3_summary_smoke.json`
- 模式：`qwen3_summary_cached_raw_jina_rank1_smoke`
- 数据：既有 raw-Jina trace 的 rank-1 缓存页
- 网络范围：仅 `127.0.0.1:8001`，未调用 Serper/Jina 网络
- 摘要字符数：384
- 首次生成：`cache_hit=false`
- Thinking：关闭
- 原始网页正文：未写入正式输出

## 7. 同卡 Mixed Search

- 样本：`fvqa_train_17`
- Query：Lovell Telescope 所属 historic county 问题
- 公开动作序列：`image_search → text_search → answer`
- 语义链路：`image_search → text_search → Jina top-5 → Qwen3 top-5 summaries → answer`
- `summarization_calls=1`；摘要聚合嵌套在 text-search tool 状态内，不单列为模型 action
- Serper：5 个结果
- Jina：requested=5、documents=5、failures=[]
- Qwen3：requested=5、num_summaries=5、failures=[]
- MMSearch：所有参数与 `hf_device_map` 均在 GPU 0，无 CPU/disk offload
- MMSearch 进程峰值显存：16,948.54 MiB
- 计算进程数量快照：Qwen3-only 为 1；双模型加载后为 2；流程结束仍为 2
- 最终答案：`Cheshire`
- Ground Truth：`Cheshire`
- Exact Match：true
- 正式输出：`/root/autodl-tmp/outputs/real_search_flow_qwen3_summary.json`

Image Search 继续使用用户已批准的官方 FVQA Lens 缓存，`fvqa_train_17` 第 5 张缩略图历史上为 404，按既有 4/5 降级口径通过。这不等价于实时 Lens 网络搜索已验证。

## 8. 缓存、安全与终验

摘要缓存目录：`/root/autodl-tmp/search_cache/qwen3_summary`。

- 5 个唯一 JSON cache
- schema：`mmsearch.qwen3_summary.v1`
- 文件名等于公共输入签名 SHA-256
- 签名包含 query/title/url/snippet、受限 Jina 输入哈希与字符数、模型 revision、prompt hash、temperature、max_tokens、Thinking、seed
- cache 不包含 API key、Authorization、prompt、Jina 原文正文
- 5 个 cache 均保留 token、耗时、finish reason 与来源元数据

凭据扫描：

- 输出：`/root/autodl-tmp/outputs/qwen3_key_scan.json`
- schema：v2
- 扫描文件数：11
- 扫描字节数：144,683
- 检查来源：`SERPER_API_KEY`（仅内存读取；值未记录）
- 精确凭据命中：0
- pass：true

严格终验：

- 输出：`/root/autodl-tmp/outputs/qwen3_step9_completion_manifest.json`
- schema：v2
- status：passed
- completed_at：`2026-08-16T12:51:31.033988+00:00`
- 5 个摘要 cache 的 schema、签名、文件名、内容摘要和 Jina 输入哈希均通过
- 正式 smoke、flow、cache 未包含受限 Jina 正文
- 服务 PID、启动时间和命令行摘要通过身份检查

非阻塞说明：vLLM 日志是在线文件，服务存活期间可能继续增长。completion manifest 固定的是凭据扫描时日志快照；若未来关服后需要不可变归档，应在停止服务后重新扫描并另存最终日志清单。

## 9. 正式证据及 SHA-256

| 证据 | SHA-256 |
|---|---|
| `reproduction/env/qwen3_install_decision.json` | `cb95f5dab8f1b67d8dab016974f1bb871ad246554465a170cf083bcaa7fd4cd9` |
| `reproduction/env/qwen3_huggingface_revision.json` | `0bfa3d41ba6ba19621911f19b1161f55cfd773290a55740a372c6b2be9e198da` |
| `reproduction/env/qwen3_summary_versions.txt` | `a8a6fb5567c40611b58bc9e2878a6f9cb32f7e81f75ba491e0bf352bd79f5233` |
| `/root/autodl-tmp/outputs/qwen3_summary_smoke.json` | `8768c255a90531bec2be6cd35c66158dcfc57c8d80a799beee186ec2821dde29` |
| `/root/autodl-tmp/outputs/real_search_flow_qwen3_summary.json` | `429d79e4ceb055a69c300e1ce6deac6b993670cb6952aeaec1b84ef6343ac929` |
| `/root/autodl-tmp/outputs/qwen3_key_scan.json` | `1ae56dadf040d61987c98d79cb424770a0bf8732153adbe8756be7d23e2902ca` |
| `/root/autodl-tmp/outputs/qwen3_step9_completion_manifest.json` | `e33fc6ae37a9ad76750122a09edf3ed3d3cb39940f237c77d04ad9807bd30ba7` |
| `reproduction/scripts/start_qwen3_summary.py` | `f600280c3bf393cd5d76f9cd84eccb56044b5974831596c5982853724b593799` |
| `reproduction/scripts/real_text_search_flow_qwen3.py` | `29b771e4d508e704c3104e3e36a74a1282007e5cd23a9197bf27c1636895a261` |
| `reproduction/scripts/finalize_qwen3_step9.py.strict` | `192607b985c9074ac12a5851233f005e8c7f52b58d8b1ba1348dc44a311b73c9` |

5 个摘要 cache 的 SHA-256 已逐项写入 completion manifest。日志扫描快照为 82,959 字节，SHA-256 `34a9694f3844d0bdc5566d3eab41c250e01df46ab77e7d3f8d4218fae52cd449`。

## 10. 当前代码与运行态注意事项

- 仓库官方 tracked 内容未被 reset/clean；`reproduction/` 仍是未跟踪工作成果，严禁 `git clean`。
- `reproduction/env/serper.env` 只能 source；不得读取、打印或写入报告。
- Qwen3 服务当前保持在线以便大步 10 复用；续跑前必须重新探活并核验 PID/GPU，不可假定 PID 136998 永久有效。
- 启动脚本：`reproduction/scripts/start_qwen3_summary.py`
- 摘要实现：`reproduction/mmsearch_tools/qwen3_summarizer.py`
- 同卡 runner：`reproduction/scripts/real_text_search_flow_qwen3.py`
- 严格扫描：`reproduction/scripts/scan_qwen3_outputs.py.strict`
- 严格终验：`reproduction/scripts/finalize_qwen3_step9.py.strict`
- 旧失败日志保留在同一 vLLM 日志中，必须按启动时间分段解读；不能把早期 DeepGEMM/FlashInfer 错误误判成最终服务仍失败。

## 11. 尚未完成

### 大步 10：四类案例与完整 Trace

- 用最终 Qwen3 runner 正式归档 Search-Free、Image Search、Text Search、Mixed Search 四类案例
- 固定候选范围、顺序、seed 和选择规则，不能事后只挑正确样本
- 至少整理 1 个失败案例及 `cases.md`
- Mixed Search 可复用本步的 `fvqa_train_17`，但仍需纳入四类统一归档

### 大步 11：分阶段批量评测

- 固定 eval IDs
- 按 `5 → 20 → 50 → 100 → 300` 分档推进
- 每档扩大前检查搜索额度、失败率、缓存命中率和成本
- 至少完成 50～100 条正式批评及 predictions/metrics/failure summary

### 大步 12：Base 对比与最终报告

- 固定 Qwen2.5-VL-7B Base revision
- 在同一 20～50 样本子集做公平对比
- 汇总 Search Ratio、EM、failure cases、版本与资源成本

## 12. 下一步与暂停点

大步 9 已完成，依用户要求在此暂停，不自动进入大步 10。

下一次继续时应：

1. 阅读本文、前置大步 1～8 记录和原始方案。
2. 只读核验 Qwen3 PID、`/v1/models`、GPU 进程与显存；若服务仍健康则直接复用，不要重复下载权重。
3. 保留全部 `reproduction/` 未跟踪文件与正式 JSON/cache。
4. 从大步 10 的固定候选选择规则和四类正式 trace 开始。
5. 大步 10 完成或出现实际问题时再次暂停等待用户确认。
