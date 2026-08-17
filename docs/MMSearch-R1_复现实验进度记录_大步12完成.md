# MMSearch-R1 复现实验进度记录：大步 12 完成

更新时间：2026-08-17 UTC

## 1. 当前结论

大步 12 已按复现方案中的 Base Model 对比与最终整理口径完成，并在此暂停：

- 使用固定 revision 的 `Qwen/Qwen2.5-VL-7B-Instruct` 作为 Base；
- Base 在大步 11 完全相同的固定 50 条 FVQA train 子集上执行 Direct Answer；
- 两侧使用相同图片、问题、Ground Truth、顺序、seed、greedy、`max_new_tokens=512` 和 strict Exact Match；
- Base 明确禁止工具调用，MMSearch 复用大步 11 的不可变自然按需搜索结果；
- Base 按 `5 → 20 → 50` 三档运行并全部通过；
- 最终 Base 为 7/50（14.0%），MMSearch 为 20/50（40.0%），MMSearch 高 26.0 个百分点；
- 配对结果、分类别指标、延迟、搜索调用、成功/失败案例、模型与数据 revision、pip freeze、GPU、Git 状态、逐样本 trace 和独立审计均已落盘；
- 独立审计重算 50/50 条 Base strict EM、固定顺序和配对四象限，结果一致；
- 69 个正式 JSON/文本文件的终验扫描未发现凭据、Authorization、原始 Jina 正文、图片 base64 或禁用字段泄漏。

至此，两份复现文档定义的大步 1–12 已执行完毕。这里的“完成”是开源权重推理、搜索工具链和小规模系统评测复现完成，不是论文 GRPO/veRL 训练复现，也不是论文完整 benchmark 复刻。

大步 11 的推荐扩展档 100/300 条、LLM-as-Judge 和训练流程仍未执行；它们不属于本次大步 12 已完成的范围。

## 2. 公平对比协议

### 2.1 固定样本

- 数据集：`lmms-lab/FVQA`；
- revision：`bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5`；
- split：`train`；
- 样本数：50；
- 类别：25 条 `search_free` + 25 条 `search_required`；
- 顺序与大步 11 完全相同；
- 冻结输入 manifest：`/root/autodl-tmp/mmsearch_step11_inputs/eval_manifest.json`；
- 输入 manifest SHA-256：`dbc28df74f3a1a0b87fd435255fda8ed73455dfe3a3d465dce9539fad37564ab`；
- 大步 10 的受控 B/C 案例未进入本次比较。

### 2.2 Base 协议

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`；
- revision：`cc594898137f460bfe9f0759e9844b3ce807cfb5`；
- 本地路径：`/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`；
- BF16、SDPA、seed=0、`do_sample=false`、`max_new_tokens=512`；
- Direct Answer，无 Image Search、Text Search、Jina、Qwen3 摘要或其他外部工具；
- 推理期间设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；
- Answer parser 优先读取 `<answer>...</answer>`，标签缺失时透明使用完整非空回复；
- Exact Match：`prediction.strip().lower() == ground_truth.strip().lower()`。

固定协议：

`/root/autodl-tmp/multimodal-search-r1/reproduction/env/step12_base_comparison_protocol.json`

SHA-256：`3a80ee1fe4685cde68335a1ad336a3cf6f8f970a71f9664f6f66e22ee3d651f5`

### 2.3 MMSearch 对照协议

- 模型：`lmms-lab/MMSearch-R1-7B`；
- revision：`3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46`；
- 复用大步 11 自然按需搜索的不可变 50 条结果，不重新执行、不重选样本；
- 大步 11 completion manifest SHA-256：`f2747c945a022d578e3d053e7112a05937b0267894dc03b6f74adeabefe2ad87`；
- 大步 11 predictions SHA-256：`beda0f6a02f750b89a5a53cb9f39dcaac56dbe047e2fb1f4ecc01290d6ac48ff`；
- 大步 11 metrics SHA-256：`ec7c289417defc57ddd5a731e6bd1a2cb0a498013d281df863ad0aaccc5f0445`。

Prompt 因模型角色不同而不同：Base 接受 Direct Answer 指令，MMSearch 使用其工具控制指令。工具权限差异是本实验要测量的系统差异，不是隐藏偏差。

## 3. Base 模型下载与校验

Base 模型按 Hugging Face 精确 commit 固定并完成本地校验：

- revision：`cc594898137f460bfe9f0759e9844b3ce807cfb5`；
- 顶层文件：16 个；
- 总字节数：16,595,981,281；
- safetensors：5 片；
- 精确文件集合、总字节数、5 片 SHA-256、index 引用和 incomplete 文件检查全部通过；
- 本地模型目录不存在 `.incomplete`；
- `Qwen2_5_VLForConditionalGeneration` 由当前 Transformers 原生加载；
- 模型加载时间：3.379441 秒；
- `hf_device_map={"": "0"}`，所有参数完整驻留 `cuda:0`。

Revision 记录：

`/root/autodl-tmp/multimodal-search-r1/reproduction/env/step12_base_huggingface_revision.json`

SHA-256：`fb8c62723161017c615f95e48e5cb9a21d3c841c9eac0dd12e5fb29414b42a48`

Artifact validation：

`/root/autodl-tmp/outputs/step12_base_artifact_validation.json`

SHA-256：`bac53be73dbd7ff14b6b92030475ad452cf04577438df61502d87c8a320c482a`

下载过程中，Hugging Face Xet 路径速度异常缓慢，因此安全中断并保留部分文件，随后改用官方 Hugging Face 普通 HTTP 路径、相同精确 revision 续传。模型内容与最终校验口径没有改变。

## 4. Base 三档结果

| 阶段 | 样本数 | Correct | Accuracy | Search calls | Search Ratio | 验收 |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 0 | 0.0% | 0 | 0.0% | pass |
| 20 | 20 | 3 | 15.0% | 0 | 0.0% | pass |
| 50 | 50 | 7 | 14.0% | 0 | 0.0% | pass |

三个阶段均满足：

- 固定前缀和顺序完整；
- 逐样本 prediction 与阶段 manifest 原子落盘；
- 断点状态与 runner/protocol/input hash 绑定；
- Base 全部驻留 GPU；
- Qwen3 服务根 PID 保持不变；
- 外部工具调用为 0；
- 序列化证据扫描通过；
- `credentials_recorded=false`。

Base 50 条最终细项：

| 指标 | 结果 |
|---|---:|
| Evaluated | 50 |
| Correct | 7 |
| Accuracy | 14.0% |
| Search-Free | 7/25（28.0%） |
| Search-Required | 0/25（0.0%） |
| Image/Text Search calls | 0/0 |
| Average generation seconds | 0.285177 |
| Median generation seconds | 0.235119 |
| Average end-to-end case seconds | 0.323203 |
| Parser: answer tags | 24 |
| Parser: plain-text fallback | 26 |
| Empty response | 0 |

## 5. Base 与 MMSearch 最终对比

| 模型/模式 | Correct | Accuracy | Search calls | Search Ratio | Avg generation | Avg case |
|---|---:|---:|---:|---:|---:|---:|
| Base Direct Answer | 7/50 | 14.0% | 0 | 0.0% | 0.285177 s | 0.323203 s |
| MMSearch Natural On-demand Search | 20/50 | 40.0% | 70 | 70.0% | 2.952620 s | 22.281700 s |

MMSearch 相对 Base：

- 准确率提高 26.0 个百分点；
- 产生 70 次搜索调用；
- 平均生成与端到端耗时明显增加；
- 50 条中 11 条出现样本级工具失败，继续沿用大步 11 的容错主口径并计为错误。

### 5.1 分类别对比

| 类别 | N | Base | MMSearch | 差值 |
|---|---:|---:|---:|---:|
| Search-Free | 25 | 7/25（28.0%） | 9/25（36.0%） | +8.0 pp |
| Search-Required | 25 | 0/25（0.0%） | 11/25（44.0%） | +44.0 pp |

本次 26 个百分点的整体差异主要来自 Search-Required 子集。这与实验设计一致地表明：在这 50 条固定样本上，允许自然按需搜索显著改善需要外部知识的案例；它不等价于证明论文全部指标已经复刻。

### 5.2 逐样本配对结果

| 配对结果 | 数量 |
|---|---:|
| Both correct | 5 |
| Base only correct | 2 |
| MMSearch only correct | 15 |
| Both wrong | 28 |

因此不能只看总体均值：MMSearch 新增解决 15 条，但也有 2 条 Base 正确而 MMSearch 错误。

## 6. Strict EM 与回答格式说明

主指标严格使用预注册的字符串比较，不做别名、冠词、标点、翻译或实体同义归一化。Base 的前 5 条中已观察到类似：

- `The Tagus River` 对 `Tagus River`；
- `Pope Julius II` 对 `Julius II`；
- 韩文影视标题对英文 Ground Truth。

这些在 strict EM 下均为 false。本步没有因观察到结果而修改指标，也没有用 LLM Judge 重新打分。故 14.0% 应理解为本次固定 prompt 和严格字符串 EM 下的结果，而不是宽松语义正确率。

## 7. GPU 与服务状态

第 50 条 Base 阶段结束时：

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition；
- 总显存：97,887 MiB；
- Qwen3 EngineCore：约 50,384 MiB；
- Base 推理进程：约 17,294 MiB；
- 总使用：约 67,692 MiB；
- 空闲：约 29,561 MiB；
- Base torch 峰值 allocated：16,836,136,960 bytes；
- Base torch 峰值 reserved：17,406,361,600 bytes；
- Base 与 Qwen3 在同一张 GPU 上成功共驻，没有 OOM 或精度降级。

Base 运行进程结束后已释放。交接时：

- GPU 总使用约 50,393 MiB，空闲约 46,860 MiB；
- Qwen3 vLLM 服务主 PID 203574、EngineCore PID 203748 仍存活；
- 服务继续使用 CUTLASS、DeepGEMM=false、FlashInfer sampler=false/native、Thinking=false。

Qwen3 常驻不是 Base 比较的必要条件，也不是结果来源；它被保留是为了保持大步 9–11 的实验现场可复核。是否关停由用户决定。

## 8. 最终证据与独立审计

### 8.1 Base 正式产物

- Base completion：`/root/autodl-tmp/outputs/step12_base_direct_v1/step12_base_completion_manifest.json`
  - SHA-256：`359036672f80b6e4101c8ed4e64c5ca2fa5d3789e81c1d699c4be344c2ea0e07`
- Base predictions：`/root/autodl-tmp/outputs/step12_base_direct_v1/predictions.jsonl`
  - SHA-256：`c1a96773652e8ac378157188d9733e1e1ff207e2a388dece6b126a9b3271b980`
- Base failure summary：`/root/autodl-tmp/outputs/step12_base_direct_v1/failure_summary.json`
  - SHA-256：`e109779b8bab4edfadd7852fecd6fc995d38785bde5f84c075d91901d4ab764c`

### 8.2 配对比较与最终报告

- Comparison metrics：`/root/autodl-tmp/outputs/step12_comparison_v1/comparison_metrics.json`
  - SHA-256：`4ec7fa191d8ef850f830d496b388de15a0f467539042f7a1d2dc918abe0be829`
- Paired outcomes：`/root/autodl-tmp/outputs/step12_comparison_v1/paired_outcomes.jsonl`
  - SHA-256：`032ad455d1e99fab487db72f836f5374d50b4825985d07a933889999a31a2c42`
- Success/failure examples：`/root/autodl-tmp/outputs/step12_comparison_v1/success_failure_examples.json`
  - SHA-256：`eec7ad0b10a6bf0983af982e77b35dd2f9b2aecd45e12c567ed63b631920336d`
- Final report：`/root/autodl-tmp/outputs/step12_comparison_v1/final_report.md`
  - SHA-256：`b7ac6ca763fb67be27c378769241f037b64f59156a00868a5a8f171dbbf75c7c`
- Step 12 completion manifest：`/root/autodl-tmp/outputs/step12_comparison_v1/step12_completion_manifest.json`
  - SHA-256：`d5c415c1373780b715e847554aa02f48a6ab039b9988aae1b336213eba5bd821`
- Independent audit：`/root/autodl-tmp/outputs/step12_comparison_v1/step12_completion_audit.json`
  - SHA-256：`f1595905b07ef7f4d3928a80c9076897fcba67a86fe492ca507e41ecd193d2ea`

同一目录另含：

- `pip_freeze.txt`；
- `gpu_info.txt`；
- `git_commit.txt`；
- `model_and_dataset_revisions.json`。

### 8.3 独立审计结果

独立审计状态：`passed`。

审计确认：

- 50 个 Base prediction 文件全部存在且与最终 JSONL 一致；
- 固定输入顺序、data ID 和类别一致；
- 50 条 Base strict EM 全部独立重算；
- Base 外部工具调用为 0；
- Base 7 条正确、MMSearch 20 条正确；
- 四种 paired outcome 全部重算一致；
- MMSearch-Base 差值为 26.0 个百分点；
- Base/comparison completion 的字节数与 SHA-256 绑定一致；
- 共扫描 69 个正式文件：60 个 JSON、9 个文本；
- 禁用 JSON 字段、图片 base64、长 base64、原始 Jina 标记、Authorization 标记和已知凭据值命中均为 0；
- `credentials_recorded=false`。

## 9. 范围限制与正确表述

可以表述为：

> 在固定 50 条平衡 FVQA train 子集、固定 seed 和 strict EM 下，Qwen2.5-VL-7B-Instruct Direct Answer 得到 14.0%，MMSearch-R1-7B 自然按需搜索得到 40.0%，后者高 26.0 个百分点；完整工具调用、失败和证据口径已记录。

不能表述为：

- 已复刻论文全部 benchmark 数字；
- 已完成 GRPO/veRL 训练复现；
- 已证明任何分布上的普遍提升；
- Base 的语义准确率只有 14.0%；
- 工具失败已经被消除；
- 大步 10 的受控路线等同于自然策略或被纳入本次统计。

## 10. 当前交接点

- 大步 1–12：已完成；
- 原方案最低 50 条批评：已完成；
- Base 公平对比：已完成；
- 最终报告与独立审计：已完成；
- 推荐 100/300 条扩展：未执行；
- LLM-as-Judge：未执行；
- GRPO/veRL 训练：未执行；
- Qwen3 服务：仍在线，等待用户决定是否关停；
- 当前应暂停，不自动扩大评测、不自动启动训练、不自动关停服务。

下一步应由用户决定：结束并归档、关停 Qwen3 以停止显存占用，或另行批准扩展到 100/300 条/训练实验。
