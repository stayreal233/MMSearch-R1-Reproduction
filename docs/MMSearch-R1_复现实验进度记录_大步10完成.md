# MMSearch-R1 复现实验进度记录：大步 10 完成

更新时间：2026-08-17 UTC

## 1. 当前结论

大步 10 已按用户批准的“自然负结果保留 + B/C 受控路径集成”口径完成：

- Case A Search-Free：自然模型策略 trace 通过；
- Case B Image Search：受控工具集成 trace 通过；
- Case C Text Search：受控工具集成 trace 通过；
- Case D Mixed Search：自然模型策略 trace 通过；
- 独立 Failure Case：自然模型策略 trace 已捕获并定位到 final answer 层；
- 五份 JSON、`cases.md`、完成 manifest 和独立审计 JSON 均已生成；
- Qwen3 健康、MMSearch/Qwen3 同卡共驻、真实工具、网络/缓存计数、凭据及正文泄漏检查均通过。

这里的“完成”严格限定为四条工程链路覆盖。它不代表自然策略已经复现出四种路径：在固定的 512 个 FVQA `train/search_required` 候选中，自然策略仍只产生 Search-Free 与 Mixed 两类路径，B/C 自然命中均为 0。受控 B/C 不得用于 Search Ratio，也不得写成模型自然选择。

这仍然不是论文完整训练复现。大步 11 批量评测与大步 12 Base 对比/最终报告尚未执行。

## 2. 为什么采用受控 B/C

先后执行了两个不可变的自然选例阶段：

1. v1 固定前 256 个候选：78 条 `answer`、178 条 `image_search → text_search → answer`，B/C 均为 0；
2. 经用户批准，v2 在相同数据池、物理顺序、seed、模型、prompt、placeholder 和严格 EM 规则下扩展到前 512 个候选，并从第 1 个候选重新跑；前 256 条与 v1 核心字段逐条一致。

v2 最终结果：

| 项目 | 结果 |
|---|---:|
| 扫描候选 | 512 |
| 全部终止为 answered | 512 |
| `answer` | 116 |
| `image_search → text_search → answer` | 396 |
| 自然 `image_search → answer` | 0 |
| 自然 `text_search → answer` | 0 |
| 严格 EM=true | 116 |
| 严格 EM=false | 396 |

因此没有继续盲目扩窗，也没有隐藏负结果。用户随后批准以受控路径完成 B/C 的工程集成覆盖。受控协议在任何正式工具调用前写入：

`/root/autodl-tmp/multimodal-search-r1/reproduction/env/step10_controlled_route_protocol.json`

SHA-256：`f18ac1cdb69e035243079c56b5abc4806609ada4d522e19ed0336d744ad5c48f`

## 3. 固定案例与语义

五个 data ID 在正式执行前固定，选择与答案正确性无关：

| Case | data_id | 执行语义 | 正式动作序列 | EM | 验收 |
|---|---|---|---|---:|---:|
| A Search-Free | `fvqa_train_0` | natural model policy | `answer` | true | pass |
| B Image Search | `fvqa_train_6` | controlled tool integration only | `image_search → answer` | true | pass |
| C Text Search | `fvqa_train_9` | controlled tool integration only | `text_search → answer` | false | pass |
| D Mixed Search | `fvqa_train_17` | natural model policy | `image_search → text_search → answer` | true | pass |
| Failure | `fvqa_train_32` | natural model policy | `answer` | false | expected failure captured |

说明：

- B 固定为自然扫描第 1 个 search-required 候选；控制器只注入第一动作 `image_search`，真实 Image Search 后由 MMSearch 生成 Answer；
- C 固定为自然扫描第 2 个 search-required 候选；控制器只注入第一动作 `text_search`，query 严格等于原问题，不使用 Ground Truth 或 candidate answers；真实 Serper/Jina/Qwen3 后由 MMSearch 生成 Answer；
- A、D、Failure 未使用控制器干预；
- 代表路径的验收独立于答案正确性，因此 C 的 EM=false 不影响 Text Search 工程路径通过；该结果已如实保留；
- B/C 的第 1 轮 trace 明确记录 `origin=controller_intervention`、`model_generated=false` 和 0 input/output tokens，后续回答轮才是模型生成。

固定选择清单：

`/root/autodl-tmp/mmsearch_step10_controlled_inputs/selection_manifest.json`

SHA-256：`b03e14d6f2dc99d6cc744f7f21ffa1a159bed44f99bd2adc443a5cd400b6bb29`

## 4. 正式工具结果

### A：Search-Free

- 工具调用：0；
- 外部网络请求：0；
- 最终 `answer`；
- EM=true；
- MMSearch 峰值分配显存：16031.31 MiB；
- 用时：1.940 秒。

### B：Image Search（受控）

- 官方 FVQA Google Lens cache：请求 5；
- 缩略图：5/5 成功；
- 缓存命中：0，实际缩略图网络尝试：5；
- Image Search 后模型直接 Answer；
- EM=true；
- MMSearch 峰值分配显存：16405.82 MiB；
- 用时：3.274 秒。

### C：Text Search（受控）

- Serper：5 个结果，cache miss，实际 API 请求 1；
- Jina：5/5 文档，cache miss，实际网络请求 5；
- Qwen3：5/5 summaries，0 failure，本机 completion 请求 5；
- 外部网络请求合计：6；本机 Qwen completion：5；
- Text Search 后模型 Answer；
- EM=false，路径与基础设施验收仍通过；
- MMSearch 峰值分配显存：16630.38 MiB；
- 用时：27.656 秒。

### D：Mixed Search（自然）

- 官方 Image Search：请求 5，4/5 返回，4 个缓存命中；
- 第 5 张历史 404 仍按用户已批准的 4/5 降级口径处理；
- Serper：5 个结果，cache hit；
- Jina：5/5，全部 cache hit；
- Qwen3 summaries：5/5，全部 cache hit；
- `image_search → text_search → answer`；
- EM=true；
- MMSearch 峰值分配显存：16947.04 MiB；
- 用时：6.229 秒。

## 5. Failure Case

固定 Failure 为 `fvqa_train_32`，是自然扫描中排除 A/B/C/D 后的首个独立、answered、严格 EM=false 案例。

- 动作序列：`answer`；
- terminal status：`answered`；
- 工具基础设施：成功；
- Exact Match：false；
- 分类层：`final_answer`；
- 审计表述：模型给出了最终答案，但按 `prediction.strip().lower() == ground_truth.strip().lower()` 为 false；没有证据支持将原因进一步归咎于视觉识别、query、搜索结果、网页读取或摘要层。

## 6. 模型、服务与同卡状态

- MMSearch：`lmms-lab/MMSearch-R1-7B`，revision `3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46`；
- Qwen3：`Qwen/Qwen3-32B-FP8`，revision `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`；
- Qwen3 服务按大步 9 相同参数重启：CUTLASS、DeepGEMM=false、FlashInfer sampler=false、Thinking=false；
- 本次服务主 PID：203574；
- 正式运行前后 Qwen3 health 均为 true；
- MMSearch load time：4.564 秒；
- MMSearch 全参数驻留 `cuda:0`，Qwen3 与 MMSearch 同卡验收通过；
- 最高案例侧 MMSearch 峰值分配显存：16947.04 MiB。

Qwen3 服务在大步 10 完成后仍保持运行，便于用户决定是否进入大步 11；在用户确认前不继续批量评测。

## 7. 安全与证据终验

完成 manifest 内 producer scan：

- 5 份案例 JSON + `cases.md`：6 个文件全部通过；
- 已知真实凭据值匹配：0；
- Authorization marker：0；
- raw Jina marker：0；
- data-image base64：0；
- 长 base64：0；
- forbidden JSON key：0。

独立终验进一步扫描正式输出目录的全部 JSON、`cases.md`、固定选择清单和受控协议，共 9 个文件：

- `known_credential_value_matches=0`；
- `authorization_marker_matches=0`；
- `raw_jina_marker_matches=0`；
- `data_image_base64_matches=0`；
- `long_base64_matches=0`；
- `forbidden_json_key_paths=[]`；
- `pass=true`。

真实 `serper.env` 仅通过 shell 静默 source；没有读取、打印或写入密钥值。

独立审计 JSON：

`/root/autodl-tmp/multimodal-search-r1/reproduction/env/step10_completion_audit.json`

SHA-256：`0bde319bb0ba317df7e3d4b42e714b965b24550e160d94146b59a01345f514c8`

## 8. 正式产物与 SHA-256

输出目录：`/root/autodl-tmp/outputs/step10_controlled_cases_v1`

| 产物 | SHA-256 |
|---|---|
| `case_A_search_free.json` | `d364308d0ecebd5a59bf5212faefd9bc53efae97bae969bb0927929c654c83e8` |
| `case_B_image_search.json` | `f8bc1a73de79f79c349a80db9532f2421e1bc7989274cf4d7244321ec6336195` |
| `case_C_text_search.json` | `96dfb0f20d05b2b86312dd406a547b5c86a7c3a01b25c9468876d8b89dcae46a` |
| `case_D_mixed_search.json` | `048850a912d344d5cf76183b9ca265ad67ce3c67d27b3a53649aca21632a7ad4` |
| `failure_case.json` | `9f1f431dbc3c767078a8dd8e1a3f8575a2b838a023d67dfb5ebbf5052a89af3e` |
| `cases.md` | `b78b6106dcf9c98b6d2c41d0bda7245f4d89b9d96831a8e59002acbc7e947d84` |
| `step10_completion_manifest.json` | `39645606cdb53d2cea7004aa56d5972541efd00b0922247182f76bb345e7edc2` |

自然 v2 负结果：

`/root/autodl-tmp/mmsearch_step10_v2/step10_candidate_selection_v2.json`

SHA-256：`a512d87f3dd24fc5a2714f58589a846f00d494132e678cc2479fd5410751b17d`

关键实现文件：

| 文件 | SHA-256 |
|---|---|
| `reproduction/scripts/step10_case_suite_qwen3.py` | `61329f76424942e240c9b633f19763413fefcc9b6e5944191527c09bc0a92c41` |
| `reproduction/scripts/step10_controlled_case_suite_qwen3.py` | `739d88a122514843aae2148ce5225575b959b46a45907c43b89d5ee8f90ffc31` |
| `reproduction/scripts/prepare_step10_controlled_inputs.py` | `862046a5980dc65bc2dd6c5508d3bc0927b4803b6ff024a24276bb5576c0c8c8` |
| `reproduction/scripts/prepare_step10_controlled_inputs_run_v2.py` | `8e14340c46473f71e358d7d6008d100d8df2e0e12b00c395b1bd0009cc24e2e5` |

## 9. 已知偏差与不能声称的内容

1. B/C 是控制器注入首个工具动作的集成案例，不是自然控制流；
2. B/C 不能用于论文 Search Ratio 或自然路径频率统计；
3. 512 条自然扫描未找到 B/C，是必须保留的真实负结果；
4. C 工程路径通过但 EM=false，不能写成答案正确；
5. D 的 Image Search 是官方 FVQA Lens 缓存集成且为 4/5 降级，不等价于实时 Lens 搜索 5/5；
6. Failure 只能定位到 final answer 层，不能无证据推断更上游的失败原因；
7. 当前属于开源权重推理与系统复现，不是论文 GRPO/veRL 训练复现。

## 10. 下一步与暂停点

大步 10 到此完成并暂停。不得自动进入大步 11。

大步 11 需要在用户确认后：

1. 先固定批量评测 ID manifest；
2. 严格按 `5 → 20 → 50 → 100 → 300` 扩容；
3. 每档检查调用量、缓存命中率、失败率、网络额度和成本；
4. Search Ratio 必须只使用自然模型策略结果，明确排除本步受控 B/C；
5. 至少完成 50～100 条后再汇总 Search-Free/Search-Required Accuracy、Search Ratio、调用次数与失败分析；
6. 每档原子落盘并保留断点，出现认证、限流、工具、CUDA/OOM或证据完整性问题立即暂停。
