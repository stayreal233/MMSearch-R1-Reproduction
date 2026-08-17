# MMSearch-R1 复现实验进度记录：大步 11 完成

更新时间：2026-08-17 UTC

## 1. 当前结论

大步 11 已按两份基准文档和用户批准的样本级容错口径完成，并在文档规定的最低强制规模 50 条处暂停：

- 评测 ID、顺序、类别、seed、模型、工具和指标定义均在正式运行前固定；
- 按 `5 → 20 → 50` 三档递增执行，每档均生成 predictions、metrics、failure summary 和阶段 manifest；
- 50 条均为 MMSearch checkpoint 的自然策略执行；
- 大步 10 的受控 B/C 未进入 Search Ratio 或准确率统计；
- 严格 Exact Match、Search Ratio、分类别准确率、搜索调用量、平均轮数、缓存/网络调用与失败分层均已汇总；
- 50 个逐样本 trace、断点状态、最终 predictions/metrics/failure summary、完成 manifest 和独立终验 JSON 均已生成；
- 50 条中 11 条出现样本级真实工具故障。它们没有重试，均按端到端错误计入固定分母，并另行报告健康回答口径；
- 全局硬故障为 0：没有 Serper 鉴权/限流、Qwen3 健康/PID、CUDA/OOM、同卡驻留、协议/hash 或证据泄漏故障；
- 独立终验逐条复算 50/50 样本并通过。

这里的“大步 11 完成”是指达到原方案最低强制批评规模 50 条。推荐扩展档 100 和 300 条尚未执行。大步 12 Base 对比与最终报告也尚未执行。

这仍然是开源权重推理与系统复现，不是论文 GRPO/veRL 训练复现，也不能将本次 50 条 FVQA train 子集结果直接等同于论文正式 benchmark 数字。

## 2. 固定评测协议

### 2.1 数据与选择规则

- 数据集：`lmms-lab/FVQA`；
- 固定 revision：`bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5`；
- split：`train`；
- 固定 50 条：25 条 `search_free` + 25 条 `search_required`；
- 排除大步 10 的 A/B/C/D/Failure 五个 data ID；
- 要求官方 FVQA Image Search cache 至少有固定 top-5 元数据；
- 使用 `sha256("mmsearch-step11-seed0|" + data_id)` 确定性排序；
- 以类别交替方式生成固定前缀：5 条为 3/2，20 条为 10/10，50 条为 25/25；
- seed=0、`do_sample=false`、BF16、SDPA、最多 3 轮、Image/Text 各最多调用 1 次；
- Image Search 使用既有 4/5 降级口径；
- Text Search 为 Serper top-5 → Jina → Qwen3 摘要，Qwen3 Thinking=false、temperature=0；
- Exact Match 固定为：`prediction.strip().lower() == ground_truth.strip().lower()`；
- Search Ratio 固定为：`sum(image_search_calls + text_search_calls) / (N * 2) * 100`。

冻结输入 manifest：

`/root/autodl-tmp/mmsearch_step11_inputs/eval_manifest.json`

SHA-256：`dbc28df74f3a1a0b87fd435255fda8ed73455dfe3a3d465dce9539fad37564ab`

### 2.2 v1 停止与 v2 容错修订

v1 在第 1 条 `fvqa_train_4724` 按原硬停止口径暂停：

- 自然路径：`image_search → text_search`；
- Image Search 4/5，另 1 张 HTTP404；
- Jina 4/5，另 1 页 HTTP403；
- Qwen summary 3/4，另 1 次 HTTP400；
- Qwen 服务本身保持健康；
- terminal=`text_search_hard_failure`，没有 final answer；
- 没有自动重试。

用户随后批准推荐容错口径。v2 在恢复前预注册，并保持 50 个 ID、顺序、模型、prompt、工具和指标不变：

- 原第 1 条失败证据按固定 SHA 导入，不重跑、不产生第二次付费或网络请求；
- 从 eval index 2 继续；
- 普通缩略图/Jina/单次 Qwen completion 故障按样本级失败持久化后继续；
- 非回答或工具失败样本在端到端准确率中计为错误；
- 另报健康回答准确率、样本工具失败率和组件失败数；
- Serper 401/403/429、Qwen health/PID、CUDA/OOM、同卡驻留、协议/hash 和泄漏问题仍是全局硬停止。

v1 协议 SHA-256：`f2fc533b824c65d5102fc10dbaebe0c3069242b00f9178c1417f6f0935c6000e`

v2 协议 SHA-256：`d8d74572a8b369b37f44a66bc314d0e3c2e6bfef081122326a34e287ad92ff17`

独立终验确认：恰好 1 条 v1 证据以原预测 SHA `534863c9b57e9fd0f9f966bef1cbc80fd9e53551fa8c822773934eea6b1a359e` 导入，且 `reexecuted=false`。

## 3. 三档扩容结果

| 阶段 | 样本数 | 端到端正确 | Accuracy | Search Ratio | 样本工具失败 | 全局硬故障 | 验收 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 1 | 20.0% | 60.0% | 2 | 0 | pass |
| 20 | 20 | 7 | 35.0% | 65.0% | 6 | 0 | pass |
| 50 | 50 | 20 | 40.0% | 70.0% | 11 | 0 | pass |

三个阶段均满足：

- 固定前缀完整；
- prediction trace 与 checkpoint 原子落盘；
- 断点恢复只扩展前缀，不重跑已完成样本；
- 阶段 Qwen health=true、Thinking=false；
- 同卡证据通过；
- serialized evidence scan pass；
- `credentials_recorded=false`。

## 4. 50 条最终指标

### 4.1 端到端指标

| 指标 | 结果 |
|---|---:|
| Evaluated | 50 |
| Strict EM correct | 20 |
| End-to-end Accuracy | 40.0% |
| Image Search calls | 40 |
| Text Search calls | 30 |
| Total Search calls | 70 |
| Search Ratio | 70.0% |
| Average turns | 2.18 |

### 4.2 分类别指标

| 类别 | N | Correct | Accuracy | Search calls | Search Ratio | Avg turns |
|---|---:|---:|---:|---:|---:|---:|
| Search-Free | 25 | 9 | 36.0% | 29 | 58.0% | 1.84 |
| Search-Required | 25 | 11 | 44.0% | 41 | 82.0% | 2.52 |

### 4.3 自然路径分布

| 动作序列 | 数量 |
|---|---:|
| `answer` | 10 |
| `image_search` | 4 |
| `image_search → answer` | 6 |
| `image_search → text_search` | 7 |
| `image_search → text_search → answer` | 23 |

以上 50 条全部标记为：

- `selection.execution_mode=natural`；
- `route_origin=natural_model_policy`；
- `controller_intervention=false`。

因此本步 Search Ratio 是自然策略统计，不包含大步 10 的受控 B/C。

## 5. 容错后的双口径报告

端到端口径保留固定分母 50：

- 样本级工具失败：11/50，22.0%；
- 非 answered terminal：11；
- 这些样本全部按 Exact Match=false 计入总准确率；
- 因工具异常导致网络计数不完整：6 条。

健康回答口径仅包含 `terminal=answered`、`tool_infrastructure_success=true` 且网络计数完整的样本：

- 健康回答样本：39；
- 其中严格 EM 正确：20；
- Healthy Answered Accuracy：51.2821%。

两种数字不得混用：40.0% 是固定 50 条端到端结果，51.2821% 是排除样本级基础设施失败后的条件准确率。主结果仍是 40.0%，健康口径只用于解释真实工具波动的影响。

## 6. 失败分析

严格 EM=false 共 30 条，按可直接观察的最早证据层分类：

| 失败层 | 数量 | 解释 |
|---|---:|---|
| final answer | 19 | 工具链健康并产出答案，但严格字符串 EM=false；不进一步臆测上游原因 |
| image search results | 4 | Image Search 未满足固定 5 请求、至少 4 返回的降级合同 |
| Jina reader | 6 | Jina 未满足本样本的固定文档/无失败合同 |
| Qwen summarization | 1 | 摘要组件状态未满足固定合同 |

组件级计数与失败层计数语义不同：

- Image item failures：26；其中很多是 4/5 的可恢复部分失败，不一定导致整条样本失败；
- Jina page failures：6；
- Qwen summary failures：3；
- Serper component failures：0；
- hard stage image_search：4；
- hard stage text_search/Jina/Qwen：7。

Failure summary 没有把所有错误都归因于模型。工具链明确失败时按实际组件归类；只有基础设施健康并输出错误答案时才归为 final answer。

## 7. 网络、缓存与调用量

50 条累计：

| 项目 | 数量 |
|---|---:|
| 外部网络请求合计 | 379 |
| 缩略图网络尝试 | 199 |
| 缩略图缓存命中 | 1 |
| Serper API 请求 | 30 |
| Serper cache hit | 0 |
| Jina 网络请求 | 150 |
| Jina cache hit | 0 |
| 本机 Qwen completion 请求 | 120 |
| Qwen summary cache hit | 0 |
| 工具 HTTP 请求合计（外部 + 本机） | 499 |

本步没有自动重试失败样本，因此这些数字不包含为第 1 条失败进行的重复调用。外部 API/网页的实时状态会影响端到端结果，这是本次复现的真实系统条件之一。

## 8. 模型、服务与 GPU 状态

- MMSearch：`lmms-lab/MMSearch-R1-7B`，revision `3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46`；
- Qwen3：`Qwen/Qwen3-32B-FP8`，revision `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`；
- Qwen3 启动合同继续沿用大步 9/10：CUTLASS、DeepGEMM=false、FlashInfer sampler=false/native、Thinking=false；
- Qwen 服务主 PID：203574；EngineCore PID：203748；
- 第 50 条阶段结束时模型列表和 health 均通过，Thinking=false；
- 完成记录中的同卡快照：Qwen EngineCore 约 50384 MiB，MMSearch 运行进程约 20804 MiB；
- MMSearch 本轮峰值分配约 16.50 GiB，峰值 reserved 约 19.64 GiB；
- 批量运行进程结束后，MMSearch 已释放；当前仅 Qwen EngineCore 保持运行，`nvidia-smi` 约 50384 MiB；
- 本地 `/v1/models` 在交接时返回 HTTP 200。

Qwen EngineCore 持续占用显存是服务常驻的预期行为，不是泄漏，也不是本步失败原因。它被保留是为了让用户决定是否进入大步 12；在用户确认前不主动关停。

## 9. 安全与独立终验

独立终验覆盖：

- 50/50 prediction 文件的路径、字节数和 SHA-256；
- 冻结 input manifest 的 50 个 data ID、顺序、source row、类别和选择 rank；
- 25/25 类别平衡；
- 每条严格 EM 重算；
- 每条 action sequence、Image/Text 调用数、turn 数与自然策略标记；
- 5/20/50 三阶段 manifest；
- protocol、runner、state、predictions、metrics 和 failure summary 的 hash 绑定；
- 最终 JSONL 与 50 个逐样本 JSON 的逐字一致性；
- Search Ratio、route、分类别指标、样本失败率和健康回答准确率重算；
- v1 第 1 条以固定 SHA 导入且没有重执行。

独立序列化扫描共检查 61 个正式文件：

- forbidden JSON key paths：0；
- data-image base64：0；
- 长 base64：0；
- raw Jina marker：0；
- Authorization marker：0；
- 已知真实凭据值匹配：0；
- pass=true。

独立审计 JSON 自身随后再次扫描，以上六类命中也均为 0，pass=true。

真实 `serper.env` 仅通过 shell 静默 source；没有打印、写入或在文档中记录密钥值。

## 10. 正式产物与 SHA-256

输出目录：`/root/autodl-tmp/outputs/step11_eval_v2`

| 产物 | SHA-256 |
|---|---|
| `step11_completion_manifest.json` | `f2747c945a022d578e3d053e7112a05937b0267894dc03b6f74adeabefe2ad87` |
| `step11_completion_audit.json` | `a23d10f0c80004c62f26fb2cb1dea81640b3712e8020c711d9104d8048441082` |
| `predictions.jsonl` | `beda0f6a02f750b89a5a53cb9f39dcaac56dbe047e2fb1f4ecc01290d6ac48ff` |
| `metrics.json` | `ec7c289417defc57ddd5a731e6bd1a2cb0a498013d281df863ad0aaccc5f0445` |
| `failure_summary.json` | `386271d9c4e9f7e8b19b929c2bc87ed264a9db82bd78320b4badd52e4eff9c07` |
| `stage_5_manifest.json` | `9ab2f11cbc762cf72594dcb5bc529a7278fa9259335d4f6aa32af616045fad0d` |
| `stage_20_manifest.json` | `b7e97c7641b3987cda180453176fd00758ea4cec2ba4fc15b37796195d942b23` |
| `stage_50_manifest.json` | `cb3926c330117c4254d3db8c6d869bbd4c51d1eea797d4963ee9e96769fa51ae` |
| `state.json` | `9885f17f47948d100adfa608a2e7e9471385fee3c2f304f47c77f08fdc3e8961` |

关键协议、输入与实现：

| 文件 | SHA-256 |
|---|---|
| `reproduction/env/step11_eval_protocol.json` | `f2fc533b824c65d5102fc10dbaebe0c3069242b00f9178c1417f6f0935c6000e` |
| `reproduction/env/step11_eval_protocol_v2.json` | `d8d74572a8b369b37f44a66bc314d0e3c2e6bfef081122326a34e287ad92ff17` |
| `/root/autodl-tmp/mmsearch_step11_inputs/eval_manifest.json` | `dbc28df74f3a1a0b87fd435255fda8ed73455dfe3a3d465dce9539fad37564ab` |
| `reproduction/scripts/prepare_step11_eval_inputs.py` | `515346a2adc37b83edb37dac17c6b77bc86082d8c3b4f84b5239b9384623016e` |
| `reproduction/scripts/step11_batch_eval_qwen3.py` | `c217da46af089fcc709065b69d951c75ca425ac2b27ae64ea2f8b159fb5b086b` |
| `reproduction/scripts/step11_batch_eval_qwen3_v2.py` | `7eb98fcd3aa98bdf83fcc92d9e5ff6b405a874c8b90b9cb163a4f326f715aca1` |
| `reproduction/scripts/audit_step11_completion.py` | `4c7b982e4bbf4398a32473f32d8b852661e7e56f4eeeff3bafb5c2966a61314e` |
| `reproduction/scripts/audit_step11_completion_v2.py` | `ea7616db35fc2da7754313bddbabbafc0f8808063d14e31d95e6ccd9a5523921` |

说明：首个独立审计实现把实际 v1/v2 的自然策略字段 `execution_mode=natural`、`route_origin=natural_model_policy` 误按早期标签名校验，因此第一次审计在读到第 1 条后停止，没有改变任何实验产物。由于宿主 `apply_patch` 更新路径出现 bwrap 故障，新增兼容入口先独立验证这两个真实字段、50 条无 controller intervention、恰好 1 条 SHA 绑定 v1 import 且 `reexecuted=false`，再执行其余全部终验。两份脚本均保留并记录 SHA，避免隐藏审计过程。

## 11. 已知限制与不能声称的内容

1. 本次只完成最低强制规模 50 条，未执行推荐的 100/300 条扩展；
2. 样本来自确定性选择的 FVQA train 平衡子集，不是论文完整评测集；
3. 主 Accuracy=40.0% 包含 11 条真实工具失败并将其计为错误；健康回答条件准确率 51.2821% 不能替代主结果；
4. Strict EM 不做别名、日期格式、词形或语义等价匹配，因此部分语义接近答案仍记 false；
5. 外部网页和缩略图具有时态性，本次 379 个外部请求的成功率会影响结果；
6. Image Search 仍是官方 FVQA Lens cache 元数据 + 缩略图下载，不等价于实时 Google Lens 5/5；
7. Search Ratio=70.0% 仅描述这 50 条自然策略子集，不应外推为论文总体 Search Ratio；
8. 大步 10 的受控 B/C 明确排除，不能用本步结果倒推其自然选择频率；
9. 本次属于开源权重推理/工具系统复现，不是训练复现。

## 12. 下一步与暂停点

大步 11 到此完成并暂停。不得自动进入大步 12。

大步 12 需要在用户确认后：

1. 固定 `Qwen2.5-VL-7B-Instruct` Base 的精确模型 revision、环境与输入子集；
2. 从本步固定 50 条中预注册公平对比的 20～50 条前缀，不得按结果挑样；
3. Base 与 MMSearch 使用同一图片、问题、Ground Truth、strict EM 和 generation 参数；
4. 明确 Base 是否允许工具。推荐先做 no-tool Base 主对照；如增加 tool-enabled Base，应作为独立次级实验；
5. 汇总 Base/MMSearch 准确率、工具调用、延迟、显存、失败和公平性限制；
6. 生成最终报告、完整环境版本、模型/dataset revisions、失败分析与证据索引；
7. 完成大步 12 后再次暂停。

Qwen3 服务当前仍保持运行。是否继续复用、先关闭释放约 50 GiB 显存，或另启 Base 环境，应由用户在进入大步 12 时确认。
