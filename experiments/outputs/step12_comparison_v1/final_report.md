# MMSearch-R1 Step 12：Base 公平对比与最终报告

生成时间：2026-08-17T05:00:51.175326+00:00

## 对比范围

- 同一固定 50 条 FVQA train 子集，Search-Free/Search-Required 各 25 条；
- Base：Qwen2.5-VL-7B-Instruct Direct Answer，零工具；
- MMSearch：MMSearch-R1-7B 自然按需搜索，复用大步 11 不可变结果；
- strict EM：`prediction.strip().lower() == ground_truth.strip().lower()`；
- 大步 10 受控 B/C 不进入本对比。

## 核心结果

| 模型 | Correct | Accuracy | Search Calls | Search Ratio |
|---|---:|---:|---:|---:|
| Base Direct Answer | 7/50 | 14.0% | 0 | 0.0% |
| MMSearch On-demand Search | 20/50 | 40.0% | 70 | 70.0% |

MMSearch 相对 Base 的准确率差：26.0 个百分点。

| 类别 | Base Accuracy | MMSearch Accuracy | MMSearch-Base |
|---|---:|---:|---:|
| Search-Free | 28.0% | 36.0% | 8.0 pp |
| Search-Required | 0.0% | 44.0% | 44.0 pp |

## 配对结果

- Both correct：5；
- Base only correct：2；
- MMSearch only correct：15；
- Both wrong：28。

## 延迟与基础设施口径

- Base 平均模型生成时间：0.285177 秒；
- MMSearch 平均模型生成时间：2.95262 秒；
- Base 平均端到端 case 时间：0.323203 秒；
- MMSearch 平均端到端 case 时间：22.2817 秒；
- MMSearch 样本级工具失败：11 条，均已按大步 11 主口径计为错误；
- Base 不调用外部工具，因此 Search Ratio 固定为 0。

## 公平性与限制

1. 两模型使用相同图片、问题、Ground Truth、顺序、seed、greedy、max_new_tokens 和 strict EM；
2. Prompt 因角色不同而不同：Base 接受 Direct Answer 指令，MMSearch 接受其训练时工具控制指令；
3. 工具权限差异是本实验的设计目标，不是隐藏偏差；
4. 本子集来自 FVQA train 的确定性平衡 50 条，不是论文完整 benchmark；
5. MMSearch 结果受实时缩略图、Serper、Jina 与摘要服务状态影响，Base 不受这些外部工具影响；
6. 本报告是开源权重推理/系统复现，不是 GRPO/veRL 训练复现；
7. 未执行 LLM-as-Judge，主指标保持 strict EM。
