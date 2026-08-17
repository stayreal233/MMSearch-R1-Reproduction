# MMSearch-R1 完整复现与工程落地全记录 (Reproduction Project)

本项目包含了针对 **MMSearch-R1**（基于多模态大模型的自主搜索与推理框架）在单卡高性能云环境（AMD Radeon PRO V620 / PRO 6000 96G 显存）上的**全链路复现、代码踩坑修复、真实多模态搜索增强、基线对照评测（大步 1~12）与交互式 Demo**。

---

## 📁 目录结构

```text
.
├── README.md                                  # 项目总览与复现指南
├── docs/                                      # 复现全过程文档与实验报告
│   ├── MMSearch-R1_PRO6000-96G_5.98元每小时_复现方案.md
│   ├── MMSearch-R1_八大步复现过程与原仓库踩坑修复详解.md
│   ├── MMSearch-R1_复现实验进度记录_大步1-8完成.md
│   ├── MMSearch-R1_复现实验进度记录_大步9完成.md
│   ├── MMSearch-R1_复现实验进度记录_大步10完成.md
│   ├── MMSearch-R1_复现实验进度记录_大步11完成.md
│   ├── MMSearch-R1_复现实验进度记录_大步12完成.md
│   └── MMSearch-R1_开源权重推理复现与训练流程报告.md
├── src/                                       # 核心代码库与复现脚本
│   ├── multimodal_search_r1/                  # 官方核心模型代码（已修复 Bug）
│   └── reproduction/                          # 自动化复现流水线、评估器与自研工具
│       ├── env/                               # 依赖配置与环境定义
│       ├── mmsearch_tools/                    # Serper / Jina 检索与 Qwen3 摘要重构工具
│       ├── scripts/                           # 各大步自动化运行与控制实验脚本
│       └── tests/                             # 单元测试与端到端 Smoke Tests
├── demo/                                      # 基于 Gradio 的交互式 Demo
│   ├── app.py
│   └── run_demo.sh
└── experiments/                               # 各阶段实验产物与评估数据
    ├── outputs/                               # 大步 9-12 的推理预测、对比指标与评审清单
    ├── step10_controlled_inputs/              # 大步 10 跨模态搜索对比测试输入用例
    └── step11_inputs/                         # 大步 11 50例 FVQA 端到端验证输入与标注
```

---

## 🚀 核心复现成果总览

1. **环境与原生 Bug 彻底修复**：
   - 修复了官方原仓库在 flash-attention / transformers 最新版本下的 forward pass 兼容性问题。
   - 解决了多模态图像分词与多步多轮检索上下文拼接中的溢出与缓存不一致缺陷。
2. **真实多模态检索与文本摘要流**：
   - 集成 Google Serper API + Jina Reader + 本地 `Qwen3-Summary` 模块，替代原有 Mock 伪搜索，实现真正与外网联动的自主图文检索闭环。
3. **严格控制变量评测（50 例 FVQA Benchmark）**：
   - **MMSearch-R1 (真实多模态搜索增强)**: 最终 Accuracy 达到 **78.0%**。
   - **Base Direct (无搜索直出基线)**: Accuracy 为 **38.0%**。
   - **性能提升**: 真实多模态搜索带来了 **+40.0%** 的绝对精度跃升。
4. **即开即用的交互式系统**：
   - 提供了支持多轮对话、上传自定义图片、实时查看思考过程（CoT）与检索源引用的 Gradio 交互界面。

---

## 🛠️ 快速开始

### 1. 环境安装
```bash
conda create -n mmsearch_infer python=3.10 -y
conda activate mmsearch_infer

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r src/reproduction/env/requirements.txt
```

### 2. 运行单步测试与评测
```bash
# 运行环境与模型加载检查
python src/reproduction/scripts/step9_qwen3_summary_smoke.py

# 运行 50 例 FVQA 端到端评测
python src/reproduction/scripts/step11_eval_v2.py
```

### 3. 启动交互式 Web Demo
```bash
cd demo
bash run_demo.sh
```

---

## 📑 详细文档索引
- [01. PRO6000-96G 复现方案](docs/MMSearch-R1_PRO6000-96G_5.98元每小时_复现方案.md)
- [02. 八大步踩坑修复详解](docs/MMSearch-R1_八大步复现过程与原仓库踩坑修复详解.md)
- [03. 大步 1~8 完成进度记录](docs/MMSearch-R1_复现实验进度记录_大步1-8完成.md)
- [04. 大步 9 检索重构进度](docs/MMSearch-R1_复现实验进度记录_大步9完成.md)
- [05. 大步 10 跨模态案例分析](docs/MMSearch-R1_复现实验进度记录_大步10完成.md)
- [06. 大步 11 50例评测报告](docs/MMSearch-R1_复现实验进度记录_大步11完成.md)
- [07. 大步 12 综合对比报告](docs/MMSearch-R1_复现实验进度记录_大步12完成.md)
- [08. 开源权重推理与训练全流程总结](docs/MMSearch-R1_开源权重推理复现与训练流程报告.md)
