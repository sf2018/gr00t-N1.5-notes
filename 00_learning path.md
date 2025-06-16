# GR00T N1 / N1.5 学习路线

---

## 第 1 步：了解模型模块结构（结合模型图）

**目标：**
- 看懂模型的组成部分（VLM、DiT、Projector、Diffusion Policy 等）
- 明白每个模块的输入/输出
- 知道哪些部分会被微调，哪些是冻结的

**推荐代码路径：**
- `gr00t/model/gr00t_n1.py`：模型主结构
- `gr00t/model/components/`：包含 vision, language, DiT, projector, diffusion 子模块

**一句话总结：**
> 语言 + 图像 → 融合编码 → 扩散预测动作（Diffusion）。

---

## 第 2 步：理解数据格式（LeRobot）

**目标：**
- 看懂 `.parquet` 文件字段：如 `observation.state`、`action`、`timestamp`
- 理解 `meta/modality.json` 中维度解释方式
- 知道视频如何与数据对齐

**阅读：**
-  `.parquet` 文件查看前几条记录
  - `gr00t/data/schema.py`
  - `gr00t/data/loader.py`

---

## 第 3 步：理解训练流程

**目标：**
- 看懂训练主入口 `scripts/gr00t_finetune.py`
- 明白 `TrainRunner` 如何初始化模型、加载数据、启动 Trainer
- 理解 `TrainingArguments` 中 `max_steps`、`gradient_accumulation_steps` 的作用

**阅读：**
- `gr00t/experiment/runner.py`
- `scripts/gr00t_finetune.py`

---

## 第 4 步：推理与部署流程

**目标：**
- 明白如何通过 `get_action(obs)` 获取动作
- 推理是“多轮调用 get_action()”，而不是一步生成完整动作

**阅读：**
- 服务端：`scripts/inference_service.py`
- 客户端：`scripts/eval_gr00t_so100.py`

---

## 第 5 步：分析 N1 vs N1.5 改进点

**目标：**
- 知道 N1.5 的具体升级点：
  - 冻结 VLM
  - LayerNorm Adapter 简化
  - 加入 FLARE 表征对齐目标
  - 引入 DreamGen 合成数据
- 明白这些改进如何提升泛化与语言理解能力

**资源：**
- 官方介绍：[GR00T N1.5 | GEAR @ NVIDIA](https://research.nvidia.com/labs/gear/gr00t-n15/)
- 比较代码（如有 N1.5 分支）：
  - `gr00t/model/gr00t_n1.py`
  - `gr00t/model/gr00t_n15.py`（或其他变种）

---

## 第 1 天：模型结构 & 数据流向

**目标：**

理解模型组成、数据流向、模块之间的连接。

理解 VLM + DiT + Diffusion 的组合逻辑。

**任务：**

阅读 gr00t/model/gr00t_n1.py 中的 GR00T_N1 类。

理解以下模块：

self.vision_model

self.language_model

self.projector

self.policy（DiffusionPolicy）

检视每个模块的输入输出。

阅读 GR00T N1.5 官网介绍，理解架构升级点。





第 2 天：数据结构与加载机制

目标：

理解 parquet 文件结构及其字段含义。

理解 modality.json 的作用和格式。

任务：

读取 parquet 示例文件：

import pandas as pd
df = pd.read_parquet("your/path/episode_000000.parquet")
print(df.head())

阅读你已有的 GR00T LeRobot 数据格式文档。

阅读源码：

gr00t/data/schema.py

gr00t/data/loader.py

练习：

编写读取数据并分析 action/state 的代码。

第 3 天：训练流程与配置理解

目标：

熟悉 CLI 启动训练的逻辑与参数设置。

理解从 CLI 到 loss 计算的调用链。

任务：

阅读以下脚本与模块：

scripts/gr00t_finetune.py

gr00t/experiment/runner.py 中的 TrainRunner

gr00t/trainers/dual_brain_trainer.py

理解关键参数：

max_steps

gradient_accumulation_steps

compute_dtype

global_batch_size

理清调用链：

CLI -> TrainRunner -> Trainer -> forward & loss -> backprop

练习：

用你自己的数据运行一次微调训练，并记录 loss 变化。

第 4 天：部署流程 & N1.5 深度比较

目标：

掌握多轮推理部署流程。

明确 N1 与 N1.5 的区别与提升。

任务：

阅读部署相关代码：

服务端：scripts/inference_service.py

客户端：scripts/eval_gr00t_so100.py

客户端核心类：gr00t/eval/service.py 中 ExternalRobotInferenceClient

阅读 N1.5 官方介绍中：

模型结构变化（VLM 冻结 + Adapter）

FLARE 目标、DreamGen 数据增强

各类评估对比数据

整理对比表格：N1 vs N1.5

练习：

实际部署一段指令测试模型响应效果，如“抓绿色方块”。

附录建议资源

理论参考：Diffusion Policy 原理

代码仓库：GR00T GitHub

数据分析工具：pandas, matplotlib, tensorboard

训练日志工具：wandb



