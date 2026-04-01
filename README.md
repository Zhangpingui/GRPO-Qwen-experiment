# GRPO 数学推理模型微调项目 (可自行运行版)

## 项目背景

本项目旨在通过 **GRPO (Generative Reward Policy Optimization)** 强化学习算法，提升大语言模型（如 `Qwen2.5-0.5B-Instruct`）在数学推理任务上的逻辑输出和格式规范能力。
原项目是基于 Jupyter Notebook 实现的实验代码，为了增强代码的可维护性和复用性，将其拆分重构为模块化的 Python 项目。通过多维度的奖励函数（如结果正确性奖励、XML 格式规范奖励、推理过程完整度奖励等），引导模型在解答数学题时能够以结构化的 `<reasoning>` 和 `<answer>` 标签形式输出完整的推导过程和最终答案。

## 目录结构

```text
grpo_project/
├── config.py             # 项目统一配置：模型路径、输出路径、Wandb信息以及 Prompt 模板和超参数
├── data.py               # 数据处理模块：负责 gsm8k 数据集的下载、解析以及系统 Prompt 的组装
├── rewards.py            # 强化学习奖励机制：定义了所有针对模型输出格式和正确性的 reward 函数
├── train.py              # 主训练入口：拼装各模块，拉起 GRPOTrainer 并执行微调与模型保存
├── inference_test.py     # 推理测试脚本：用于测试未经 GRPO 微调的原始模型表现，以供对比
└── README.md             # 项目说明文档 (本文档)
```

## 模块详解

### 1. `config.py` (配置中心)
集中管理了实验所需的所有超参数，包括：
- 模型名称：`Qwen2.5-0.5B-Instruct`
- 训练 Prompt 模板和预期的 XML 格式输出模板
- `GRPOConfig` 的所有关键参数（学习率、Batch Size、梯度累加、梯度裁剪等）
- Wandb 日志记录的项目名称及 API Key（需自行填写）。

### 2. `data.py` (数据处理)
基于 `datasets` 库加载经典的数学推理数据集 `gsm8k`。
该模块会自动清理和格式化问题内容，并从原本的标注中提取出纯净的答案部分，最后合并 `SYSTEM_PROMPT` 为适合模型消化的格式。

### 3. `rewards.py` (奖励函数)
这是 GRPO 算法的核心部分，包含了 5 个不同的评分函数（共计最高可获得 3.5 分的基础评分体系）：
- **`correctness_reward_func`**: 答案完全正确且符合格式（2 分）
- **`int_reward_func`**: 答案包含在 `<answer>` 标签中且为整数（0.5 分）
- **`strict_format_reward_func`**: 答案严格符合特定的 XML 格式和换行要求（0.5 分）
- **`soft_format_reward_func`**: 格式有一定容差，包含 XML 标签即可（0.5 分）
- **`xmlcount_reward_func`**: 按标签出现的次数和位置进行打分（最高 0.5 分）

### 4. `train.py` (训练入口)
主运行程序，它会：
1. 根据 `config.py` 判断是否登录并初始化 Weights & Biases (wandb) 用于实验追踪。
2. 加载 `gsm8k` 数据集。
3. 加载基础模型和 Tokenizer。
4. 使用 `GRPOTrainer` 结合收集到的奖励函数群和数据集进行训练。
5. 训练完成后将模型保存到 `outputs/Qwen2.5-0.5B-reasoning-GRPO` 目录。

### 5. `inference_test.py` (原始模型测试)
一个独立的评估脚本。你可以运行此脚本直接观察 `Qwen2.5-0.5B-Instruct` 在未经过我们设定的 GRPO 强化学习前，面对数学推理问题时的原始输出格式（通常比较杂乱），以此作为训练前后的效果对比 Baseline。

## 运行指南

### 前置依赖
在运行本项目前，请确保你已经安装了以下关键库（建议在虚拟环境中运行）：
```bash
pip install torch transformers datasets trl wandb
```
*(如果需要使用 vLLM 加速，请确保安装对应的 `vllm` 版本，并在 `config.py` 中开启 `use_vllm=True`)*

### 步骤 1：配置 Wandb (可选)
如果需要使用 Wandb 跟踪你的训练损失和奖励变化，请在当前项目根目录下基于 `.env.example` 创建一个 `.env` 文件，或者直接在终端设置环境变量：

1. 复制 `.env.example` 为 `.env` 文件：
```bash
cp .env.example .env
```
2. 编辑 `.env` 文件，填入你的项目名称和 API Key：
```env
WANDB_API_KEY=你的_wandb_api_key
WANDB_PROJECT=你的项目名称
```
*提示：由于本项目涉及读取 `.env` 文件，你可能需要安装 `python-dotenv` 库 (`pip install python-dotenv`)。*

### 步骤 2：运行 Baseline 推理测试 (可选)
运行测试脚本，查看模型在未进行 GRPO 强化学习时的乱序输出表现：
```bash
python inference_test.py
```

### 步骤 3：开始训练
运行主入口文件开始微调过程：
```bash
python train.py
```
训练过程可能需要一定时间（取决于你的显卡性能），训练完成后，模型会自动保存在 `outputs/Qwen2.5-0.5B-reasoning-GRPO` 中。

## 注意事项
- 本项目针对 `Qwen2.5-0.5B-Instruct` 做了参数适配。如果需要替换为更大的模型，请务必关注显存占用，并适时在 `config.py` 中调小 `per_device_train_batch_size` 配合增大 `gradient_accumulation_steps`，或者开启 `use_vllm`。
- 如果你的网络环境连接 Hugging Face 较慢，可以在启动前配置镜像环境变量。