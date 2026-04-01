import os

# 模型配置
MODEL_NAME = "Qwen2.5-0.5B-Instruct"

# Prompt 模板设置
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# 最终期望的输出格式模板
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# 训练参数配置
OUTPUT_DIR = "outputs/Qwen2.5-0.5B-reasoning-GRPO"
RUN_NAME = "Qwen2.5-0.5B-GRPO-gsm8k"
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "project name")
WANDB_KEY = os.getenv("WANDB_API_KEY", "")
