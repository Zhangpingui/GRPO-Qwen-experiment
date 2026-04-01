import os
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# 尝试导入 dotenv 解析 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import MODEL_NAME, OUTPUT_DIR, RUN_NAME, WANDB_PROJECT, WANDB_KEY
from data import get_gsm8k_questions
from rewards import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func
)

def main():
    # 配置并登录 Wandb (如果有 key)
    use_wandb = False
    if WANDB_KEY and WANDB_KEY.strip():
        print("Logging into wandb...")
        wandb.login(key=WANDB_KEY)
        wandb.init(project=WANDB_PROJECT, name=RUN_NAME)
        use_wandb = True
    
    print("Loading dataset gsm8k...")
    dataset = get_gsm8k_questions(split="train")
    
    # 强化学习配置参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,               # 输出目录
        run_name=RUN_NAME,                   # wandb 中的项目名称
        learning_rate=5e-6,                  # 强化学习学习率设置的比较小
        adam_beta1=0.9,                      # adam优化器
        adam_beta2=0.99,
        weight_decay=0.1,                    # 正则
        warmup_ratio=0.1,                    # 学习率预热比例
        lr_scheduler_type='cosine',          # 学习率衰减策略
        logging_steps=1,
        bf16=True,                           # 混合精度训练
        per_device_train_batch_size=8,       # 总的batch = per_device_train_batch_size * 显卡数
        gradient_accumulation_steps=4,       # 累计gradient_accumulation_steps个batch更新一次模型
        num_generations=8,                   # GRPO中每个q输出num_generations个o
        max_prompt_length=256,               # 限制prompt长度
        max_completion_length=200,           # 限制模型输出上限 
        num_train_epochs=1,
        save_steps=100,                      # 每save_steps步保存一次模型
        max_grad_norm=0.1,                   # 梯度裁剪
        log_on_each_node=False,
        use_vllm=False,
        vllm_gpu_memory_utilization=.3,      # vllm 加速参数
        vllm_device="cuda:0",
        report_to="wandb" if use_wandb else "none"
    )

    print(f"Loading model and tokenizer ({MODEL_NAME})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GRPO 等对齐训练通常需要 pad_token
    tokenizer.pad_token = tokenizer.eos_token

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
