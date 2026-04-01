import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from config import MODEL_NAME

def test_inference():
    """
    测试未经 GRPO 强化学习之前的模型推理表现。
    """
    print(f"Loading {MODEL_NAME} for original inference test...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading gsm8k dataset...")
    data = load_dataset('gsm8k')
    
    prompt = data['train'][0]['question']
    true_answer = data['train'][0]['answer']
    
    print("\n" + "="*50)
    print(f"Question: {prompt}")
    print(f"True Answer: {true_answer}")
    print("="*50 + "\n")

    # 按照 Qwen 要求的格式构造数据
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating response...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    # 只提取答案部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # 解码得到文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\nModel Output:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    print("\n输出可能比较混乱，说明未经强化学习的模型还需提升逻辑输出格式。")

if __name__ == "__main__":
    test_inference()
