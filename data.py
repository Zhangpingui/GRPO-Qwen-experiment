from datasets import load_dataset, Dataset
from config import SYSTEM_PROMPT

def extract_hash_answer(text: str) -> str | None:
    """
    按照 gsm8k 数据集的格式，提取出真实答案
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split: str = "train") -> Dataset:
    """
    加载并处理 gsm8k 数据集，应用系统提示模板
    """
    data = load_dataset('gsm8k')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role':'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore
