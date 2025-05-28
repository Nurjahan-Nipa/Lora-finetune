# 1_load_data.py
from datasets import load_dataset, disable_caching

disable_caching()

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction: {instruction}\n Response:"""
answer_template = """{response}"""

def _add_text(rec):
    instruction = rec["instruction"]
    response = rec["response"]
    if not instruction or not response:
        raise ValueError(f"Missing data in: {rec}")
    rec["prompt"] = prompt_template.format(instruction=instruction)
    rec["answer"] = answer_template.format(response=response)
    rec["text"] = rec["prompt"] + rec["answer"]
    return rec

def get_dataset(sample_size=200):
    dataset = load_dataset("MBZUAI/LaMini-instruction", split="train")
    small_dataset = dataset.select(range(sample_size)).map(_add_text)
    return small_dataset

if __name__ == "__main__":
    ds = get_dataset()
    print(ds[0])
