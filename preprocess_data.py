# 3_preprocess_data.py
# 3_preprocess_data.py
import copy
from functools import partial
from transformers import DataCollatorForSeq2Seq
from datasets import DatasetDict
from load_data import get_dataset  # normal import now

def preprocess_batch(batch, tokenizer, max_length=256):
    inputs = tokenizer(batch["text"], max_length=max_length, truncation=True, padding="max_length")
    inputs["labels"] = copy.deepcopy(inputs["input_ids"])
    return inputs

def prepare_data(tokenizer, max_length=256):
    ds = get_dataset()
    processed = ds.map(lambda x: preprocess_batch(x, tokenizer, max_length), batched=True,
                       remove_columns=["instruction", "response", "prompt", "answer"])
    filtered = processed.filter(lambda rec: len(rec["input_ids"]) <= max_length)
    split_dataset = filtered.train_test_split(test_size=14, seed=0)
    return split_dataset, DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None, max_length=max_length, pad_to_multiple_of=8, padding="max_length"
    )

if __name__ == "__main__":
    from prepare_model import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer()
    splits, collator = prepare_data(tokenizer)
    print(splits)
