# 2_prepare_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

def load_model_and_tokenizer(model_id="databricks/dolly-v2-3b"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 8-bit quantization using BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def apply_lora(model):
    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value"],
    )
    # prepare_model_for_int8_training is no longer needed in newer PEFT versions
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
