# run_inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

Instruction: {instruction}

Response:"""

def postprocess(response, original_prompt):
    # Remove the original prompt from the response
    response = response.replace(original_prompt, "").strip()
    # Clean up any remaining formatting
    if response.startswith("Response:"):
        response = response[9:].strip()
    return response

def run_inference(prompt):
    # Load base model and tokenizer
    base_model_id = "databricks/dolly-v2-3b"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, "dolly-3b-lora")
    
    # Prepare prompt
    full_prompt = prompt_template.format(instruction=prompt)
    
    # Tokenize
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return postprocess(response, full_prompt)

if __name__ == "__main__":
    print("🤖 LoRA Fine-tuned Dolly Model - Ready for Questions!")
    print("=" * 50)
    
    # Test with your original question
    user_prompt = "List 5 reasons why someone should learn to cook"
    print(f"Question: {user_prompt}")
    print(f"Answer: {run_inference(user_prompt)}")
    
    print("\n" + "=" * 50)
    
    # Interactive mode
    while True:
        user_input = input("\nEnter your question (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.strip():
            print(f"Answer: {run_inference(user_input)}")
        else:
            print("Please enter a valid question.")