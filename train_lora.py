# 4_train_lora.py
from transformers import TrainingArguments, Trainer
from prepare_model import load_model_and_tokenizer, apply_lora
from preprocess_data import prepare_data

def train():
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    splits, collator = prepare_data(tokenizer)

    training_args = TrainingArguments(
        output_dir="dolly-3b-lora",
        overwrite_output_dir=True,
        fp16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        data_collator=collator,
    )

    model.config.use_cache = False
    trainer.train()
    trainer.model.save_pretrained("dolly-3b-lora")
    trainer.save_model("dolly-3b-lora")
    trainer.model.config.save_pretrained("dolly-3b-lora")

if __name__ == "__main__":
    train()
