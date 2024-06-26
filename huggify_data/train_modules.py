import gc
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from trl import SFTTrainer


class LlamaTrainer:
    def __init__(self, model_name, dataset_name, new_model, huggingface_token):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.new_model = new_model
        self.hf_token = huggingface_token
        self.device_map = {"": 0}
        self.bnb_config = self.configure_bitsandbytes()
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def configure_bitsandbytes(self):
        compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        return bnb_config

    def load_model(self):
        print("Starting model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map=self.device_map,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        print("Model loading completed.")
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def configure_lora(self, lora_r=64, lora_alpha=16, lora_dropout=0.1):
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return peft_config

    def configure_training_arguments(
        self,
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        weight_decay=0.001,
        save_steps=0,
        logging_steps=25,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
    ):
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            gradient_checkpointing=True,  # Enable gradient checkpointing
            report_to="tensorboard",
        )
        return training_arguments

    def train_model(
        self, training_arguments, peft_config, max_seq_length=None, packing=False
    ):
        print("Starting dataset loading...")
        dataset = load_dataset(self.dataset_name, split="train")
        print("Dataset loading completed.")

        print("Starting training...")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=packing,
        )

        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    "CUDA out of memory error occurred, attempting to clear cache and retry."
                )
                self.clear_cuda_cache()
                trainer.train()
                print("Training completed.")

        try:
            trainer.model.save_pretrained(self.new_model)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    "CUDA out of memory error occurred, attempting to clear cache and retry."
                )
                self.clear_cuda_cache()
                trainer.model.save_pretrained(self.new_model)
                print("Model saved.")

        # Explicitly free up GPU memory
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

        self.merge_and_save_model()

    def merge_and_save_model(self):
        print("Starting model merging and saving...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map=self.device_map,
            )
            model = PeftModel.from_pretrained(base_model, self.new_model)
            model = model.merge_and_unload()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    "CUDA out of memory error occurred, attempting to clear cache and retry."
                )
                self.clear_cuda_cache()
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    return_dict=True,
                    torch_dtype=torch.float16,
                    device_map=self.device_map,
                )
                model = PeftModel.from_pretrained(base_model, self.new_model)
                model = model.merge_and_unload()

        print("Pushing model to the Hugging Face Hub...")
        self.tokenizer.push_to_hub(self.new_model, use_temp_dir=False)
        model.push_to_hub(self.new_model, use_temp_dir=False, token=self.hf_token)
        print("Model pushing completed.")

    def clear_cuda_cache(self):
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared.")
