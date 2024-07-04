import gc
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from trl import SFTTrainer


class LlamaTrainer:
    """
    A class to handle the training and uploading of a fine-tuned Llama model using the Hugging Face transformers library.
    """

    def __init__(
        self, model_name: str, dataset_name: str, new_model: str, huggingface_token: str
    ):
        """
        Initialize the LlamaTrainer class.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset_name (str): The name of the dataset to use for training.
            new_model (str): The name of the new fine-tuned model to be saved.
            huggingface_token (str): The Hugging Face token for authentication.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.new_model = new_model
        self.hf_token = huggingface_token
        self.device_map = {"": 0}
        self.bnb_config = self.configure_bitsandbytes()
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def configure_bitsandbytes(self) -> BitsAndBytesConfig:
        """
        Configure the BitsAndBytes quantization settings.

        Returns:
            BitsAndBytesConfig: The configuration object for BitsAndBytes.
        """
        compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        return bnb_config

    def load_model(self) -> AutoModelForCausalLM:
        """
        Load the pre-trained model with quantization settings.

        Returns:
            AutoModelForCausalLM: The loaded model.
        """
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

    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load the tokenizer for the pre-trained model.

        Returns:
            AutoTokenizer: The loaded tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def configure_lora(
        self, lora_r: int = 64, lora_alpha: int = 16, lora_dropout: float = 0.1
    ) -> LoraConfig:
        """
        Configure the LoRA (Low-Rank Adaptation) settings.

        Args:
            lora_r (int, optional): The rank of the LoRA matrices. Default is 64.
            lora_alpha (int, optional): The scaling factor for the LoRA matrices. Default is 16.
            lora_dropout (float, optional): The dropout rate for LoRA layers. Default is 0.1.

        Returns:
            LoraConfig: The configuration object for LoRA.
        """
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
        output_dir: str = "./results",
        num_train_epochs: int = 4,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.001,
        save_steps: int = 0,
        logging_steps: int = 25,
        fp16: bool = False,
        bf16: bool = False,
        max_grad_norm: float = 0.3,
        max_steps: int = -1,
        warmup_ratio: float = 0.03,
        group_by_length: bool = True,
        lr_scheduler_type: str = "cosine",
    ) -> TrainingArguments:
        """
        Configure the training arguments.

        Args:
            output_dir (str, optional): The output directory for saving results. Default is "./results".
            num_train_epochs (int, optional): The number of training epochs. Default is 4.
            per_device_train_batch_size (int, optional): The batch size per device. Default is 2.
            gradient_accumulation_steps (int, optional): The number of gradient accumulation steps. Default is 2.
            learning_rate (float, optional): The learning rate for training. Default is 2e-4.
            weight_decay (float, optional): The weight decay for optimization. Default is 0.001.
            save_steps (int, optional): The number of steps between each save. Default is 0.
            logging_steps (int, optional): The number of steps between each log. Default is 25.
            fp16 (bool, optional): Use FP16 precision. Default is False.
            bf16 (bool, optional): Use BF16 precision. Default is False.
            max_grad_norm (float, optional): The maximum gradient norm. Default is 0.3.
            max_steps (int, optional): The maximum number of training steps. Default is -1 (no limit).
            warmup_ratio (float, optional): The warmup ratio for the learning rate scheduler. Default is 0.03.
            group_by_length (bool, optional): Whether to group sequences by length. Default is True.
            lr_scheduler_type (str, optional): The type of learning rate scheduler. Default is "cosine".

        Returns:
            TrainingArguments: The configuration object for training arguments.
        """
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
        self,
        training_arguments: TrainingArguments,
        peft_config: LoraConfig,
        max_seq_length: int = None,
        packing: bool = False,
    ):
        """
        Train the model with the given dataset and configuration.

        Args:
            training_arguments (TrainingArguments): The training arguments.
            peft_config (LoraConfig): The LoRA configuration.
            max_seq_length (int, optional): The maximum sequence length for training. Default is None.
            packing (bool, optional): Whether to use packing for training. Default is False.
        """
        print("Starting dataset loading...")
        dataset = load_dataset(self.dataset_name, split="train")
        print("Dataset loading completed.")

        print("Trainer defined...")
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
            print("Start training...")
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    "CUDA out of memory error occurred, attempting to clear cache and retry."
                )
                self.clear_cuda_cache()
                print("Start training...")
                trainer.train()

        print("Save model...")
        trainer.model.save_pretrained(self.new_model)
        self.tokenizer.save_pretrained(self.new_model)
        print("Training and saving completed.")

        # Push to hub
        self.push_model_to_hub(trainer)

    def push_model_to_hub(self, trainer: SFTTrainer):
        """
        Push the trained model to the Hugging Face Hub.

        Args:
            trainer (SFTTrainer): The trainer object containing the trained model.
        """
        print("Starting model merging and saving...")
        try:
            print("Pushing model to the Hugging Face Hub...")
            self.tokenizer.push_to_hub(self.new_model, use_temp_dir=False)
            trainer.model.push_to_hub(
                self.new_model, use_temp_dir=False, token=self.hf_token
            )
            print("Model pushing completed.")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    "CUDA out of memory error occurred, attempting to clear cache and retry."
                )
                self.clear_cuda_cache()
                print("Pushing model to the Hugging Face Hub...")
                self.tokenizer.push_to_hub(self.new_model, use_temp_dir=False)
                trainer.model.push_to_hub(
                    self.new_model, use_temp_dir=False, token=self.hf_token
                )
                print("Model pushing completed.")

    def clear_cuda_cache(self):
        """
        Clear the CUDA cache to free up memory.
        """
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared.")

    def load_model_and_tokenizer(
        self, base_model_path: str, new_model_path: str
    ) -> tuple:
        """
        Load the model and tokenizer.

        Args:
            base_model_path (str): The path to the base model.
            new_model_path (str): The path to the new model.

        Returns:
            tuple: A tuple containing the loaded model and tokenizer.
        """
        # Load the config and base model
        config = PeftConfig.from_pretrained(new_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(base_model, new_model_path)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(new_model_path)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def generate_response(self, model, tokenizer, prompt: str, max_len: int) -> str:
        """
        Generates a response for the given prompt using the loaded model and tokenizer.

        Args:
            model: The loaded language model.
            tokenizer: The loaded tokenizer.
            prompt (str): The input text prompt.
            max_len (int): The max length of output.

        Returns:
            str: The generated response text.
        """
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate the response
        outputs = model.generate(
            **inputs, max_length=max_len, do_sample=True, top_k=50, top_p=0.95
        )

        # Decode the output tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
