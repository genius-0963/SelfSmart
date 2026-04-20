"""
Model Loader - Load and prepare base models for fine-tuning
Supports Llama 3, Mistral, and Phi-3 models with quantization.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Production-grade model loader for LLM fine-tuning.
    Supports quantization and LoRA preparation.
    """
    
    MODEL_CONFIGS = {
        "mistral-7b": {
            "model_name": "mistralai/Mistral-7B-v0.1",
            "requires_vram": 12,  # GB
            "context_length": 8192,
            "recommended": True
        },
        "llama-3-8b": {
            "model_name": "meta-llama/Meta-Llama-3-8B",
            "requires_vram": 16,
            "context_length": 8192,
            "recommended": False
        },
        "phi-3-mini": {
            "model_name": "microsoft/Phi-3-mini-4k-instruct",
            "requires_vram": 8,
            "context_length": 4096,
            "recommended": False
        }
    }
    
    def __init__(
        self,
        model_key: str = "mistral-7b",
        use_quantization: bool = True,
        quantization_bits: int = 4
    ):
        """
        Initialize model loader.
        
        Args:
            model_key: Key from MODEL_CONFIGS
            use_quantization: Whether to use quantization
            quantization_bits: Bits for quantization (4 or 8)
        """
        if model_key not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        self.model_key = model_key
        self.model_config = self.MODEL_CONFIGS[model_key]
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        logger.info(f"Model loader initialized for {model_key}")
    
    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load the tokenizer for the model.
        
        Returns:
            AutoTokenizer instance
        """
        logger.info(f"Loading tokenizer for {self.model_config['model_name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model_name'],
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info("Tokenizer loaded successfully")
        return self.tokenizer
    
    def load_model(self) -> AutoModelForCausalLM:
        """
        Load the base model with optional quantization.
        Automatically detects and uses MPS for M1 Macs.
        
        Returns:
            AutoModelForCausalLM instance
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        
        # Detect device
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Metal Performance Shaders) for M1 Mac")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA for NVIDIA GPU")
        else:
            device = "cpu"
            logger.info("Using CPU (will be slow)")
        
        logger.info(f"Loading model {self.model_config['model_name']} with quantization={self.use_quantization}")
        
        # Configure quantization (not available on MPS)
        quantization_config = None
        if self.use_quantization and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.quantization_bits == 4,
                load_in_8bit=self.quantization_bits == 8,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.use_quantization and device == "mps":
            logger.warning("Quantization not supported on MPS, loading full model")
            self.use_quantization = False
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model_name'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if device == "mps" else (torch.float16 if self.use_quantization else torch.float32),
            trust_remote_code=True
        )
        
        # Move to device
        if device != "auto":
            self.model = self.model.to(device)
        
        logger.info("Model loaded successfully")
        return self.model
    
    def prepare_for_training(self) -> AutoModelForCausalLM:
        """
        Prepare model for LoRA training.
        
        Returns:
            Prepared model
        """
        if self.model is None:
            self.load_model()
        
        logger.info("Preparing model for LoRA training")
        
        # Prepare model for k-bit training if using quantization
        if self.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        logger.info("Model prepared for training")
        return self.model
    
    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None
    ) -> AutoModelForCausalLM:
        """
        Apply LoRA adapters to the model.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: Dropout rate
            target_modules: Target modules for LoRA
            
        Returns:
            Model with LoRA adapters
        """
        if self.model is None:
            self.prepare_for_training()
        
        # Default target modules based on model
        if target_modules is None:
            if "mistral" in self.model_key or "llama" in self.model_key:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        logger.info(f"Applying LoRA with r={r}, alpha={lora_alpha}")
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA adapters applied successfully")
        return self.peft_model
    
    def get_training_arguments(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500
    ) -> TrainingArguments:
        """
        Get training arguments for fine-tuning.
        
        Args:
            output_dir: Output directory for checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
            save_steps: Checkpoint save frequency
            eval_steps: Evaluation frequency
            
        Returns:
            TrainingArguments instance
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.use_quantization,
            bf16=not self.use_quantization,
            optim="paged_adamw_32bit",
            report_to=["tensorboard"],
            logging_dir=f"{output_dir}/logs",
            save_total_limit=3,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0
        )
    
    def save_model(self, output_dir: str):
        """Save the model and tokenizer."""
        if self.peft_model is not None:
            self.peft_model.save_pretrained(output_dir)
            logger.info(f"LoRA model saved to {output_dir}")
        elif self.model is not None:
            self.model.save_pretrained(output_dir)
            logger.info(f"Base model saved to {output_dir}")
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available models with their requirements."""
        return cls.MODEL_CONFIGS
    
    @classmethod
    def get_recommended_model(cls) -> str:
        """Get the recommended model for your resources."""
        for model_key, config in cls.MODEL_CONFIGS.items():
            if config.get("recommended", False):
                return model_key
        return list(cls.MODEL_CONFIGS.keys())[0]
