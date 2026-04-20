"""
Local LLM Inference - Generate responses from fine-tuned models
Production-grade inference engine for local LLM deployment.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Represents an LLM response"""
    content: str
    finish_reason: str
    usage: Dict[str, int]
    model: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LocalLLMClient:
    """
    Production-grade local LLM client for inference.
    Supports loading fine-tuned models and streaming responses.
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        use_quantization: bool = True,
        device: str = "auto"
    ):
        """
        Initialize local LLM client.
        
        Args:
            model_path: Path to fine-tuned model or base model
            base_model_path: Path to base model (if using LoRA)
            use_quantization: Whether to use quantization
            device: Device to use (auto, cuda, cpu)
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.use_quantization = use_quantization
        self.device = device
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Local LLM client initialized for model: {model_path}")
    
    def load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if self.base_model_path:
            # Load base model and apply LoRA
            logger.info(f"Loading base model from {self.base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # Load fine-tuned model directly
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ) -> LLMResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            LLMResponse object
        """
        if self.model is None:
            self.load_model()
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new response (remove prompt)
        response_text = generated_text[len(prompt):].strip()
        
        return LLMResponse(
            content=response_text,
            finish_reason="stop",
            usage={
                "prompt_tokens": inputs['input_ids'].shape[1],
                "completion_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
                "total_tokens": outputs.shape[1]
            },
            model=self.model_path
        )
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the model.
        
        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Yields:
            String chunks of the response
        """
        if self.model is None:
            self.load_model()
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate with streaming
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Get the new token
                new_token = outputs[0][-1].item()
                
                # Check for EOS
                if new_token == self.tokenizer.eos_token_id:
                    break
                
                # Decode the new token
                new_text = self.tokenizer.decode([new_token], skip_special_tokens=True)
                yield new_text
                
                # Append to input for next iteration
                input_ids = torch.cat([input_ids, outputs[0][-1:]], dim=1)
                
                # Check length
                if input_ids.shape[1] >= 2048:
                    break
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a prompt string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        formatted = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted += f"### System:\n{content}\n\n"
            elif role == "user":
                formatted += f"### User:\n{content}\n\n"
            elif role == "assistant":
                formatted += f"### Assistant:\n{content}\n\n"
        
        formatted += "### Assistant:\n"
        return formatted
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        torch.cuda.empty_cache()
        logger.info("Model unloaded, memory freed")
