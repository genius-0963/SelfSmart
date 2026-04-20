"""
Data Preprocessor - Clean and format training data
Production-grade data preprocessing for LLM fine-tuning.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib
from collections import Counter

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Production-grade data preprocessor for LLM training.
    Handles cleaning, deduplication, and formatting.
    """
    
    def __init__(
        self,
        input_dir: str = "./training_data",
        output_dir: str = "./processed_data",
        min_text_length: int = 100,
        max_text_length: int = 10000
    ):
        """
        Initialize data preprocessor.
        
        Args:
            input_dir: Directory with raw data
            output_dir: Directory for processed data
            min_text_length: Minimum text length to keep
            max_text_length: Maximum text length to keep
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        # Statistics
        self.stats = {
            "total_input": 0,
            "total_output": 0,
            "duplicates_removed": 0,
            "too_short": 0,
            "too_long": 0,
            "filtered": 0
        }
        
        logger.info(f"Data preprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text content.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\'\"]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([\.!?])\1+', r'\1', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def deduplicate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entries based on content hash.
        
        Args:
            data: List of data items
            
        Returns:
            Deduplicated data
        """
        seen_hashes = set()
        unique_data = []
        
        for item in data:
            # Create hash of content
            content = item.get("content", "")
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_data.append(item)
            else:
                self.stats["duplicates_removed"] += 1
        
        logger.info(f"Removed {self.stats['duplicates_removed']} duplicates")
        return unique_data
    
    def filter_by_length(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter data by text length.
        
        Args:
            data: List of data items
            
        Returns:
            Filtered data
        """
        filtered_data = []
        
        for item in data:
            content = item.get("content", "")
            content_length = len(content)
            
            if content_length < self.min_text_length:
                self.stats["too_short"] += 1
                continue
            
            if content_length > self.max_text_length:
                self.stats["too_long"] += 1
                # Truncate instead of filtering
                item["content"] = content[:self.max_text_length]
                filtered_data.append(item)
            else:
                filtered_data.append(item)
        
        logger.info(f"Filtered {self.stats['too_short']} too short, {self.stats['too_long']} truncated")
        return filtered_data
    
    def format_for_training(
        self,
        data: List[Dict[str, Any]],
        format_type: str = "instruction"
    ) -> List[Dict[str, str]]:
        """
        Format data for training.
        
        Args:
            data: List of data items
            format_type: Format type (instruction, completion, or conversation)
            
        Returns:
            Formatted training data
        """
        formatted_data = []
        
        for item in data:
            if format_type == "instruction":
                # Instruction format: prompt + response
                formatted = {
                    "instruction": item.get("title", ""),
                    "input": "",
                    "output": self.clean_text(item.get("content", ""))
                }
                formatted_data.append(formatted)
            
            elif format_type == "completion":
                # Completion format: just the text
                formatted = {
                    "text": self.clean_text(item.get("content", ""))
                }
                formatted_data.append(formatted)
            
            elif format_type == "conversation":
                # Conversation format: simulate Q&A
                formatted = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": item.get("title", "")
                        },
                        {
                            "from": "gpt",
                            "value": self.clean_text(item.get("content", ""))
                        }
                    ]
                }
                formatted_data.append(formatted)
        
        logger.info(f"Formatted {len(formatted_data)} items as {format_type}")
        return formatted_data
    
    def process_file(
        self,
        input_file: str,
        output_file: str,
        format_type: str = "instruction"
    ) -> List[Dict[str, str]]:
        """
        Process a single file.
        
        Args:
            input_file: Input filename
            output_file: Output filename
            format_type: Format type
            
        Returns:
            Processed data
        """
        input_path = self.input_dir / input_file
        
        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            return []
        
        # Load data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.stats["total_input"] = len(data)
        logger.info(f"Loaded {len(data)} items from {input_file}")
        
        # Clean text
        for item in data:
            if "content" in item:
                item["content"] = self.clean_text(item["content"])
        
        # Deduplicate
        data = self.deduplicate(data)
        
        # Filter by length
        data = self.filter_by_length(data)
        
        # Format for training
        formatted_data = self.format_for_training(data, format_type)
        
        # Save processed data
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        self.stats["total_output"] = len(formatted_data)
        logger.info(f"Processed {len(formatted_data)} items to {output_file}")
        
        return formatted_data
    
    def process_all(
        self,
        format_type: str = "instruction"
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Process all files in input directory.
        
        Args:
            format_type: Format type
            
        Returns:
            Dictionary of processed data by source
        """
        all_processed = {}
        
        # Find all JSON files
        json_files = list(self.input_dir.glob("*.json"))
        
        for json_file in json_files:
            output_file = f"processed_{json_file.name}"
            
            try:
                processed_data = self.process_file(
                    json_file.name,
                    output_file,
                    format_type
                )
                
                all_processed[json_file.stem] = processed_data
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        # Save statistics
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Processing complete. Stats: {self.stats}")
        return all_processed
    
    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self.stats
    
    def create_train_val_split(
        self,
        input_file: str,
        train_split: float = 0.9
    ) -> tuple:
        """
        Create train/validation split from processed data.
        
        Args:
            input_file: Input filename
            train_split: Training split ratio
            
        Returns:
            Tuple of (train_data, val_data)
        """
        input_path = self.output_dir / input_file
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Shuffle data
        import random
        random.shuffle(data)
        
        # Split
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Save splits
        train_path = self.output_dir / f"train_{input_file}"
        val_path = self.output_dir / f"val_{input_file}"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created train/val split: {len(train_data)} train, {len(val_data)} val")
        
        return train_data, val_data
