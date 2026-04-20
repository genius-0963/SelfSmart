"""
Continuous Learner - Ongoing model fine-tuning loop
Production-grade continuous learning pipeline for LLM updates.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import shutil

from src.llm_training.data_collector import DataCollector
from src.llm_training.data_preprocessor import DataPreprocessor
from src.llm_training.lora_trainer import LoRATrainer
from src.llm_training.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ContinuousLearner:
    """
    Production-grade continuous learning pipeline.
    Periodically collects new data and fine-tunes the model.
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_path: str,
        data_dir: str = "./training_data",
        output_dir: str = "./model_updates",
        learning_interval_hours: int = 24,
        min_new_samples: int = 100
    ):
        """
        Initialize continuous learner.
        
        Args:
            model_path: Path to current fine-tuned model
            base_model_path: Path to base model
            data_dir: Directory for training data
            output_dir: Directory for model updates
            learning_interval_hours: Hours between learning cycles
            min_new_samples: Minimum new samples to trigger training
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_interval = timedelta(hours=learning_interval_hours)
        self.min_new_samples = min_new_samples
        
        self.last_learning_time = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_learning_cycles": 0,
            "total_samples_processed": 0,
            "last_learning_time": None,
            "model_versions": []
        }
        
        logger.info("Continuous learner initialized")
    
    async def collect_new_data(self) -> Dict[str, list]:
        """
        Collect new data from internet sources.
        
        Returns:
            Dictionary of collected data by source
        """
        logger.info("Starting data collection")
        
        async with DataCollector(output_dir=str(self.data_dir)) as collector:
            # Collect from various sources
            data = await collector.collect_all(
                wikipedia_count=50,
                hacker_news_count=50
            )
        
        return data
    
    def process_new_data(self) -> list:
        """
        Process newly collected data.
        
        Returns:
            Processed training data
        """
        logger.info("Processing new data")
        
        preprocessor = DataPreprocessor(
            input_dir=str(self.data_dir),
            output_dir=str(self.data_dir / "processed")
        )
        
        # Process all files
        processed_data = preprocessor.process_all(format_type="instruction")
        
        # Combine all processed data
        all_data = []
        for source_data in processed_data.values():
            all_data.extend(source_data)
        
        # Save combined data
        combined_path = self.data_dir / "processed" / "combined_training_data.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {len(all_data)} samples")
        return all_data
    
    def fine_tune_model(
        self,
        training_data_path: str,
        num_epochs: int = 1
    ):
        """
        Fine-tune the model with new data.
        
        Args:
            training_data_path: Path to training data
            num_epochs: Number of training epochs
        """
        logger.info(f"Starting fine-tuning with {num_epochs} epochs")
        
        # Load base model
        model_loader = ModelLoader(model_key="mistral-7b")
        model = model_loader.load_model()
        tokenizer = model_loader.load_tokenizer()
        
        # Create trainer
        trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir=str(self.output_dir / f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
        
        # Train
        trainer.train(
            train_data_path=training_data_path,
            num_train_epochs=num_epochs
        )
        
        # Update statistics
        self.stats["total_learning_cycles"] += 1
        self.stats["last_learning_time"] = datetime.now().isoformat()
        
        # Save model version
        model_version = {
            "timestamp": datetime.now().isoformat(),
            "samples_processed": self.stats["total_samples_processed"],
            "cycle": self.stats["total_learning_cycles"]
        }
        self.stats["model_versions"].append(model_version)
        
        logger.info("Fine-tuning complete")
    
    def should_trigger_learning(self, new_samples_count: int) -> bool:
        """
        Determine if learning should be triggered.
        
        Args:
            new_samples_count: Number of new samples collected
            
        Returns:
            True if learning should trigger
        """
        # Check minimum samples
        if new_samples_count < self.min_new_samples:
            logger.info(f"Not enough samples ({new_samples_count} < {self.min_new_samples})")
            return False
        
        # Check time since last learning
        if self.last_learning_time:
            time_since_last = datetime.now() - self.last_learning_time
            if time_since_last < self.learning_interval:
                logger.info(f"Learning interval not reached ({time_since_last} < {self.learning_interval})")
                return False
        
        return True
    
    async def learning_cycle(self):
        """
        Execute one complete learning cycle.
        """
        logger.info("Starting learning cycle")
        
        try:
            # Collect new data
            new_data = await self.collect_new_data()
            total_samples = sum(len(data) for data in new_data.values())
            
            # Process data
            processed_data = self.process_new_data()
            
            # Check if we should trigger learning
            if self.should_trigger_learning(len(processed_data)):
                # Fine-tune model
                training_data_path = self.data_dir / "processed" / "combined_training_data.json"
                self.fine_tune_model(str(training_data_path), num_epochs=1)
                
                # Update last learning time
                self.last_learning_time = datetime.now()
                
                # Save statistics
                self.save_stats()
                
                logger.info("Learning cycle completed successfully")
            else:
                logger.info("Learning cycle skipped (conditions not met)")
                
        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
    
    async def start(self):
        """Start the continuous learning loop."""
        logger.info("Starting continuous learning loop")
        self.is_running = True
        
        while self.is_running:
            try:
                await self.learning_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.learning_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def stop(self):
        """Stop the continuous learning loop."""
        logger.info("Stopping continuous learning loop")
        self.is_running = False
    
    def save_stats(self):
        """Save learning statistics."""
        stats_path = self.output_dir / "learning_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")
    
    def load_stats(self):
        """Load learning statistics."""
        stats_path = self.output_dir / "learning_stats.json"
        
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
            
            logger.info("Statistics loaded")
        else:
            logger.info("No existing statistics found")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current learning statistics."""
        return self.stats
