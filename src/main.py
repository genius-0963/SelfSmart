"""
SmartSelf AI - Main Application Entry Point
Professional self-learning chatbot with continuous knowledge acquisition.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings
from src.learning.continuous_learner import ContinuousLearner, LearningConfig


def setup_logging(settings):
    """Configure application logging"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO if not settings.debug else logging.DEBUG,
        format=log_format,
        handlers=[
            logging.FileHandler(settings.logs_dir / "app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


async def main():
    """Main application entry point"""
    # Load settings
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Project root: {settings.project_root}")
    
    # Create learning configuration
    learning_config = LearningConfig(
        max_concurrent_crawls=settings.max_concurrent_crawls,
        crawl_rate_limit=settings.crawl_rate_limit,
        daily_crawl_limit=settings.daily_crawl_limit,
        min_quality_score=settings.min_quality_score
    )
    
    # Initialize continuous learner
    learner = ContinuousLearner(learning_config)
    
    try:
        # Start learning
        await learner.start_learning()
        
        logger.info("SmartSelf AI is running. Press Ctrl+C to stop.")
        
        # Keep the application running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await learner.stop_learning()
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
