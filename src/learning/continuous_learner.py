"""
SmartSelf Learning Chatbot - Continuous Learning System
Manages the continuous learning pipeline for the chatbot.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import schedule
import time
from pathlib import Path

from src.crawler.web_crawler import WebCrawler, RSSCrawler, CrawlResult
from src.processor.content_processor import ContentProcessor, ProcessedContent
from src.knowledge.knowledge_integrator import KnowledgeIntegrator
from src.api.free_api_client import FreeAPIClient

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Configuration for continuous learning"""
    # Crawling settings
    max_concurrent_crawls: int = 10
    crawl_rate_limit: int = 1  # requests per second per domain
    daily_crawl_limit: int = 1000
    
    # Quality thresholds
    min_quality_score: float = 0.3
    min_relevance_score: float = 0.2
    
    # Learning schedule
    real_time_interval: int = 300  # 5 minutes
    hourly_interval: int = 3600    # 1 hour
    daily_interval: int = 86400    # 24 hours
    
    # Data sources
    rss_feeds: List[str] = None
    crawl_domains: List[str] = None
    api_endpoints: List[str] = None
    
    def __post_init__(self):
        if self.rss_feeds is None:
            self.rss_feeds = [
                "https://feeds.bbci.co.uk/news/rss.xml",
                "https://rss.cnn.com/rss/edition.rss",
                "https://feeds.reuters.com/reuters/topNews",
                "https://techcrunch.com/feed/",
                "https://www.wired.com/feed/rss",
            ]
        
        if self.crawl_domains is None:
            self.crawl_domains = [
                "en.wikipedia.org",
                "github.com",
                "stackoverflow.com",
                "medium.com",
                "dev.to",
                "towardsdatascience.com"
            ]
        
        if self.api_endpoints is None:
            self.api_endpoints = [
                "https://api.github.com/events",
                "https://hacker-news.firebaseio.com/v0/newstories.json"
            ]


@dataclass
class LearningStats:
    """Statistics for learning progress"""
    total_urls_crawled: int = 0
    successful_crawls: int = 0
    content_processed: int = 0
    knowledge_added: int = 0
    duplicates_found: int = 0
    learning_sessions: int = 0
    last_learning_time: Optional[datetime] = None
    average_quality_score: float = 0.0
    top_topics: List[str] = None
    
    def __post_init__(self):
        if self.top_topics is None:
            self.top_topics = []


class ContinuousLearner:
    """
    Main continuous learning system that orchestrates
    web crawling, content processing, and knowledge integration.
    """
    
    def __init__(self, config: LearningConfig):
        """Initialize continuous learner"""
        self.config = config
        self.stats = LearningStats()
        self.is_running = False
        self.learning_task = None
        
        # Initialize components
        self.web_crawler = None
        self.rss_crawler = None
        self.content_processor = ContentProcessor()
        self.knowledge_integrator = KnowledgeIntegrator()
        self.free_api_client = FreeAPIClient()
        
        # Learning state
        self.learning_history: List[Dict[str, Any]] = []
        self.active_sources: Set[str] = set()
        
        logger.info("Continuous learner initialized")
    
    async def start_learning(self):
        """Start the continuous learning process"""
        if self.is_running:
            logger.warning("Learning is already running")
            return
        
        self.is_running = True
        logger.info("Starting continuous learning...")
        
        # Start learning loop
        self.learning_task = asyncio.create_task(self._learning_loop())
        
        # Schedule periodic tasks
        self._schedule_periodic_tasks()
        
        logger.info("Continuous learning started")
    
    async def stop_learning(self):
        """Stop the continuous learning process"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Continuous learning stopped")
    
    def _schedule_periodic_tasks(self):
        """Schedule periodic learning tasks"""
        # Real-time learning (RSS feeds, APIs)
        schedule.every(self.config.real_time_interval).seconds.do(
            lambda: asyncio.create_task(self._real_time_learning())
        )
        
        # Hourly updates
        schedule.every(self.config.hourly_interval).seconds.do(
            lambda: asyncio.create_task(self._hourly_learning())
        )
        
        # Daily deep crawl
        schedule.every(self.config.daily_interval).seconds.do(
            lambda: asyncio.create_task(self._daily_learning())
        )
        
        logger.info("Periodic tasks scheduled")
    
    async def _learning_loop(self):
        """Main learning loop"""
        while self.is_running:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _real_time_learning(self):
        """Real-time learning from RSS feeds and free APIs"""
        logger.info("Starting real-time learning...")
        
        try:
            # Initialize crawlers if needed
            if self.rss_crawler is None:
                self.rss_crawler = RSSCrawler()
            
            async with self.rss_crawler:
                # Crawl RSS feeds
                rss_results = await self._crawl_rss_feeds()
                
                # Fetch from free APIs
                api_content = await self._learn_from_free_apis()
                
                # Combine results
                all_results = rss_results + api_content
                
                # Process content
                processed_content = await self._process_content(all_results)
                
                # Integrate knowledge
                await self._integrate_knowledge(processed_content)
                
                # Update stats
                self._update_stats(all_results, processed_content)
                
            logger.info(f"Real-time learning completed: {len(processed_content)} items processed")
            
        except Exception as e:
            logger.error(f"Error in real-time learning: {e}")
    
    async def _hourly_learning(self):
        """Hourly learning from selected sources"""
        logger.info("Starting hourly learning...")
        
        try:
            # Crawl high-priority sources
            urls = await self._get_hourly_urls()
            
            if urls:
                crawl_results = await self._crawl_urls(urls)
                processed_content = await self._process_content(crawl_results)
                await self._integrate_knowledge(processed_content)
                self._update_stats(crawl_results, processed_content)
            
            logger.info(f"Hourly learning completed: {len(urls)} URLs crawled")
            
        except Exception as e:
            logger.error(f"Error in hourly learning: {e}")
    
    async def _daily_learning(self):
        """Daily deep crawl and learning"""
        logger.info("Starting daily learning...")
        
        try:
            # Deep crawl of all domains
            urls = await self._get_daily_urls()
            
            if urls:
                crawl_results = await self._crawl_urls(urls)
                processed_content = await self._process_content(crawl_results)
                await self._integrate_knowledge(processed_content)
                self._update_stats(crawl_results, processed_content)
                
                # Perform maintenance tasks
                await self._perform_maintenance()
            
            logger.info(f"Daily learning completed: {len(urls)} URLs crawled")
            
        except Exception as e:
            logger.error(f"Error in daily learning: {e}")
    
    async def _crawl_rss_feeds(self) -> List[CrawlResult]:
        """Crawl RSS feeds for new content"""
        all_results = []
        
        for feed_url in self.config.rss_feeds:
            try:
                feed_results = await self.rss_crawler.crawl_feed(feed_url)
                all_results.extend(feed_results)
                self.active_sources.add(feed_url)
                
            except Exception as e:
                logger.error(f"Error crawling RSS feed {feed_url}: {e}")
        
        return all_results
    
    async def _learn_from_free_apis(self) -> List[CrawlResult]:
        """Learn from free public APIs that don't require authentication"""
        api_results = []
        
        try:
            async with self.free_api_client:
                # Fetch random content from multiple APIs
                api_content = await self.free_api_client.fetch_all_random_content()
                
                # Convert API content to CrawlResult format
                for source, content in api_content.items():
                    if content:
                        crawl_result = self._api_content_to_crawl_result(source, content)
                        if crawl_result:
                            api_results.append(crawl_result)
                            self.active_sources.add(f"api_{source}")
                
                logger.info(f"Fetched content from {len(api_results)} free APIs")
                
        except Exception as e:
            logger.error(f"Error learning from free APIs: {e}")
        
        return api_results
    
    def _api_content_to_crawl_result(self, source: str, content: Dict[str, Any]) -> Optional[CrawlResult]:
        """Convert API content to CrawlResult format"""
        try:
            # Extract text content based on source type
            text_content = ""
            url = f"https://free-api-{source}.generated"
            
            if source == "wikipedia":
                text_content = f"{content.get('title', '')}: {content.get('extract', '')}"
                url = content.get('url', url)
            elif source == "joke":
                if 'joke' in content:
                    text_content = content['joke']
                elif 'setup' in content and 'punchline' in content:
                    text_content = f"{content['setup']} - {content['punchline']}"
            elif source == "quote":
                text_content = f'"{content.get("content", "")}" - {content.get("author", "")}'
            elif source == "advice":
                text_content = content.get('advice', '')
            elif source == "activity":
                text_content = f"Activity: {content.get('activity', '')} (Type: {content.get('type', '')})"
            elif source == "trivia":
                text_content = f"Question: {content.get('question', '')} Answer: {content.get('correct_answer', '')}"
            
            if not text_content:
                return None
            
            # Create CrawlResult
            return CrawlResult(
                url=url,
                title=content.get('title', source.replace('_', ' ').title()),
                content=text_content,
                metadata=content,
                quality_score=0.8,  # Free APIs generally have good quality
                source_type="api"
            )
            
        except Exception as e:
            logger.error(f"Error converting API content: {e}")
            return None
    
    async def _get_hourly_urls(self) -> List[str]:
        """Get URLs for hourly crawling"""
        # For now, return some high-priority URLs
        # In production, this would be more sophisticated
        urls = [
            "https://en.wikipedia.org/wiki/Main_Page",
            "https://github.com/trending",
            "https://stackoverflow.com/questions",
            "https://news.ycombinator.com"
        ]
        
        return urls[:self.config.daily_crawl_limit // 10]  # Limit hourly crawls
    
    async def _get_daily_urls(self) -> List[str]:
        """Get URLs for daily deep crawling"""
        # Generate URLs from domains
        urls = []
        
        for domain in self.config.crawl_domains:
            # In production, this would use sitemaps, APIs, or discovery
            base_urls = [
                f"https://{domain}",
                f"https://{domain}/browse",
                f"https://{domain}/trending",
                f"https://{domain}/popular"
            ]
            urls.extend(base_urls)
        
        return urls[:self.config.daily_crawl_limit]
    
    async def _crawl_urls(self, urls: List[str]) -> List[CrawlResult]:
        """Crawl a list of URLs"""
        if self.web_crawler is None:
            self.web_crawler = WebCrawler(
                max_concurrent=self.config.max_concurrent_crawls,
                rate_limit=self.config.crawl_rate_limit
            )
        
        async with self.web_crawler:
            results = await self.web_crawler.crawl_urls_batch(urls)
            
            for url in urls:
                self.active_sources.add(url)
        
        return results
    
    async def _process_content(self, crawl_results: List[CrawlResult]) -> List[ProcessedContent]:
        """Process crawled content"""
        if not crawl_results:
            return []
        
        # Filter by quality threshold
        filtered_results = [
            result for result in crawl_results
            if result.quality_score >= self.config.min_quality_score
        ]
        
        # Process content
        processed_content = await self.content_processor.process_content_batch(filtered_results)
        
        # Filter by relevance threshold
        relevant_content = [
            content for content in processed_content
            if content.relevance_score >= self.config.min_relevance_score
        ]
        
        return relevant_content
    
    async def _integrate_knowledge(self, processed_content: List[ProcessedContent]):
        """Integrate processed content into knowledge base"""
        if not processed_content:
            return
        
        try:
            # Add to knowledge base
            await self.knowledge_integrator.batch_integrate(processed_content)
            
            # Update knowledge count
            self.stats.knowledge_added += len(processed_content)
            
        except Exception as e:
            logger.error(f"Error integrating knowledge: {e}")
    
    def _update_stats(self, crawl_results: List[CrawlResult], processed_content: List[ProcessedContent]):
        """Update learning statistics"""
        self.stats.total_urls_crawled += len(crawl_results)
        self.stats.successful_crawls += len([r for r in crawl_results if r])
        self.stats.content_processed += len(processed_content)
        self.stats.learning_sessions += 1
        self.stats.last_learning_time = datetime.utcnow()
        
        # Calculate average quality score
        if processed_content:
            avg_quality = sum(c.quality_score for c in processed_content) / len(processed_content)
            self.stats.average_quality_score = (
                (self.stats.average_quality_score * (self.stats.learning_sessions - 1) + avg_quality) /
                self.stats.learning_sessions
            )
            
            # Update top topics
            all_topics = []
            for content in processed_content:
                all_topics.extend(content.topics)
            
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            self.stats.top_topics = sorted(
                topic_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            self.stats.top_topics = [topic for topic, count in self.stats.top_topics]
        
        # Record learning session
        session_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'urls_crawled': len(crawl_results),
            'content_processed': len(processed_content),
            'quality_scores': [c.quality_score for c in processed_content],
            'topics': list(set(all_topics)) if processed_content else []
        }
        
        self.learning_history.append(session_data)
        
        # Keep only last 100 sessions
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
    
    async def _perform_maintenance(self):
        """Perform maintenance tasks"""
        logger.info("Performing maintenance tasks...")
        
        try:
            # Cleanup old data
            await self.knowledge_integrator.cleanup_old_data()
            
            # Optimize knowledge base
            await self.knowledge_integrator.optimize_index()
            
            # Export learning history
            await self._export_learning_history()
            
        except Exception as e:
            logger.error(f"Error in maintenance: {e}")
    
    async def _export_learning_history(self):
        """Export learning history to file"""
        try:
            history_file = Path("learning_history.json")
            
            export_data = {
                'stats': asdict(self.stats),
                'history': self.learning_history,
                'config': asdict(self.config),
                'active_sources': list(self.active_sources),
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Learning history exported to {history_file}")
            
        except Exception as e:
            logger.error(f"Error exporting learning history: {e}")
    
    async def get_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report"""
        return {
            'current_stats': asdict(self.stats),
            'is_running': self.is_running,
            'active_sources_count': len(self.active_sources),
            'recent_sessions': self.learning_history[-10:],
            'component_stats': {
                'crawler': self.web_crawler.get_crawl_stats() if self.web_crawler else None,
                'processor': self.content_processor.get_processing_stats(),
                'knowledge_integrator': await self.knowledge_integrator.get_stats()
            },
            'learning_efficiency': {
                'success_rate': self.stats.successful_crawls / max(self.stats.total_urls_crawled, 1),
                'content_yield': self.stats.content_processed / max(self.stats.successful_crawls, 1),
                'knowledge_yield': self.stats.knowledge_added / max(self.stats.content_processed, 1),
                'average_quality': self.stats.average_quality_score
            }
        }
    
    async def manual_learning_session(self, urls: List[str]) -> Dict[str, Any]:
        """Perform a manual learning session with specific URLs"""
        logger.info(f"Starting manual learning session with {len(urls)} URLs")
        
        try:
            crawl_results = await self._crawl_urls(urls)
            processed_content = await self._process_content(crawl_results)
            await self._integrate_knowledge(processed_content)
            self._update_stats(crawl_results, processed_content)
            
            return {
                'success': True,
                'urls_crawled': len(crawl_results),
                'content_processed': len(processed_content),
                'knowledge_added': len(processed_content),
                'average_quality': sum(c.quality_score for c in processed_content) / max(len(processed_content), 1)
            }
            
        except Exception as e:
            logger.error(f"Error in manual learning session: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get current learning progress"""
        return {
            'stats': asdict(self.stats),
            'is_running': self.is_running,
            'active_sources': list(self.active_sources),
            'config': asdict(self.config),
            'recent_activity': self.learning_history[-5:] if self.learning_history else []
        }


# Learning scheduler for managing multiple learning tasks
class LearningScheduler:
    """Scheduler for managing learning tasks and priorities"""
    
    def __init__(self):
        self.tasks = []
        self.running = False
        self.scheduler_task = None
    
    def add_task(self, task_func, priority: int = 0, delay: int = 0):
        """Add a learning task to the scheduler"""
        self.tasks.append({
            'func': task_func,
            'priority': priority,
            'delay': delay,
            'next_run': datetime.utcnow() + timedelta(seconds=delay)
        })
        
        # Sort by priority
        self.tasks.sort(key=lambda x: x['priority'], reverse=True)
    
    async def start_scheduler(self):
        """Start the task scheduler"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop_scheduler(self):
        """Stop the task scheduler"""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for task in self.tasks:
                    if current_time >= task['next_run']:
                        # Execute task
                        try:
                            if asyncio.iscoroutinefunction(task['func']):
                                await task['func']
                            else:
                                task['func']()
                            
                            # Update next run time (simple: add delay)
                            task['next_run'] = current_time + timedelta(seconds=task['delay'])
                            
                        except Exception as e:
                            logger.error(f"Error executing scheduled task: {e}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
