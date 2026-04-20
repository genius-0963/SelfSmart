"""
Data Collector - Gather training data from the internet
Production-grade data collection for LLM fine-tuning.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Production-grade data collector for internet data.
    Supports web crawling, API integration, and dataset downloading.
    """
    
    def __init__(
        self,
        output_dir: str = "./training_data",
        max_concurrent_requests: int = 10,
        delay_between_requests: float = 1.0
    ):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to store collected data
            max_concurrent_requests: Max concurrent HTTP requests
            delay_between_requests: Delay between requests (seconds)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_requests = max_concurrent_requests
        self.delay_between_requests = delay_between_requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Data sources configuration
        self.sources = {
            "wikipedia": {
                "base_url": "https://en.wikipedia.org/api/rest_v1",
                "enabled": True
            },
            "hacker_news": {
                "base_url": "https://hacker-news.firebaseio.com/v0",
                "enabled": True
            },
            "reddit": {
                "base_url": "https://www.reddit.com",
                "enabled": False  # Requires API key
            }
        }
        
        logger.info(f"Data collector initialized with output directory: {self.output_dir}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True,
            force_close=False,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'SmartSelf-AI-DataCollector/1.0'
            },
            trust_env=False
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_wikipedia_articles(
        self,
        count: int = 100,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch Wikipedia articles for training data.
        
        Args:
            count: Number of articles to fetch
            categories: Specific categories to fetch
            
        Returns:
            List of article data
        """
        if not self.sources["wikipedia"]["enabled"]:
            logger.warning("Wikipedia source is disabled")
            return []
        
        logger.info(f"Fetching {count} Wikipedia articles")
        
        articles = []
        
        try:
            # Fetch random articles
            for _ in range(count):
                await asyncio.sleep(self.delay_between_requests)
                
                try:
                    async with self.session.get(
                        f"{self.sources['wikipedia']['base_url']}/page/random/summary"
                    ) as response:
                        if response.status == 200:
                            article = await response.json()
                            
                            # Extract content
                            article_data = {
                                "source": "wikipedia",
                                "title": article.get("title", ""),
                                "content": article.get("extract", ""),
                                "url": article.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                "timestamp": datetime.now().isoformat(),
                                "categories": article.get("categories", [])
                            }
                            
                            articles.append(article_data)
                            
                except Exception as e:
                    logger.error(f"Error fetching Wikipedia article: {e}")
                    continue
            
            logger.info(f"Successfully fetched {len(articles)} Wikipedia articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error in Wikipedia fetching: {e}")
            return []
    
    async def fetch_hacker_news_stories(
        self,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch Hacker News stories for training data.
        
        Args:
            count: Number of stories to fetch
            
        Returns:
            List of story data
        """
        if not self.sources["hacker_news"]["enabled"]:
            logger.warning("Hacker News source is disabled")
            return []
        
        logger.info(f"Fetching {count} Hacker News stories")
        
        stories = []
        
        try:
            # Get top story IDs
            async with self.session.get(
                f"{self.sources['hacker_news']['base_url']}/topstories.json"
            ) as response:
                if response.status == 200:
                    story_ids = await response.json()
                    story_ids = story_ids[:count]
                    
                    # Fetch story details
                    for story_id in story_ids:
                        await asyncio.sleep(self.delay_between_requests)
                        
                        try:
                            async with self.session.get(
                                f"{self.sources['hacker_news']['base_url']}/item/{story_id}.json"
                            ) as story_response:
                                if story_response.status == 200:
                                    story = await story_response.json()
                                    
                                    story_data = {
                                        "source": "hacker_news",
                                        "title": story.get("title", ""),
                                        "content": story.get("text", story.get("title", "")),
                                        "url": story.get("url", ""),
                                        "timestamp": datetime.now().isoformat(),
                                        "score": story.get("score", 0),
                                        "comments": story.get("descendants", 0)
                                    }
                                    
                                    stories.append(story_data)
                                    
                        except Exception as e:
                            logger.error(f"Error fetching story {story_id}: {e}")
                            continue
            
            logger.info(f"Successfully fetched {len(stories)} Hacker News stories")
            return stories
            
        except Exception as e:
            logger.error(f"Error in Hacker News fetching: {e}")
            return []
    
    async def crawl_url(
        self,
        url: str,
        max_depth: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Crawl a single URL and extract content.
        
        Args:
            url: URL to crawl
            max_depth: Maximum crawl depth
            
        Returns:
            Extracted content or None
        """
        try:
            await asyncio.sleep(self.delay_between_requests)
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # Extract metadata
                title = soup.find('title')
                title_text = title.get_text() if title else ""
                
                return {
                    "source": "web_crawl",
                    "title": title_text,
                    "content": text[:10000],  # Limit content length
                    "url": url,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return None
    
    async def collect_from_urls(
        self,
        urls: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Collect data from a list of URLs.
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of collected data
        """
        logger.info(f"Crawling {len(urls)} URLs")
        
        tasks = [self.crawl_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = []
        for result in results:
            if isinstance(result, dict):
                data.append(result)
        
        logger.info(f"Successfully crawled {len(data)} URLs")
        return data
    
    def save_data(
        self,
        data: List[Dict[str, Any]],
        filename: str
    ):
        """
        Save collected data to file.
        
        Args:
            data: Data to save
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} items to {output_path}")
    
    def load_data(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded data
        """
        input_path = self.output_dir / filename
        
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            return []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} items from {input_path}")
        return data
    
    async def collect_all(
        self,
        wikipedia_count: int = 50,
        hacker_news_count: int = 50,
        urls: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect data from all enabled sources.
        
        Args:
            wikipedia_count: Number of Wikipedia articles
            hacker_news_count: Number of Hacker News stories
            urls: List of URLs to crawl
            
        Returns:
            Dictionary of collected data by source
        """
        all_data = {}
        
        # Collect from Wikipedia
        if self.sources["wikipedia"]["enabled"]:
            wikipedia_data = await self.fetch_wikipedia_articles(wikipedia_count)
            all_data["wikipedia"] = wikipedia_data
            self.save_data(wikipedia_data, "wikipedia_data.json")
        
        # Collect from Hacker News
        if self.sources["hacker_news"]["enabled"]:
            hn_data = await self.fetch_hacker_news_stories(hacker_news_count)
            all_data["hacker_news"] = hn_data
            self.save_data(hn_data, "hacker_news_data.json")
        
        # Collect from URLs
        if urls:
            crawl_data = await self.collect_from_urls(urls)
            all_data["web_crawl"] = crawl_data
            self.save_data(crawl_data, "web_crawl_data.json")
        
        total_items = sum(len(data) for data in all_data.values())
        logger.info(f"Collected total of {total_items} items from all sources")
        
        return all_data
