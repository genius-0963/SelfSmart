"""
SmartSelf Learning Chatbot - Web Crawler
Intelligent web crawling system for continuous knowledge acquisition.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import feedparser
import trafilatura
from newspaper import Article
from aiolimiter import AsyncLimiter
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Result from crawling a URL"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    source_type: str
    quality_score: float
    language: str


class WebCrawler:
    """
    Intelligent web crawler with rate limiting and content extraction.
    """
    
    def __init__(self, max_concurrent: int = 10, rate_limit: int = 1):
        """
        Initialize web crawler.
        
        Args:
            max_concurrent: Maximum concurrent requests
            rate_limit: Requests per second per domain
        """
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.domain_limiters = {}  # Per-domain rate limiters
        self.session = None
        self.visited_urls: Set[str] = set()
        self.crawl_history: List[Dict[str, Any]] = []
        
        # User agent and headers
        self.headers = {
            'User-Agent': 'SmartSelf-Learning-Bot/1.0 (Educational Purpose)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        logger.info("Web crawler initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=self.max_concurrent)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_domain_limiter(self, domain: str) -> AsyncLimiter:
        """Get or create rate limiter for domain"""
        if domain not in self.domain_limiters:
            self.domain_limiters[domain] = AsyncLimiter(self.rate_limit, 1)
        return self.domain_limiters[domain]
    
    async def crawl_url(self, url: str, source_type: str = "web") -> Optional[CrawlResult]:
        """
        Crawl a single URL and extract content.
        
        Args:
            url: URL to crawl
            source_type: Type of source (web, rss, api)
            
        Returns:
            CrawlResult with extracted content or None if failed
        """
        try:
            domain = urlparse(url).netloc
            limiter = self._get_domain_limiter(domain)
            
            async with self.semaphore:
                async with limiter:
                    if url in self.visited_urls:
                        logger.debug(f"URL already visited: {url}")
                        return None
                    
                    self.visited_urls.add(url)
                    
                    async with self.session.get(url) as response:
                        if response.status != 200:
                            logger.warning(f"HTTP {response.status} for {url}")
                            return None
                        
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' not in content_type:
                            logger.debug(f"Skipping non-HTML content: {content_type}")
                            return None
                        
                        html = await response.text()
                        
                        # Extract content using multiple methods
                        result = await self._extract_content(url, html, source_type)
                        
                        if result and result.quality_score > 0.3:
                            self.crawl_history.append({
                                'url': url,
                                'timestamp': datetime.utcnow(),
                                'success': True,
                                'quality_score': result.quality_score
                            })
                            return result
                        else:
                            logger.debug(f"Low quality content for {url}")
                            return None
                            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            self.crawl_history.append({
                'url': url,
                'timestamp': datetime.utcnow(),
                'success': False,
                'error': str(e)
            })
            return None
    
    async def _extract_content(self, url: str, html: str, source_type: str) -> Optional[CrawlResult]:
        """Extract high-quality content from HTML"""
        try:
            # Method 1: Newspaper3k for news articles
            if source_type in ["web", "news"]:
                article = Article(url)
                article.set_html(html)
                article.parse()
                
                if article.title and article.text and len(article.text) > 200:
                    return CrawlResult(
                        url=url,
                        title=article.title,
                        content=article.text,
                        metadata={
                            'authors': article.authors,
                            'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                            'top_image': article.top_image,
                            'movies': article.movies,
                            'keywords': article.keywords,
                            'summary': article.summary,
                            'extraction_method': 'newspaper3k'
                        },
                        timestamp=datetime.utcnow(),
                        source_type=source_type,
                        quality_score=self._calculate_quality_score(article.text),
                        language=article.meta_lang or 'en'
                    )
            
            # Method 2: Trafilatura for general content
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                include_links=True
            )
            
            if extracted and len(extracted) > 200:
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ''
                
                return CrawlResult(
                    url=url,
                    title=title_text,
                    content=extracted,
                    metadata={
                        'extraction_method': 'trafilatura',
                        'content_length': len(extracted)
                    },
                    timestamp=datetime.utcnow(),
                    source_type=source_type,
                    quality_score=self._calculate_quality_score(extracted),
                        language='en'  # Default, can be detected later
                )
            
            # Method 3: BeautifulSoup fallback
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ''
            
            # Try to find main content
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_=lambda x: x and ('content' in x.lower() or 'main' in x.lower()))
            )
            
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) > 200:
                return CrawlResult(
                    url=url,
                    title=title_text,
                    content=text,
                    metadata={
                        'extraction_method': 'beautifulsoup',
                        'content_length': len(text)
                    },
                    timestamp=datetime.utcnow(),
                    source_type=source_type,
                    quality_score=self._calculate_quality_score(text),
                    language='en'
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for extracted content"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length score
        length = len(text)
        if length > 500:
            score += 0.3
        elif length > 200:
            score += 0.2
        
        # Sentence structure
        sentences = text.split('.')
        if len(sentences) > 5:
            score += 0.2
        
        # Word diversity
        words = text.lower().split()
        unique_words = set(words)
        if len(unique_words) / max(len(words), 1) > 0.3:
            score += 0.2
        
        # Content indicators
        content_indicators = ['however', 'therefore', 'because', 'although', 'moreover', 'furthermore']
        if any(indicator in text.lower() for indicator in content_indicators):
            score += 0.2
        
        # Penalty for repetitive content
        if len(set(words)) / max(len(words), 1) < 0.2:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    async def crawl_urls_batch(self, urls: List[str], source_type: str = "web") -> List[CrawlResult]:
        """Crawl multiple URLs concurrently"""
        tasks = [self.crawl_url(url, source_type) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, CrawlResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch crawling error: {result}")
        
        logger.info(f"Successfully crawled {len(valid_results)}/{len(urls)} URLs")
        return valid_results
    
    def get_crawl_stats(self) -> Dict[str, Any]:
        """Get crawling statistics"""
        successful = sum(1 for h in self.crawl_history if h.get('success', False))
        total = len(self.crawl_history)
        
        return {
            'total_urls_crawled': total,
            'successful_crawls': successful,
            'success_rate': successful / max(total, 1),
            'unique_urls_visited': len(self.visited_urls),
            'active_domains': len(self.domain_limiters),
            'last_crawl_time': max([h['timestamp'] for h in self.crawl_history]) if self.crawl_history else None
        }
    
    def export_crawl_history(self, filename: str):
        """Export crawl history to file"""
        with open(filename, 'w') as f:
            json.dump(self.crawl_history, f, indent=2, default=str)
        logger.info(f"Crawl history exported to {filename}")


class RSSCrawler:
    """Specialized crawler for RSS feeds"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def crawl_feed(self, feed_url: str) -> List[CrawlResult]:
        """Crawl RSS feed and return articles"""
        try:
            # Parse feed
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"Feed parsing warning for {feed_url}: {feed.bozo_exception}")
            
            articles = []
            
            for entry in feed.entries:
                # Extract article URL
                article_url = entry.get('link')
                if not article_url:
                    continue
                
                # Create basic article info
                title = entry.get('title', '')
                summary = entry.get('summary', '') or entry.get('description', '')
                
                # Combine title and summary for content
                content = f"{title}\n\n{summary}"
                
                # Extract metadata
                metadata = {
                    'feed_url': feed_url,
                    'feed_title': feed.feed.get('title', ''),
                    'published': entry.get('published'),
                    'author': entry.get('author'),
                    'tags': [tag.get('term') for tag in entry.get('tags', [])],
                    'source_type': 'rss'
                }
                
                article = CrawlResult(
                    url=article_url,
                    title=title,
                    content=content,
                    metadata=metadata,
                    timestamp=datetime.utcnow(),
                    source_type='rss',
                    quality_score=0.7,  # RSS feeds generally have good quality
                    language='en'
                )
                
                articles.append(article)
            
            logger.info(f"Extracted {len(articles)} articles from {feed_url}")
            return articles
            
        except Exception as e:
            logger.error(f"Error crawling RSS feed {feed_url}: {e}")
            return []


# Example usage and configuration
DEFAULT_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.reuters.com/reuters/topNews",
    "https://techcrunch.com/feed/",
    "https://www.wired.com/feed/rss",
]

DEFAULT_CRAWL_DOMAINS = [
    "en.wikipedia.org",
    "github.com",
    "stackoverflow.com",
    "medium.com",
    "dev.to",
    "towardsdatascience.com"
]
