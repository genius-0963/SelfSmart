"""
DeepSeek API Client - Production-Grade Implementation
Handles LLM interactions with streaming, context management, and error recovery.
"""

import asyncio
import aiohttp
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List
from datetime import datetime
import json
from dataclasses import dataclass, field

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a conversation message"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Represents an LLM response"""
    content: str
    finish_reason: str
    usage: Dict[str, int]
    model: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)


class DeepSeekClient:
    """
    Production-grade DeepSeek API client with streaming and context management.
    Implements retry logic, rate limiting, and proper error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DeepSeek client"""
        settings = get_settings()
        self.api_key = api_key or settings.deepseek_api_key
        
        if not self.api_key:
            logger.warning("DeepSeek API key not configured")
            raise ValueError("DeepSeek API key is required")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_retries = 3
        self.retry_delay = 2.0
        self.timeout = 60.0  # Increased from 30 to 60 seconds
        
        # Rate limiting
        self.request_semaphore = asyncio.Semaphore(10)  # Max concurrent requests
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        logger.info("DeepSeek client initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.timeout, connect=30)  # Increased connect timeout
        connector = aiohttp.TCPConnector(
            limit=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            force_close=False,
            enable_cleanup_closed=True,
            limit_per_host=5  # Limit connections per host to avoid rate limiting
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'SmartSelf-AI/1.0'
            },
            trust_env=False,  # Don't use system proxy
            skip_auto_headers=['Accept-Encoding']  # Skip auto headers that might cause issues
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request_with_retry(
        self,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic and rate limiting.
        Production-grade error handling and recovery.
        """
        async with self.request_semaphore:
            await asyncio.sleep(self.rate_limit_delay)
            
            for attempt in range(self.max_retries):
                try:
                    url = f"{self.base_url}{endpoint}"
                    async with self.session.post(url, json=payload) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:
                            # Rate limit hit
                            retry_after = float(response.headers.get('Retry-After', self.retry_delay))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        elif response.status >= 500:
                            # Server error, retry
                            error_text = await response.text()
                            logger.error(f"Server error (attempt {attempt + 1}): {error_text}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                                continue
                            else:
                                raise Exception(f"Server error after {self.max_retries} attempts")
                        else:
                            # Client error
                            error_text = await response.text()
                            error_data = await response.json() if response.headers.get('content-type') == 'application/json' else {}
                            raise Exception(f"API error {response.status}: {error_text} - {error_data}")
                
                except asyncio.TimeoutError as e:
                    logger.error(f"Timeout error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        raise Exception(f"Connection timeout after {self.max_retries} attempts: {e}")
                except aiohttp.ClientConnectorError as e:
                    logger.error(f"Connection error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        raise Exception(f"Connection failed after {self.max_retries} attempts: {e}")
                except aiohttp.ClientError as e:
                    logger.error(f"Network error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        raise Exception(f"Network error after {self.max_retries} attempts: {e}")
            
            raise Exception("Max retries exceeded")
    
    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> LLMResponse:
        """
        Send chat completion request to DeepSeek API.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            stream: Whether to stream response
            
        Returns:
            LLMResponse with content and metadata
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Convert Message objects to API format
        api_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response_data = await self._make_request_with_retry("/chat/completions", payload)
            
            choice = response_data["choices"][0]
            
            return LLMResponse(
                content=choice["message"]["content"],
                finish_reason=choice.get("finish_reason", "stop"),
                usage=response_data.get("usage", {}),
                model=response_data.get("model", self.model)
            )
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def chat_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion response from DeepSeek API.
        Yields chunks of text as they arrive.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Yields:
            String chunks of the response
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Convert Message objects to API format
        api_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        async with self.request_semaphore:
            await asyncio.sleep(self.rate_limit_delay)
            
            try:
                url = f"{self.base_url}/chat/completions"
                async with self.session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Stream API error {response.status}: {error_text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or line == 'data: [DONE]':
                            continue
                        
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse stream chunk: {e}")
                                continue
                
            except aiohttp.ClientError as e:
                logger.error(f"Streaming error: {e}")
                raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            response_data = await self._make_request_with_retry("/models", {})
            return response_data
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def create_system_prompt(self, context: Optional[str] = None) -> Message:
        """
        Create a system prompt with optional context.
        Production-grade prompt engineering.
        """
        base_prompt = """You are SmartSelf AI, an intelligent assistant that continuously learns from the internet. You are:
- Helpful and accurate
- Concise but thorough
- Able to cite sources when relevant
- Context-aware and conversational
- Professional yet friendly

When responding:
1. Provide clear, direct answers
2. If uncertain, acknowledge limitations
3. Use natural, conversational language
4. Break down complex topics
5. Adapt to user's expertise level
"""
        
        if context:
            base_prompt += f"\n\nContext:\n{context}"
        
        return Message(role="system", content=base_prompt)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Rough approximation: ~4 characters per token for English.
        """
        return len(text) // 4
