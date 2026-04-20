"""
Free API Client - No Authentication Required
Integrates various free public APIs that don't require API keys.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class FreeAPIClient:
    """Client for accessing free public APIs without authentication"""
    
    def __init__(self):
        self.session = None
        # Disable proxy for API calls to avoid DNS resolution issues
        self.connector = aiohttp.TCPConnector(
            limit=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            force_close=False,
            enable_cleanup_closed=True
        )
        self.base_apis = {
            # Wikipedia API - No auth required
            "wikipedia": {
                "base_url": "https://en.wikipedia.org/api/rest_v1",
                "endpoints": {
                    "random": "/page/random/summary",
                    "search": "/page/summary/{title}"
                }
            },
            # Hacker News - No auth required
            "hacker_news": {
                "base_url": "https://hacker-news.firebaseio.com/v0",
                "endpoints": {
                    "top_stories": "/topstories.json",
                    "new_stories": "/newstories.json",
                    "best_stories": "/beststories.json",
                    "item": "/item/{id}.json"
                }
            },
            # GitHub Public Events - No auth required (rate limited)
            "github": {
                "base_url": "https://api.github.com",
                "endpoints": {
                    "events": "/events",
                    "trending": "/search/repositories?q=stars:>1000&sort=stars&order=desc&per_page=10"
                }
            },
            # JSONPlaceholder - Fake data for testing
            "jsonplaceholder": {
                "base_url": "https://jsonplaceholder.typicode.com",
                "endpoints": {
                    "posts": "/posts",
                    "users": "/users",
                    "comments": "/comments"
                }
            },
            # Chuck Norris Jokes - No auth
            "chuck_norris": {
                "base_url": "https://api.chucknorris.io/jokes",
                "endpoints": {
                    "random": "/random",
                    "categories": "/categories"
                }
            },
            # Official Joke API - No auth
            "official_jokes": {
                "base_url": "https://official-joke-api.appspot.com",
                "endpoints": {
                    "random": "/random_joke",
                    "programming": "/jokes/programming/random"
                }
            },
            # Dog CEO - Dog images - No auth
            "dog_ceo": {
                "base_url": "https://dog.ceo/api",
                "endpoints": {
                    "random": "/breeds/image/random",
                    "all_breeds": "/breeds/list/all"
                }
            },
            # Cat Facts - No auth
            "cat_facts": {
                "base_url": "https://cat-fact.herokuapp.com",
                "endpoints": {
                    "random": "/facts/random"
                }
            },
            # Bored API - Activity suggestions - No auth
            "bored": {
                "base_url": "https://www.boredapi.com/api",
                "endpoints": {
                    "random": "/activity"
                }
            },
            # Agify - Age prediction - No auth
            "agify": {
                "base_url": "https://api.agify.io",
                "endpoints": {
                    "predict": "?name={name}"
                }
            },
            # Genderize - Gender prediction - No auth
            "genderize": {
                "base_url": "https://api.genderize.io",
                "endpoints": {
                    "predict": "?name={name}"
                }
            },
            # Nationalize - Nationality prediction - No auth
            "nationalize": {
                "base_url": "https://api.nationalize.io",
                "endpoints": {
                    "predict": "?name={name}"
                }
            },
            # Advice Slip - Advice API - No auth
            "advice": {
                "base_url": "https://api.adviceslip.com",
                "endpoints": {
                    "random": "/advice"
                }
            },
            # Quotable - Quotes API - No auth
            "quotable": {
                "base_url": "https://api.quotable.io",
                "endpoints": {
                    "random": "/random",
                    "quotes": "/quotes?limit=10"
                }
            },
            # Trivia - Trivia questions - No auth
            "trivia": {
                "base_url": "https://opentdb.com",
                "endpoints": {
                    "random": "/api.php?amount=1&type=multiple"
                }
            },
            # Fake Store - E-commerce data - No auth
            "fake_store": {
                "base_url": "https://fakestoreapi.com",
                "endpoints": {
                    "products": "/products",
                    "categories": "/products/categories"
                }
            },
            # CoinGecko - Crypto prices - No auth (rate limited)
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "endpoints": {
                    "ping": "/ping",
                    "coins": "/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false"
                }
            },
            # Open Library - Books API - No auth
            "open_library": {
                "base_url": "https://openlibrary.org",
                "endpoints": {
                    "random": "/api/books?bibkeys=OLID:OL1M&jscmd=data&format=json",
                    "search": "/search.json?q={query}"
                }
            },
            # Rest Countries - Country data - No auth
            "rest_countries": {
                "base_url": "https://restcountries.com/v3.1",
                "endpoints": {
                    "all": "/all",
                    "random": "/random"
                }
            },
            # IP API - IP geolocation - No auth
            "ip_api": {
                "base_url": "http://ip-api.com/json",
                "endpoints": {
                    "lookup": "/{ip}"
                }
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Create session with connector to avoid DNS issues
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        # Clear proxy environment variables to avoid DNS resolution issues
        import os
        proxies = None
        if not os.getenv('NO_PROXY'):
            # If no NO_PROXY is set, try without proxy
            proxies = None
        else:
            # Use proxy settings if explicitly configured
            proxies = None  # Disable proxy to avoid DNS issues
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=self.connector,
            headers={'User-Agent': 'SmartSelf-Learning-Chatbot/1.0'},
            trust_env=False  # Don't use system proxy settings
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_wikipedia_random(self) -> Optional[Dict[str, Any]]:
        """Fetch a random Wikipedia article"""
        try:
            url = f"{self.base_apis['wikipedia']['base_url']}{self.base_apis['wikipedia']['endpoints']['random']}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'source': 'wikipedia',
                        'title': data.get('title'),
                        'extract': data.get('extract'),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page')
                    }
        except Exception as e:
            logger.error(f"Error fetching Wikipedia: {e}")
        return None
    
    async def fetch_wikipedia_search(self, title: str) -> Optional[Dict[str, Any]]:
        """Search Wikipedia for a specific title"""
        try:
            endpoint = self.base_apis['wikipedia']['endpoints']['search'].format(title=title)
            url = f"{self.base_apis['wikipedia']['base_url']}{endpoint}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'source': 'wikipedia',
                        'title': data.get('title'),
                        'extract': data.get('extract'),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page')
                    }
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
        return None
    
    async def fetch_hacker_news_stories(self, story_type: str = "top", limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch Hacker News stories"""
        try:
            story_types = {
                "top": "top_stories",
                "new": "new_stories", 
                "best": "best_stories"
            }
            
            # Get story IDs
            endpoint = story_types.get(story_type, "top_stories")
            url = f"{self.base_apis['hacker_news']['base_url']}{self.base_apis['hacker_news']['endpoints'][endpoint]}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    story_ids = await response.json()
                    story_ids = story_ids[:limit]  # Limit results
                    
                    # Fetch story details
                    stories = []
                    for story_id in story_ids:
                        story_url = f"{self.base_apis['hacker_news']['base_url']}/item/{story_id}.json"
                        async with self.session.get(story_url) as story_response:
                            if story_response.status == 200:
                                story_data = await story_response.json()
                                stories.append({
                                    'source': 'hacker_news',
                                    'title': story_data.get('title'),
                                    'url': story_data.get('url'),
                                    'score': story_data.get('score'),
                                    'by': story_data.get('by'),
                                    'time': datetime.fromtimestamp(story_data.get('time', 0)).isoformat()
                                })
                    
                    return stories
        except Exception as e:
            logger.error(f"Error fetching Hacker News: {e}")
        return []
    
    async def fetch_github_trending(self) -> List[Dict[str, Any]]:
        """Fetch trending GitHub repositories"""
        try:
            url = f"{self.base_apis['github']['base_url']}{self.base_apis['github']['endpoints']['trending']}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    repos = []
                    for repo in data.get('items', []):
                        repos.append({
                            'source': 'github',
                            'name': repo.get('full_name'),
                            'description': repo.get('description'),
                            'url': repo.get('html_url'),
                            'stars': repo.get('stargazers_count'),
                            'language': repo.get('language')
                        })
                    return repos
        except Exception as e:
            logger.error(f"Error fetching GitHub trending: {e}")
        return []
    
    async def fetch_joke(self, joke_type: str = "chuck") -> Optional[Dict[str, Any]]:
        """Fetch a random joke"""
        try:
            if joke_type == "chuck":
                url = f"{self.base_apis['chuck_norris']['base_url']}{self.base_apis['chuck_norris']['endpoints']['random']}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'source': 'chuck_norris',
                            'joke': data.get('value'),
                            'categories': data.get('categories')
                        }
            elif joke_type == "official":
                url = f"{self.base_apis['official_jokes']['base_url']}{self.base_apis['official_jokes']['endpoints']['random']}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'source': 'official_jokes',
                            'setup': data.get('setup'),
                            'punchline': data.get('punchline'),
                            'type': data.get('type')
                        }
        except Exception as e:
            logger.error(f"Error fetching joke: {e}")
        return None
    
    async def fetch_quote(self) -> Optional[Dict[str, Any]]:
        """Fetch a random quote"""
        try:
            url = f"{self.base_apis['quotable']['base_url']}{self.base_apis['quotable']['endpoints']['random']}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'source': 'quotable',
                        'content': data.get('content'),
                        'author': data.get('author'),
                        'tags': data.get('tags')
                    }
        except Exception as e:
            logger.error(f"Error fetching quote: {e}")
        return None
    
    async def fetch_advice(self) -> Optional[Dict[str, Any]]:
        """Fetch random advice"""
        try:
            url = f"{self.base_apis['advice']['base_url']}{self.base_apis['advice']['endpoints']['random']}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'source': 'advice',
                        'advice': data.get('slip', {}).get('advice')
                    }
        except Exception as e:
            logger.error(f"Error fetching advice: {e}")
        return None
    
    async def fetch_trivia(self) -> Optional[Dict[str, Any]]:
        """Fetch a trivia question"""
        try:
            url = f"{self.base_apis['trivia']['base_url']}{self.base_apis['trivia']['endpoints']['random']}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('response_code') == 0 and data.get('results'):
                        question = data['results'][0]
                        return {
                            'source': 'trivia',
                            'question': question.get('question'),
                            'correct_answer': question.get('correct_answer'),
                            'incorrect_answers': question.get('incorrect_answers'),
                            'category': question.get('category'),
                            'difficulty': question.get('difficulty')
                        }
        except Exception as e:
            logger.error(f"Error fetching trivia: {e}")
        return None
    
    async def fetch_countries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch country information"""
        try:
            url = f"{self.base_apis['rest_countries']['base_url']}{self.base_apis['rest_countries']['endpoints']['random']}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    countries = []
                    for country in data[:limit]:
                        countries.append({
                            'source': 'rest_countries',
                            'name': country.get('name', {}).get('common'),
                            'capital': country.get('capital'),
                            'region': country.get('region'),
                            'population': country.get('population'),
                            'languages': country.get('languages')
                        })
                    return countries
        except Exception as e:
            logger.error(f"Error fetching countries: {e}")
        return []
    
    async def fetch_crypto_prices(self) -> List[Dict[str, Any]]:
        """Fetch cryptocurrency prices"""
        try:
            url = f"{self.base_apis['coingecko']['base_url']}{self.base_apis['coingecko']['endpoints']['coins']}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    cryptos = []
                    for coin in data:
                        cryptos.append({
                            'source': 'coingecko',
                            'name': coin.get('name'),
                            'symbol': coin.get('symbol'),
                            'current_price': coin.get('current_price'),
                            'market_cap': coin.get('market_cap'),
                            'price_change_24h': coin.get('price_change_percentage_24h')
                        })
                    return cryptos
        except Exception as e:
            logger.error(f"Error fetching crypto prices: {e}")
        return []
    
    async def fetch_activity_suggestion(self) -> Optional[Dict[str, Any]]:
        """Fetch a random activity suggestion"""
        try:
            url = f"{self.base_apis['bored']['base_url']}{self.base_apis['bored']['endpoints']['random']}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'source': 'bored',
                        'activity': data.get('activity'),
                        'type': data.get('type'),
                        'participants': data.get('participants'),
                        'price': data.get('price')
                    }
        except Exception as e:
            logger.error(f"Error fetching activity: {e}")
        return None
    
    async def fetch_all_random_content(self) -> Dict[str, Any]:
        """Fetch random content from multiple APIs"""
        results = {}
        
        # Fetch from multiple APIs in parallel
        tasks = [
            self.fetch_wikipedia_random(),
            self.fetch_joke(),
            self.fetch_quote(),
            self.fetch_advice(),
            self.fetch_activity_suggestion(),
            self.fetch_trivia()
        ]
        
        api_names = ['wikipedia', 'joke', 'quote', 'advice', 'activity', 'trivia']
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for api_name, response in zip(api_names, responses):
            if not isinstance(response, Exception) and response:
                results[api_name] = response
        
        return results
    
    def get_available_apis(self) -> List[str]:
        """Get list of available free APIs"""
        return list(self.base_apis.keys())
