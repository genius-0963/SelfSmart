#!/usr/bin/env python3
"""
LLM Pipeline for ChatGPT/Gemini Integration
RAG pipeline with context retrieval and LLM generation
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

import aiohttp
from openai import AsyncOpenAI
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMPipeline:
    """LLM Pipeline with RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider.lower()
        self.client = None
        self.fallback_client = None
        
        # Get proxy settings
        http_proxy = os.getenv("HTTP_PROXY")
        https_proxy = os.getenv("HTTPS_PROXY")
        
        # Configure httpx client with proxy if available
        http_client = None
        if http_proxy or https_proxy:
            http_client = httpx.AsyncClient(
                proxies={
                    "http://": http_proxy,
                    "https://": https_proxy or http_proxy
                }
            )
        
        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=api_key, http_client=http_client)
        elif self.provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                http_client=http_client
            )
            # Set up OpenAI as fallback
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.fallback_client = AsyncOpenAI(api_key=openai_key, http_client=http_client)
        elif self.provider == "gemini":
            # For Gemini, we'll use HTTP requests with proxy
            self.proxy_config = {}
            if http_proxy or https_proxy:
                self.proxy_config = {
                    "http": http_proxy,
                    "https": https_proxy or http_proxy
                }
            pass
    
    async def generate_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate response using LLM with retrieved context"""
        
        if self.provider == "openai":
            return await self._generate_openai_response(query, context, conversation_history)
        elif self.provider == "deepseek":
            return await self._generate_deepseek_response(query, context, conversation_history)
        elif self.provider == "gemini":
            return await self._generate_gemini_response(query, context, conversation_history)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _generate_openai_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        
        # Build context string
        context_text = self._build_context_string(context)
        
        # Build conversation history
        messages = self._build_conversation_messages(query, context_text, conversation_history)
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" for better responses
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return {
                "response": response.choices[0].message.content,
                "sources": [ctx.get("metadata", {}).get("url", "") for ctx in context],
                "confidence": 0.9,  # High confidence for LLM responses
                "model": "openai",
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "response": f"I apologize, but I encountered an error generating a response. Error: {str(e)}",
                "sources": [],
                "confidence": 0.1,
                "model": "openai",
                "error": str(e)
            }
    
    async def _generate_deepseek_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate response using DeepSeek API (OpenAI-compatible)"""
        
        # Build context string
        context_text = self._build_context_string(context)
        
        # Build conversation history
        messages = self._build_conversation_messages(query, context_text, conversation_history)
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",  # DeepSeek's model
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return {
                "response": response.choices[0].message.content,
                "sources": [ctx.get("metadata", {}).get("url", "") for ctx in context],
                "confidence": 0.9,  # High confidence for LLM responses
                "model": "deepseek",
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            
            # Check if it's a balance issue and fallback to OpenAI
            if "Insufficient Balance" in str(e) and self.fallback_client:
                logger.info("DeepSeek balance insufficient, falling back to OpenAI...")
                try:
                    response = await self.fallback_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    
                    return {
                        "response": response.choices[0].message.content,
                        "sources": [ctx.get("metadata", {}).get("url", "") for ctx in context],
                        "confidence": 0.9,
                        "model": "openai-fallback",
                        "tokens_used": response.usage.total_tokens if response.usage else 0,
                        "note": "Response generated using OpenAI fallback due to DeepSeek balance limits"
                    }
                except Exception as fallback_error:
                    logger.error(f"OpenAI fallback also failed: {fallback_error}")
                    return {
                        "response": f"I apologize, but both DeepSeek and OpenAI APIs are unavailable. Error: {str(e)}",
                        "sources": [],
                        "confidence": 0.1,
                        "model": "deepseek",
                        "error": str(e)
                    }
            else:
                return {
                    "response": f"I apologize, but I encountered an error generating a response. Error: {str(e)}",
                    "sources": [],
                    "confidence": 0.1,
                    "model": "deepseek",
                    "error": str(e)
                }
    
    async def _generate_gemini_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate response using Gemini API"""
        
        # Build context string
        context_text = self._build_context_string(context)
        
        # Build prompt
        prompt = self._build_gemini_prompt(query, context_text, conversation_history)
        
        try:
            # Configure session with proxy if available
            connector = None
            if hasattr(self, 'proxy_config') and self.proxy_config:
                connector = aiohttp.TCPConnector()
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                trust_env=True  # This will use environment proxy settings
            ) as session:
                headers = {
                    "Content-Type": "application/json",
                }
                
                data = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 1000,
                    }
                }
                
                # Using Gemini 2.5 Flash API
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={self.api_key}"
                
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if "candidates" in result and len(result["candidates"]) > 0:
                            generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                            
                            return {
                                "response": generated_text,
                                "sources": [ctx.get("metadata", {}).get("url", "") for ctx in context],
                                "confidence": 0.9,
                                "model": "gemini",
                                "tokens_used": len(generated_text.split())
                            }
                        else:
                            return {
                                "response": "I apologize, but I couldn't generate a response.",
                                "sources": [],
                                "confidence": 0.1,
                                "model": "gemini",
                                "error": "No candidates in response"
                            }
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API error: {response.status} - {error_text}")
                        return {
                            "response": f"I apologize, but I encountered an error with the Gemini API.",
                            "sources": [],
                            "confidence": 0.1,
                            "model": "gemini",
                            "error": f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "response": f"I apologize, but I encountered an error generating a response. Error: {str(e)}",
                "sources": [],
                "confidence": 0.1,
                "model": "gemini",
                "error": str(e)
            }
    
    def _build_context_string(self, context: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        if not context:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, ctx in enumerate(context[:5], 1):  # Limit to top 5 results
            metadata = ctx.get("metadata", {})
            title = metadata.get("title", f"Document {i}")
            url = metadata.get("url", "")
            content = ctx.get("content", "")
            
            # Truncate content if too long
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_parts.append(f"[Source {i}: {title}]({url})\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _build_conversation_messages(
        self, 
        query: str, 
        context_text: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Build conversation messages for OpenAI API"""
        
        system_prompt = """You are a helpful AI assistant with access to a knowledge base. 
        Use the provided context to answer the user's question accurately and comprehensively.
        If the context doesn't contain enough information, say so politely.
        Always cite your sources when using information from the context.
        Be conversational and helpful."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:  # Keep last 10 messages
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        return messages
    
    def _build_gemini_prompt(
        self, 
        query: str, 
        context_text: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """Build prompt for Gemini API"""
        
        prompt = f"""You are a helpful AI assistant with access to a knowledge base. 
Use the provided context to answer the user's question accurately and comprehensively.
If the context doesn't contain enough information, say so politely.
Always cite your sources when using information from the context.
Be conversational and helpful.

Context:
{context_text}

Question: {query}

Please provide a helpful response based on the context above."""
        
        # Add conversation history if provided
        if conversation_history:
            prompt += "\n\nRecent conversation:\n"
            for msg in conversation_history[-5:]:  # Keep last 5 messages
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
        
        return prompt


class ConversationManager:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 20):
        self.conversations = {}  # session_id -> conversation history
        self.max_history = max_history
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for session"""
        return self.conversations.get(session_id, [])
    
    def clear_history(self, session_id: str):
        """Clear conversation history for session"""
        if session_id in self.conversations:
            del self.conversations[session_id]


# Global instances
llm_pipeline = None
conversation_manager = ConversationManager()


def initialize_llm_pipeline(api_key: str, provider: str = "openai"):
    """Initialize the LLM pipeline"""
    global llm_pipeline
    llm_pipeline = LLMPipeline(api_key, provider)
    return llm_pipeline


def get_llm_pipeline():
    """Get the LLM pipeline instance"""
    return llm_pipeline


def get_conversation_manager():
    """Get the conversation manager instance"""
    return conversation_manager
