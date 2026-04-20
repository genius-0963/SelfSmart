#!/usr/bin/env python3
"""
Self-Learning Chatbot - Focused Implementation
A chatbot that continuously learns from the internet
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Load environment
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

# Import simplified learning chatbot
import sys
sys.path.append('/Users/subh/Documents/selfsmart')
from simple_learning_chatbot import create_learning_chatbot, get_chatbot

# Create app
app = FastAPI(
    title="Self-Learning Chatbot",
    description="AI chatbot that continuously learns from the internet",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Request models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class LearnRequest(BaseModel):
    urls: list[str]
    force_learn: bool = False

# Global chatbot instance
learning_chatbot = None

@app.get("/")
async def root():
    """Serve the frontend interface"""
    return FileResponse("frontend/index.html")

@app.get("/api")
async def api_info():
    return {
        "service": "Self-Learning Chatbot",
        "version": "1.0.0",
        "description": "AI chatbot that continuously learns from the internet",
        "capabilities": [
            "Continuous web learning",
            "Knowledge integration",
            "Natural language conversation",
            "Source citation"
        ]
    }

@app.get("/health")
async def health():
    if learning_chatbot is None:
        return {"status": "initializing", "chatbot_ready": False}
    
    try:
        status = await learning_chatbot.get_chatbot_status()
        return {
            "status": "healthy",
            "chatbot_ready": True,
            "learning_status": status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "chatbot_ready": False,
            "error": str(e)
        }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat with the self-learning bot"""
    try:
        chatbot = await get_chatbot()
        response = await chatbot.chat(req.message, req.session_id)
        return {
            "response": response.response,
            "sources": response.sources,
            "confidence": response.confidence,
            "session_id": req.session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/learn")
async def learn_from_urls(req: LearnRequest):
    """Manually trigger learning from specific URLs"""
    try:
        chatbot = await get_chatbot()
        result = await chatbot.learn_from_urls(req.urls)
        return {
            "status": "learning_completed",
            "urls_count": len(req.urls),
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning error: {str(e)}")

@app.get("/api/status")
async def get_learning_status():
    """Get detailed learning system status"""
    try:
        chatbot = await get_chatbot()
        status = await chatbot.get_status()
        return {
            "chatbot_status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

@app.get("/api/knowledge/{topic}")
async def get_knowledge_about_topic(topic: str, limit: int = 5):
    """Get knowledge about a specific topic"""
    try:
        chatbot = await get_chatbot()
        # Use the search functionality to get knowledge about the topic
        search_results = chatbot.vector_store.search(topic, n_results=limit)
        return {
            "topic": topic,
            "knowledge_items": search_results,
            "count": len(search_results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge retrieval error: {str(e)}")

@app.post("/api/learning/toggle")
async def toggle_learning():
    """Toggle continuous learning on/off"""
    if learning_chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # This would need to be implemented in the learning chatbot
        # For now, return a placeholder response
        return {
            "message": "Learning toggle functionality to be implemented",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Toggle error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the learning chatbot with LLM pipeline"""
    try:
        print(" Initializing Self-Learning Chatbot...")
        print(" Setting up vector store...")
        print(" Starting web crawler...")
        
        # Initialize LLM pipeline
        from llm_pipeline import initialize_llm_pipeline
        
        # Get API key from environment (prioritize working APIs)
        openai_key = os.getenv("OPENAI_API_KEY")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if openai_key:
            print(" Initializing OpenAI pipeline (reliable)...")
            initialize_llm_pipeline(openai_key, "openai")
            print(" OpenAI pipeline ready!")
        elif deepseek_key:
            print(" Initializing DeepSeek pipeline (cost-effective)...")
            initialize_llm_pipeline(deepseek_key, "deepseek")
            print(" DeepSeek pipeline ready!")
        elif gemini_key:
            print(" Initializing Gemini pipeline...")
            initialize_llm_pipeline(gemini_key, "gemini")
            print(" Gemini pipeline ready!")
        else:
            print(" WARNING: No LLM API key found!")
            print(" Set OPENAI_API_KEY, DEEPSEEK_API_KEY or GEMINI_API_KEY in your .env file")
            print(" Chatbot will use basic responses only")
        
        # Initialize the chatbot (will be created lazily when needed)
        chatbot = await get_chatbot()
        
        print(" Self-Learning Chatbot ready!")
        print(" API available at: http://localhost:8000")
        print(" Try: POST /api/chat with {\"message\": \"Hello, what can you tell me about AI?\"}")
        print(" Or: POST /api/learn with {\"urls\": [\"https://example.com\"]} to teach me")
        
    except Exception as e:
        print(f" Failed to initialize chatbot: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print(" Starting Self-Learning Chatbot Server...")
    print(" This bot continuously learns from the internet!")
    print()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
