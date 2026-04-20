"""
RAG (Retrieval-Augmented Generation) Service - Production-Grade Implementation
Enhances LLM responses with knowledge from the integrated knowledge base.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from src.knowledge.knowledge_integrator import KnowledgeIntegrator
from src.llm.deepseek_client import Message, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class RetrievedKnowledge:
    """Represents retrieved knowledge piece"""
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any]


@dataclass
class RAGContext:
    """Represents the RAG context for a query"""
    query: str
    retrieved_knowledge: List[RetrievedKnowledge]
    enhanced_prompt: str
    timestamp: datetime


class RAGService:
    """
    Production-grade RAG service that enhances LLM responses with knowledge
    from the integrated knowledge base using semantic search.
    """
    
    def __init__(self, knowledge_integrator: Optional[KnowledgeIntegrator] = None):
        """Initialize RAG service"""
        self.knowledge_integrator = knowledge_integrator
        self.max_knowledge_pieces = 5
        self.min_relevance_score = 0.5
        self.use_rag = True
        
        if self.knowledge_integrator is None:
            try:
                self.knowledge_integrator = KnowledgeIntegrator()
                logger.info("RAG service initialized with knowledge integrator")
            except Exception as e:
                logger.warning(f"Could not initialize knowledge integrator: {e}")
                self.knowledge_integrator = None
                self.use_rag = False
        else:
            logger.info("RAG service initialized with provided knowledge integrator")
    
    async def retrieve_relevant_knowledge(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievedKnowledge]:
        """
        Retrieve relevant knowledge pieces for a query using semantic search.
        
        Args:
            query: User query
            top_k: Number of top results to retrieve
            
        Returns:
            List of RetrievedKnowledge objects
        """
        if not self.use_rag or self.knowledge_integrator is None:
            logger.debug("RAG disabled, skipping knowledge retrieval")
            return []
        
        try:
            # Search vector store for relevant documents
            if self.knowledge_integrator.vector_store:
                search_results = await self.knowledge_integrator.vector_store.search(
                    query=query,
                    n_results=top_k
                )
                
                knowledge_pieces = []
                for result in search_results:
                    if result.get('distance', 1.0) <= (1.0 - self.min_relevance_score):
                        metadata = result.get('metadata', {})
                        knowledge_piece = RetrievedKnowledge(
                            content=result.get('document', ''),
                            source=metadata.get('source_url', 'knowledge base'),
                            relevance_score=1.0 - result.get('distance', 1.0),
                            metadata=metadata
                        )
                        knowledge_pieces.append(knowledge_piece)
                
                # Sort by relevance and limit
                knowledge_pieces.sort(key=lambda x: x.relevance_score, reverse=True)
                return knowledge_pieces[:self.max_knowledge_pieces]
            
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []
    
    def build_enhanced_prompt(
        self,
        query: str,
        knowledge: List[RetrievedKnowledge],
        conversation_history: Optional[List[Message]] = None
    ) -> str:
        """
        Build an enhanced prompt that includes retrieved knowledge.
        Production-grade prompt engineering.
        
        Args:
            query: User query
            knowledge: Retrieved knowledge pieces
            conversation_history: Previous conversation context
            
        Returns:
            Enhanced prompt string
        """
        if not knowledge:
            # No relevant knowledge found, return original query
            return query
        
        # Build knowledge context
        knowledge_context = "Relevant knowledge from the system's learning:\n\n"
        for i, piece in enumerate(knowledge, 1):
            knowledge_context += f"{i}. {piece.content}\n"
            knowledge_context += f"   Source: {piece.source}\n"
            knowledge_context += f"   Relevance: {piece.relevance_score:.2f}\n\n"
        
        # Build enhanced prompt
        enhanced_prompt = f"""User Query: {query}

{knowledge_context}

Instructions:
- Use the provided knowledge to answer the user's question
- If the knowledge is insufficient, acknowledge this and provide general guidance
- Cite sources when using specific information from the knowledge
- Maintain natural, conversational tone
- Be accurate and honest about what you know from the knowledge vs general knowledge

Answer:"""
        
        return enhanced_prompt
    
    async def enhance_query(
        self,
        query: str,
        conversation_history: Optional[List[Message]] = None
    ) -> Tuple[str, List[RetrievedKnowledge]]:
        """
        Enhance a query with retrieved knowledge.
        
        Args:
            query: Original user query
            conversation_history: Previous conversation context
            
        Returns:
            Tuple of (enhanced_query, retrieved_knowledge)
        """
        # Retrieve relevant knowledge
        knowledge = await self.retrieve_relevant_knowledge(query)
        
        if not knowledge:
            logger.debug("No relevant knowledge found, returning original query")
            return query, knowledge
        
        # Build enhanced prompt
        enhanced_query = self.build_enhanced_prompt(query, knowledge, conversation_history)
        
        logger.info(f"Enhanced query with {len(knowledge)} knowledge pieces")
        return enhanced_query, knowledge
    
    async def process_llm_response(
        self,
        llm_response: LLMResponse,
        retrieved_knowledge: List[RetrievedKnowledge]
    ) -> LLMResponse:
        """
        Process LLM response and add knowledge sources.
        
        Args:
            llm_response: Original LLM response
            retrieved_knowledge: Knowledge used for enhancement
            
        Returns:
            Enhanced LLM response with sources
        """
        if not retrieved_knowledge:
            return llm_response
        
        # Extract unique sources
        sources = list(set([piece.source for piece in retrieved_knowledge]))
        
        # Add sources to response
        llm_response.sources = sources
        
        return llm_response
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics"""
        stats = {
            "rag_enabled": self.use_rag,
            "knowledge_integrator_available": self.knowledge_integrator is not None,
            "max_knowledge_pieces": self.max_knowledge_pieces,
            "min_relevance_score": self.min_relevance_score
        }
        
        if self.knowledge_integrator and self.knowledge_integrator.vector_store:
            try:
                vector_stats = asyncio.create_task(
                    self.knowledge_integrator.vector_store.get_stats()
                )
                stats["vector_store"] = vector_stats.result()
            except Exception as e:
                logger.warning(f"Could not get vector store stats: {e}")
        
        return stats
    
    def enable_rag(self, enabled: bool = True):
        """Enable or disable RAG"""
        self.use_rag = enabled
        logger.info(f"RAG {'enabled' if enabled else 'disabled'}")
    
    def set_relevance_threshold(self, threshold: float):
        """Set minimum relevance score for knowledge retrieval"""
        self.min_relevance_score = max(0.0, min(1.0, threshold))
        logger.info(f"Relevance threshold set to {self.min_relevance_score}")
