"""
SmartSelf Learning Chatbot - Main Chatbot Interface
Intelligent chatbot that continuously learns from the internet.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict

from src.knowledge.knowledge_integrator import KnowledgeIntegrator
from src.learning.continuous_learner import ContinuousLearner, LearningConfig

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: List[str] = None
    confidence: float = 0.0
    learning_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.learning_info is None:
            self.learning_info = {}


@dataclass
class ChatResponse:
    """Chat response structure"""
    response: str
    sources: List[str]
    confidence: float
    learning_status: Dict[str, Any]
    related_topics: List[str]
    follow_up_questions: List[str]
    processing_time: float


class LearningChatbot:
    """
    Main chatbot class that combines conversational AI with continuous learning.
    """
    
    def __init__(self, learning_config: LearningConfig = None):
        """Initialize learning chatbot"""
        self.learning_config = learning_config or LearningConfig()
        
        # Initialize components
        self.knowledge_integrator = KnowledgeIntegrator()
        self.continuous_learner = ContinuousLearner(self.learning_config)
        
        # Chat state
        self.conversation_history: List[ChatMessage] = []
        self.session_id = None
        self.is_learning = False
        
        # Statistics
        self.chat_stats = {
            'total_conversations': 0,
            'total_messages': 0,
            'knowledge_used': 0,
            'average_response_time': 0.0,
            'user_satisfaction': 0.0
        }
        
        logger.info("Learning chatbot initialized")
    
    async def start(self):
        """Start the chatbot and continuous learning"""
        try:
            # Start continuous learning
            await self.continuous_learner.start_learning()
            self.is_learning = True
            
            logger.info("Learning chatbot started successfully")
            
        except Exception as e:
            logger.error(f"Error starting chatbot: {e}")
            raise
    
    async def stop(self):
        """Stop the chatbot and continuous learning"""
        try:
            # Stop continuous learning
            await self.continuous_learner.stop_learning()
            self.is_learning = False
            
            logger.info("Learning chatbot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping chatbot: {e}")
    
    async def chat(self, message: str, session_id: str = None) -> ChatResponse:
        """
        Process a chat message and generate response.
        
        Args:
            message: User message
            session_id: Optional session ID for conversation context
            
        Returns:
            ChatResponse with generated response and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Set session ID
            if session_id:
                self.session_id = session_id
            
            # Add user message to history
            user_message = ChatMessage(
                role='user',
                content=message,
                timestamp=datetime.utcnow()
            )
            self.conversation_history.append(user_message)
            
            # Generate response
            response = await self._generate_response(message)
            
            # Add assistant message to history
            assistant_message = ChatMessage(
                role='assistant',
                content=response.response,
                timestamp=datetime.utcnow(),
                sources=response.sources,
                confidence=response.confidence,
                learning_info=response.learning_status
            )
            self.conversation_history.append(assistant_message)
            
            # Update statistics
            self._update_chat_stats(start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            
            # Return error response
            error_response = ChatResponse(
                response="I'm sorry, I encountered an error while processing your message. Please try again.",
                sources=[],
                confidence=0.0,
                learning_status={'error': str(e)},
                related_topics=[],
                follow_up_questions=[],
                processing_time=0.0
            )
            
            return error_response
    
    async def _generate_response(self, message: str) -> ChatResponse:
        """Generate response using RAG and learning knowledge"""
        try:
            # Search knowledge base
            search_results = await self.knowledge_integrator.search_knowledge(
                message, 
                search_type="hybrid"
            )
            
            # Get conversation context
            context = self._get_conversation_context()
            
            # Generate response based on search results
            response_text = await self._generate_response_from_results(
                message, 
                search_results, 
                context
            )
            
            # Extract sources
            sources = [result.get('metadata', {}).get('source_url', '') for result in search_results[:5]]
            sources = [s for s in sources if s]  # Remove empty strings
            
            # Calculate confidence
            confidence = self._calculate_confidence(search_results, response_text)
            
            # Get related topics
            related_topics = self._extract_related_topics(search_results)
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(message, search_results)
            
            # Get learning status
            learning_status = await self.continuous_learner.get_learning_progress()
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - datetime.utcnow()).total_seconds()  # Placeholder
            
            return ChatResponse(
                response=response_text,
                sources=sources,
                confidence=confidence,
                learning_status=learning_status,
                related_topics=related_topics,
                follow_up_questions=follow_up_questions,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def _generate_response_from_results(
        self, 
        message: str, 
        search_results: List[Dict[str, Any]], 
        context: str
    ) -> str:
        """Generate response from search results"""
        if not search_results:
            return self._generate_fallback_response(message)
        
        try:
            # Extract relevant content
            relevant_content = []
            for result in search_results[:5]:  # Use top 5 results
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                
                if content:
                    relevant_content.append({
                        'content': content[:1000],  # Limit length
                        'title': metadata.get('title', ''),
                        'source': metadata.get('source_url', ''),
                        'quality_score': metadata.get('quality_score', 0.0)
                    })
            
            # Generate response using template-based approach
            # In production, this would use an LLM
            response = self._template_based_response(message, relevant_content, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response from results: {e}")
            return self._generate_fallback_response(message)
    
    def _template_based_response(
        self, 
        message: str, 
        relevant_content: List[Dict[str, Any]], 
        context: str
    ) -> str:
        """Generate response using template-based approach"""
        # This is a simplified template-based response generator
        # In production, this would use a sophisticated LLM
        
        message_lower = message.lower()
        
        # Check for question patterns
        if any(q in message_lower for q in ['what', 'how', 'why', 'when', 'where', 'who']):
            if relevant_content:
                best_content = max(relevant_content, key=lambda x: x.get('quality_score', 0))
                
                response = f"Based on what I've learned:\n\n{best_content['content'][:500]}..."
                
                if best_content.get('source'):
                    response += f"\n\nSource: {best_content['source']}"
                
                return response
            else:
                return "I don't have specific information about that in my current knowledge base. I'm continuously learning from the internet, so I may have an answer soon!"
        
        # Check for general conversation
        elif any(g in message_lower for g in ['hello', 'hi', 'hey', 'how are you']):
            return f"Hello! I'm a learning chatbot that continuously acquires knowledge from the internet. I've learned about {len(relevant_content)} relevant topics. How can I help you today?"
        
        # Default response
        else:
            if relevant_content:
                topics = [content.get('title', '') for content in relevant_content[:3]]
                topics = [t for t in topics if t]
                
                if topics:
                    return f"I can help you with information about: {', '.join(topics)}. What specific aspect would you like to know more about?"
                else:
                    return "I have some information that might be relevant to your query. Could you please be more specific about what you'd like to know?"
            else:
                return "I'm still learning about that topic. I continuously acquire knowledge from various sources on the internet. Could you try rephrasing your question or ask about something else?"
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate fallback response when no search results are available"""
        return ("I'm continuously learning from the internet to expand my knowledge base. "
                "While I don't have specific information about your query right now, "
                "I'm constantly acquiring new knowledge. Please try again later or ask about a different topic.")
    
    def _get_conversation_context(self) -> str:
        """Get conversation context for response generation"""
        if not self.conversation_history:
            return ""
        
        # Get last 5 messages for context
        recent_messages = self.conversation_history[-5:]
        context_parts = []
        
        for msg in recent_messages:
            if msg.role == 'user':
                context_parts.append(f"User: {msg.content}")
            else:
                context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(
        self, 
        search_results: List[Dict[str, Any]], 
        response_text: str
    ) -> float:
        """Calculate confidence score for response"""
        if not search_results:
            return 0.1  # Low confidence for fallback responses
        
        # Base confidence on search result quality and relevance
        max_score = max([result.get('score', 0) for result in search_results])
        avg_quality = sum([result.get('metadata', {}).get('quality_score', 0) 
                           for result in search_results]) / len(search_results)
        
        # Combine factors
        confidence = (max_score * 0.6) + (avg_quality * 0.4)
        
        # Adjust based on response length
        if len(response_text) < 50:
            confidence *= 0.7  # Penalty for very short responses
        elif len(response_text) > 500:
            confidence *= 1.2  # Bonus for detailed responses
        
        return min(1.0, max(0.0, confidence))
    
    def _extract_related_topics(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """Extract related topics from search results"""
        topics = set()
        
        for result in search_results:
            metadata = result.get('metadata', {})
            result_topics = metadata.get('topics', [])
            topics.update(result_topics)
        
        return list(topics)[:10]  # Limit to 10 topics
    
    def _generate_follow_up_questions(
        self, 
        message: str, 
        search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate follow-up questions based on search results"""
        questions = []
        
        if not search_results:
            return [
                "What would you like to learn more about?",
                "Can you provide more context for your question?",
                "Is there a specific aspect you're interested in?"
            ]
        
        # Extract topics from results
        topics = self._extract_related_topics(search_results)
        
        # Generate questions based on topics
        if topics:
            for topic in topics[:3]:
                questions.append(f"Can you tell me more about {topic}?")
        
        # Add general questions
        questions.extend([
            "How does this relate to what you're working on?",
            "Would you like me to find more recent information?",
            "Is there a specific application you're interested in?"
        ])
        
        return questions[:5]  # Limit to 5 questions
    
    def _update_chat_stats(self, start_time: datetime):
        """Update chat statistics"""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.chat_stats['total_messages'] += 1
        
        # Update average response time
        if self.chat_stats['total_messages'] == 1:
            self.chat_stats['average_response_time'] = processing_time
        else:
            current_avg = self.chat_stats['average_response_time']
            n = self.chat_stats['total_messages']
            new_avg = ((current_avg * (n - 1)) + processing_time) / n
            self.chat_stats['average_response_time'] = new_avg
    
    async def get_chatbot_status(self) -> Dict[str, Any]:
        """Get comprehensive chatbot status"""
        try:
            learning_progress = await self.continuous_learner.get_learning_progress()
            knowledge_stats = await self.knowledge_integrator.get_stats()
            
            return {
                'is_learning': self.is_learning,
                'session_id': self.session_id,
                'conversation_length': len(self.conversation_history),
                'chat_stats': self.chat_stats,
                'learning_progress': learning_progress,
                'knowledge_stats': knowledge_stats,
                'health_status': await self.knowledge_integrator.health_check()
            }
            
        except Exception as e:
            logger.error(f"Error getting chatbot status: {e}")
            return {
                'is_learning': self.is_learning,
                'error': str(e)
            }
    
    async def manual_learn(self, urls: List[str]) -> Dict[str, Any]:
        """Manually trigger learning from specific URLs"""
        try:
            result = await self.continuous_learner.manual_learning_session(urls)
            return result
            
        except Exception as e:
            logger.error(f"Error in manual learning: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.session_id = None
        logger.info("Conversation history cleared")
    
    def export_conversation_history(self, filename: str):
        """Export conversation history to file"""
        try:
            export_data = {
                'session_id': self.session_id,
                'export_timestamp': datetime.utcnow().isoformat(),
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat(),
                        'sources': msg.sources,
                        'confidence': msg.confidence,
                        'learning_info': msg.learning_info
                    }
                    for msg in self.conversation_history
                ],
                'stats': self.chat_stats
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Conversation history exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting conversation history: {e}")
            raise
    
    async def get_knowledge_about_topic(self, topic: str) -> Dict[str, Any]:
        """Get detailed knowledge about a specific topic"""
        try:
            # Search for topic
            search_results = await self.knowledge_integrator.search_knowledge(
                topic, 
                search_type="hybrid"
            )
            
            # Group results by topic
            topic_results = {}
            for result in search_results:
                metadata = result.get('metadata', {})
                result_topics = metadata.get('topics', [])
                
                for result_topic in result_topics:
                    if result_topic not in topic_results:
                        topic_results[result_topic] = []
                    topic_results[result_topic].append(result)
            
            # Format response
            response = {
                'topic': topic,
                'total_results': len(search_results),
                'related_topics': list(topic_results.keys()),
                'top_results': search_results[:5],
                'topic_breakdown': {
                    topic_name: {
                        'count': len(results),
                        'avg_quality': sum([r.get('metadata', {}).get('quality_score', 0) for r in results]) / len(results),
                        'top_result': max(results, key=lambda x: x.get('score', 0))
                    }
                    for topic_name, results in topic_results.items()
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting knowledge about topic: {e}")
            return {
                'topic': topic,
                'error': str(e),
                'total_results': 0
            }
    
    async def rate_response(self, message_index: int, rating: float) -> Dict[str, Any]:
        """Rate a chatbot response for learning improvement"""
        try:
            if 0 <= message_index < len(self.conversation_history):
                message = self.conversation_history[message_index]
                
                # Update user satisfaction
                self.chat_stats['user_satisfaction'] = (
                    (self.chat_stats['user_satisfaction'] * (self.chat_stats['total_messages'] - 1) + rating) /
                    self.chat_stats['total_messages']
                )
                
                # Store feedback for learning (in production, this would be more sophisticated)
                feedback_data = {
                    'message_index': message_index,
                    'rating': rating,
                    'message_content': message.content,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # In production, this would be stored in a database
                logger.info(f"User feedback received: {feedback_data}")
                
                return {
                    'success': True,
                    'message': 'Thank you for your feedback!',
                    'updated_satisfaction': self.chat_stats['user_satisfaction']
                }
            else:
                return {
                    'success': False,
                    'error': 'Invalid message index'
                }
                
        except Exception as e:
            logger.error(f"Error rating response: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Chatbot factory for easy initialization
def create_learning_chatbot(
    max_concurrent_crawls: int = 10,
    crawl_rate_limit: int = 1,
    daily_crawl_limit: int = 1000,
    min_quality_score: float = 0.3
) -> LearningChatbot:
    """Factory function to create a learning chatbot with custom configuration"""
    
    config = LearningConfig(
        max_concurrent_crawls=max_concurrent_crawls,
        crawl_rate_limit=crawl_rate_limit,
        daily_crawl_limit=daily_crawl_limit,
        min_quality_score=min_quality_score
    )
    
    return LearningChatbot(config)


# Example usage
async def main():
    """Example usage of the learning chatbot"""
    # Create chatbot
    chatbot = create_learning_chatbot()
    
    try:
        # Start chatbot
        await chatbot.start()
        
        # Example conversation
        response1 = await chatbot.chat("What is machine learning?")
        print(f"Bot: {response1.response}")
        
        response2 = await chatbot.chat("Can you explain neural networks?")
        print(f"Bot: {response2.response}")
        
        # Get status
        status = await chatbot.get_chatbot_status()
        print(f"Chatbot Status: {json.dumps(status, indent=2, default=str)}")
        
        # Manual learning
        learn_result = await chatbot.manual_learn([
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        ])
        print(f"Learning Result: {learn_result}")
        
    finally:
        # Stop chatbot
        await chatbot.stop()


if __name__ == "__main__":
    asyncio.run(main())
