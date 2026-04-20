"""
SmartSelf Learning Chatbot - Content Processor
Advanced content processing pipeline for quality control and knowledge extraction.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import hashlib
import spacy
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ProcessedContent:
    """Processed content with metadata"""
    id: str
    title: str
    content: str
    summary: str
    topics: List[str]
    entities: List[Dict[str, Any]]
    quality_score: float
    relevance_score: float
    language: str
    metadata: Dict[str, Any]
    timestamp: datetime
    embeddings: Optional[np.ndarray] = None


class ContentProcessor:
    """
    Advanced content processing with quality control, 
    deduplication, and knowledge extraction.
    """
    
    def __init__(self):
        """Initialize content processor"""
        self.nlp = None
        self.quality_classifier = None
        self.tfidf_vectorizer = None
        self.content_hashes = set()
        self.processed_count = 0
        self.duplicate_count = 0
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Content processor initialized")
    
    def _initialize_models(self):
        """Initialize NLP and ML models"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded")
        except OSError:
            logger.warning("SpaCy model not found, using basic processing")
            self.nlp = None
        
        try:
            # Load quality classification model
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.quality_classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name
            )
            logger.info("Quality classifier loaded")
        except Exception as e:
            logger.warning(f"Could not load quality classifier: {e}")
            self.quality_classifier = None
        
        # Initialize TF-IDF vectorizer for deduplication
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    async def process_content_batch(
        self, 
        crawl_results: List[Any]
    ) -> List[ProcessedContent]:
        """
        Process a batch of crawled content.
        
        Args:
            crawl_results: List of CrawlResult objects
            
        Returns:
            List of ProcessedContent objects
        """
        processed_contents = []
        
        for crawl_result in crawl_results:
            try:
                processed = await self.process_single_content(crawl_result)
                if processed:
                    processed_contents.append(processed)
            except Exception as e:
                logger.error(f"Error processing content {crawl_result.url}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_contents)}/{len(crawl_results)} content items")
        return processed_contents
    
    async def process_single_content(self, crawl_result: Any) -> Optional[ProcessedContent]:
        """Process a single content item"""
        try:
            # Extract text content
            title = crawl_result.title
            content = crawl_result.content
            
            if not content or len(content.strip()) < 100:
                logger.debug(f"Content too short: {crawl_result.url}")
                return None
            
            # Clean and normalize content
            cleaned_content = self._clean_text(content)
            cleaned_title = self._clean_text(title)
            
            # Check for duplicates
            content_hash = self._generate_content_hash(cleaned_content)
            if content_hash in self.content_hashes:
                self.duplicate_count += 1
                logger.debug(f"Duplicate content detected: {crawl_result.url}")
                return None
            
            self.content_hashes.add(content_hash)
            
            # Language detection
            language = self._detect_language(cleaned_content)
            if language != 'en':
                logger.debug(f"Non-English content detected: {language}")
                # Could implement translation here
            
            # Quality assessment
            quality_score = await self._assess_quality(cleaned_content)
            if quality_score < 0.3:
                logger.debug(f"Low quality content: {quality_score}")
                return None
            
            # Extract knowledge
            topics = await self._extract_topics(cleaned_content)
            entities = await self._extract_entities(cleaned_content)
            summary = self._generate_summary(cleaned_content)
            
            # Relevance scoring
            relevance_score = self._calculate_relevance(topics, entities)
            
            # Create processed content
            processed = ProcessedContent(
                id=self._generate_content_id(crawl_result.url),
                title=cleaned_title,
                content=cleaned_content,
                summary=summary,
                topics=topics,
                entities=entities,
                quality_score=quality_score,
                relevance_score=relevance_score,
                language=language,
                metadata={
                    'source_url': crawl_result.url,
                    'source_type': crawl_result.source_type,
                    'original_metadata': crawl_result.metadata,
                    'content_length': len(cleaned_content),
                    'word_count': len(cleaned_content.split()),
                    'processing_timestamp': datetime.utcnow().isoformat()
                },
                timestamp=datetime.utcnow()
            )
            
            self.processed_count += 1
            return processed
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\!]{2,}', '!', text)
        text = re.sub(r'[\?]{2,}', '?', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([\.!\?,;:])', r'\1', text)
        text = re.sub(r'([\.!\?,;:])\s+', r'\1 ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        return text.strip()
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        # Normalize content for hashing
        normalized = content.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _generate_content_id(self, url: str) -> str:
        """Generate unique content ID"""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def _detect_language(self, text: str) -> str:
        """Detect content language"""
        try:
            if len(text) < 50:
                return 'en'  # Default for short text
            
            language = detect(text)
            return language
        except:
            return 'en'  # Default on error
    
    async def _assess_quality(self, content: str) -> float:
        """Assess content quality using multiple metrics"""
        quality_score = 0.0
        
        # Length score
        word_count = len(content.split())
        if word_count > 500:
            quality_score += 0.2
        elif word_count > 200:
            quality_score += 0.1
        
        # Readability score
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if 10 <= avg_sentence_length <= 25:
            quality_score += 0.2
        elif 5 <= avg_sentence_length <= 35:
            quality_score += 0.1
        
        # Vocabulary diversity
        words = content.lower().split()
        unique_words = set(words)
        if len(unique_words) / max(len(words), 1) > 0.3:
            quality_score += 0.2
        
        # Content structure indicators
        structure_indicators = ['however', 'therefore', 'because', 'although', 'moreover', 'furthermore', 'in conclusion']
        if any(indicator in content.lower() for indicator in structure_indicators):
            quality_score += 0.1
        
        # ML-based quality assessment
        if self.quality_classifier:
            try:
                # Use sentiment analysis as a proxy for quality
                result = self.quality_classifier(content[:512])  # Limit length
                if result[0]['label'] == 'POSITIVE':
                    quality_score += 0.2
            except:
                pass
        
        # Penalty for repetitive content
        if len(unique_words) / max(len(words), 1) < 0.2:
            quality_score -= 0.2
        
        # Penalty for very short content
        if word_count < 50:
            quality_score -= 0.3
        
        return max(0.0, min(1.0, quality_score))
    
    async def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content"""
        topics = []
        
        # Simple keyword-based topic extraction
        topic_keywords = {
            'technology': ['technology', 'software', 'computer', 'programming', 'artificial intelligence', 'machine learning', 'data', 'algorithm'],
            'business': ['business', 'company', 'market', 'economy', 'finance', 'investment', 'revenue', 'profit'],
            'science': ['science', 'research', 'study', 'experiment', 'discovery', 'theory', 'hypothesis', 'analysis'],
            'health': ['health', 'medicine', 'disease', 'treatment', 'patient', 'medical', 'healthcare', 'therapy'],
            'education': ['education', 'learning', 'school', 'university', 'student', 'teacher', 'course', 'knowledge'],
            'politics': ['politics', 'government', 'policy', 'election', 'democracy', 'political', 'vote', 'campaign'],
            'sports': ['sports', 'game', 'player', 'team', 'competition', 'match', 'tournament', 'athlete'],
            'entertainment': ['entertainment', 'movie', 'music', 'film', 'celebrity', 'show', 'performance', 'art']
        }
        
        content_lower = content.lower()
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score >= 2:  # At least 2 keywords to qualify
                topics.append(topic)
        
        # Use spaCy for more advanced topic extraction if available
        if self.nlp:
            try:
                doc = self.nlp(content[:100000])  # Limit length for performance
                
                # Extract noun phrases as potential topics
                noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
                
                # Count frequency of noun phrases
                phrase_freq = {}
                for phrase in noun_phrases:
                    phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
                
                # Add most frequent phrases as topics
                top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                for phrase, freq in top_phrases:
                    if freq >= 2 and len(phrase) > 3:
                        topics.append(phrase.replace(' ', '_'))
                
            except Exception as e:
                logger.warning(f"SpaCy topic extraction failed: {e}")
        
        return list(set(topics))[:10]  # Limit to 10 topics
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from content"""
        entities = []
        
        if self.nlp:
            try:
                doc = self.nlp(content[:100000])  # Limit length
                
                for ent in doc.ents:
                    if len(ent.text.strip()) > 2:  # Filter out very short entities
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 1.0  # SpaCy doesn't provide confidence scores
                        })
                
            except Exception as e:
                logger.warning(f"SpaCy entity extraction failed: {e}")
        else:
            # Fallback: basic regex-based entity extraction
            entities = self._extract_entities_basic(content)
        
        return entities
    
    def _extract_entities_basic(self, content: str) -> List[Dict[str, Any]]:
        """Basic entity extraction using regex patterns"""
        entities = []
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, content):
            entities.append({
                'text': match.group(),
                'label': 'EMAIL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8
            })
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for match in re.finditer(url_pattern, content):
            entities.append({
                'text': match.group(),
                'label': 'URL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        # Dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        for match in re.finditer(date_pattern, content):
            entities.append({
                'text': match.group(),
                'label': 'DATE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.7
            })
        
        return entities
    
    def _generate_summary(self, content: str) -> str:
        """Generate content summary"""
        # Simple extractive summarization
        sentences = content.split('.')
        
        # Filter sentences by length and importance
        important_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                # Check for importance indicators
                importance_words = ['important', 'significant', 'key', 'major', 'critical', 'essential', 'crucial']
                if any(word in sentence.lower() for word in importance_words):
                    important_sentences.append(sentence)
        
        # If no important sentences found, use first few sentences
        if not important_sentences:
            important_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
        
        # Create summary
        summary = '. '.join(important_sentences[:3])
        return summary if summary else content[:200] + "..." if len(content) > 200 else content
    
    def _calculate_relevance(self, topics: List[str], entities: List[Dict[str, Any]]) -> float:
        """Calculate relevance score based on topics and entities"""
        relevance_score = 0.0
        
        # Topic relevance
        if topics:
            relevance_score += min(len(topics) * 0.1, 0.5)
        
        # Entity relevance
        if entities:
            relevance_score += min(len(entities) * 0.05, 0.3)
        
        # High-value entity types
        high_value_entities = ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']
        high_value_count = sum(1 for e in entities if e.get('label') in high_value_entities)
        relevance_score += min(high_value_count * 0.1, 0.2)
        
        return min(1.0, relevance_score)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get content processing statistics"""
        return {
            'processed_count': self.processed_count,
            'duplicate_count': self.duplicate_count,
            'duplicate_rate': self.duplicate_count / max(self.processed_count + self.duplicate_count, 1),
            'unique_content_hashes': len(self.content_hashes),
            'models_loaded': {
                'spacy': self.nlp is not None,
                'quality_classifier': self.quality_classifier is not None
            }
        }
    
    def export_processed_content(self, processed_contents: List[ProcessedContent], filename: str):
        """Export processed content to file"""
        export_data = []
        for content in processed_contents:
            export_data.append({
                'id': content.id,
                'title': content.title,
                'content': content.content,
                'summary': content.summary,
                'topics': content.topics,
                'entities': content.entities,
                'quality_score': content.quality_score,
                'relevance_score': content.relevance_score,
                'language': content.language,
                'metadata': content.metadata,
                'timestamp': content.timestamp.isoformat()
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} processed content items to {filename}")


# Utility functions for content processing
def merge_similar_content(contents: List[ProcessedContent], similarity_threshold: float = 0.8) -> List[ProcessedContent]:
    """Merge similar content items to reduce redundancy"""
    if not contents:
        return contents
    
    # Create TF-IDF vectors
    texts = [content.content for content in contents]
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find and merge similar content
    merged_contents = []
    used_indices = set()
    
    for i, content in enumerate(contents):
        if i in used_indices:
            continue
        
        # Find similar items
        similar_indices = [j for j in range(len(contents)) 
                          if j != i and j not in used_indices 
                          and similarity_matrix[i][j] > similarity_threshold]
        
        if similar_indices:
            # Merge content
            similar_contents = [content] + [contents[j] for j in similar_indices]
            merged_content = merge_content_items(similar_contents)
            merged_contents.append(merged_content)
            used_indices.update(similar_indices)
        else:
            merged_contents.append(content)
        
        used_indices.add(i)
    
    return merged_contents


def merge_content_items(contents: List[ProcessedContent]) -> ProcessedContent:
    """Merge multiple content items into one"""
    if not contents:
        raise ValueError("Cannot merge empty content list")
    
    # Use the first content as base
    base = contents[0]
    
    # Merge topics
    all_topics = []
    for content in contents:
        all_topics.extend(content.topics)
    merged_topics = list(set(all_topics))
    
    # Merge entities
    all_entities = []
    for content in contents:
        all_entities.extend(content.entities)
    
    # Remove duplicate entities
    unique_entities = []
    seen_entities = set()
    for entity in all_entities:
        entity_key = (entity['text'], entity['label'])
        if entity_key not in seen_entities:
            unique_entities.append(entity)
            seen_entities.add(entity_key)
    
    # Create merged content
    merged_content = ProcessedContent(
        id=base.id,
        title=base.title,
        content=base.content,  # Keep original content
        summary=base.summary,
        topics=merged_topics,
        entities=unique_entities,
        quality_score=max(c.quality_score for c in contents),
        relevance_score=max(c.relevance_score for c in contents),
        language=base.language,
        metadata={
            **base.metadata,
            'merged_from': [c.id for c in contents[1:]],
            'merge_count': len(contents)
        },
        timestamp=base.timestamp
    )
    
    return merged_content
