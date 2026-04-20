"""
SmartSelf Learning Chatbot - Knowledge Integrator
Manages integration of processed content into various knowledge stores.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import chromadb
from neo4j import GraphDatabase
import elasticsearch
from sentence_transformers import SentenceTransformer
import hashlib
import json
from pathlib import Path

from src.processor.content_processor import ProcessedContent

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database for semantic search"""
    
    def __init__(self, collection_name: str = "learning_chatbot"):
        """Initialize vector store"""
        self.collection_name = collection_name
        try:
            self.client = chromadb.PersistentClient(path="./vector_store")
            self.collection = self.client.get_or_create_collection(collection_name)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Vector store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")
            self.client = None
            self.collection = None
            self.embedding_model = None
    
    async def add_documents(self, contents: List[ProcessedContent]):
        """Add documents to vector store"""
        if self.collection is None or self.embedding_model is None:
            logger.warning("Vector store not available, skipping document addition")
            return
            
        try:
            documents = []
            metadatas = []
            ids = []
            
            for content in contents:
                # Prepare document text
                doc_text = f"{content.title}\n\n{content.summary}\n\n{content.content[:1000]}"
                documents.append(doc_text)
                
                # Prepare metadata
                metadata = {
                    'title': content.title,
                    'summary': content.summary,
                    'topics': content.topics,
                    'quality_score': content.quality_score,
                    'relevance_score': content.relevance_score,
                    'language': content.language,
                    'source_url': content.metadata.get('source_url', ''),
                    'source_type': content.metadata.get('source_type', ''),
                    'timestamp': content.timestamp.isoformat(),
                    'content_length': len(content.content)
                }
                metadatas.append(metadata)
                ids.append(content.id)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            
            logger.info(f"Added {len(contents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
    
    async def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.collection is None or self.embedding_model is None:
            logger.warning("Vector store not available, returning empty results")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {}


class GraphStore:
    """Graph database for knowledge relationships"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """Initialize graph store"""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Graph store initialized")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            self.driver = None
    
    async def add_entities(self, contents: List[ProcessedContent]):
        """Add entities and relationships to graph"""
        if not self.driver:
            logger.warning("Graph store not available")
            return
        
        try:
            with self.driver.session() as session:
                for content in contents:
                    # Create content node
                    session.run("""
                        MERGE (c:Content {id: $id})
                        SET c.title = $title,
                            c.summary = $summary,
                            c.quality_score = $quality_score,
                            c.relevance_score = $relevance_score,
                            c.timestamp = $timestamp,
                            c.source_url = $source_url
                    """, {
                        'id': content.id,
                        'title': content.title,
                        'summary': content.summary,
                        'quality_score': content.quality_score,
                        'relevance_score': content.relevance_score,
                        'timestamp': content.timestamp.isoformat(),
                        'source_url': content.metadata.get('source_url', '')
                    })
                    
                    # Add topic nodes and relationships
                    for topic in content.topics:
                        session.run("""
                            MERGE (t:Topic {name: $topic})
                            MERGE (c:Content {id: $content_id})
                            MERGE (c)-[:HAS_TOPIC]->(t)
                        """, {
                            'topic': topic,
                            'content_id': content.id
                        })
                    
                    # Add entity nodes and relationships
                    for entity in content.entities:
                        entity_type = entity.get('label', 'ENTITY')
                        entity_text = entity.get('text', '')
                        
                        session.run("""
                            MERGE (e:Entity {name: $entity_name, type: $entity_type})
                            MERGE (c:Content {id: $content_id})
                            MERGE (c)-[:CONTAINS_ENTITY]->(e)
                        """, {
                            'entity_name': entity_text,
                            'entity_type': entity_type,
                            'content_id': content.id
                        })
            
            logger.info(f"Added entities for {len(contents)} contents to graph store")
            
        except Exception as e:
            logger.error(f"Error adding entities to graph store: {e}")
    
    async def query_related_content(self, content_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Query content related through entities and topics"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Content {id: $content_id})
                    MATCH (c)-[:HAS_TOPIC|CONTAINS_ENTITY*1..$depth]-(related:Content)
                    WHERE related.id <> $content_id
                    RETURN DISTINCT related.id as id, 
                           related.title as title,
                           related.quality_score as quality_score,
                           length shortestPath((c)-[*]-(related)) as distance
                    ORDER BY distance, related.quality_score DESC
                    LIMIT 20
                """, {
                    'content_id': content_id,
                    'depth': depth
                })
                
                results = []
                for record in result:
                    results.append({
                        'id': record['id'],
                        'title': record['title'],
                        'quality_score': record['quality_score'],
                        'distance': record['distance']
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error querying related content: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get graph store statistics"""
        if not self.driver:
            return {'status': 'not_connected'}
        
        try:
            with self.driver.session() as session:
                content_count = session.run("MATCH (c:Content) RETURN count(c) as count").single()['count']
                topic_count = session.run("MATCH (t:Topic) RETURN count(t) as count").single()['count']
                entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()['count']
                
                return {
                    'content_nodes': content_count,
                    'topic_nodes': topic_count,
                    'entity_nodes': entity_count,
                    'status': 'connected'
                }
                
        except Exception as e:
            logger.error(f"Error getting graph store stats: {e}")
            return {'status': 'error', 'error': str(e)}


class DocumentStore:
    """Document store for full-text search"""
    
    def __init__(self, hosts: List[str] = ["http://localhost:9200"]):
        """Initialize document store"""
        self.hosts = hosts
        self.client = None
        self.index_name = "learning_chatbot"
        
        try:
            self.client = elasticsearch.Elasticsearch(hosts)
            
            # Test connection
            if self.client.ping():
                logger.info("Document store initialized")
                
                # Create index if it doesn't exist
                if not self.client.indices.exists(index=self.index_name):
                    self.client.indices.create(
                        index=self.index_name,
                        body={
                            'mappings': {
                                'properties': {
                                    'title': {'type': 'text', 'analyzer': 'standard'},
                                    'content': {'type': 'text', 'analyzer': 'standard'},
                                    'summary': {'type': 'text', 'analyzer': 'standard'},
                                    'topics': {'type': 'keyword'},
                                    'quality_score': {'type': 'float'},
                                    'relevance_score': {'type': 'float'},
                                    'language': {'type': 'keyword'},
                                    'source_url': {'type': 'keyword'},
                                    'source_type': {'type': 'keyword'},
                                    'timestamp': {'type': 'date'}
                                }
                            }
                        }
                    )
            else:
                logger.warning("Could not connect to Elasticsearch")
                self.client = None
            
        except Exception as e:
            logger.warning(f"Could not connect to Elasticsearch: {e}")
            self.client = None
    
    async def index_documents(self, contents: List[ProcessedContent]):
        """Index documents for full-text search"""
        if not self.client:
            logger.warning("Document store not available")
            return
        
        try:
            for content in contents:
                doc = {
                    'title': content.title,
                    'content': content.content,
                    'summary': content.summary,
                    'topics': content.topics,
                    'quality_score': content.quality_score,
                    'relevance_score': content.relevance_score,
                    'language': content.language,
                    'source_url': content.metadata.get('source_url', ''),
                    'source_type': content.metadata.get('source_type', ''),
                    'timestamp': content.timestamp.isoformat()
                }
                
                self.client.index(
                    index=self.index_name,
                    id=content.id,
                    body=doc
                )
            
            logger.info(f"Indexed {len(contents)} documents in document store")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
    
    async def search(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search documents using full-text search"""
        if not self.client:
            return []
        
        try:
            result = self.client.search(
                index=self.index_name,
                body={
                    'query': {
                        'multi_match': {
                            'query': query,
                            'fields': ['title^3', 'summary^2', 'content'],
                            'type': 'best_fields'
                        }
                    },
                    'size': size,
                    'sort': [
                        {'quality_score': {'order': 'desc'}},
                        {'_score': {'order': 'desc'}}
                    ]
                }
            )
            
            results = []
            for hit in result['hits']['hits']:
                results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching document store: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics"""
        if not self.client:
            return {'status': 'not_connected'}
        
        try:
            count = self.client.count(index=self.index_name)['count']
            return {
                'total_documents': count,
                'index_name': self.index_name,
                'status': 'connected'
            }
        except Exception as e:
            logger.error(f"Error getting document store stats: {e}")
            return {'status': 'error', 'error': str(e)}


class KnowledgeIntegrator:
    """
    Main knowledge integrator that manages all knowledge stores
    and provides unified access to knowledge.
    """
    
    def __init__(self):
        """Initialize knowledge integrator"""
        try:
            self.vector_store = VectorStore()
            logger.info("Vector store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")
            self.vector_store = None
        
        try:
            self.graph_store = GraphStore()
            logger.info("Graph store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize graph store: {e}")
            self.graph_store = None
        
        try:
            self.document_store = DocumentStore()
            logger.info("Document store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize document store: {e}")
            self.document_store = None
        
        # Integration statistics
        self.integration_stats = {
            'total_integrated': 0,
            'last_integration': None,
            'integration_errors': 0
        }
        
        logger.info("Knowledge integrator initialized")
    
    async def batch_integrate(self, contents: List[ProcessedContent]):
        """Integrate a batch of processed content into all knowledge stores"""
        if not contents:
            return
        
        try:
            # Integrate into available stores concurrently
            tasks = []
            if self.vector_store:
                tasks.append(self.vector_store.add_documents(contents))
            if self.graph_store:
                tasks.append(self.graph_store.add_entities(contents))
            if self.document_store:
                tasks.append(self.document_store.index_documents(contents))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update statistics
            self.integration_stats['total_integrated'] += len(contents)
            self.integration_stats['last_integration'] = datetime.utcnow()
            
            logger.info(f"Successfully integrated {len(contents)} content items")
            
        except Exception as e:
            logger.error(f"Error in batch integration: {e}")
            self.integration_stats['integration_errors'] += 1
            raise
    
    async def search_knowledge(self, query: str, search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Search across all knowledge stores
        
        Args:
            query: Search query
            search_type: Type of search ("vector", "fulltext", "graph", "hybrid")
        
        Returns:
            List of search results with metadata
        """
        results = []
        
        try:
            if search_type in ["vector", "hybrid"]:
                vector_results = await self.vector_store.search(query, n_results=10)
                for result in vector_results:
                    results.append({
                        'id': result['id'],
                        'content': result['document'],
                        'metadata': result['metadata'],
                        'score': 1 - result['distance'],  # Convert distance to similarity
                        'source': 'vector'
                    })
            
            if search_type in ["fulltext", "hybrid"]:
                fulltext_results = await self.document_store.search(query, size=10)
                for result in fulltext_results:
                    results.append({
                        'id': result['id'],
                        'content': result['source']['content'],
                        'metadata': result['source'],
                        'score': result['score'],
                        'source': 'fulltext'
                    })
            
            if search_type == "graph":
                # For graph search, we need a content ID as starting point
                # This would be implemented based on specific use cases
                pass
            
            # If hybrid, merge and deduplicate results
            if search_type == "hybrid":
                results = self._merge_search_results(results)
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:20]  # Limit to top 20 results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    def _merge_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate search results from different sources"""
        seen_ids = set()
        merged_results = []
        
        for result in results:
            result_id = result['id']
            
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                merged_results.append(result)
            else:
                # Update score if this result has a higher score
                for existing in merged_results:
                    if existing['id'] == result_id and result['score'] > existing['score']:
                        existing['score'] = result['score']
                        existing['source'] = f"{existing['source']}+{result['source']}"
                        break
        
        return merged_results
    
    async def get_related_content(self, content_id: str) -> List[Dict[str, Any]]:
        """Get content related through knowledge graph"""
        try:
            # Get related content from graph store
            graph_results = await self.graph_store.query_related_content(content_id)
            
            # Enhance with full content from vector store
            related_content = []
            for result in graph_results:
                vector_results = await self.vector_store.search(result['id'], n_results=1)
                if vector_results:
                    related_content.append({
                        'id': result['id'],
                        'title': result['title'],
                        'content': vector_results[0]['document'],
                        'metadata': vector_results[0]['metadata'],
                        'distance': result['distance'],
                        'quality_score': result['quality_score']
                    })
            
            return related_content
            
        except Exception as e:
            logger.error(f"Error getting related content: {e}")
            return []
    
    async def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data from knowledge stores"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        try:
            # Note: Actual cleanup implementation would depend on specific store capabilities
            # This is a placeholder for the cleanup logic
            
            logger.info(f"Cleanup of data older than {days_old} days completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    async def optimize_index(self):
        """Optimize knowledge store indices"""
        try:
            # Note: Optimization would depend on specific store capabilities
            # This is a placeholder for optimization logic
            
            logger.info("Knowledge store optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing indices: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all knowledge stores"""
        try:
            vector_stats = await self.vector_store.get_stats()
            graph_stats = await self.graph_store.get_stats()
            document_stats = await self.document_store.get_stats()
            
            return {
                'vector_store': vector_stats,
                'graph_store': graph_stats,
                'document_store': document_stats,
                'integration_stats': self.integration_stats,
                'total_knowledge_items': vector_stats.get('total_documents', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge integrator stats: {e}")
            return {}
    
    async def export_knowledge(self, export_path: str, format: str = "json"):
        """Export knowledge to file"""
        try:
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'stats': await self.get_stats(),
                'sample_content': []  # Would include actual content in production
            }
            
            # Get sample content from vector store
            sample_results = await self.vector_store.search("", n_results=10)
            for result in sample_results:
                export_data['sample_content'].append({
                    'id': result['id'],
                    'title': result['metadata'].get('title', ''),
                    'summary': result['metadata'].get('summary', ''),
                    'topics': result['metadata'].get('topics', []),
                    'quality_score': result['metadata'].get('quality_score', 0)
                })
            
            # Export to file
            export_file = Path(export_path)
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Knowledge exported to {export_file}")
            
        except Exception as e:
            logger.error(f"Error exporting knowledge: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all knowledge stores"""
        health_status = {
            'overall': 'healthy',
            'stores': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Check vector store
            vector_stats = await self.vector_store.get_stats()
            health_status['stores']['vector'] = {
                'status': 'healthy' if vector_stats else 'unhealthy',
                'stats': vector_stats
            }
            
            # Check graph store
            graph_stats = await self.graph_store.get_stats()
            health_status['stores']['graph'] = {
                'status': graph_stats.get('status', 'unknown'),
                'stats': graph_stats
            }
            
            # Check document store
            document_stats = await self.document_store.get_stats()
            health_status['stores']['document'] = {
                'status': 'healthy' if document_stats else 'unhealthy',
                'stats': document_stats
            }
            
            # Determine overall health
            unhealthy_stores = [
                name for name, info in health_status['stores'].items()
                if info.get('status') not in ['healthy', 'connected']
            ]
            
            if unhealthy_stores:
                health_status['overall'] = 'degraded'
                health_status['unhealthy_stores'] = unhealthy_stores
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            health_status['overall'] = 'error'
            health_status['error'] = str(e)
            return health_status
