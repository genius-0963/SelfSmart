"""
SmartSelf Learning Chatbot - Knowledge Export/Import System
Comprehensive knowledge base export and import functionality.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import pickle
import gzip
import hashlib
from pathlib import Path
import zipfile
import shutil

from src.knowledge.knowledge_integrator import KnowledgeIntegrator
from src.processor.content_processor import ProcessedContent

logger = logging.getLogger(__name__)


class KnowledgeExporter:
    """Knowledge base export system"""
    
    def __init__(self, knowledge_integrator: KnowledgeIntegrator):
        """Initialize knowledge exporter"""
        self.knowledge_integrator = knowledge_integrator
        self.export_formats = ['json', 'pickle', 'csv', 'xml', 'yaml']
        self.compression_formats = ['none', 'gzip', 'zip']
        
        logger.info("Knowledge exporter initialized")
    
    async def export_knowledge(
        self,
        output_path: str,
        format: str = 'json',
        compression: str = 'none',
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Export knowledge base to file
        
        Args:
            output_path: Output file path
            format: Export format (json, pickle, csv, xml, yaml)
            compression: Compression format (none, gzip, zip)
            filters: Optional filters for export
            include_metadata: Include metadata in export
            batch_size: Batch size for processing
            
        Returns:
            Export result information
        """
        try:
            # Validate parameters
            if format not in self.export_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            if compression not in self.compression_formats:
                raise ValueError(f"Unsupported compression: {compression}")
            
            # Create output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect knowledge data
            knowledge_data = await self._collect_knowledge_data(filters, batch_size)
            
            # Format data
            formatted_data = await self._format_data(knowledge_data, format, include_metadata)
            
            # Apply compression
            final_path = await self._apply_compression(formatted_data, output_path, format, compression)
            
            # Generate export summary
            summary = {
                'success': True,
                'output_path': str(final_path),
                'format': format,
                'compression': compression,
                'items_exported': len(knowledge_data),
                'file_size': final_path.stat().st_size if final_path.exists() else 0,
                'export_timestamp': datetime.utcnow().isoformat(),
                'filters': filters or {}
            }
            
            logger.info(f"Knowledge exported successfully: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error exporting knowledge: {e}")
            return {
                'success': False,
                'error': str(e),
                'export_timestamp': datetime.utcnow().isoformat()
            }
    
    async def _collect_knowledge_data(
        self,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Collect knowledge data from all stores"""
        knowledge_data = []
        
        # Get vector store data
        vector_data = await self._get_vector_store_data(filters, batch_size)
        knowledge_data.extend(vector_data)
        
        # Get graph store data
        graph_data = await self._get_graph_store_data(filters, batch_size)
        knowledge_data.extend(graph_data)
        
        # Get document store data
        document_data = await self._get_document_store_data(filters, batch_size)
        knowledge_data.extend(document_data)
        
        logger.info(f"Collected {len(knowledge_data)} knowledge items")
        return knowledge_data
    
    async def _get_vector_store_data(
        self,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get data from vector store"""
        try:
            # Get all documents from vector store
            collection = self.knowledge_integrator.vector_store.collection
            
            # Get collection stats
            count = collection.count()
            logger.info(f"Vector store has {count} documents")
            
            # Get all documents (in batches)
            all_data = []
            offset = 0
            
            while offset < count:
                # Get batch
                results = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=['metadatas', 'documents']
                )
                
                # Convert to export format
                for i, doc_id in enumerate(results['ids']):
                    data = {
                        'id': doc_id,
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'source': 'vector_store',
                        'export_timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Apply filters
                    if self._passes_filters(data, filters):
                        all_data.append(data)
                
                offset += batch_size
            
            logger.info(f"Exported {len(all_data)} items from vector store")
            return all_data
            
        except Exception as e:
            logger.error(f"Error getting vector store data: {e}")
            return []
    
    async def _get_graph_store_data(
        self,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get data from graph store"""
        try:
            if not self.knowledge_integrator.graph_store.driver:
                logger.warning("Graph store not connected")
                return []
            
            graph_data = []
            
            with self.knowledge_integrator.graph_store.driver.session() as session:
                # Get all content nodes
                result = session.run("""
                    MATCH (c:Content)
                    RETURN c.id as id, c.title as title, c.summary as summary,
                           c.quality_score as quality_score, c.relevance_score as relevance_score,
                           c.timestamp as timestamp, c.source_url as source_url
                    LIMIT $batch_size
                """, batch_size=batch_size)
                
                for record in result:
                    data = {
                        'id': record['id'],
                        'title': record['title'],
                        'summary': record['summary'],
                        'quality_score': record['quality_score'],
                        'relevance_score': record['relevance_score'],
                        'timestamp': record['timestamp'],
                        'source_url': record['source_url'],
                        'source': 'graph_store',
                        'export_timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Apply filters
                    if self._passes_filters(data, filters):
                        graph_data.append(data)
            
            logger.info(f"Exported {len(graph_data)} items from graph store")
            return graph_data
            
        except Exception as e:
            logger.error(f"Error getting graph store data: {e}")
            return []
    
    async def _get_document_store_data(
        self,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get data from document store"""
        try:
            if not self.knowledge_integrator.document_store.client:
                logger.warning("Document store not connected")
                return []
            
            document_data = []
            
            # Search all documents
            result = self.knowledge_integrator.document_store.client.search(
                index=self.knowledge_integrator.document_store.index_name,
                body={
                    'query': {'match_all': {}},
                    'size': batch_size,
                    '_source': ['title', 'content', 'summary', 'topics', 'quality_score', 'source_url', 'timestamp']
                }
            )
            
            for hit in result['hits']['hits']:
                data = {
                    'id': hit['_id'],
                    'title': hit['_source'].get('title', ''),
                    'content': hit['_source'].get('content', ''),
                    'summary': hit['_source'].get('summary', ''),
                    'topics': hit['_source'].get('topics', []),
                    'quality_score': hit['_source'].get('quality_score', 0),
                    'source_url': hit['_source'].get('source_url', ''),
                    'timestamp': hit['_source'].get('timestamp'),
                    'source': 'document_store',
                    'export_timestamp': datetime.utcnow().isoformat()
                }
                
                # Apply filters
                if self._passes_filters(data, filters):
                    document_data.append(data)
            
            logger.info(f"Exported {len(document_data)} items from document store")
            return document_data
            
        except Exception as e:
            logger.error(f"Error getting document store data: {e}")
            return []
    
    def _passes_filters(self, data: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Check if data passes filters"""
        if not filters:
            return True
        
        # Quality score filter
        if 'min_quality_score' in filters:
            if data.get('quality_score', 0) < filters['min_quality_score']:
                return False
        
        # Source filter
        if 'sources' in filters:
            if data.get('source') not in filters['sources']:
                return False
        
        # Date range filter
        if 'date_range' in filters:
            date_range = filters['date_range']
            if 'start' in date_range and data.get('timestamp'):
                if data['timestamp'] < date_range['start']:
                    return False
            if 'end' in date_range and data.get('timestamp'):
                if data['timestamp'] > date_range['end']:
                    return False
        
        # Topics filter
        if 'topics' in filters:
            required_topics = set(filters['topics'])
            data_topics = set(data.get('topics', []))
            if not required_topics.issubset(data_topics):
                return False
        
        return True
    
    async def _format_data(
        self,
        data: List[Dict[str, Any]],
        format: str,
        include_metadata: bool = True
    ) -> Union[str, bytes]:
        """Format data for export"""
        if format == 'json':
            export_data = {
                'metadata': {
                    'export_version': '1.0',
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'total_items': len(data),
                    'format': 'json',
                    'includes_metadata': include_metadata
                } if include_metadata else None,
                'data': data
            }
            return json.dumps(export_data, indent=2, default=str)
        
        elif format == 'pickle':
            return pickle.dumps(data)
        
        elif format == 'csv':
            # Convert to CSV format
            import csv
            import io
            
            if not data:
                return ""
            
            output = io.StringIO()
            
            # Get all possible fields
            all_fields = set()
            for item in data:
                all_fields.update(item.keys())
            
            fieldnames = sorted(all_fields)
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in data:
                # Convert complex objects to strings
                row = {}
                for field in fieldnames:
                    value = item.get(field, '')
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    row[field] = value
                
                writer.writerow(row)
            
            return output.getvalue()
        
        elif format == 'xml':
            # Convert to XML format
            import xml.etree.ElementTree as ET
            
            root = ET.Element('knowledge_export')
            
            if include_metadata:
                metadata = ET.SubElement(root, 'metadata')
                ET.SubElement(metadata, 'export_version').text = '1.0'
                ET.SubElement(metadata, 'export_timestamp').text = datetime.utcnow().isoformat()
                ET.SubElement(metadata, 'total_items').text = str(len(data))
                ET.SubElement(metadata, 'format').text = 'xml'
            
            data_element = ET.SubElement(root, 'data')
            
            for item in data:
                item_element = ET.SubElement(data_element, 'item')
                for key, value in item.items():
                    child = ET.SubElement(item_element, key)
                    child.text = str(value)
            
            return ET.tostring(root, encoding='unicode')
        
        elif format == 'yaml':
            # Convert to YAML format
            import yaml
            
            export_data = {
                'metadata': {
                    'export_version': '1.0',
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'total_items': len(data),
                    'format': 'yaml',
                    'includes_metadata': include_metadata
                } if include_metadata else None,
                'data': data
            }
            
            return yaml.dump(export_data, default_flow_style=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _apply_compression(
        self,
        data: Union[str, bytes],
        output_path: str,
        format: str,
        compression: str
    ) -> Path:
        """Apply compression to exported data"""
        output_path = Path(output_path)
        
        if compression == 'none':
            # No compression
            if isinstance(data, str):
                output_path.write_text(data, encoding='utf-8')
            else:
                output_path.write_bytes(data)
            
            return output_path
        
        elif compression == 'gzip':
            # Gzip compression
            gzip_path = output_path.with_suffix(f'.{format}.gz')
            
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            with gzip.open(gzip_path, 'wb') as f:
                f.write(data_bytes)
            
            return gzip_path
        
        elif compression == 'zip':
            # Zip compression
            zip_path = output_path.with_suffix('.zip')
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                if isinstance(data, str):
                    zf.writestr(f'knowledge_export.{format}', data.encode('utf-8'))
                else:
                    zf.writestr(f'knowledge_export.{format}', data)
            
            return zip_path
        
        else:
            raise ValueError(f"Unsupported compression: {compression}")


class KnowledgeImporter:
    """Knowledge base import system"""
    
    def __init__(self, knowledge_integrator: KnowledgeIntegrator):
        """Initialize knowledge importer"""
        self.knowledge_integrator = knowledge_integrator
        self.supported_formats = ['json', 'pickle', 'csv', 'xml', 'yaml']
        self.compression_formats = ['none', 'gzip', 'zip']
        
        logger.info("Knowledge importer initialized")
    
    async def import_knowledge(
        self,
        input_path: str,
        format: str = 'auto',
        compression: str = 'auto',
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000,
        overwrite: bool = False,
        validate_data: bool = True
    ) -> Dict[str, Any]:
        """
        Import knowledge from file
        
        Args:
            input_path: Input file path
            format: Import format (auto, json, pickle, csv, xml, yaml)
            compression: Compression format (auto, none, gzip, zip)
            filters: Optional filters for import
            batch_size: Batch size for processing
            overwrite: Overwrite existing data
            validate_data: Validate data before import
            
        Returns:
            Import result information
        """
        try:
            # Detect format and compression if auto
            if format == 'auto':
                format = self._detect_format(input_path)
            
            if compression == 'auto':
                compression = self._detect_compression(input_path)
            
            # Validate parameters
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            if compression not in self.compression_formats:
                raise ValueError(f"Unsupported compression: {compression}")
            
            # Decompress if needed
            decompressed_data = await self._decompress_file(input_path, compression)
            
            # Parse data
            parsed_data = await self._parse_data(decompressed_data, format)
            
            # Validate data
            if validate_data:
                validation_result = await self._validate_data(parsed_data)
                if not validation_result['valid']:
                    raise ValueError(f"Data validation failed: {validation_result['errors']}")
            
            # Apply filters
            filtered_data = self._apply_import_filters(parsed_data, filters)
            
            # Import data
            import_result = await self._import_data(filtered_data, batch_size, overwrite)
            
            # Generate import summary
            summary = {
                'success': True,
                'input_path': input_path,
                'format': format,
                'compression': compression,
                'items_imported': import_result['imported_count'],
                'items_skipped': import_result['skipped_count'],
                'items_failed': import_result['failed_count'],
                'validation_passed': validate_data,
                'import_timestamp': datetime.utcnow().isoformat(),
                'filters': filters or {}
            }
            
            logger.info(f"Knowledge imported successfully: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error importing knowledge: {e}")
            return {
                'success': False,
                'error': str(e),
                'import_timestamp': datetime.utcnow().isoformat()
            }
    
    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        format_map = {
            '.json': 'json',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.csv': 'csv',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        
        return format_map.get(extension, 'json')
    
    def _detect_compression(self, file_path: str) -> str:
        """Detect compression from extension"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.gz':
            return 'gzip'
        elif path.suffix.lower() == '.zip':
            return 'zip'
        else:
            return 'none'
    
    async def _decompress_file(self, file_path: str, compression: str) -> Union[str, bytes]:
        """Decompress file if needed"""
        if compression == 'none':
            # No compression
            path = Path(file_path)
            if path.suffix in ['.json', '.xml', '.yaml', '.yml', '.csv']:
                return path.read_text(encoding='utf-8')
            else:
                return path.read_bytes()
        
        elif compression == 'gzip':
            # Gzip decompression
            with gzip.open(file_path, 'rb') as f:
                data = f.read()
            
            # Try to decode as text first
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                return data
        
        elif compression == 'zip':
            # Zip decompression
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Get the first file in the zip
                file_list = zf.namelist()
                if not file_list:
                    raise ValueError("Zip file is empty")
                
                # Extract the first file
                with zf.open(file_list[0]) as f:
                    data = f.read()
                
                # Try to decode as text first
                try:
                    return data.decode('utf-8')
                except UnicodeDecodeError:
                    return data
        
        else:
            raise ValueError(f"Unsupported compression: {compression}")
    
    async def _parse_data(self, data: Union[str, bytes], format: str) -> List[Dict[str, Any]]:
        """Parse data from file"""
        if format == 'json':
            parsed = json.loads(data) if isinstance(data, str) else json.loads(data.decode('utf-8'))
            
            # Handle different JSON structures
            if isinstance(parsed, dict) and 'data' in parsed:
                return parsed['data']
            elif isinstance(parsed, list):
                return parsed
            else:
                raise ValueError("Invalid JSON structure for import")
        
        elif format == 'pickle':
            if isinstance(data, bytes):
                return pickle.loads(data)
            else:
                return pickle.loads(data.encode('utf-8'))
        
        elif format == 'csv':
            import csv
            import io
            
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            reader = csv.DictReader(io.StringIO(data))
            return list(reader)
        
        elif format == 'xml':
            import xml.etree.ElementTree as ET
            
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            root = ET.fromstring(data)
            
            # Find data element
            data_element = root.find('data')
            if data_element is None:
                raise ValueError("No data element found in XML")
            
            parsed_data = []
            for item_element in data_element.findall('item'):
                item_data = {}
                for child in item_element:
                    item_data[child.tag] = child.text
                parsed_data.append(item_data)
            
            return parsed_data
        
        elif format == 'yaml':
            import yaml
            
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            parsed = yaml.safe_load(data)
            
            # Handle different YAML structures
            if isinstance(parsed, dict) and 'data' in parsed:
                return parsed['data']
            elif isinstance(parsed, list):
                return parsed
            else:
                raise ValueError("Invalid YAML structure for import")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _validate_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate imported data"""
        errors = []
        warnings = []
        
        if not isinstance(data, list):
            errors.append("Data must be a list")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        for i, item in enumerate(data[:10]):  # Validate first 10 items
            if not isinstance(item, dict):
                errors.append(f"Item {i} is not a dictionary")
                continue
            
            # Check required fields
            required_fields = ['id', 'content']
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item {i} missing required field: {field}")
            
            # Check data types
            if 'quality_score' in item:
                try:
                    float(item['quality_score'])
                except (ValueError, TypeError):
                    errors.append(f"Item {i} has invalid quality_score")
            
            if 'timestamp' in item:
                # Try to parse timestamp
                try:
                    datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    warnings.append(f"Item {i} has invalid timestamp format")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_items': len(data)
        }
    
    def _apply_import_filters(
        self,
        data: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply filters to imported data"""
        if not filters:
            return data
        
        filtered_data = []
        
        for item in data:
            # Quality score filter
            if 'min_quality_score' in filters:
                if item.get('quality_score', 0) < filters['min_quality_score']:
                    continue
            
            # Source filter
            if 'sources' in filters:
                if item.get('source') not in filters['sources']:
                    continue
            
            # Date range filter
            if 'date_range' in filters:
                date_range = filters['date_range']
                if 'start' in date_range and item.get('timestamp'):
                    try:
                        item_time = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                        if item_time < date_range['start']:
                            continue
                    except:
                        continue
                
                if 'end' in date_range and item.get('timestamp'):
                    try:
                        item_time = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                        if item_time > date_range['end']:
                            continue
                    except:
                        continue
            
            # Topics filter
            if 'topics' in filters:
                required_topics = set(filters['topics'])
                item_topics = set(item.get('topics', []))
                if not required_topics.intersection(item_topics):
                    continue
            
            filtered_data.append(item)
        
        return filtered_data
    
    async def _import_data(
        self,
        data: List[Dict[str, Any]],
        batch_size: int = 1000,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Import data into knowledge stores"""
        imported_count = 0
        skipped_count = 0
        failed_count = 0
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            try:
                # Convert to ProcessedContent objects
                processed_contents = []
                for item in batch:
                    try:
                        processed_content = await self._convert_to_processed_content(item)
                        if processed_content:
                            processed_contents.append(processed_content)
                    except Exception as e:
                        logger.error(f"Error converting item {item.get('id', 'unknown')}: {e}")
                        failed_count += 1
                
                # Import batch
                if processed_contents:
                    if overwrite:
                        # Remove existing items
                        await self._remove_existing_items(processed_contents)
                    
                    await self.knowledge_integrator.batch_integrate(processed_contents)
                    imported_count += len(processed_contents)
                else:
                    skipped_count += len(batch)
                
            except Exception as e:
                logger.error(f"Error importing batch {i//batch_size}: {e}")
                failed_count += len(batch)
        
        return {
            'imported_count': imported_count,
            'skipped_count': skipped_count,
            'failed_count': failed_count
        }
    
    async def _convert_to_processed_content(self, item: Dict[str, Any]) -> Optional[ProcessedContent]:
        """Convert imported item to ProcessedContent"""
        try:
            # Extract required fields
            content_id = item.get('id')
            content = item.get('content', '')
            title = item.get('title', '')
            
            if not content_id or not content:
                return None
            
            # Extract optional fields
            summary = item.get('summary', '')
            topics = item.get('topics', [])
            entities = item.get('entities', [])
            quality_score = item.get('quality_score', 0.5)
            relevance_score = item.get('relevance_score', 0.5)
            language = item.get('language', 'en')
            source_url = item.get('source_url', '')
            source_type = item.get('source', 'imported')
            
            # Parse timestamp
            timestamp = datetime.utcnow()
            if 'timestamp' in item:
                try:
                    timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                except:
                    pass
            
            # Create metadata
            metadata = {
                'source_url': source_url,
                'source_type': source_type,
                'import_timestamp': datetime.utcnow().isoformat(),
                'original_id': content_id,
                'content_length': len(content)
            }
            
            # Create ProcessedContent
            processed_content = ProcessedContent(
                id=content_id,
                title=title,
                content=content,
                summary=summary,
                topics=topics,
                entities=entities,
                quality_score=quality_score,
                relevance_score=relevance_score,
                language=language,
                metadata=metadata,
                timestamp=timestamp
            )
            
            return processed_content
            
        except Exception as e:
            logger.error(f"Error converting item to ProcessedContent: {e}")
            return None
    
    async def _remove_existing_items(self, processed_contents: List[ProcessedContent]):
        """Remove existing items before overwrite"""
        try:
            # Get IDs to remove
            ids_to_remove = [content.id for content in processed_contents]
            
            # Remove from vector store
            if ids_to_remove:
                self.knowledge_integrator.vector_store.collection.delete(ids=ids_to_remove)
            
            # Remove from graph store (if implemented)
            # This would need to be implemented based on your graph store
            
            # Remove from document store
            if ids_to_remove:
                for content_id in ids_to_remove:
                    try:
                        self.knowledge_integrator.document_store.client.delete(
                            index=self.knowledge_integrator.document_store.index_name,
                            id=content_id
                        )
                    except:
                        pass  # Ignore if item doesn't exist
            
            logger.info(f"Removed {len(ids_to_remove)} existing items for overwrite")
            
        except Exception as e:
            logger.error(f"Error removing existing items: {e}")


class KnowledgeManager:
    """High-level knowledge export/import manager"""
    
    def __init__(self, knowledge_integrator: KnowledgeIntegrator):
        """Initialize knowledge manager"""
        self.knowledge_integrator = knowledge_integrator
        self.exporter = KnowledgeExporter(knowledge_integrator)
        self.importer = KnowledgeImporter(knowledge_integrator)
        
        logger.info("Knowledge manager initialized")
    
    async def export_knowledge_snapshot(
        self,
        output_dir: str,
        name: str = None,
        formats: List[str] = None,
        compression: str = 'gzip'
    ) -> Dict[str, Any]:
        """Create a complete knowledge snapshot"""
        try:
            name = name or f"knowledge_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            formats = formats or ['json', 'pickle']
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = {}
            
            for format in formats:
                output_path = output_dir / f"{name}.{format}"
                
                result = await self.exporter.export_knowledge(
                    str(output_path),
                    format=format,
                    compression=compression
                )
                
                results[format] = result
            
            # Create snapshot manifest
            manifest = {
                'snapshot_name': name,
                'created_at': datetime.utcnow().isoformat(),
                'formats': formats,
                'compression': compression,
                'results': results,
                'total_items': sum(r.get('items_exported', 0) for r in results.values())
            }
            
            manifest_path = output_dir / f"{name}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            return {
                'success': True,
                'snapshot_name': name,
                'output_dir': str(output_dir),
                'manifest': manifest,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error creating knowledge snapshot: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def import_knowledge_snapshot(
        self,
        input_path: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Import a complete knowledge snapshot"""
        try:
            input_path = Path(input_path)
            
            # Check if it's a directory (snapshot with manifest)
            if input_path.is_dir():
                # Find manifest file
                manifest_files = list(input_path.glob("*_manifest.json"))
                if not manifest_files:
                    raise ValueError("No manifest file found in snapshot directory")
                
                manifest_path = manifest_files[0]
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Import all formats in the snapshot
                results = {}
                for format in manifest['formats']:
                    file_path = input_path / f"{manifest['snapshot_name']}.{format}"
                    if file_path.exists():
                        result = await self.importer.import_knowledge(
                            str(file_path),
                            format=format,
                            compression=manifest['compression'],
                            overwrite=overwrite
                        )
                        results[format] = result
                
                return {
                    'success': True,
                    'snapshot_name': manifest['snapshot_name'],
                    'results': results
                }
            
            else:
                # Single file import
                result = await self.importer.import_knowledge(
                    str(input_path),
                    overwrite=overwrite
                )
                
                return {
                    'success': result['success'],
                    'result': result
                }
            
        except Exception as e:
            logger.error(f"Error importing knowledge snapshot: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def transfer_knowledge(
        self,
        source_integrator: KnowledgeIntegrator,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Transfer knowledge from another integrator"""
        try:
            # Create temporary exporter and importer
            temp_exporter = KnowledgeExporter(source_integrator)
            
            # Export to temporary file
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                export_result = await temp_exporter.export_knowledge(
                    temp_file.name,
                    format='json',
                    compression='none',
                    filters=filters
                )
                
                if not export_result['success']:
                    return export_result
                
                # Import from temporary file
                import_result = await self.importer.import_knowledge(
                    temp_file.name,
                    format='json',
                    compression='none',
                    batch_size=batch_size
                )
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
                return {
                    'success': import_result['success'],
                    'exported_items': export_result['items_exported'],
                    'imported_items': import_result.get('items_imported', 0),
                    'failed_items': import_result.get('items_failed', 0)
                }
            
        except Exception as e:
            logger.error(f"Error transferring knowledge: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Example usage
async def main():
    """Example usage of knowledge export/import"""
    from knowledge.knowledge_integrator import KnowledgeIntegrator
    
    # Initialize integrator
    integrator = KnowledgeIntegrator()
    await integrator.batch_integrate([])  # Initialize
    
    # Create knowledge manager
    manager = KnowledgeManager(integrator)
    
    # Export knowledge snapshot
    export_result = await manager.export_knowledge_snapshot(
        output_dir="./exports",
        name="test_snapshot",
        formats=['json', 'pickle'],
        compression='gzip'
    )
    
    print(f"Export result: {export_result}")
    
    # Import knowledge snapshot
    import_result = await manager.import_knowledge_snapshot(
        input_path="./exports/test_snapshot_manifest.json",
        overwrite=False
    )
    
    print(f"Import result: {import_result}")


if __name__ == "__main__":
    asyncio.run(main())
