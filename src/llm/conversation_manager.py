"""
Conversation Manager - Production-Grade Implementation
Manages conversation history with database storage and context retrieval.
"""

import asyncio
import logging
import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import uuid

from src.llm.deepseek_client import Message
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """Represents a conversation session"""
    id: str
    title: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """
    Production-grade conversation manager with SQLite database storage.
    Handles conversation persistence, retrieval, and context management.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize conversation manager"""
        settings = get_settings()
        
        if db_path is None:
            db_path = str(settings.data_dir / "conversations.db")
        
        self.db_path = db_path
        self.max_context_messages = 10  # Last N messages for context
        self.max_conversation_age_days = 30
        
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Conversation manager initialized with database at {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages (conversation_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_updated 
            ON conversations (updated_at DESC)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Database schema initialized")
    
    async def create_conversation(
        self,
        title: str = "New Conversation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            metadata: Optional metadata
            
        Returns:
            Created Conversation object
        """
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    now.isoformat(),
                    now.isoformat(),
                    json.dumps(metadata or {})
                )
            )
            
            conn.commit()
            
            conversation = Conversation(
                id=conversation_id,
                title=title,
                created_at=now,
                updated_at=now,
                metadata=metadata or {}
            )
            
            logger.info(f"Created conversation {conversation_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message object
        """
        message_id = str(uuid.uuid4())
        now = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Add message
            cursor.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    now.isoformat(),
                    json.dumps(metadata or {})
                )
            )
            
            # Update conversation timestamp
            cursor.execute(
                """
                UPDATE conversations 
                SET updated_at = ?
                WHERE id = ?
                """,
                (now.isoformat(), conversation_id)
            )
            
            conn.commit()
            
            message = Message(
                role=role,
                content=content,
                timestamp=now,
                metadata=metadata or {}
            )
            
            logger.debug(f"Added message {message_id} to conversation {conversation_id}")
            return message
            
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def get_conversation(
        self,
        conversation_id: str
    ) -> Optional[Conversation]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation object or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get conversation metadata
            cursor.execute(
                """
                SELECT id, title, created_at, updated_at, metadata
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get messages
            cursor.execute(
                """
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                """,
                (conversation_id,)
            )
            
            messages = []
            for msg_row in cursor.fetchall():
                messages.append(Message(
                    role=msg_row[0],
                    content=msg_row[1],
                    timestamp=datetime.fromisoformat(msg_row[2]),
                    metadata=json.loads(msg_row[3]) if msg_row[3] else {}
                ))
            
            conversation = Conversation(
                id=row[0],
                title=row[1],
                messages=messages,
                created_at=datetime.fromisoformat(row[2]),
                updated_at=datetime.fromisoformat(row[3]),
                metadata=json.loads(row[4]) if row[4] else {}
            )
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error retrieving conversation: {e}")
            return None
        finally:
            conn.close()
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None
    ) -> List[Message]:
        """
        Get context messages for a conversation (last N messages).
        
        Args:
            conversation_id: Conversation ID
            max_messages: Maximum number of messages to retrieve
            
        Returns:
            List of Message objects
        """
        limit = max_messages or self.max_context_messages
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (conversation_id, limit)
            )
            
            messages = []
            for row in cursor.fetchall():
                messages.append(Message(
                    role=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    metadata=json.loads(row[3]) if row[3] else {}
                ))
            
            # Reverse to get chronological order
            messages.reverse()
            
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}")
            return []
        finally:
            conn.close()
    
    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """
        List conversations ordered by update time.
        
        Args:
            limit: Maximum number of conversations to return
            offset: Offset for pagination
            
        Returns:
            List of Conversation objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT id, title, created_at, updated_at, metadata
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    id=row[0],
                    title=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4]) if row[4] else {}
                ))
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
        finally:
            conn.close()
    
    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str
    ) -> bool:
        """
        Update conversation title.
        
        Args:
            conversation_id: Conversation ID
            title: New title
            
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (title, datetime.now().isoformat(), conversation_id)
            )
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error updating conversation title: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Messages will be deleted automatically due to CASCADE
            cursor.execute(
                """
                DELETE FROM conversations
                WHERE id = ?
                """,
                (conversation_id,)
            )
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    async def cleanup_old_conversations(self) -> int:
        """
        Delete conversations older than max age.
        
        Returns:
            Number of conversations deleted
        """
        cutoff_date = datetime.now() - timedelta(days=self.max_conversation_age_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                DELETE FROM conversations
                WHERE created_at < ?
                """,
                (cutoff_date.isoformat(),)
            )
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old conversations")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Count conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            # Count messages
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            
            # Average messages per conversation
            avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
            
            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "average_messages_per_conversation": round(avg_messages, 2),
                "max_context_messages": self.max_context_messages,
                "database_path": self.db_path
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
        finally:
            conn.close()
