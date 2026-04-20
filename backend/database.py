"""
Database - PostgreSQL async operations for storing articles and briefs
Uses asyncpg for high-performance async database access
"""

import asyncpg
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone

load_dotenv()


class Database:
    """Async PostgreSQL database interface"""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize database connection
        
        Args:
            connection_string: PostgreSQL connection string
                             Format: postgresql://user:pass@host:port/dbname
        """
        self.connection_string = connection_string or self._get_default_connection()
        self.pool = None
    
    def _get_default_connection(self) -> str:
        """Get default connection string from environment or use default"""
        import os
        return os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/technews"
        )
    
    async def connect(self):
        """Create connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10
            )
            print("✅ Database connected")
            
            # Initialize schema
            await self._init_schema()
            
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            raise
    
    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            print("✅ Database disconnected")
    
    async def _init_schema(self):
        """Initialize database schema"""
        
        schema = """
        -- Articles table
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            article_id VARCHAR(32) UNIQUE NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            canonical_url TEXT NOT NULL,
            source VARCHAR(255) NOT NULL,
            published_date TIMESTAMPTZ NOT NULL,
            summary TEXT,
            content TEXT,
            word_count INTEGER,
            reading_time_minutes INTEGER,
            is_syndicated BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_published_date ON articles(published_date);
        CREATE INDEX IF NOT EXISTS idx_source ON articles(source);
        CREATE INDEX IF NOT EXISTS idx_article_id ON articles(article_id);
        
        -- News briefs table
        CREATE TABLE IF NOT EXISTS briefs (
            id SERIAL PRIMARY KEY,
            cluster_id VARCHAR(64) UNIQUE NOT NULL,
            title TEXT NOT NULL,
            key_points JSONB NOT NULL,
            why_it_matters TEXT NOT NULL,
            sources JSONB NOT NULL,
            background_context TEXT,
            technical_glossary JSONB,
            examples TEXT,
            categories JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            articles_count INTEGER NOT NULL,
            article_ids JSONB NOT NULL
        );
        
        CREATE INDEX IF NOT EXISTS idx_briefs_created_at ON briefs(created_at);
        CREATE INDEX IF NOT EXISTS idx_briefs_categories ON briefs USING GIN(categories);
        
        -- Categories table (for fast lookups)
        CREATE TABLE IF NOT EXISTS categories (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE NOT NULL,
            brief_count INTEGER DEFAULT 0,
            last_updated TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Collection metadata table (tracks collection runs)
        CREATE TABLE IF NOT EXISTS collection_metadata (
            id SERIAL PRIMARY KEY,
            collection_timestamp TIMESTAMPTZ DEFAULT NOW(),
            articles_fetched INTEGER DEFAULT 0,
            articles_stored INTEGER DEFAULT 0,
            clusters_created INTEGER DEFAULT 0,
            briefs_generated INTEGER DEFAULT 0,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_collection_timestamp ON collection_metadata(collection_timestamp);
        
        -- Full-text search indexes
        CREATE INDEX IF NOT EXISTS idx_articles_search 
            ON articles USING GIN(to_tsvector('english', title || ' ' || COALESCE(content, '')));
        
        CREATE INDEX IF NOT EXISTS idx_briefs_search 
            ON briefs USING GIN(to_tsvector('english', title || ' ' || why_it_matters));
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(schema)
        
        print("✅ Database schema initialized")
    
    # Articles Operations
    
    async def store_articles(self, articles: List[Dict[str, Any]]) -> List[int]:
        """Store cleaned articles in database"""
        
        stored_ids = []
        
        query = """
        INSERT INTO articles (
            article_id, title, url, canonical_url, source,
            published_date, summary, content, word_count,
            reading_time_minutes, is_syndicated
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (article_id) DO NOTHING
        RETURNING id
        """
        
        async with self.pool.acquire() as conn:
            for article in articles:
                try:
                    result = await conn.fetchrow(
                        query,
                        article["article_id"],
                        article["title"],
                        article["url"],
                        article["canonical_url"],
                        article["source"],
                        article["published_date"],
                        article.get("summary"),
                        article.get("content"),
                        article.get("word_count"),
                        article.get("reading_time_minutes"),
                        article.get("is_syndicated", False)
                    )
                    
                    if result:
                        stored_ids.append(result["id"])
                        
                except Exception as e:
                    print(f"⚠️  Error storing article: {e}")
                    continue
        
        return stored_ids
    
    async def get_articles_count(self, start_date: datetime, end_date: datetime) -> int:
        """Get count of articles in date range"""
        
        query = """
        SELECT COUNT(*) as count
        FROM articles
        WHERE published_date >= $1 AND published_date <= $2
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(query, start_date, end_date)
            return result["count"]
    
    # Briefs Operations
    
    async def store_briefs(self, briefs: List[Dict[str, Any]]) -> List[int]:
        """Store news briefs in database"""
        
        stored_ids = []
        
        query = """
        INSERT INTO briefs (
            cluster_id, title, key_points, why_it_matters,
            sources, background_context, technical_glossary,
            examples, categories, articles_count, article_ids
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (cluster_id) DO UPDATE SET
            title = EXCLUDED.title,
            key_points = EXCLUDED.key_points,
            why_it_matters = EXCLUDED.why_it_matters,
            sources = EXCLUDED.sources,
            background_context = EXCLUDED.background_context,
            technical_glossary = EXCLUDED.technical_glossary,
            examples = EXCLUDED.examples,
            categories = EXCLUDED.categories,
            articles_count = EXCLUDED.articles_count,
            article_ids = EXCLUDED.article_ids
        RETURNING id
        """
        
        async with self.pool.acquire() as conn:
            for brief in briefs:
                try:
                    result = await conn.fetchrow(
                        query,
                        brief["cluster_id"],
                        brief["title"],
                        json.dumps(brief["key_points"]),
                        brief["why_it_matters"],
                        json.dumps(brief["sources"]),
                        brief.get("background_context"),
                        json.dumps(brief.get("technical_glossary")) if brief.get("technical_glossary") else None,
                        brief.get("examples"),
                        json.dumps(brief["categories"]),
                        brief["articles_count"],
                        json.dumps(brief["article_ids"])
                    )
                    
                    if result:
                        stored_ids.append(result["id"])
                        
                        # Update categories
                        await self._update_categories(conn, brief["categories"])
                        
                except Exception as e:
                    print(f"⚠️  Error storing brief: {e}")
                    continue
        
        return stored_ids
    
    async def get_briefs(self, start_date: datetime, end_date: datetime, 
                        category_filter: Optional[str] = None) -> List[Any]:
        """Get briefs in date range, optionally filtered by category"""
        
        if category_filter:
            query = """
            SELECT * FROM briefs
            WHERE created_at >= $1 AND created_at <= $2
            AND categories @> $3
            ORDER BY created_at DESC
            """
            filter_json = json.dumps([category_filter])
        else:
            query = """
            SELECT * FROM briefs
            WHERE created_at >= $1 AND created_at <= $2
            ORDER BY created_at DESC
            """
        
        async with self.pool.acquire() as conn:
            if category_filter:
                rows = await conn.fetch(query, start_date, end_date, filter_json)
            else:
                rows = await conn.fetch(query, start_date, end_date)
            
            return [self._row_to_brief(row) for row in rows]
    
    async def adaptive_get_briefs(
        self, 
        category_filter: Optional[str] = None,
        min_briefs: int = 10
    ) -> Tuple[List[Any], int]:
        """
        Adaptively fetch briefs with expanding time window until minimum count reached
        
        Strategy:
        1. Try last 24 hours
        2. If < min_briefs, expand to 48 hours
        3. If still < min_briefs, expand to 7 days
        4. If still < min_briefs, expand to 30 days
        
        Args:
            category_filter: Optional category to filter by
            min_briefs: Minimum number of briefs to return (default 10)
            
        Returns:
            Tuple of (briefs_list, days_used)
        """
        end_date = datetime.now(timezone.utc)        

        # Time windows to try (in days)
        time_windows = [1, 2, 7, 30]
        
        for days in time_windows:
            start_date = end_date - timedelta(days=days)
            briefs = await self.get_briefs(start_date, end_date, category_filter)
            
            if len(briefs) >= min_briefs:
                print(f"📊 Found {len(briefs)} briefs in last {days} day(s)")
                return briefs, days
        
        # If we still don't have enough after 30 days, return what we have
        print(f"📊 Found {len(briefs)} briefs in last 30 days (less than minimum {min_briefs})")
        return briefs, 30
    
    async def get_brief_by_id(self, brief_id: int) -> Optional[Any]:
        """Get brief by ID"""
        
        query = "SELECT * FROM briefs WHERE id = $1"
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, brief_id)
            
            if row:
                return self._row_to_brief(row)
            return None
    
    async def search_briefs(self, search_query: str, start_date: datetime, 
                           end_date: datetime) -> List[Any]:
        """Full-text search on briefs"""
        
        query = """
        SELECT *, ts_rank(
            to_tsvector('english', title || ' ' || why_it_matters),
            plainto_tsquery('english', $1)
        ) AS rank
        FROM briefs
        WHERE created_at >= $2 AND created_at <= $3
        AND to_tsvector('english', title || ' ' || why_it_matters) @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC, created_at DESC
        LIMIT 50
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, search_query, start_date, end_date)
            return [self._row_to_brief(row) for row in rows]
    
    async def get_category_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all categories"""
        
        query = """
        SELECT name, brief_count, last_updated
        FROM categories
        ORDER BY brief_count DESC
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            
            return [
                {
                    "name": row["name"],
                    "count": row["brief_count"],
                    "last_updated": row["last_updated"]
                }
                for row in rows
            ]
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        
        async with self.pool.acquire() as conn:
            articles_count = await conn.fetchval("SELECT COUNT(*) FROM articles")
            briefs_count = await conn.fetchval("SELECT COUNT(*) FROM briefs")
            categories_count = await conn.fetchval("SELECT COUNT(*) FROM categories")
            
            latest_article = await conn.fetchrow(
                "SELECT published_date FROM articles ORDER BY published_date DESC LIMIT 1"
            )
            
            latest_brief = await conn.fetchrow(
                "SELECT created_at FROM briefs ORDER BY created_at DESC LIMIT 1"
            )
            
            return {
                "total_articles": articles_count,
                "total_briefs": briefs_count,
                "total_categories": categories_count,
                "latest_article_date": latest_article["published_date"] if latest_article else None,
                "latest_brief_date": latest_brief["created_at"] if latest_brief else None
            }
    
    # Collection Metadata Operations
    
    async def store_collection_metadata(
        self,
        articles_fetched: int,
        articles_stored: int,
        clusters_created: int,
        briefs_generated: int,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Store metadata about a collection run
        
        Args:
            articles_fetched: Number of raw articles fetched
            articles_stored: Number of articles stored in DB
            clusters_created: Number of clusters created
            briefs_generated: Number of briefs generated
            success: Whether the collection succeeded
            error_message: Error message if failed
        """
        query = """
        INSERT INTO collection_metadata (
            collection_timestamp,
            articles_fetched,
            articles_stored,
            clusters_created,
            briefs_generated,
            success,
            error_message
        ) VALUES (NOW(), $1, $2, $3, $4, $5, $6)
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                articles_fetched,
                articles_stored,
                clusters_created,
                briefs_generated,
                success,
                error_message
            )
        
        print(f"✅ Collection metadata stored: {articles_fetched} fetched, {briefs_generated} briefs")
    
    async def get_last_collection_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last collection run
        
        Returns:
            Dictionary with collection metadata or None if no collections yet
        """
        query = """
        SELECT 
            collection_timestamp,
            articles_fetched,
            articles_stored,
            clusters_created,
            briefs_generated,
            success,
            error_message
        FROM collection_metadata
        ORDER BY collection_timestamp DESC
        LIMIT 1
        """
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query)
            
            if row:
                return {
                    "timestamp": row["collection_timestamp"],
                    "articles_fetched": row["articles_fetched"],
                    "articles_stored": row["articles_stored"],
                    "clusters_created": row["clusters_created"],
                    "briefs_generated": row["briefs_generated"],
                    "success": row["success"],
                    "error_message": row["error_message"],
                    "relative_time": self.calculate_relative_time(row["collection_timestamp"])
                }
            
            return None
    
    # Helper methods
    
    @staticmethod
    def calculate_relative_time(timestamp: datetime) -> str:
        """
        Convert datetime to relative time string
        
        Args:
            timestamp: Datetime to convert
            
        Returns:
            Relative time string like "2h ago", "Yesterday", "3d ago"
        """
        now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
        delta = now - timestamp
        
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return "Just now"
        elif seconds < 3600:  # Less than 1 hour
            minutes = int(seconds / 60)
            return f"{minutes}m ago" if minutes > 1 else "1m ago"
        elif seconds < 7200:  # Less than 2 hours
            return "1h ago"
        elif seconds < 86400:  # Less than 24 hours
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        elif seconds < 172800:  # Less than 48 hours
            return "Yesterday"
        elif seconds < 604800:  # Less than 7 days
            days = int(seconds / 86400)
            return f"{days}d ago"
        elif seconds < 2592000:  # Less than 30 days
            weeks = int(seconds / 604800)
            return f"{weeks}w ago" if weeks > 1 else "1w ago"
        else:
            months = int(seconds / 2592000)
            return f"{months}mo ago" if months > 1 else "1mo ago"
    
    async def _update_categories(self, conn, categories: List[str]):
        """Update category counts"""
        
        for category in categories:
            query = """
            INSERT INTO categories (name, brief_count, last_updated)
            VALUES ($1, 1, NOW())
            ON CONFLICT (name) DO UPDATE SET
                brief_count = categories.brief_count + 1,
                last_updated = NOW()
            """
            
            await conn.execute(query, category)
    
    def _row_to_brief(self, row) -> Any:
        """Convert database row to brief object"""
        from main import NewsBrief
        
        return NewsBrief(
            id=row["id"],
            cluster_id=row["cluster_id"],
            title=row["title"],
            key_points=json.loads(row["key_points"]),
            why_it_matters=row["why_it_matters"],
            sources=json.loads(row["sources"]),
            background_context=row["background_context"],
            technical_glossary=json.loads(row["technical_glossary"]) if row["technical_glossary"] else None,
            examples=row["examples"],
            categories=json.loads(row["categories"]),
            created_at=row["created_at"],
            articles_count=row["articles_count"]
        )