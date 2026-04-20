"""
News Processor - Cleans, normalizes and validates news articles
Handles deduplication, date conversion, and content quality filtering
"""

import re
from typing import List, Dict, Any, Set
from datetime import datetime, timezone
import hashlib
from dateutil import parser as date_parser
import pytz


class NewsProcessor:
    """Processes and normalizes news articles"""
    
    # Minimum content length for valid articles
    MIN_CONTENT_LENGTH = 100
    MIN_TITLE_LENGTH = 10
    
    # Syndication indicators
    SYNDICATION_MARKERS = [
        "originally published",
        "first appeared",
        "syndicated from",
        "cross-posted",
        "republished with permission"
    ]
    
    def __init__(self):
        self.seen_urls = set()
        self.seen_hashes = set()
    
    def clean_and_normalize(self, raw_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.seen_urls = set()   # ✅ reset at start of each run
        self.seen_hashes = set()    
        """
        Clean and normalize all articles
        
        Pipeline:
        1. Validate required fields
        2. Normalize dates to ET
        3. Clean title and content
        4. Deduplicate
        5. Filter thin/syndicated content
        6. Add metadata
        
        Args:
            raw_articles: List of raw article dictionaries
            
        Returns:
            List of cleaned and normalized articles
        """
        cleaned = []
        
        for article in raw_articles:
            try:
                # Step 1: Validate
                if not self._validate_article(article):
                    continue
                
                # Step 2: Normalize
                normalized = self._normalize_article(article)
                
                # Step 3: Deduplicate
                if self._is_duplicate(normalized):
                    continue
                
                # Step 4: Check content quality
                if not self._check_content_quality(normalized):
                    continue
                
                # Step 5: Detect syndication
                if self._is_syndicated(normalized):
                    normalized["is_syndicated"] = True
                else:
                    normalized["is_syndicated"] = False
                
                # Step 6: Add metadata
                normalized = self._add_metadata(normalized)
                
                cleaned.append(normalized)
                
            except Exception as e:
                print(f"⚠️  Error processing article: {e}")
                continue
        
        print(f"✅ Processed {len(cleaned)}/{len(raw_articles)} articles (removed {len(raw_articles) - len(cleaned)} duplicates/invalid)")
        return cleaned
    
    def _validate_article(self, article: Dict[str, Any]) -> bool:
        """Check if article has required fields"""
        required_fields = ["title", "url", "source"]
        
        for field in required_fields:
            if field not in article or not article[field]:
                return False
        
        # Check minimum title length
        if len(article["title"]) < self.MIN_TITLE_LENGTH:
            return False
        
        return True
    
    def _normalize_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize article fields
        
        - Clean title (remove extra whitespace, fix encoding)
        - Convert URL to canonical form
        - Convert date to ET timezone
        - Clean summary and content
        """
        normalized = {}
        
        # Title
        normalized["title"] = self._clean_title(article["title"])
        
        # URL (already cleaned by fetcher, but ensure lowercase)
        normalized["url"] = article["url"].strip()
        
        # Canonical URL for deduplication
        normalized["canonical_url"] = self._get_canonical_url(normalized["url"])
        
        # Source
        normalized["source"] = article["source"].strip()
        
        # Date - Convert to ET
        normalized["published_date"] = self._convert_to_et(article.get("published_date"))
        
        # Summary
        normalized["summary"] = self._clean_text(article.get("summary", ""))
        
        # Content
        normalized["content"] = self._clean_text(article.get("content", ""))
        
        # Preserve raw entry for debugging
        normalized["raw_entry"] = article.get("raw_entry")
        
        return normalized
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize title"""
        # Remove extra whitespace
        title = " ".join(title.split())
        
        # Fix common encoding issues
        # Fix common encoding issues
        replacements = {
            "\u00e2\u20ac\u2122": "'",    # Fixes â€™
            "\u00e2\u20ac\u0153": '"',    # Fixes â€œ
            "\u00e2\u20ac\u009d": '"',    # Fixes â€ 
            "\u00e2\u20ac\u2013": "-",    # Fixes â€“ (En dash)
            "\u00e2\u20ac\u2014": "—",    # Fixes â€” (Em dash)
            "&amp;": "&",
            "&quot;": '"',
            "&lt;": "<",
            "&gt;": ">",
        }
        
        for old, new in replacements.items():
            title = title.replace(old, new)
        
        # Remove trailing ellipsis or dash
        title = title.rstrip("…-–—")
        
        return title.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean summary/content text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Read more:",
            "Continue reading:",
            "Full article:",
            "Source:",
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Fix encoding
        replacements = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "&amp;": "&",
            "&quot;": '"',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _get_canonical_url(self, url: str) -> str:
        """Get canonical form of URL for deduplication"""
        # Remove protocol
        canonical = url.lower()
        canonical = canonical.replace("https://", "").replace("http://", "")
        
        # Remove www
        canonical = canonical.replace("www.", "")
        
        # Remove trailing slash
        canonical = canonical.rstrip("/")
        
        # Remove query string (already removed by fetcher, but double check)
        if "?" in canonical:
            canonical = canonical.split("?")[0]
        
        return canonical
    
    def _convert_to_et(self, pub_date: Any) -> datetime:
        """Convert publication date to Eastern Time"""
        if not pub_date:
            return datetime.now(pytz.timezone("America/New_York"))
        
        # If already datetime
        if isinstance(pub_date, datetime):
            dt = pub_date
        else:
            # Try to parse string
            try:
                dt = date_parser.parse(str(pub_date))
            except (ValueError, OverflowError):  # 
                return datetime.now(pytz.timezone("America/New_York"))  
        
        # Ensure timezone aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to ET
        et_tz = pytz.timezone("America/New_York")
        dt_et = dt.astimezone(pytz.timezone("America/New_York"))
        
        return dt_et
    
    def _is_duplicate(self, article: Dict[str, Any]) -> bool:
        """
        Check if article is duplicate
        
        Uses both URL and content hash for detection
        """
        # Check URL
        canonical_url = article["canonical_url"]
        if canonical_url in self.seen_urls:
            return True
        
        # Check content hash
        content_hash = self._compute_content_hash(article)
        if content_hash in self.seen_hashes:
            return True
        
        # Not a duplicate - add to seen sets
        self.seen_urls.add(canonical_url)
        self.seen_hashes.add(content_hash)
        
        return False
    
    def _compute_content_hash(self, article: Dict[str, Any]) -> str:
        """Compute hash of article content for deduplication"""
        # Use title + first 200 chars of content
        content = article["title"].lower()
        if article.get("content"):
            content += article["content"][:200].lower()
        elif article.get("summary"):
            content += article["summary"][:200].lower()
        
        # Remove all whitespace for better matching
        content = re.sub(r'\s+', '', content)
        
        # Compute hash
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_content_quality(self, article: Dict[str, Any]) -> bool:
        """
        Check if article has sufficient content quality
        
        Filters:
        - Thin content (too short)
        - Missing dates
        - Broken/incomplete articles
        """
        # Check for content
        content = article.get("content", "") or article.get("summary", "")
        
        if len(content) < self.MIN_CONTENT_LENGTH:
            return False
        
        # Check for date
        if not article.get("published_date"):
            return False
        
        # Check for common error patterns
        error_patterns = [
            "404 not found",
            "page not found",
            "access denied",
            "subscription required",
        ]
        
        content_lower = content.lower()
        for pattern in error_patterns:
            if pattern in content_lower:
                return False
        
        return True
    
    def _is_syndicated(self, article: Dict[str, Any]) -> bool:
        """Detect if article is syndicated/cross-posted"""
        content = (article.get("content", "") + " " + article.get("summary", "")).lower()
        
        for marker in self.SYNDICATION_MARKERS:
            if marker in content:
                return True
        
        return False
    
    def _add_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Add additional metadata to article"""
        # Word count
        content = article.get("content", "") or article.get("summary", "")
        article["word_count"] = len(content.split())
        
        # Reading time (average 200 words per minute)
        article["reading_time_minutes"] = max(1, article["word_count"] // 200)
        
        # Article ID (hash of canonical URL)
        article["article_id"] = hashlib.sha256(
            article["canonical_url"].encode()
        ).hexdigest()[:16]
        
        return article