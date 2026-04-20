"""
News Fetcher - Collects tech news from multiple sources
Supports RSS feeds and web scraping with rate limiting
"""

import aiohttp
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timezone
import feedparser
from bs4 import BeautifulSoup
import hashlib
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


class NewsFetcher:
    """Fetches tech news from multiple authoritative sources"""
    
    # Comprehensive list of tech news sources
    SOURCES = {
        "TechCrunch": "https://techcrunch.com/feed/",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
        "Hacker News": "https://news.ycombinator.com/rss",
        "VentureBeat": "https://venturebeat.com/feed/",
        "Wired": "https://www.wired.com/feed/rss",
        "ZDNet": "https://www.zdnet.com/news/rss.xml",
        "The Next Web": "https://thenextweb.com/feed/",
        "ReadWrite": "https://readwrite.com/feed/",
    }
    
    # Blog sources for deep tech content
    BLOG_SOURCES = {
        "OpenAI Blog": "https://openai.com/blog/rss/",
        "Google AI Blog": "https://ai.googleblog.com/feeds/posts/default",
        "DeepMind Blog": "https://deepmind.google/blog/rss.xml",
        "Anthropic": "https://www.anthropic.com/news/rss.xml",
        "NVIDIA Blog": "https://blogs.nvidia.com/feed/",
        "AWS News": "https://aws.amazon.com/blogs/aws/feed/",
    }
    
    def __init__(self):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    async def fetch_all_sources(self, hours: int = 48) -> List[Dict[str, Any]]:
        """
        Fetch articles from all sources
        
        Args:
            hours: How many hours back to fetch (default 48)
            
        Returns:
            List of raw article dictionaries
        """
        all_sources = {**self.SOURCES, **self.BLOG_SOURCES}
        
        tasks = []
        for source_name, url in all_sources.items():
            tasks.append(self._fetch_feed(source_name, url, hours))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and filter out errors
        articles = []
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
            elif isinstance(result, Exception):
                print(f"⚠️  Error fetching source: {result}")
        
        print(f"📰 Total articles fetched: {len(articles)}")
        return articles
    
    async def _fetch_feed(self, source_name: str, url: str, hours: int) -> List[Dict[str, Any]]:
        """
        Fetch and parse RSS feed from a single source
        
        Args:
            source_name: Name of the source
            url: RSS feed URL
            hours: Time window in hours
            
        Returns:
            List of article dictionaries
        """
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                content = await response.text()
            
            # Parse RSS feed
            feed = feedparser.parse(content)
            
            articles = []
            cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
            
            for entry in feed.entries:
                try:
                    # Parse publication date
                    pub_date = self._parse_date(entry)
                    
                    # Filter by time window
                    if pub_date and pub_date.timestamp() < cutoff_time:
                        continue
                    
                    # Extract article data
                    article = {
                        "title": entry.get("title", "").strip(),
                        "url": self._get_canonical_url(entry),
                        "source": source_name,
                        "published_date": pub_date or datetime.now(timezone.utc),
                        "summary": self._extract_summary(entry),
                        "content": self._extract_content(entry),
                        "raw_entry": entry
                    }
                    
                    # Only add if we have essential fields
                    if article["title"] and article["url"]:
                        articles.append(article)
                        
                except Exception as e:
                    print(f"⚠️  Error parsing entry from {source_name}: {e}")
                    continue
            
            print(f"✅ {source_name}: {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"❌ Error fetching {source_name}: {e}")
            return []
    
    def _parse_date(self, entry) -> datetime:
        """Parse publication date from feed entry"""
        # Try different date fields
        for date_field in ["published_parsed", "updated_parsed", "created_parsed"]:
            if hasattr(entry, date_field):
                time_tuple = getattr(entry, date_field)
                if time_tuple:
                    try:
                        return datetime(*time_tuple[:6], tzinfo=timezone.utc)
                    except (ValueError, TypeError, OverflowError):  # ✅
                        pass
        
        # Fallback to current time
        return datetime.now(timezone.utc)
    
    def _get_canonical_url(self, entry) -> str:
        """Extract canonical URL and remove tracking parameters"""
        url = entry.get("link", "")
        
        if not url:
            return ""
        
        # Parse URL
        parsed = urlparse(url)
        
        # Remove tracking parameters
        tracking_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "ref", "_ga", "mc_cid", "mc_eid"
        }
        
        # Parse query string
        query_params = parse_qs(parsed.query)
        
        # Filter out tracking params
        clean_params = {
            k: v for k, v in query_params.items() 
            if k.lower() not in tracking_params
        }
        
        # Rebuild query string
        clean_query = urlencode(clean_params, doseq=True)
        
        # Reconstruct URL
        clean_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            clean_query,
            ""  # Remove fragment
        ))
        
        return clean_url
    
    def _extract_summary(self, entry) -> str:
        """Extract and clean summary text"""
        summary = ""
        
        # Try different summary fields
        for field in ["summary", "description", "content"]:
            if hasattr(entry, field):
                value = getattr(entry, field)
                if isinstance(value, list) and value:
                    summary = value[0].get("value", "")
                elif isinstance(value, str):
                    summary = value
                
                if summary:
                    break
        
        # Clean HTML
        if summary:
            summary = self._strip_html(summary)
        
        return summary[:500]  # Limit length
    
    def _extract_content(self, entry) -> str:
        """Extract full content if available"""
        content = ""
        
        # Try content field
        if hasattr(entry, "content") and entry.content:
            if isinstance(entry.content, list):
                content = entry.content[0].get("value", "")
        
        # Fallback to summary
        if not content and hasattr(entry, "summary"):
            content = entry.summary
        
        # Clean HTML
        if content:
            content = self._strip_html(content)
        
        return content[:2000]  # Limit length
    
    def _strip_html(self, html_text: str) -> str:
        """Remove HTML tags and clean text"""
        if not html_text:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(html_text, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)
        
        return text
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.session and not self.session.closed:
            try:
                asyncio.create_task(self.session.close())
            except:
                pass