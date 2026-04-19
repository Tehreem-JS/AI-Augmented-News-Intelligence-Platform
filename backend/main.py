"""
Tech News Analyst Agent - Main FastAPI Application
Context-Aware News Aggregation and Analysis System

IMPORTANT: This file does NOT include API keys.
You need to configure Claude API access separately.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from news_fetcher import NewsFetcher
from news_processor import NewsProcessor
from cluster_engine import ClusterEngine
from brief_generator import BriefGenerator
from database import Database
from scheduler import NewsScheduler
from dotenv import load_dotenv
import os

load_dotenv()  # This looks for a .env file and loads the variables
# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(" Starting Tech News Analyst Agent...")
    await db.connect()
    
    # Start scheduler
    scheduler.start()
    print(" Scheduler started - Daily news collection enabled")
    
    yield
    
    # Shutdown
    print(" Shutting down...")
    scheduler.shutdown()
    await db.disconnect()

# Initialize FastAPI app
app = FastAPI(
    title="Tech News Analyst Agent",
    description="Context-Aware Tech News Aggregation and Analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db = Database()
fetcher = NewsFetcher()
processor = NewsProcessor()
cluster_engine = ClusterEngine()
brief_generator = BriefGenerator()
scheduler = NewsScheduler(db, fetcher, processor, cluster_engine, brief_generator)


# Pydantic Models
class NewsArticle(BaseModel):
    id: Optional[int] = None
    title: str
    url: str
    source: str
    published_date: datetime
    summary: Optional[str] = None
    content: Optional[str] = None
    category: List[str] = []
    

class NewsBrief(BaseModel):
    id: Optional[int] = None
    cluster_id: str
    title: str
    key_points: List[str]
    why_it_matters: str
    sources: List[Dict[str, str]]
    background_context: Optional[str] = None
    technical_glossary: Optional[Dict[str, str]] = None
    examples: Optional[str] = None
    categories: List[str]
    created_at: datetime
    articles_count: int
    relative_time: Optional[str] = None  # "2h ago", "Yesterday", etc.


class SystemStatus(BaseModel):
    """System status and collection metadata"""
    last_updated: str  # "3 hours ago"
    last_updated_timestamp: Optional[datetime] = None
    next_update: str  # "6:00 PM ET"
    articles_collected_today: int
    briefs_generated_today: int
    total_briefs_shown: int
    time_window_used: str  # "Last 24 hours", "Last 7 days", etc.


class TimeTier(BaseModel):
    """Time-based tier of news briefs"""
    label: str  # "Breaking Today", "This Week", "Recent"
    briefs: List[NewsBrief]
    count: int


class DashboardResponse(BaseModel):
    """Enhanced dashboard response with tiered sections and system status"""
    date: str
    greeting: str
    system_status: SystemStatus
    top_headlines: List[NewsBrief]
    time_tiers: List[TimeTier]
    all_briefs: Dict[str, List[NewsBrief]]
    total_articles: int
    last_updated: datetime


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "Tech News Analyst Agent",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    topic: Optional[str] = Query(None, description="Filter by topic (e.g., 'AI', 'Cybersecurity')")
):
    """
    Get the main dashboard with categorized news briefs
    
    Features:
    - Adaptive time window (guarantees 10-15 briefs minimum)
    - Tiered sections: Breaking (24h) / This Week (2-7d) / Recent (8-30d)
    - Relative timestamps on each brief
    - System status metadata
    """
    try:
        # Step 1: Fetch briefs with adaptive time window
        briefs, days_used = await db.adaptive_get_briefs(
            category_filter=topic,
            min_briefs=10
        )
        
        # If no briefs exist at all, trigger collection
        if not briefs:
            await collect_and_process_news()
            briefs, days_used = await db.adaptive_get_briefs(
                category_filter=topic,
                min_briefs=10
            )
        
        # Step 2: Add relative time to each brief
        now = datetime.now()
        for brief in briefs:
            brief.relative_time = db.calculate_relative_time(brief.created_at)
        
        # Step 3: Sort by recency
        sorted_briefs = sorted(briefs, key=lambda x: x.created_at, reverse=True)
        
        # Step 4: Create time tiers
        time_tiers = _categorize_briefs_by_time(sorted_briefs)
        
        # Step 5: Get top 5 headlines (most recent)
        top_headlines = sorted_briefs[:5]
        
        # Step 6: Group by categories
        categorized = {}
        for brief in sorted_briefs:
            for category in brief.categories:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(brief)
        
        # Step 7: Get system status
        system_status = await _get_system_status(days_used, len(sorted_briefs))
        
        # Step 8: Get total articles count
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_used)
        total_articles = await db.get_articles_count(start_date, end_date)
        
        # Step 9: Generate greeting
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good Morning"
        elif hour < 18:
            greeting = "Good Afternoon"
        else:
            greeting = "Good Evening"
        
        return DashboardResponse(
            date=datetime.now().strftime("%B %d, %Y"),
            greeting=greeting,
            system_status=system_status,
            top_headlines=top_headlines,
            time_tiers=time_tiers,
            all_briefs=categorized,
            total_articles=total_articles,
            last_updated=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")


def _categorize_briefs_by_time(briefs: List[NewsBrief]) -> List[TimeTier]:
    """
    Categorize briefs into time tiers
    
    Tiers:
    - Breaking Today: Last 24 hours
    - This Week: 2-7 days ago
    - Recent: 8-30 days ago
    """
    now = datetime.now()
    
    breaking = []  # Last 24 hours
    this_week = []  # 2-7 days
    recent = []  # 8-30 days
    
    for brief in briefs:
        age_hours = (now - brief.created_at).total_seconds() / 3600
        
        if age_hours <= 24:
            breaking.append(brief)
        elif age_hours <= 168:  # 7 days
            this_week.append(brief)
        else:
            recent.append(brief)
    
    tiers = []
    
    if breaking:
        tiers.append(TimeTier(
            label="Breaking Today",
            briefs=breaking,
            count=len(breaking)
        ))
    
    if this_week:
        tiers.append(TimeTier(
            label="This Week",
            briefs=this_week,
            count=len(this_week)
        ))
    
    if recent:
        tiers.append(TimeTier(
            label="Recent",
            briefs=recent,
            count=len(recent)
        ))
    
    return tiers


async def _get_system_status(days_used: int, total_briefs: int) -> SystemStatus:
    """
    Get system status metadata
    
    Includes:
    - Last collection time
    - Next scheduled update
    - Collection stats for today
    - Time window used for current view
    """
    # Get last collection info
    last_collection = await db.get_last_collection_info()
    
    if last_collection:
        last_updated = last_collection["relative_time"]
        last_updated_timestamp = last_collection["timestamp"]
        articles_today = last_collection["articles_fetched"]
        briefs_today = last_collection["briefs_generated"]
    else:
        last_updated = "Never"
        last_updated_timestamp = None
        articles_today = 0
        briefs_today = 0
    
    # Get next run time
    next_update = scheduler.get_next_run_time()
    
    # Format time window
    if days_used == 1:
        time_window = "Last 24 hours"
    elif days_used == 2:
        time_window = "Last 48 hours"
    elif days_used == 7:
        time_window = "Last 7 days"
    else:
        time_window = f"Last {days_used} days"
    
    return SystemStatus(
        last_updated=last_updated,
        last_updated_timestamp=last_updated_timestamp,
        next_update=next_update,
        articles_collected_today=articles_today,
        briefs_generated_today=briefs_today,
        total_briefs_shown=total_briefs,
        time_window_used=time_window
    )


@app.get("/api/briefs/{brief_id}", response_model=NewsBrief)
async def get_brief_detail(brief_id: int):
    """Get detailed view of a specific news brief"""
    try:
        brief = await db.get_brief_by_id(brief_id)
        if not brief:
            raise HTTPException(status_code=404, detail="Brief not found")
        
        # Add relative time
        brief.relative_time = db.calculate_relative_time(brief.created_at)
        
        return brief
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
async def get_categories():
    """Get all available news categories with counts"""
    try:
        categories = await db.get_category_stats()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collect")
async def trigger_collection():
    """Manually trigger news collection and processing"""
    try:
        result = await collect_and_process_news()
        return {
            "status": "success",
            "message": "News collection completed",
            "articles_fetched": result["articles_fetched"],
            "briefs_generated": result["briefs_generated"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collection failed: {str(e)}")


@app.get("/api/search")
async def search_news(
    query: str = Query(..., min_length=2),
    days: int = Query(7, ge=1, le=30)
):
    """Search news briefs by keyword"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        results = await db.search_briefs(query, start_date, end_date)
        
        # Add relative time to each result
        for result in results:
            result.relative_time = db.calculate_relative_time(result.created_at)
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    """Get overall system statistics"""
    try:
        stats = await db.get_system_stats()
        
        # Add last collection info
        last_collection = await db.get_last_collection_info()
        if last_collection:
            stats["last_collection"] = last_collection
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Core Processing Function
async def collect_and_process_news() -> Dict[str, Any]:
    """
    Main pipeline: Fetch → Clean → Cluster → Generate Briefs
    """
    print("📡 Starting news collection pipeline...")
    
    stats = {
        "articles_fetched": 0,
        "articles_stored": 0,
        "clusters_created": 0,
        "briefs_generated": 0
    }
    
    try:
        # Step 1: Fetch articles from multiple sources
        raw_articles = await fetcher.fetch_all_sources()
        stats["articles_fetched"] = len(raw_articles)
        print(f"✅ Fetched {len(raw_articles)} raw articles")
        
        # Step 2: Clean and normalize
        cleaned_articles = processor.clean_and_normalize(raw_articles)
        print(f"✅ Cleaned {len(cleaned_articles)} articles")
        
        # Step 3: Store articles in database
        stored_ids = await db.store_articles(cleaned_articles)
        stats["articles_stored"] = len(stored_ids)
        print(f"✅ Stored {len(stored_ids)} articles")
        
        # Step 4: Cluster similar articles
        clusters = cluster_engine.cluster_articles(cleaned_articles)
        stats["clusters_created"] = len(clusters)
        print(f"✅ Created {len(clusters)} clusters")
        
        # Step 5: Generate briefs for each cluster
        briefs = []
        for cluster in clusters:
            brief = await brief_generator.generate_brief(cluster)
            briefs.append(brief)
        
        stats["briefs_generated"] = len(briefs)
        print(f"✅ Generated {len(briefs)} news briefs")
        
        # Step 6: Store briefs
        await db.store_briefs(briefs)
        
        # Step 7: Store collection metadata
        await db.store_collection_metadata(
            articles_fetched=stats["articles_fetched"],
            articles_stored=stats["articles_stored"],
            clusters_created=stats["clusters_created"],
            briefs_generated=stats["briefs_generated"],
            success=True
        )
        
        return stats
        
    except Exception as e:
        print(f"❌ Error in collection pipeline: {e}")
        
        # Store error metadata
        await db.store_collection_metadata(
            articles_fetched=stats.get("articles_fetched", 0),
            articles_stored=stats.get("articles_stored", 0),
            clusters_created=stats.get("clusters_created", 0),
            briefs_generated=stats.get("briefs_generated", 0),
            success=False,
            error_message=str(e)
        )
        
        raise


if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)