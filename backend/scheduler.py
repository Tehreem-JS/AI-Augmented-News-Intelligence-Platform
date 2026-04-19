"""
Scheduler - Daily automated news collection and processing
Uses APScheduler for reliable job scheduling
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import asyncio


class NewsScheduler:
    """Manages scheduled news collection jobs"""
    
    def __init__(self, db, fetcher, processor, cluster_engine, brief_generator):
        """
        Initialize scheduler with all pipeline components
        
        Args:
            db: Database instance
            fetcher: NewsFetcher instance
            processor: NewsProcessor instance
            cluster_engine: ClusterEngine instance
            brief_generator: BriefGenerator instance
        """
        self.db = db
        self.fetcher = fetcher
        self.processor = processor
        self.cluster_engine = cluster_engine
        self.brief_generator = brief_generator
        
        self.scheduler = AsyncIOScheduler()
        self._configure_jobs()
    
    def _configure_jobs(self):
        """Configure scheduled jobs"""
        
        # Daily news collection at 6:00 AM ET
        self.scheduler.add_job(
            self.daily_collection,
            CronTrigger(hour=6, minute=0, timezone="US/Eastern"),
            id="daily_collection",
            name="Daily News Collection",
            replace_existing=True
        )
        
        # Refresh at 12:00 PM ET (midday update)
        self.scheduler.add_job(
            self.daily_collection,
            CronTrigger(hour=12, minute=0, timezone="US/Eastern"),
            id="midday_refresh",
            name="Midday News Refresh",
            replace_existing=True
        )
        
        # Evening update at 6:00 PM ET
        self.scheduler.add_job(
            self.daily_collection,
            CronTrigger(hour=18, minute=0, timezone="US/Eastern"),
            id="evening_refresh",
            name="Evening News Refresh",
            replace_existing=True
        )
        
        print("📅 Scheduled jobs configured:")
        print("   - Daily collection: 6:00 AM ET")
        print("   - Midday refresh: 12:00 PM ET")
        print("   - Evening refresh: 6:00 PM ET")
    
    def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        print("✅ Scheduler started")
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
        print("✅ Scheduler stopped")
    
    def get_next_run_time(self) -> str:
        """
        Get the next scheduled run time as a formatted string
        
        Returns:
            Formatted time string like "6:00 PM ET" or "Unknown" if no jobs scheduled
        """
        jobs = self.scheduler.get_jobs()
        if not jobs:
            return "Unknown"
        
        # Get the earliest next run time
        next_run_times = [job.next_run_time for job in jobs if job.next_run_time]
        
        if not next_run_times:
            return "Unknown"
        
        next_run = min(next_run_times)
        
        # Format as "6:00 PM ET"
        return next_run.strftime("%I:%M %p ET")
    
    async def daily_collection(self):
        """
        Main scheduled job: collect and process news
        
        Pipeline:
        1. Fetch articles from all sources
        2. Clean and normalize
        3. Store in database
        4. Cluster similar articles
        5. Generate briefs
        6. Store briefs
        7. Store collection metadata
        """
        print(f"\n{'='*60}")
        print(f"🔄 Starting scheduled news collection - {datetime.now()}")
        print(f"{'='*60}\n")
        
        # Track stats
        stats = {
            "articles_fetched": 0,
            "articles_stored": 0,
            "clusters_created": 0,
            "briefs_generated": 0
        }
        
        try:
            # Step 1: Fetch
            print("📡 Step 1: Fetching articles...")
            raw_articles = await self.fetcher.fetch_all_sources(hours=48)
            stats["articles_fetched"] = len(raw_articles)
            print(f"   Fetched: {len(raw_articles)} articles")
            
            if not raw_articles:
                print("⚠️  No articles fetched. Skipping this run.")
                # Store metadata even for empty runs
                await self.db.store_collection_metadata(
                    articles_fetched=0,
                    articles_stored=0,
                    clusters_created=0,
                    briefs_generated=0,
                    success=True
                )
                return
            
            # Step 2: Clean and normalize
            print("\n🧹 Step 2: Cleaning and normalizing...")
            cleaned_articles = self.processor.clean_and_normalize(raw_articles)
            print(f"   Cleaned: {len(cleaned_articles)} articles")
            
            if not cleaned_articles:
                print("⚠️  No valid articles after cleaning. Skipping this run.")
                await self.db.store_collection_metadata(
                    articles_fetched=stats["articles_fetched"],
                    articles_stored=0,
                    clusters_created=0,
                    briefs_generated=0,
                    success=True
                )
                return
            
            # Step 3: Store articles
            print("\n💾 Step 3: Storing articles...")
            stored_ids = await self.db.store_articles(cleaned_articles)
            stats["articles_stored"] = len(stored_ids)
            print(f"   Stored: {len(stored_ids)} new articles")
            
            # Step 4: Cluster
            print("\n🔗 Step 4: Clustering articles...")
            clusters = self.cluster_engine.cluster_articles(cleaned_articles)
            stats["clusters_created"] = len(clusters)
            print(f"   Created: {len(clusters)} clusters")
            
            # Step 5: Generate briefs
            print("\n📝 Step 5: Generating briefs...")
            briefs = []
            for i, cluster in enumerate(clusters, 1):
                print(f"   Generating brief {i}/{len(clusters)}: {cluster['main_title'][:60]}...")
                brief = await self.brief_generator.generate_brief(cluster)
                briefs.append(brief)
            
            stats["briefs_generated"] = len(briefs)
            print(f"   Generated: {len(briefs)} briefs")
            
            # Step 6: Store briefs
            print("\n💾 Step 6: Storing briefs...")
            brief_ids = await self.db.store_briefs(briefs)
            print(f"   Stored: {len(brief_ids)} briefs")
            
            # Step 7: Store collection metadata
            print("\n📊 Step 7: Storing collection metadata...")
            await self.db.store_collection_metadata(
                articles_fetched=stats["articles_fetched"],
                articles_stored=stats["articles_stored"],
                clusters_created=stats["clusters_created"],
                briefs_generated=stats["briefs_generated"],
                success=True
            )
            
            # Summary
            print(f"\n{'='*60}")
            print(f"✅ Collection complete!")
            print(f"   Articles fetched: {stats['articles_fetched']}")
            print(f"   Articles stored: {stats['articles_stored']}")
            print(f"   Clusters created: {stats['clusters_created']}")
            print(f"   Briefs generated: {stats['briefs_generated']}")
            print(f"   Time: {datetime.now()}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n❌ Error during scheduled collection: {e}")
            import traceback
            traceback.print_exc()
            
            # Store error metadata
            await self.db.store_collection_metadata(
                articles_fetched=stats.get("articles_fetched", 0),
                articles_stored=stats.get("articles_stored", 0),
                clusters_created=stats.get("clusters_created", 0),
                briefs_generated=stats.get("briefs_generated", 0),
                success=False,
                error_message=str(e)
            )