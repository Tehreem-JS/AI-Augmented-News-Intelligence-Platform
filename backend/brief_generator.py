
import os
from typing import Dict, Any, List
import json
from datetime import datetime
import asyncio  

class BriefGenerator:
    """Generates comprehensive news briefs using Groq API"""
    
    def __init__(self):
        """
        Initialize Brief Generator
        
        ⚠️ CONFIGURATION REQUIRED:
        You need to set GROQ_API_KEY environment variable or configure it below.
        """
        # OPTION 1: Read from environment (RECOMMENDED)
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # OPTION 2: Direct configuration (UNCOMMENT AND ADD YOUR KEY)
        # self.api_key = "gsk_your-api-key-here"  # ⚠️ REPLACE WITH YOUR KEY
        
        if not self.api_key:
            print("⚠️  WARNING: GROQ_API_KEY not configured!")
            print("   Set environment variable or configure in brief_generator.py")
            print("   Brief generation will fail without a valid API key.")
        
        self.model =  "llama-3.1-8b-instant" # or "llama-3.3-70b-versatile" 
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    async def generate_brief(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive brief for a news cluster
        
        The brief includes:
        1. Title (concise, informative)
        2. Key Points (3-5 bullet points)
        3. Why It Matters (impact analysis)
        4. Sources (with links)
        5. Background Context (historical evolution)
        6. Technical Glossary (key terms explained)
        7. Examples/Illustrations
        8. Category Tags
        
        Args:
            cluster: Dictionary containing clustered articles
            
        Returns:
            Brief dictionary with all components
            
        ⚠️ COST: Each call costs ~$0.001-0.005 depending on content length
        """
        articles = cluster["articles"]
        
        # Prepare article summaries for Claude
        article_summaries = self._prepare_article_summaries(articles)
        
        # Create prompt for Claude
        prompt = self._create_brief_prompt(
            cluster["main_title"],
            article_summaries,
            cluster["entities"],
            cluster["category"]
        )
        
        try:
            await asyncio.sleep(3)
            # Call Claude API
            response = await self._call_claude_api(prompt)
            
            # Parse response
            brief_data = self._parse_claude_response(response)
            
            # Add metadata
            brief = {
                "cluster_id": cluster["cluster_id"],
                "title": brief_data.get("title", cluster["main_title"]),
                "key_points": brief_data.get("key_points", []),
                "why_it_matters": brief_data.get("why_it_matters", ""),
                "sources": self._extract_sources(articles),
                "background_context": brief_data.get("background_context"),
                "technical_glossary": brief_data.get("technical_glossary"),
                "examples": brief_data.get("examples"),
                "categories": cluster["category"],
                "created_at": datetime.now(),
                "articles_count": len(articles),
                "article_ids": [a["article_id"] for a in articles]
            }
            
            return brief
            
        except Exception as e:
            print(f"⚠️  Error generating brief with groq API: {e}")
            # Fallback to deterministic brief
            return self._create_fallback_brief(cluster)
    
    def _prepare_article_summaries(self, articles: List[Dict[str, Any]]) -> str:
        """Prepare article summaries for Claude prompt"""
        summaries = []
        
        for i, article in enumerate(articles[:5], 1):  # Limit to 5 articles
            summary = f"""
Article {i}:
Source: {article['source']}
Title: {article['title']}
Date: {article['published_date'].strftime('%Y-%m-%d %H:%M ET')}
Content: {article.get('content', article.get('summary', ''))[:500]}
"""
            summaries.append(summary)
        
        return "\n---\n".join(summaries)
    
    def _create_brief_prompt(self, main_title: str, article_summaries: str, 
                            entities: List[str], categories: List[str]) -> str:
        """Create structured prompt for groq"""
        
        prompt = f"""You are a tech news analyst creating a comprehensive news brief.

CONTEXT:
Main Topic: {main_title}
Key Entities: {', '.join(entities) if entities else 'N/A'}
Categories: {', '.join(categories)}

ARTICLES TO ANALYZE:
{article_summaries}

CREATE A COMPREHENSIVE NEWS BRIEF with the following structure (respond ONLY in JSON format):

{{
  "title": "Clear, informative title (60 chars max)",
  
  "key_points": [
    "First key development or announcement",
    "Second important point or detail",
    "Third significant aspect or implication",
    "Fourth point if relevant (3-5 points total)"
  ],
  
  "why_it_matters": "2-3 sentences explaining: Industry impact, Who is affected, Future implications. Be specific and concrete.",
  
  "background_context": "If this news requires historical context, explain: What came before? What previous versions/models existed? What were their limitations? What led to this development? Include specific examples and timeline. If no background needed, set to null.",
  
  "technical_glossary": {{
    "term1": "One-line clear explanation",
    "term2": "One-line clear explanation",
    "term3": "One-line clear explanation"
  }} OR null if not technical,
  
  "examples": "If relevant, provide a simple example, use case, or scenario that illustrates the concept. Keep it practical and relatable. Set to null if not needed."
}}

IMPORTANT GUIDELINES:
- For AI/ML topics: Explain model architectures, training methods, capabilities
- For products: Include version history and evolution
- For companies: Provide context on their position in the market
- For technical concepts: Break down complexity into understandable terms
- Use specific numbers, dates, and facts when available
- Keep each field concise but informative

Respond ONLY with the JSON object, no markdown formatting, no explanation."""

        return prompt
    
    async def _call_claude_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call Groq API for brief generation
        
        ⚠️ REQUIRES: Valid GROQ_API_KEY
        ⚠️ COST: FREE (with rate limits)
        """
        if not self.api_key:
            raise Exception("GROQ_API_KEY not configured. Cannot call Groq API.")
        
        import aiohttp
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Groq API error {response.status}: {error_text}")
                
                data = await response.json()
                return data
    
    def _parse_claude_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Groq's JSON response"""
        
        # Extract content from OpenAI-compatible format
        choices = response.get("choices", [])
        
        if not choices:
            raise ValueError("Empty response from Groq")
        
        # Get text from message content
        text = choices[0].get("message", {}).get("content", "")
        
        # Remove markdown code fences if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Parse JSON
        try:
            brief_data = json.loads(text)
            return brief_data
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Response text: {text[:200]}")
            raise
    
    def _extract_sources(self, articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract credible sources with links"""
        sources = []
        
        # Limit to top 3 sources
        for article in articles[:3]:
            source = {
                "name": article["source"],
                "title": article["title"],
                "url": article["url"],
                "date": article["published_date"].strftime("%Y-%m-%d")
            }
            sources.append(source)
        
        return sources
    
    def _create_fallback_brief(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a deterministic brief if API fails
        
        This is used when:
        - API key is not configured
        - API call fails
        - Rate limit exceeded
        """
        articles = cluster["articles"]
        
        # Simple key points from article titles
        key_points = []
        for article in articles[:5]:
            # Extract first sentence from content or use title
            content = article.get("content", article.get("summary", ""))
            if content:
                sentences = content.split(". ")
                if sentences:
                    key_points.append(sentences[0])
            else:
                key_points.append(article["title"])
        
        return {
            "cluster_id": cluster["cluster_id"],
            "title": cluster["main_title"],
            "key_points": key_points[:5],
            "why_it_matters": "This is a developing story in the tech industry. " + 
                            "Further details and analysis will be added as more information becomes available.",
            "sources": self._extract_sources(articles),
            "background_context": None,
            "technical_glossary": None,
            "examples": None,
            "categories": cluster["category"],
            "created_at": datetime.now(),
            "articles_count": len(articles),
            "article_ids": [a.get("article_id", str(i)) for i, a in enumerate(articles)]
        }

