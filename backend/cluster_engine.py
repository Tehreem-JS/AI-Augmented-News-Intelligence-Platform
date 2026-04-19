"""
Cluster Engine - Groups similar articles using NLP techniques
Uses TF-IDF + keyword fingerprinting + entity overlap for robust clustering
"""

from typing import List, Dict, Any, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import re


class ClusterEngine:
    """Clusters similar news articles deterministically"""

    SIMILARITY_THRESHOLD = 0.15      # Lower TF-IDF threshold — semantic boost handles precision
    ENTITY_OVERLAP_BONUS = 0.20      # Added to score when named entities overlap significantly
    KEYWORD_FINGERPRINT_BONUS = 0.25 # Added when 2+ rare keywords match exactly
    FINAL_MERGE_THRESHOLD = 0.12     # Second-pass: merge near-miss clusters

    MIN_CLUSTER_SIZE = 1
    MAX_CLUSTER_SIZE = 20

    # --- stop words for keyword fingerprinting (beyond sklearn's list) ---
    _EXTRA_STOPS = {
        "new", "says", "say", "said", "report", "reports", "reported",
        "share", "shares", "via", "use", "using", "used", "make", "makes",
        "week", "month", "year", "day", "today", "just", "now", "latest",
        "tech", "technology", "software", "update", "release",
    }

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.85,
            sublinear_tf=True,          # log(1+tf) dampens very common terms
        )

    # Max articles to process — keeps API calls manageable
    MAX_ARTICLES = 35

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not articles:
            return []

        print(f"🔗 Clustering {len(articles)} articles...")

        # Cap articles before clustering to control downstream API usage
        if len(articles) > self.MAX_ARTICLES:
            articles = self._select_top_articles(articles, self.MAX_ARTICLES)
            print(f"✂️  Capped to {len(articles)} articles (MAX_ARTICLES={self.MAX_ARTICLES})")

        texts = self._extract_texts(articles)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        except Exception as e:
            print(f"⚠️ Vectorization error: {e}")
            return self._create_single_clusters(articles)

        if tfidf_matrix.shape[0] == 0:
            return self._create_single_clusters(articles)

        # Base cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix).astype(np.float32)

        # Boost similarity with entity overlap + keyword fingerprints
        fingerprints = [self._keyword_fingerprint(a) for a in articles]
        entity_sets  = [self._extract_entity_set(a) for a in articles]
        similarity_matrix = self._apply_bonuses(
            similarity_matrix, articles, fingerprints, entity_sets
        )

        # Zero out self-similarity
        np.fill_diagonal(similarity_matrix, 0.0)

        # Cluster via Union-Find (handles transitive links correctly)
        labels = self._union_find_clustering(similarity_matrix, len(articles))

        # Second pass: merge clusters whose *centroid* similarity is high
        labels = self._merge_close_clusters(labels, tfidf_matrix)

        clusters_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters_dict[label].append(idx)

        clusters = self._create_cluster_objects(clusters_dict, articles)
        print(f"✅ Created {len(clusters)} clusters from {len(articles)} articles")
        return clusters

    # ------------------------------------------------------------------
    # Smart article selection (cap with diversity)
    # ------------------------------------------------------------------

    def _select_top_articles(
        self, articles: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """
        Select `limit` articles prioritising:
        1. Recency — newer articles ranked first
        2. Source diversity — avoid filling the cap with one RSS feed
        """
        # Sort newest first
        sorted_articles = sorted(
            articles, key=lambda a: a["published_date"], reverse=True
        )

        selected: List[Dict[str, Any]] = []
        source_counts: Dict[str, int] = defaultdict(int)

        # Max articles allowed per source (floor at 2 so rare sources aren't squeezed out)
        per_source_cap = max(2, limit // 5)

        # First pass — fill up respecting per-source cap
        for article in sorted_articles:
            source = article.get("source", "unknown")
            if source_counts[source] < per_source_cap:
                selected.append(article)
                source_counts[source] += 1
            if len(selected) >= limit:
                break

        # Second pass — if we're still under limit, fill remaining slots in recency order
        if len(selected) < limit:
            already = set(id(a) for a in selected)
            for article in sorted_articles:
                if id(article) not in already:
                    selected.append(article)
                if len(selected) >= limit:
                    break

        # Log source distribution
        dist = ", ".join(f"{s}:{c}" for s, c in sorted(source_counts.items()))
        print(f"   Source distribution: {dist}")

        return selected

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_texts(self, articles: List[Dict[str, Any]]) -> List[str]:
        texts = []
        for article in articles:
            title   = article["title"]
            summary = article.get("summary", "")
            content = article.get("content", "")[:600]
            # Title repeated 3× so it dominates the TF-IDF space
            combined = " ".join(filter(None, [title, title, title, summary, content]))
            texts.append(combined)
        return texts

    # ------------------------------------------------------------------
    # Keyword fingerprint  (rare, meaningful tokens from the title only)
    # ------------------------------------------------------------------

    def _keyword_fingerprint(self, article: Dict[str, Any]) -> Set[str]:
        title = article["title"].lower()
        tokens = re.findall(r"[a-z][a-z0-9\-\.]+", title)
        stops  = self.vectorizer.get_stop_words() if hasattr(self.vectorizer, "get_stop_words") else set()
        stops  = (stops or set()) | self._EXTRA_STOPS
        return {t for t in tokens if len(t) > 3 and t not in stops}

    # ------------------------------------------------------------------
    # Named-entity extraction (fast regex, no spaCy dependency)
    # ------------------------------------------------------------------

    _ENTITY_PATTERN = re.compile(
        r"\b(?:"
        # Companies
        r"OpenAI|Anthropic|Google|DeepMind|Microsoft|Apple|Amazon|Meta|Tesla"
        r"|NVIDIA|AMD|Intel|Qualcomm|Samsung|Huawei|ByteDance|xAI|Mistral|Cohere"
        # Models / products
        r"|GPT-?[3-9o]|GPT-4o?|Claude|Gemini|Llama|Grok|Copilot|Bard|Sora"
        r"|ChatGPT|Midjourney|Stable\s*Diffusion|Dall-?E"
        # Topics
        r"|AGI|LLM|AI|ML|API|GPU|CPU|5G|VR|AR|XR|IoT|SaaS|IPO"
        r"|Python|JavaScript|Rust|TypeScript|Java|Go"
        r"|AWS|Azure|GCP|Kubernetes|Docker|Linux"
        r")\b",
        re.IGNORECASE,
    )

    def _extract_entity_set(self, article: Dict[str, Any]) -> Set[str]:
        text = article["title"] + " " + article.get("summary", "")
        return {m.upper().replace(" ", "") for m in self._ENTITY_PATTERN.findall(text)}

    # ------------------------------------------------------------------
    # Similarity bonuses
    # ------------------------------------------------------------------

    def _apply_bonuses(
        self,
        sim: np.ndarray,
        articles: List[Dict[str, Any]],
        fingerprints: List[Set[str]],
        entity_sets: List[Set[str]],
    ) -> np.ndarray:
        n = len(articles)
        for i in range(n):
            for j in range(i + 1, n):
                bonus = 0.0

                # --- entity overlap ---
                ei, ej = entity_sets[i], entity_sets[j]
                if ei and ej:
                    overlap = len(ei & ej) / max(len(ei | ej), 1)
                    if overlap >= 0.5:
                        bonus += self.ENTITY_OVERLAP_BONUS
                    elif overlap >= 0.25:
                        bonus += self.ENTITY_OVERLAP_BONUS * 0.5

                # --- keyword fingerprint overlap ---
                fi, fj = fingerprints[i], fingerprints[j]
                if fi and fj:
                    shared = len(fi & fj)
                    if shared >= 3:
                        bonus += self.KEYWORD_FINGERPRINT_BONUS
                    elif shared >= 2:
                        bonus += self.KEYWORD_FINGERPRINT_BONUS * 0.6
                    elif shared >= 1 and len(fi | fj) <= 6:
                        # Small vocab titles with 1 shared rare word
                        bonus += self.KEYWORD_FINGERPRINT_BONUS * 0.3

                if bonus > 0:
                    sim[i, j] = min(1.0, sim[i, j] + bonus)
                    sim[j, i] = sim[i, j]

        return sim

    # ------------------------------------------------------------------
    # Union-Find clustering  (correctly handles transitive similarity)
    # ------------------------------------------------------------------

    def _union_find_clustering(self, sim: np.ndarray, n: int) -> List[int]:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Connect all pairs above threshold
        rows, cols = np.where(sim >= self.SIMILARITY_THRESHOLD)
        for i, j in zip(rows, cols):
            if i < j:
                union(int(i), int(j))

        return [find(i) for i in range(n)]

    # ------------------------------------------------------------------
    # Second pass: merge clusters whose TF-IDF centroids are close
    # ------------------------------------------------------------------

    def _merge_close_clusters(self, labels: List[int], tfidf_matrix) -> List[int]:
        unique_labels = list(set(labels))
        if len(unique_labels) <= 1:
            return labels

        # Build centroid for each cluster
        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            label_to_indices[lbl].append(idx)

        label_list = list(label_to_indices.keys())
        centroids  = np.vstack([
            np.asarray(tfidf_matrix[label_to_indices[lbl]].mean(axis=0))
            for lbl in label_list
        ])

        centroid_sim = cosine_similarity(centroids)
        np.fill_diagonal(centroid_sim, 0.0)

        # Union-Find on centroids
        parent = list(range(len(label_list)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                # Merge smaller cluster into larger
                size_a = len(label_to_indices[label_list[ra]])
                size_b = len(label_to_indices[label_list[rb]])
                if size_a >= size_b:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        rows, cols = np.where(centroid_sim >= self.FINAL_MERGE_THRESHOLD)
        for i, j in zip(rows, cols):
            if i < j:
                union(int(i), int(j))

        # Remap article labels
        root_of = {lbl: label_list[find(k)] for k, lbl in enumerate(label_list)}
        return [root_of[lbl] for lbl in labels]

    # ------------------------------------------------------------------
    # Cluster object construction
    # ------------------------------------------------------------------

    def _create_cluster_objects(
        self,
        clusters_dict: Dict[int, List[int]],
        articles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        cluster_objects = []

        for cluster_id, article_indices in clusters_dict.items():
            cluster_articles = [articles[i] for i in article_indices]
            cluster_articles.sort(key=lambda x: x["published_date"], reverse=True)

            main_title = cluster_articles[0]["title"]
            entities   = self._extract_entities(cluster_articles)
            category   = self._infer_category(cluster_articles, entities)

            cluster_objects.append({
                "cluster_id":    f"cluster_{cluster_id}_{hash(main_title) % 10000}",
                "articles":      cluster_articles,
                "main_title":    main_title,
                "article_count": len(cluster_articles),
                "entities":      entities,
                "category":      category,
                "date_range": {
                    "start": min(a["published_date"] for a in cluster_articles),
                    "end":   max(a["published_date"] for a in cluster_articles),
                },
            })

        cluster_objects.sort(
            key=lambda x: (x["article_count"], x["date_range"]["end"]),
            reverse=True,
        )
        return cluster_objects

    # ------------------------------------------------------------------
    # Entity extraction (for metadata, not clustering)
    # ------------------------------------------------------------------

    def _extract_entities(self, articles: List[Dict[str, Any]]) -> List[str]:
        entities: Dict[str, int] = defaultdict(int)
        for article in articles:
            text    = article["title"] + " " + article.get("content", "")
            matches = self._ENTITY_PATTERN.findall(text)
            for m in matches:
                entities[m.upper().replace(" ", "")] += 1
        return [e for e, _ in sorted(entities.items(), key=lambda x: -x[1])[:10]]

    # ------------------------------------------------------------------
    # Category inference
    # ------------------------------------------------------------------

    def _infer_category(
        self, articles: List[Dict[str, Any]], entities: List[str]
    ) -> List[str]:
        all_text = " ".join(
            a["title"] + " " + a.get("content", "")[:200] for a in articles
        ).lower()

        category_keywords = {
            "AI":           ["artificial intelligence", "ai model", "machine learning", "deep learning",
                             "neural network", "gpt", "claude", "gemini", "llm", "generative ai"],
            "ML Research":  ["research paper", "arxiv", "algorithm", "training", "dataset",
                             "benchmark", "transformer"],
            "Cybersecurity":["security", "vulnerability", "breach", "hack", "exploit",
                             "malware", "ransomware", "zero-day"],
            "Big Tech":     ["google", "microsoft", "apple", "amazon", "meta", "tesla",
                             "acquisition", "earnings"],
            "Cloud/DevOps": ["cloud", "aws", "azure", "gcp", "kubernetes", "docker",
                             "serverless", "devops"],
            "Hardware":     ["chip", "processor", "gpu", "cpu", "nvidia", "amd", "intel",
                             "semiconductor"],
            "Startups":     ["startup", "funding", "series a", "series b", "venture",
                             "valuation", "ipo", "unicorn"],
            "Data Science": ["data", "analytics", "sql", "database", "big data", "etl"],
            "AI Tools":     ["chatgpt", "copilot", "api", "sdk", "developer", "plugin"],
            "Frontend":     ["javascript", "react", "vue", "angular", "css", "frontend"],
        }

        categories: Set[str] = set()
        for category, keywords in category_keywords.items():
            if any(kw in all_text for kw in keywords):
                categories.add(category)

        entity_map = {
            frozenset(["OPENAI", "CLAUDE", "GEMINI", "GPT-4", "GPT-5", "CHATGPT"]): "AI",
            frozenset(["AWS", "AZURE", "GCP", "KUBERNETES", "DOCKER"]):             "Cloud/DevOps",
            frozenset(["NVIDIA", "AMD", "INTEL"]):                                  "Hardware",
        }
        for entity_set, cat in entity_map.items():
            if any(e in entity_set for e in entities):
                categories.add(cat)

        return list(categories) if categories else ["Tech News"]

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _create_single_clusters(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "cluster_id":    f"cluster_{i}_{hash(a['title']) % 10000}",
                "articles":      [a],
                "main_title":    a["title"],
                "article_count": 1,
                "entities":      [],
                "category":      ["Tech News"],
                "date_range":    {"start": a["published_date"], "end": a["published_date"]},
            }
            for i, a in enumerate(articles)
        ]