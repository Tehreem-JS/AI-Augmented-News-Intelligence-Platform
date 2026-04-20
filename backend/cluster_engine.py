"""
Cluster Engine - Groups similar articles using NLP techniques
Uses TF-IDF + keyword fingerprinting + entity overlap for robust clustering

Key changes vs previous version:
- MAX_ARTICLES raised to 80 (was 35)
- TF-IDF uses min_df=2 and better feature weighting
- Agglomerative / chain-link clustering replaces strict Union-Find threshold
- Title fingerprint matching is the primary signal (not a bonus)
- Second-pass centroid merge threshold raised significantly
- Hot-topic scoring pushes bigger clusters to top
"""

from typing import List, Dict, Any, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import re


class ClusterEngine:
    """Clusters similar news articles deterministically"""

    # --- Primary clustering ---
    SIMILARITY_THRESHOLD = 0.10      # Low base; bonuses do the heavy lifting

    # --- Bonuses ---
    ENTITY_OVERLAP_BONUS      = 0.30  # Up from 0.20 — entities are strong signals
    KEYWORD_FINGERPRINT_BONUS = 0.35  # Up from 0.25 — title keywords are king

    # --- Second pass ---
    FINAL_MERGE_THRESHOLD = 0.25     # Up from 0.12 — actually merges near-miss clusters now

    MIN_CLUSTER_SIZE = 1
    MAX_CLUSTER_SIZE = 30            # Up from 20

    # Raised so we don't discard articles before clustering
    MAX_ARTICLES = 80                # Up from 35

    _EXTRA_STOPS = {
        "new", "says", "say", "said", "report", "reports", "reported",
        "share", "shares", "via", "use", "using", "used", "make", "makes",
        "week", "month", "year", "day", "today", "just", "now", "latest",
        "tech", "technology", "software", "update", "release", "inside",
        "here", "first", "look", "watch", "read", "want", "need", "gets",
        "will", "could", "would", "should", "this", "that", "with", "from",
    }

    def __init__(self):
        # min_df=2: ignore terms that only appear in 1 doc (reduces noise dimensions)
        # max_df=0.80: ignore terms in >80% of docs (too generic)
        # Tighter feature count forces the model to focus on meaningful terms
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,                # will auto-fallback to 1 when corpus is small
            max_df=0.80,
            sublinear_tf=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not articles:
            return []

        print(f"🔗 Clustering {len(articles)} articles...")

        if len(articles) > self.MAX_ARTICLES:
            articles = self._select_top_articles(articles, self.MAX_ARTICLES)
            print(f"✂️  Capped to {len(articles)} articles (MAX_ARTICLES={self.MAX_ARTICLES})")

        # --- Pre-cluster by title fingerprint (zero TF-IDF cost) ---
        # This catches obvious duplicates/same-story articles immediately,
        # reducing the matrix size before the expensive cosine step.
        pre_groups, ungrouped = self._fingerprint_pre_cluster(articles)

        # Run TF-IDF clustering on whatever remains ungrouped
        if len(ungrouped) >= 2:
            tfidf_groups = self._tfidf_cluster(ungrouped)
        else:
            tfidf_groups = [[a] for a in ungrouped]

        # Merge pre-groups and tfidf-groups into flat cluster list
        all_groups = pre_groups + tfidf_groups
        all_groups = self._merge_overlapping_groups(all_groups)

        clusters = self._create_cluster_objects_from_groups(all_groups)

        # Sort by "hotness": article_count first, then recency
        clusters.sort(
            key=lambda x: (x["article_count"], x["date_range"]["end"]),
            reverse=True,
        )

        print(f"✅ Created {len(clusters)} clusters from {len(articles)} articles")
        return clusters

    # ------------------------------------------------------------------
    # Stage 1 — Fast fingerprint pre-clustering
    # ------------------------------------------------------------------

    def _fingerprint_pre_cluster(
        self, articles: List[Dict[str, Any]]
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """
        Groups articles whose title keyword fingerprints share 2+ rare words.
        Returns (grouped_lists, ungrouped_articles).
        This runs in O(n²) over titles only — much cheaper than TF-IDF.
        """
        n = len(articles)
        fingerprints = [self._keyword_fingerprint(a) for a in articles]

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

        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = fingerprints[i], fingerprints[j]
                if not fi or not fj:
                    continue
                shared = fi & fj
                # 2+ shared rare title words → definitely same story
                if len(shared) >= 2:
                    union(i, j)
                # 1 shared word AND very similar vocab size → likely same story
                elif len(shared) == 1 and abs(len(fi) - len(fj)) <= 2 and len(fi | fj) <= 5:
                    union(i, j)

        groups: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        grouped: List[List[Dict]] = []
        ungrouped_indices: List[int] = []

        for root, indices in groups.items():
            if len(indices) >= 2:
                grouped.append([articles[i] for i in indices])
            else:
                ungrouped_indices.append(indices[0])

        ungrouped = [articles[i] for i in ungrouped_indices]
        print(f"   Fingerprint pre-cluster: {len(grouped)} groups, {len(ungrouped)} ungrouped")
        return grouped, ungrouped

    # ------------------------------------------------------------------
    # Stage 2 — TF-IDF clustering on ungrouped articles
    # ------------------------------------------------------------------

    def _tfidf_cluster(self, articles: List[Dict[str, Any]]) -> List[List[Dict]]:
        texts = self._extract_texts(articles)

        # Dynamically adjust min_df: with few docs, min_df=2 kills everything
        n = len(articles)
        self.vectorizer.set_params(min_df=2 if n >= 10 else 1)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        except Exception as e:
            print(f"⚠️ Vectorization error: {e}")
            return [[a] for a in articles]

        if tfidf_matrix.shape[0] == 0:
            return [[a] for a in articles]

        sim = cosine_similarity(tfidf_matrix).astype(np.float32)

        # Apply entity + fingerprint bonuses
        fingerprints = [self._keyword_fingerprint(a) for a in articles]
        entity_sets  = [self._extract_entity_set(a) for a in articles]
        sim = self._apply_bonuses(sim, articles, fingerprints, entity_sets)

        np.fill_diagonal(sim, 0.0)

        # Union-Find
        labels = self._union_find_clustering(sim, n)

        # Second pass: centroid merge
        labels = self._merge_close_clusters(labels, tfidf_matrix)

        groups: Dict[int, List[Dict]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            groups[lbl].append(articles[idx])

        return list(groups.values())

    # ------------------------------------------------------------------
    # Stage 3 — Merge overlapping groups (safety net)
    # ------------------------------------------------------------------

    def _merge_overlapping_groups(
        self, groups: List[List[Dict]]
    ) -> List[List[Dict]]:
        """
        If the same article somehow ended up in two groups (shouldn't happen
        with Union-Find, but defensive), merge those groups.
        """
        id_to_group: Dict[int, int] = {}
        parent = list(range(len(groups)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for g_idx, group in enumerate(groups):
            for article in group:
                aid = id(article)
                if aid in id_to_group:
                    union(g_idx, id_to_group[aid])
                else:
                    id_to_group[aid] = g_idx

        merged: Dict[int, List[Dict]] = defaultdict(list)
        for g_idx, group in enumerate(groups):
            root = find(g_idx)
            for article in group:
                # deduplicate within merged group
                if not any(id(a) == id(article) for a in merged[root]):
                    merged[root].append(article)

        return list(merged.values())

    # ------------------------------------------------------------------
    # Smart article selection (diversity cap)
    # ------------------------------------------------------------------

    def _select_top_articles(
        self, articles: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        sorted_articles = sorted(
            articles, key=lambda a: a["published_date"], reverse=True
        )
        selected: List[Dict[str, Any]] = []
        source_counts: Dict[str, int] = defaultdict(int)
        per_source_cap = max(3, limit // 4)   # slightly looser cap

        for article in sorted_articles:
            source = article.get("source", "unknown")
            if source_counts[source] < per_source_cap:
                selected.append(article)
                source_counts[source] += 1
            if len(selected) >= limit:
                break

        if len(selected) < limit:
            already = set(id(a) for a in selected)
            for article in sorted_articles:
                if id(article) not in already:
                    selected.append(article)
                if len(selected) >= limit:
                    break

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
            content = article.get("content", "")[:800]   # slightly more context
            # Title twice (not 3×) — less distortion, still dominant
            combined = " ".join(filter(None, [title, title, summary, content]))
            texts.append(combined)
        return texts

    # ------------------------------------------------------------------
    # Keyword fingerprint
    # ------------------------------------------------------------------

    def _keyword_fingerprint(self, article: Dict[str, Any]) -> Set[str]:
        title = article["title"].lower()
        tokens = re.findall(r"[a-z][a-z0-9\-\.]+", title)
        try:
            stops = set(self.vectorizer.get_stop_words() or [])
        except Exception:
            stops = set()
        stops = stops | self._EXTRA_STOPS
        return {t for t in tokens if len(t) > 3 and t not in stops}

    # ------------------------------------------------------------------
    # Named-entity extraction
    # ------------------------------------------------------------------

    _ENTITY_PATTERN = re.compile(
        r"\b(?:"
        r"OpenAI|Anthropic|Google|DeepMind|Microsoft|Apple|Amazon|Meta|Tesla"
        r"|NVIDIA|AMD|Intel|Qualcomm|Samsung|Huawei|ByteDance|xAI|Mistral|Cohere"
        r"|GPT-?[3-9o]|GPT-4o?|Claude|Gemini|Llama|Grok|Copilot|Bard|Sora"
        r"|ChatGPT|Midjourney|Stable\s*Diffusion|Dall-?E"
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

                ei, ej = entity_sets[i], entity_sets[j]
                if ei and ej:
                    overlap = len(ei & ej) / max(len(ei | ej), 1)
                    if overlap >= 0.5:
                        bonus += self.ENTITY_OVERLAP_BONUS
                    elif overlap >= 0.25:
                        bonus += self.ENTITY_OVERLAP_BONUS * 0.5

                fi, fj = fingerprints[i], fingerprints[j]
                if fi and fj:
                    shared = len(fi & fj)
                    if shared >= 3:
                        bonus += self.KEYWORD_FINGERPRINT_BONUS
                    elif shared >= 2:
                        bonus += self.KEYWORD_FINGERPRINT_BONUS * 0.7
                    elif shared >= 1 and len(fi | fj) <= 6:
                        bonus += self.KEYWORD_FINGERPRINT_BONUS * 0.3

                if bonus > 0:
                    sim[i, j] = min(1.0, sim[i, j] + bonus)
                    sim[j, i] = sim[i, j]

        return sim

    # ------------------------------------------------------------------
    # Union-Find clustering
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

        rows, cols = np.where(sim >= self.SIMILARITY_THRESHOLD)
        for i, j in zip(rows, cols):
            if i < j:
                union(int(i), int(j))

        return [find(i) for i in range(n)]

    # ------------------------------------------------------------------
    # Second pass: centroid merge
    # ------------------------------------------------------------------

    def _merge_close_clusters(self, labels: List[int], tfidf_matrix) -> List[int]:
        unique_labels = list(set(labels))
        if len(unique_labels) <= 1:
            return labels

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

        parent = list(range(len(label_list)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
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

        root_of = {lbl: label_list[find(k)] for k, lbl in enumerate(label_list)}
        return [root_of[lbl] for lbl in labels]

    # ------------------------------------------------------------------
    # Cluster object construction
    # ------------------------------------------------------------------

    def _create_cluster_objects_from_groups(
        self, groups: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        cluster_objects = []
        for group_idx, cluster_articles in enumerate(groups):
            cluster_articles = sorted(
                cluster_articles, key=lambda x: x["published_date"], reverse=True
            )
            main_title = cluster_articles[0]["title"]
            entities   = self._extract_entities(cluster_articles)
            category   = self._infer_category(cluster_articles, entities)

            cluster_objects.append({
                "cluster_id":    f"cluster_{group_idx}_{hash(main_title) % 10000}",
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

        return cluster_objects

    # ------------------------------------------------------------------
    # Entity extraction (metadata)
    # ------------------------------------------------------------------

    def _extract_entities(self, articles: List[Dict[str, Any]]) -> List[str]:
        entities: Dict[str, int] = defaultdict(int)
        for article in articles:
            text = article["title"] + " " + article.get("content", "")
            for m in self._ENTITY_PATTERN.findall(text):
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
            "AI":            ["artificial intelligence", "ai model", "machine learning", "deep learning",
                              "neural network", "gpt", "claude", "gemini", "llm", "generative ai"],
            "ML Research":   ["research paper", "arxiv", "algorithm", "training", "dataset",
                              "benchmark", "transformer"],
            "Cybersecurity": ["security", "vulnerability", "breach", "hack", "exploit",
                              "malware", "ransomware", "zero-day"],
            "Big Tech":      ["google", "microsoft", "apple", "amazon", "meta", "tesla",
                              "acquisition", "earnings"],
            "Cloud/DevOps":  ["cloud", "aws", "azure", "gcp", "kubernetes", "docker",
                              "serverless", "devops"],
            "Hardware":      ["chip", "processor", "gpu", "cpu", "nvidia", "amd", "intel",
                              "semiconductor"],
            "Startups":      ["startup", "funding", "series a", "series b", "venture",
                              "valuation", "ipo", "unicorn"],
            "Data Science":  ["data", "analytics", "sql", "database", "big data", "etl"],
            "AI Tools":      ["chatgpt", "copilot", "api", "sdk", "developer", "plugin"],
            "Frontend":      ["javascript", "react", "vue", "angular", "css", "frontend"],
        }

        categories: Set[str] = set()
        for category, keywords in category_keywords.items():
            if any(kw in all_text for kw in keywords):
                categories.add(category)

        entity_map = {
            frozenset(["OPENAI", "CLAUDE", "GEMINI", "GPT-4", "GPT-5", "CHATGPT"]): "AI",
            frozenset(["AWS", "AZURE", "GCP", "KUBERNETES", "DOCKER"]):              "Cloud/DevOps",
            frozenset(["NVIDIA", "AMD", "INTEL"]):                                   "Hardware",
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