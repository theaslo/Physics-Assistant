#!/usr/bin/env python3
"""
Semantic Similarity Search System for Physics Content
Implements vector similarity search with hybrid approaches combining semantic and exact matching
"""
import os
import json
import logging
import asyncio
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import math

# Third-party imports
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import redis
from neo4j import GraphDatabase
import asyncpg

# Local imports
from .vector_embeddings import PhysicsEmbeddingManager, EmbeddingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Types of search strategies"""
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    GRAPH_ENHANCED = "graph_enhanced"

@dataclass
class SearchResult:
    """Search result with metadata"""
    node_id: int
    content_type: str
    title: str
    content: str
    similarity_score: float
    rank: int
    metadata: Dict[str, Any]
    matched_terms: List[str] = None
    explanation: str = None

@dataclass
class SearchQuery:
    """Structured search query"""
    text: str
    search_type: SearchType = SearchType.HYBRID
    content_types: List[str] = None  # Filter by content types
    difficulty_level: str = None  # beginner, intermediate, advanced
    category: str = None  # physics category filter
    limit: int = 10
    min_similarity: float = 0.3
    boost_exact_matches: bool = True
    include_related: bool = True  # Include graph-related content

class PhysicsQueryProcessor:
    """Processes and analyzes physics queries for better search"""
    
    def __init__(self):
        # Physics-specific keywords and patterns
        self.physics_concepts = {
            'kinematics': ['velocity', 'acceleration', 'position', 'displacement', 'speed', 'motion'],
            'forces': ['force', 'newton', 'friction', 'tension', 'normal', 'weight', 'gravity'],
            'energy': ['energy', 'work', 'power', 'kinetic', 'potential', 'conservation'],
            'momentum': ['momentum', 'impulse', 'collision', 'conservation'],
            'rotational': ['torque', 'angular', 'rotation', 'moment', 'inertia'],
            'oscillations': ['oscillation', 'wave', 'frequency', 'amplitude', 'period'],
            'thermodynamics': ['temperature', 'heat', 'entropy', 'thermal', 'gas'],
            'electromagnetism': ['electric', 'magnetic', 'field', 'charge', 'current']
        }
        
        # Mathematical expressions patterns
        self.math_patterns = [
            r'[a-zA-Z]\s*=\s*[^,\n]+',  # Equations like v = at
            r'\d+[\+\-\*\/]\d+',         # Basic math operations
            r'√\([^)]+\)',               # Square roots
            r'\d+\^\d+',                 # Exponents
            r'sin|cos|tan|log|ln',       # Trigonometric/logarithmic functions
        ]
        
        # Units and quantities
        self.physics_units = [
            'm/s', 'km/h', 'm/s²', 'N', 'J', 'W', 'kg', 'm', 's', 'K', 'A', 'V', 'Ω',
            'Hz', 'Pa', 'C', 'F', 'H', 'T', 'Wb', 'rad', 'sr', 'mol', 'cd'
        ]
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze physics query to extract key information"""
        query_lower = query.lower()
        
        analysis = {
            'original_query': query,
            'identified_concepts': [],
            'math_expressions': [],
            'units_mentioned': [],
            'question_type': self._identify_question_type(query),
            'difficulty_indicators': self._identify_difficulty(query),
            'suggested_categories': []
        }
        
        # Identify physics concepts
        for category, concepts in self.physics_concepts.items():
            for concept in concepts:
                if concept in query_lower:
                    analysis['identified_concepts'].append(concept)
                    if category not in analysis['suggested_categories']:
                        analysis['suggested_categories'].append(category)
        
        # Extract mathematical expressions
        for pattern in self.math_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            analysis['math_expressions'].extend(matches)
        
        # Identify units
        for unit in self.physics_units:
            if unit in query:
                analysis['units_mentioned'].append(unit)
        
        return analysis
    
    def _identify_question_type(self, query: str) -> str:
        """Identify the type of physics question"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return 'definition'
        elif any(word in query_lower for word in ['how to', 'calculate', 'find', 'solve']):
            return 'calculation'
        elif any(word in query_lower for word in ['why', 'explain', 'reason']):
            return 'explanation'
        elif any(word in query_lower for word in ['example', 'problem', 'exercise']):
            return 'example'
        elif '?' in query:
            return 'question'
        else:
            return 'general'
    
    def _identify_difficulty(self, query: str) -> str:
        """Identify difficulty level from query"""
        query_lower = query.lower()
        
        beginner_indicators = ['basic', 'simple', 'introduction', 'beginner', 'elementary']
        advanced_indicators = ['advanced', 'complex', 'detailed', 'derive', 'proof']
        
        if any(indicator in query_lower for indicator in advanced_indicators):
            return 'advanced'
        elif any(indicator in query_lower for indicator in beginner_indicators):
            return 'beginner'
        else:
            return 'intermediate'
    
    def expand_query(self, query: str) -> str:
        """Expand query with related physics terms"""
        analysis = self.analyze_query(query)
        expanded_terms = [query]
        
        # Add synonyms and related terms
        for concept in analysis['identified_concepts']:
            if concept == 'velocity':
                expanded_terms.append('speed rate of change position')
            elif concept == 'acceleration':
                expanded_terms.append('rate of change velocity')
            elif concept == 'force':
                expanded_terms.append('newton push pull interaction')
            elif concept == 'energy':
                expanded_terms.append('work capacity joule')
            # Add more concept expansions as needed
        
        return ' '.join(expanded_terms)

class SemanticSearchEngine:
    """High-performance semantic search engine for physics content"""
    
    def __init__(self, embedding_manager: PhysicsEmbeddingManager,
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 redis_host: str = 'localhost', redis_port: int = 6379, redis_password: str = None,
                 postgres_config: dict = None):
        
        self.embedding_manager = embedding_manager
        self.query_processor = PhysicsQueryProcessor()
        
        # Database connections
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, password=redis_password, 
            decode_responses=False  # Keep binary for FAISS indices
        )
        self.postgres_config = postgres_config
        self.postgres_pool = None
        
        # Search indices
        self.faiss_indices = {}  # FAISS indices per content type
        self.tfidf_vectorizers = {}  # TF-IDF vectorizers
        self.tfidf_matrices = {}  # TF-IDF matrices
        self.content_metadata = {}  # Metadata for all content
        
        # Configuration
        self.config = EmbeddingConfig()
        
        # Cache
        self.search_cache = {}
        self.cache_expiry = 1800  # 30 minutes
    
    async def initialize(self, rebuild_indices: bool = False):
        """Initialize search engine and build indices"""
        logger.info("🚀 Initializing Semantic Search Engine")
        
        # Initialize PostgreSQL if config provided
        if self.postgres_config:
            try:
                self.postgres_pool = await asyncpg.create_pool(**self.postgres_config)
                logger.info("✅ PostgreSQL connection pool initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
        
        # Build or load search indices
        if rebuild_indices or not await self._load_indices_from_cache():
            await self.build_search_indices()
        
        logger.info("✅ Semantic search engine initialized")
    
    async def close(self):
        """Close database connections"""
        if self.neo4j_driver:
            await asyncio.to_thread(self.neo4j_driver.close)
        if self.redis_client:
            await asyncio.to_thread(self.redis_client.close)
        if self.postgres_pool:
            await self.postgres_pool.close()
        logger.info("🔒 Semantic search engine connections closed")
    
    async def build_search_indices(self):
        """Build FAISS and TF-IDF indices for fast search"""
        logger.info("🔨 Building search indices...")
        
        # Get all embeddings and content
        content_data = await self._load_all_content_with_embeddings()
        
        if not content_data:
            logger.error("❌ No content data found. Generate embeddings first.")
            return
        
        # Build indices for each content type
        for content_type, data in content_data.items():
            await self._build_faiss_index(content_type, data)
            await self._build_tfidf_index(content_type, data)
        
        # Cache indices
        await self._cache_indices()
        
        logger.info("✅ Search indices built successfully")
    
    async def _load_all_content_with_embeddings(self) -> Dict[str, Dict]:
        """Load all content with embeddings from database"""
        content_data = {}
        
        # Try PostgreSQL first (more efficient for large datasets)
        if self.postgres_pool:
            content_data = await self._load_from_postgres()
        
        # Fallback to Neo4j if PostgreSQL not available
        if not content_data:
            content_data = await self._load_from_neo4j()
        
        return content_data
    
    async def _load_from_postgres(self) -> Dict[str, Dict]:
        """Load content and embeddings from PostgreSQL"""
        content_data = {}
        
        # Determine active model
        model_name = self.embedding_manager.active_model
        table_name = f"node_embeddings_{model_name.replace('-', '_')}"
        
        try:
            async with self.postgres_pool.acquire() as conn:
                query = f"""
                SELECT node_id, content_type, content_text, embedding
                FROM {table_name}
                ORDER BY content_type, node_id
                """
                
                results = await conn.fetch(query)
                
                for row in results:
                    content_type = row['content_type']
                    
                    if content_type not in content_data:
                        content_data[content_type] = {
                            'embeddings': [],
                            'texts': [],
                            'metadata': []
                        }
                    
                    # Convert embedding back to numpy array
                    embedding = np.array(row['embedding'])
                    
                    content_data[content_type]['embeddings'].append(embedding)
                    content_data[content_type]['texts'].append(row['content_text'])
                    content_data[content_type]['metadata'].append({
                        'node_id': row['node_id'],
                        'content_type': content_type
                    })
                
                # Convert lists to numpy arrays
                for content_type in content_data:
                    content_data[content_type]['embeddings'] = np.array(
                        content_data[content_type]['embeddings']
                    )
                
                logger.info(f"📊 Loaded content from PostgreSQL: {len(results)} items")
                
        except Exception as e:
            logger.error(f"Failed to load from PostgreSQL: {e}")
            return {}
        
        return content_data
    
    async def _load_from_neo4j(self) -> Dict[str, Dict]:
        """Load content and embeddings from Neo4j"""
        model_name = self.embedding_manager.active_model
        embedding_field = f"embedding_{model_name.replace('-', '_')}"
        
        queries = {
            'concepts': f"""
                MATCH (c:Concept)
                WHERE c.{embedding_field} IS NOT NULL
                RETURN id(c) as node_id, c.name as name, c.description as description,
                       c.{embedding_field} as embedding, labels(c) as labels
            """,
            'formulas': f"""
                MATCH (f:Formula)
                WHERE f.{embedding_field} IS NOT NULL
                RETURN id(f) as node_id, f.name as name, f.expression as expression,
                       f.{embedding_field} as embedding, labels(f) as labels
            """,
            'problems': f"""
                MATCH (p:Problem)
                WHERE p.{embedding_field} IS NOT NULL
                RETURN id(p) as node_id, p.title as title, p.description as description,
                       p.{embedding_field} as embedding, labels(p) as labels
            """,
            'explanations': f"""
                MATCH (e:Explanation)
                WHERE e.{embedding_field} IS NOT NULL
                RETURN id(e) as node_id, e.title as title, e.content as content,
                       e.{embedding_field} as embedding, labels(e) as labels
            """
        }
        
        content_data = {}
        
        def run_query(query):
            with self.neo4j_driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result]
        
        for content_type, query in queries.items():
            try:
                results = await asyncio.to_thread(run_query, query)
                
                if results:
                    embeddings = []
                    texts = []
                    metadata = []
                    
                    for item in results:
                        embeddings.append(np.array(item['embedding']))
                        
                        # Create text representation
                        if content_type == 'concepts':
                            text = f"{item['name']}: {item['description']}"
                        elif content_type == 'formulas':
                            text = f"{item['name']}: {item.get('expression', '')}"
                        elif content_type == 'problems':
                            text = f"{item['title']}: {item['description']}"
                        elif content_type == 'explanations':
                            text = f"{item['title']}: {item['content']}"
                        
                        texts.append(text)
                        metadata.append({
                            'node_id': item['node_id'],
                            'content_type': content_type,
                            'original_data': item
                        })
                    
                    content_data[content_type] = {
                        'embeddings': np.array(embeddings),
                        'texts': texts,
                        'metadata': metadata
                    }
                
                logger.info(f"📊 Loaded {len(results)} {content_type} from Neo4j")
                
            except Exception as e:
                logger.error(f"Failed to load {content_type} from Neo4j: {e}")
        
        return content_data
    
    async def _build_faiss_index(self, content_type: str, data: Dict):
        """Build FAISS index for content type"""
        embeddings = data['embeddings']
        
        if len(embeddings) == 0:
            logger.warning(f"No embeddings found for {content_type}")
            return
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype(np.float32))
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity after normalization)
        index.add(embeddings.astype(np.float32))
        
        self.faiss_indices[content_type] = index
        self.content_metadata[content_type] = data['metadata']
        
        logger.info(f"🔍 Built FAISS index for {content_type}: {index.ntotal} vectors, dim={dimension}")
    
    async def _build_tfidf_index(self, content_type: str, data: Dict):
        """Build TF-IDF index for keyword search"""
        texts = data['texts']
        
        if not texts:
            logger.warning(f"No texts found for {content_type}")
            return
        
        # Create TF-IDF vectorizer with physics-specific preprocessing
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            lowercase=True
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            self.tfidf_vectorizers[content_type] = vectorizer
            self.tfidf_matrices[content_type] = tfidf_matrix
            
            logger.info(f"📝 Built TF-IDF index for {content_type}: {tfidf_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Failed to build TF-IDF index for {content_type}: {e}")
    
    async def _cache_indices(self):
        """Cache indices in Redis for faster loading"""
        try:
            # Note: In production, you might want to use a more efficient serialization
            # format like pickle or custom binary format for FAISS indices
            logger.info("💾 Caching search indices...")
            
            cache_key = f"search_indices_{self.embedding_manager.active_model}"
            cache_data = {
                'content_types': list(self.faiss_indices.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
            await asyncio.to_thread(
                self.redis_client.setex,
                cache_key,
                self.cache_expiry * 2,  # Cache indices longer
                json.dumps(cache_data)
            )
            
            logger.info("✅ Indices cached")
            
        except Exception as e:
            logger.warning(f"Failed to cache indices: {e}")
    
    async def _load_indices_from_cache(self) -> bool:
        """Load indices from cache if available"""
        try:
            cache_key = f"search_indices_{self.embedding_manager.active_model}"
            cached_data = await asyncio.to_thread(self.redis_client.get, cache_key)
            
            if cached_data:
                # In a full implementation, you would deserialize FAISS indices here
                # For now, just check if cache exists
                logger.info("📋 Found cached indices")
                return False  # Always rebuild for now
            
        except Exception as e:
            logger.warning(f"Failed to load cached indices: {e}")
        
        return False
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic search with optional hybrid approach"""
        
        # Check cache first
        cache_key = f"search:{hash(query.text)}:{query.search_type.value}:{query.limit}"
        cached_results = await self._get_cached_results(cache_key)
        if cached_results:
            return cached_results
        
        # Analyze query
        query_analysis = self.query_processor.analyze_query(query.text)
        
        # Get query embedding
        expanded_query = self.query_processor.expand_query(query.text)
        query_embedding = await self.embedding_manager.generate_embeddings(
            expanded_query, self.embedding_manager.active_model
        )
        
        # Perform search based on strategy
        if query.search_type == SearchType.SEMANTIC_ONLY:
            results = await self._semantic_search(query, query_embedding)
        elif query.search_type == SearchType.KEYWORD_ONLY:
            results = await self._keyword_search(query)
        elif query.search_type == SearchType.HYBRID:
            results = await self._hybrid_search(query, query_embedding, query_analysis)
        elif query.search_type == SearchType.GRAPH_ENHANCED:
            results = await self._graph_enhanced_search(query, query_embedding, query_analysis)
        else:
            results = await self._hybrid_search(query, query_embedding, query_analysis)
        
        # Filter and rank results
        filtered_results = await self._filter_and_rank_results(results, query, query_analysis)
        
        # Cache results
        await self._cache_results(cache_key, filtered_results)
        
        return filtered_results
    
    async def _semantic_search(self, query: SearchQuery, query_embedding: np.ndarray) -> List[SearchResult]:
        """Pure semantic search using vector similarity"""
        all_results = []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search in each content type
        content_types_to_search = query.content_types or list(self.faiss_indices.keys())
        
        for content_type in content_types_to_search:
            if content_type not in self.faiss_indices:
                continue
            
            index = self.faiss_indices[content_type]
            metadata = self.content_metadata[content_type]
            
            # Search for similar vectors
            k = min(query.limit * 2, index.ntotal)  # Get more results for filtering
            similarities, indices = index.search(query_embedding, k)
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity < query.min_similarity:
                    continue
                
                if idx >= len(metadata):
                    continue
                
                meta = metadata[idx]
                
                result = SearchResult(
                    node_id=meta['node_id'],
                    content_type=content_type,
                    title=meta.get('title', f"{content_type}_{meta['node_id']}"),
                    content=self._get_content_text(meta),
                    similarity_score=float(similarity),
                    rank=i + 1,
                    metadata=meta
                )
                
                all_results.append(result)
        
        return all_results
    
    async def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Keyword-based search using TF-IDF"""
        all_results = []
        
        content_types_to_search = query.content_types or list(self.tfidf_vectorizers.keys())
        
        for content_type in content_types_to_search:
            if content_type not in self.tfidf_vectorizers:
                continue
            
            vectorizer = self.tfidf_vectorizers[content_type]
            tfidf_matrix = self.tfidf_matrices[content_type]
            metadata = self.content_metadata[content_type]
            
            # Transform query
            query_vector = vectorizer.transform([query.text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-query.limit*2:][::-1]
            
            for rank, idx in enumerate(top_indices):
                similarity = similarities[idx]
                
                if similarity < query.min_similarity:
                    continue
                
                meta = metadata[idx]
                
                result = SearchResult(
                    node_id=meta['node_id'],
                    content_type=content_type,
                    title=meta.get('title', f"{content_type}_{meta['node_id']}"),
                    content=self._get_content_text(meta),
                    similarity_score=similarity,
                    rank=rank + 1,
                    metadata=meta
                )
                
                all_results.append(result)
        
        return all_results
    
    async def _hybrid_search(self, query: SearchQuery, query_embedding: np.ndarray, 
                           query_analysis: Dict) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword approaches"""
        
        # Get semantic results
        semantic_results = await self._semantic_search(query, query_embedding)
        
        # Get keyword results
        keyword_results = await self._keyword_search(query)
        
        # Combine and re-rank results
        combined_results = self._combine_search_results(
            semantic_results, keyword_results, 
            semantic_weight=0.7, keyword_weight=0.3
        )
        
        return combined_results
    
    async def _graph_enhanced_search(self, query: SearchQuery, query_embedding: np.ndarray,
                                   query_analysis: Dict) -> List[SearchResult]:
        """Graph-enhanced search using knowledge graph relationships"""
        
        # Start with hybrid search
        base_results = await self._hybrid_search(query, query_embedding, query_analysis)
        
        if not base_results or not query.include_related:
            return base_results
        
        # Expand results using graph relationships
        enhanced_results = []
        seen_node_ids = set()
        
        for result in base_results:
            enhanced_results.append(result)
            seen_node_ids.add(result.node_id)
            
            # Find related nodes in the graph
            related_nodes = await self._get_related_graph_nodes(
                result.node_id, result.content_type, max_related=3
            )
            
            for related_node in related_nodes:
                if related_node['node_id'] not in seen_node_ids:
                    related_result = SearchResult(
                        node_id=related_node['node_id'],
                        content_type=related_node['content_type'],
                        title=related_node.get('title', ''),
                        content=related_node.get('content', ''),
                        similarity_score=result.similarity_score * 0.8,  # Reduce score for related
                        rank=len(enhanced_results) + 1,
                        metadata=related_node,
                        explanation=f"Related to: {result.title}"
                    )
                    enhanced_results.append(related_result)
                    seen_node_ids.add(related_node['node_id'])
        
        return enhanced_results
    
    async def _get_related_graph_nodes(self, node_id: int, content_type: str, 
                                     max_related: int = 3) -> List[Dict]:
        """Get related nodes from the knowledge graph"""
        
        query = """
        MATCH (n)-[r]-(related)
        WHERE id(n) = $node_id
        RETURN id(related) as node_id, labels(related) as labels,
               related.name as name, related.title as title,
               related.description as description, related.content as content,
               type(r) as relationship_type
        LIMIT $max_related
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {'node_id': node_id, 'max_related': max_related})
                return [record.data() for record in result]
        
        try:
            results = await asyncio.to_thread(run_query)
            
            related_nodes = []
            for item in results:
                content_type_from_labels = 'concept'  # Default
                if 'Formula' in item['labels']:
                    content_type_from_labels = 'formula'
                elif 'Problem' in item['labels']:
                    content_type_from_labels = 'problem'
                elif 'Explanation' in item['labels']:
                    content_type_from_labels = 'explanation'
                
                related_nodes.append({
                    'node_id': item['node_id'],
                    'content_type': content_type_from_labels,
                    'title': item.get('title') or item.get('name', ''),
                    'content': item.get('content') or item.get('description', ''),
                    'relationship_type': item['relationship_type']
                })
            
            return related_nodes
            
        except Exception as e:
            logger.error(f"Failed to get related nodes: {e}")
            return []
    
    def _combine_search_results(self, semantic_results: List[SearchResult], 
                               keyword_results: List[SearchResult],
                               semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> List[SearchResult]:
        """Combine semantic and keyword search results with weighted scoring"""
        
        # Create mapping of results by node_id
        semantic_map = {r.node_id: r for r in semantic_results}
        keyword_map = {r.node_id: r for r in keyword_results}
        
        # Combine results
        all_node_ids = set(semantic_map.keys()) | set(keyword_map.keys())
        combined_results = []
        
        for node_id in all_node_ids:
            semantic_result = semantic_map.get(node_id)
            keyword_result = keyword_map.get(node_id)
            
            if semantic_result and keyword_result:
                # Both found - combine scores
                combined_score = (
                    semantic_weight * semantic_result.similarity_score +
                    keyword_weight * keyword_result.similarity_score
                )
                result = semantic_result  # Use semantic result as base
                result.similarity_score = combined_score
                result.matched_terms = getattr(keyword_result, 'matched_terms', [])
                
            elif semantic_result:
                # Only semantic
                result = semantic_result
                result.similarity_score *= semantic_weight
                
            else:
                # Only keyword
                result = keyword_result
                result.similarity_score *= keyword_weight
            
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
    
    async def _filter_and_rank_results(self, results: List[SearchResult], 
                                     query: SearchQuery, query_analysis: Dict) -> List[SearchResult]:
        """Filter and rank results based on query parameters"""
        
        filtered_results = []
        
        for result in results:
            # Apply filters
            if query.content_types and result.content_type not in query.content_types:
                continue
            
            # Difficulty level filtering (if metadata available)
            if query.difficulty_level:
                result_difficulty = result.metadata.get('difficulty_level')
                if result_difficulty and result_difficulty != query.difficulty_level:
                    continue
            
            # Category filtering
            if query.category:
                result_category = result.metadata.get('category')
                if result_category and result_category != query.category:
                    continue
            
            # Boost exact matches
            if query.boost_exact_matches:
                query_lower = query.text.lower()
                content_lower = result.content.lower()
                
                if query_lower in content_lower:
                    result.similarity_score *= 1.2  # 20% boost
                
                # Boost for matching identified concepts
                for concept in query_analysis.get('identified_concepts', []):
                    if concept in content_lower:
                        result.similarity_score *= 1.1  # 10% boost per concept
            
            filtered_results.append(result)
        
        # Sort by similarity score
        filtered_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit results
        filtered_results = filtered_results[:query.limit]
        
        # Update ranks
        for i, result in enumerate(filtered_results):
            result.rank = i + 1
        
        return filtered_results
    
    def _get_content_text(self, metadata: Dict) -> str:
        """Extract displayable text from metadata"""
        if 'original_data' in metadata:
            data = metadata['original_data']
            if 'description' in data:
                return data['description']
            elif 'content' in data:
                return data['content']
            elif 'expression' in data:
                return data['expression']
        
        return str(metadata.get('node_id', ''))
    
    async def _get_cached_results(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        try:
            cached_data = await asyncio.to_thread(self.redis_client.get, cache_key)
            if cached_data:
                # In a full implementation, you would deserialize SearchResult objects
                return None  # Skip caching for now
        except Exception as e:
            logger.warning(f"Failed to get cached results: {e}")
        return None
    
    async def _cache_results(self, cache_key: str, results: List[SearchResult]):
        """Cache search results"""
        try:
            # In a full implementation, you would serialize SearchResult objects
            await asyncio.to_thread(
                self.redis_client.setex,
                cache_key,
                self.cache_expiry,
                "cached"  # Placeholder
            )
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
    
    # Convenience search methods
    
    async def search_concepts(self, query_text: str, limit: int = 10) -> List[SearchResult]:
        """Search for physics concepts"""
        query = SearchQuery(
            text=query_text,
            content_types=['concepts'],
            limit=limit,
            search_type=SearchType.HYBRID
        )
        return await self.search(query)
    
    async def search_problems(self, query_text: str, difficulty: str = None, limit: int = 10) -> List[SearchResult]:
        """Search for physics problems"""
        query = SearchQuery(
            text=query_text,
            content_types=['problems'],
            difficulty_level=difficulty,
            limit=limit,
            search_type=SearchType.HYBRID
        )
        return await self.search(query)
    
    async def search_formulas(self, query_text: str, limit: int = 10) -> List[SearchResult]:
        """Search for physics formulas"""
        query = SearchQuery(
            text=query_text,
            content_types=['formulas'],
            limit=limit,
            search_type=SearchType.SEMANTIC_ONLY  # Formulas benefit more from semantic search
        )
        return await self.search(query)
    
    async def search_explanations(self, query_text: str, limit: int = 10) -> List[SearchResult]:
        """Search for physics explanations"""
        query = SearchQuery(
            text=query_text,
            content_types=['explanations'],
            limit=limit,
            search_type=SearchType.HYBRID
        )
        return await self.search(query)

# Example usage and testing functions
async def test_semantic_search():
    """Test function for semantic search"""
    from .vector_embeddings import get_physics_embedding_manager
    
    # Configuration
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "physics_graph_password_2024"
    
    postgres_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'physics_assistant',
        'user': 'physics_user',
        'password': 'physics_secure_password_2024'
    }
    
    # Initialize embedding manager
    async with get_physics_embedding_manager(
        neo4j_uri, neo4j_user, neo4j_password,
        postgres_config=postgres_config,
        models_to_load=['sentence_transformer']
    ) as embedding_manager:
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            embedding_manager, neo4j_uri, neo4j_user, neo4j_password,
            postgres_config=postgres_config
        )
        
        await search_engine.initialize()
        
        try:
            # Test different types of searches
            test_queries = [
                "What is velocity?",
                "How to calculate kinetic energy?",
                "Newton's laws of motion",
                "F = ma formula",
                "Conservation of momentum problems"
            ]
            
            for query_text in test_queries:
                print(f"\n🔍 Searching: '{query_text}'")
                
                query = SearchQuery(
                    text=query_text,
                    search_type=SearchType.HYBRID,
                    limit=5
                )
                
                results = await search_engine.search(query)
                
                print(f"📊 Found {len(results)} results:")
                for i, result in enumerate(results[:3]):  # Show top 3
                    print(f"  {i+1}. {result.title} (score: {result.similarity_score:.3f})")
                    print(f"     Type: {result.content_type}")
                    print(f"     Content: {result.content[:100]}...")
                    
        finally:
            await search_engine.close()

if __name__ == "__main__":
    asyncio.run(test_semantic_search())