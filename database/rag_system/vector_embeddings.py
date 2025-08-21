#!/usr/bin/env python3
"""
Vector Embeddings Generation System for Physics Content
Generates and manages embeddings for all physics concepts, formulas, problems, and explanations
"""
import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import hashlib
from contextlib import asynccontextmanager

# Third-party imports
import openai
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import redis
from neo4j import GraphDatabase
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingConfig:
    """Configuration for embedding models and parameters"""
    
    def __init__(self):
        self.models = {
            'sentence_transformer': 'all-MiniLM-L6-v2',  # Fast, good for general text
            'physics_optimized': 'sentence-transformers/all-mpnet-base-v2',  # Better for domain-specific
            'openai_model': 'text-embedding-ada-002',  # High quality but requires API key
            'math_specialized': 'allenai/specter2',  # Scientific papers/equations
        }
        
        self.default_model = 'sentence_transformer'
        self.embedding_dimensions = {
            'sentence_transformer': 384,
            'physics_optimized': 768,
            'openai_model': 1536,
            'math_specialized': 768
        }
        
        # Physics-specific preprocessing
        self.physics_terms_substitution = {
            'velocity': 'speed rate of change position',
            'acceleration': 'rate of change velocity',
            'momentum': 'mass times velocity',
            'force': 'mass times acceleration',
            'energy': 'capacity to do work',
            'work': 'force times displacement',
            'power': 'work per time rate of energy transfer',
            'torque': 'rotational force moment',
            'angular velocity': 'rotational speed rate of angular change',
            'frequency': 'cycles per second oscillations',
            'wavelength': 'distance between wave peaks',
            'amplitude': 'maximum displacement from equilibrium'
        }

class EmbeddingModel:
    """Base class for embedding models"""
    
    def __init__(self, model_name: str, config: EmbeddingConfig):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def load_model(self):
        """Load the embedding model"""
        raise NotImplementedError
    
    async def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts into embeddings"""
        raise NotImplementedError
    
    def preprocess_physics_text(self, text: str) -> str:
        """Preprocess physics text for better embeddings"""
        text_lower = text.lower()
        
        # Substitute physics terms with expanded descriptions
        for term, description in self.config.physics_terms_substitution.items():
            if term in text_lower:
                text_lower = text_lower.replace(term, f"{term} {description}")
        
        # Handle mathematical notation (basic cleanup)
        text_lower = text_lower.replace('δ', 'delta change in ')
        text_lower = text_lower.replace('∆', 'delta change in ')
        text_lower = text_lower.replace('∂', 'partial derivative ')
        text_lower = text_lower.replace('∫', 'integral ')
        text_lower = text_lower.replace('∑', 'sum ')
        text_lower = text_lower.replace('θ', 'theta angle ')
        text_lower = text_lower.replace('ω', 'omega angular frequency ')
        text_lower = text_lower.replace('α', 'alpha angular acceleration ')
        
        return text_lower

class SentenceTransformerModel(EmbeddingModel):
    """Sentence-BERT based embedding model"""
    
    async def load_model(self):
        """Load SentenceTransformer model"""
        try:
            model_name = self.config.models.get(self.model_name, self.config.models['sentence_transformer'])
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"✅ Loaded SentenceTransformer model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load SentenceTransformer: {e}")
            return False
    
    async def encode(self, texts: Union[str, List[str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts using SentenceTransformer"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess physics text
        processed_texts = [self.preprocess_physics_text(text) for text in texts]
        
        try:
            embeddings = await asyncio.to_thread(
                self.model.encode,
                processed_texts,
                batch_size=batch_size,
                show_progress_bar=len(processed_texts) > 100,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model"""
    
    def __init__(self, model_name: str, config: EmbeddingConfig, api_key: Optional[str] = None):
        super().__init__(model_name, config)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
    
    async def load_model(self):
        """Initialize OpenAI client"""
        if not self.api_key:
            logger.warning("⚠️ OpenAI API key not found, skipping OpenAI model")
            return False
        
        try:
            # Test the API
            await asyncio.to_thread(
                openai.embeddings.create,
                input="test",
                model=self.config.models['openai_model']
            )
            logger.info("✅ OpenAI embedding model initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI model: {e}")
            return False
    
    async def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using OpenAI API"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess physics text
        processed_texts = [self.preprocess_physics_text(text) for text in texts]
        
        try:
            # Process in batches to avoid rate limits
            batch_size = 20
            all_embeddings = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                response = await asyncio.to_thread(
                    openai.embeddings.create,
                    input=batch,
                    model=self.config.models['openai_model']
                )
                
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if i + batch_size < len(processed_texts):
                    await asyncio.sleep(0.1)
            
            return np.array(all_embeddings)
        except Exception as e:
            logger.error(f"Failed to encode texts with OpenAI: {e}")
            raise

class PhysicsEmbeddingManager:
    """Manages embeddings for physics content in the knowledge graph"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 redis_host: str = 'localhost', redis_port: int = 6379, redis_password: str = None,
                 postgres_config: dict = None):
        self.config = EmbeddingConfig()
        
        # Database connections
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, password=redis_password, 
            decode_responses=True
        )
        self.postgres_config = postgres_config
        self.postgres_pool = None
        
        # Embedding models
        self.models: Dict[str, EmbeddingModel] = {}
        self.active_model = None
        
        # Caching
        self.embedding_cache = {}
        self.cache_expiry = 3600  # 1 hour
    
    async def initialize(self, models_to_load: List[str] = None):
        """Initialize embedding models and database connections"""
        if models_to_load is None:
            models_to_load = ['sentence_transformer']
        
        logger.info("🚀 Initializing Physics Embedding Manager")
        
        # Initialize PostgreSQL if config provided
        if self.postgres_config:
            try:
                self.postgres_pool = await asyncpg.create_pool(**self.postgres_config)
                logger.info("✅ PostgreSQL connection pool initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
        
        # Load embedding models
        for model_name in models_to_load:
            try:
                if model_name == 'openai_model':
                    model = OpenAIEmbeddingModel(model_name, self.config)
                else:
                    model = SentenceTransformerModel(model_name, self.config)
                
                success = await model.load_model()
                if success:
                    self.models[model_name] = model
                    if not self.active_model:
                        self.active_model = model_name
                        
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        
        if not self.models:
            raise RuntimeError("No embedding models could be loaded")
        
        logger.info(f"✅ Initialized {len(self.models)} embedding models")
        logger.info(f"🎯 Active model: {self.active_model}")
    
    async def close(self):
        """Close database connections"""
        if self.neo4j_driver:
            await asyncio.to_thread(self.neo4j_driver.close)
        if self.redis_client:
            await asyncio.to_thread(self.redis_client.close)
        if self.postgres_pool:
            await self.postgres_pool.close()
        logger.info("🔒 Embedding manager connections closed")
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content caching"""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _cache_embedding(self, content_hash: str, embedding: np.ndarray, model_name: str):
        """Cache embedding in Redis"""
        try:
            cache_key = f"embedding:{model_name}:{content_hash}"
            embedding_json = json.dumps(embedding.tolist())
            await asyncio.to_thread(
                self.redis_client.setex,
                cache_key,
                self.cache_expiry,
                embedding_json
            )
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    async def _get_cached_embedding(self, content_hash: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding from Redis"""
        try:
            cache_key = f"embedding:{model_name}:{content_hash}"
            cached_data = await asyncio.to_thread(self.redis_client.get, cache_key)
            if cached_data:
                return np.array(json.loads(cached_data))
        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
        return None
    
    async def generate_embeddings(self, texts: Union[str, List[str]], model_name: str = None) -> np.ndarray:
        """Generate embeddings for texts"""
        if model_name is None:
            model_name = self.active_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache for individual texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            content_hash = self._get_content_hash(text)
            cached_embedding = await self._get_cached_embedding(content_hash, model_name)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await model.encode(uncached_texts)
            
            # Cache new embeddings and fill placeholders
            for i, (embedding, original_idx) in enumerate(zip(new_embeddings, uncached_indices)):
                embeddings[original_idx] = embedding
                
                # Cache the embedding
                content_hash = self._get_content_hash(uncached_texts[i])
                await self._cache_embedding(content_hash, embedding, model_name)
        
        return np.array(embeddings)
    
    async def get_all_graph_content(self) -> Dict[str, List[Dict]]:
        """Extract all content from Neo4j knowledge graph"""
        queries = {
            'concepts': """
                MATCH (c:Concept)
                RETURN id(c) as node_id, c.name as name, c.description as description,
                       c.category as category, c.difficulty_level as difficulty, labels(c) as labels
            """,
            'formulas': """
                MATCH (f:Formula)
                RETURN id(f) as node_id, f.name as name, f.expression as expression,
                       f.description as description, f.units as units, labels(f) as labels
            """,
            'problems': """
                MATCH (p:Problem)
                RETURN id(p) as node_id, p.title as title, p.description as description,
                       p.difficulty_level as difficulty, p.problem_statement as statement, labels(p) as labels
            """,
            'explanations': """
                MATCH (e:Explanation)
                RETURN id(e) as node_id, e.title as title, e.content as content,
                       e.explanation_type as type, labels(e) as labels
            """
        }
        
        content = {}
        
        def run_query(query):
            with self.neo4j_driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result]
        
        for content_type, query in queries.items():
            try:
                results = await asyncio.to_thread(run_query, query)
                content[content_type] = results
                logger.info(f"📊 Extracted {len(results)} {content_type}")
            except Exception as e:
                logger.error(f"Failed to extract {content_type}: {e}")
                content[content_type] = []
        
        return content
    
    async def generate_content_embeddings(self, model_name: str = None) -> Dict[str, Dict]:
        """Generate embeddings for all graph content"""
        if model_name is None:
            model_name = self.active_model
        
        logger.info(f"🔄 Generating embeddings using model: {model_name}")
        
        # Get all content from graph
        content = await self.get_all_graph_content()
        
        embeddings_data = {}
        total_items = sum(len(items) for items in content.values())
        processed_items = 0
        
        for content_type, items in content.items():
            if not items:
                continue
            
            logger.info(f"🔄 Processing {len(items)} {content_type}...")
            
            # Prepare texts for embedding
            texts = []
            metadata = []
            
            for item in items:
                # Create comprehensive text representation
                if content_type == 'concepts':
                    text = f"Physics concept: {item['name']}. Description: {item['description']}. Category: {item.get('category', 'unknown')}."
                elif content_type == 'formulas':
                    text = f"Physics formula: {item['name']}. Expression: {item.get('expression', '')}. Description: {item['description']}. Units: {item.get('units', '')}."
                elif content_type == 'problems':
                    text = f"Physics problem: {item['title']}. Description: {item['description']}. Statement: {item.get('statement', '')}."
                elif content_type == 'explanations':
                    text = f"Physics explanation: {item['title']}. Content: {item['content']}. Type: {item.get('type', '')}."
                else:
                    text = str(item)
                
                texts.append(text)
                metadata.append({
                    'node_id': item['node_id'],
                    'content_type': content_type,
                    'original_data': item
                })
            
            # Generate embeddings
            try:
                embeddings = await self.generate_embeddings(texts, model_name)
                
                embeddings_data[content_type] = {
                    'embeddings': embeddings,
                    'metadata': metadata,
                    'texts': texts
                }
                
                processed_items += len(items)
                logger.info(f"✅ Generated {len(embeddings)} embeddings for {content_type} ({processed_items}/{total_items})")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for {content_type}: {e}")
        
        logger.info(f"🎉 Generated embeddings for {processed_items} total items")
        return embeddings_data
    
    async def store_embeddings_in_neo4j(self, embeddings_data: Dict[str, Dict], model_name: str = None):
        """Store embeddings back in Neo4j nodes"""
        if model_name is None:
            model_name = self.active_model
        
        logger.info(f"💾 Storing embeddings in Neo4j for model: {model_name}")
        
        def store_batch(node_updates):
            with self.neo4j_driver.session() as session:
                for node_id, embedding in node_updates:
                    # Convert numpy array to list for JSON storage
                    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    
                    query = f"""
                    MATCH (n) WHERE id(n) = $node_id
                    SET n.embedding_{model_name.replace('-', '_')} = $embedding,
                        n.embedding_model = $model_name,
                        n.embedding_updated_at = $updated_at
                    RETURN n
                    """
                    
                    session.run(query, {
                        'node_id': node_id,
                        'embedding': embedding_list,
                        'model_name': model_name,
                        'updated_at': datetime.now().isoformat()
                    })
        
        total_stored = 0
        batch_size = 50  # Process in batches to avoid memory issues
        
        for content_type, data in embeddings_data.items():
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            logger.info(f"💾 Storing {len(embeddings)} embeddings for {content_type}...")
            
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                
                node_updates = [
                    (meta['node_id'], embedding)
                    for meta, embedding in zip(batch_metadata, batch_embeddings)
                ]
                
                try:
                    await asyncio.to_thread(store_batch, node_updates)
                    total_stored += len(node_updates)
                except Exception as e:
                    logger.error(f"Failed to store batch for {content_type}: {e}")
        
        logger.info(f"✅ Stored {total_stored} embeddings in Neo4j")
    
    async def store_embeddings_in_postgres(self, embeddings_data: Dict[str, Dict], model_name: str = None):
        """Store embeddings in PostgreSQL for efficient vector search"""
        if not self.postgres_pool or model_name is None:
            logger.warning("PostgreSQL not available or model_name not specified")
            return
        
        if model_name is None:
            model_name = self.active_model
        
        logger.info(f"💾 Storing embeddings in PostgreSQL for model: {model_name}")
        
        # Create embeddings table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS node_embeddings_{model_name.replace('-', '_')} (
            id SERIAL PRIMARY KEY,
            node_id BIGINT NOT NULL,
            content_type VARCHAR(50) NOT NULL,
            content_text TEXT NOT NULL,
            embedding vector({self.config.embedding_dimensions.get(model_name, 384)}) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node_id, model_name)
        );
        
        CREATE INDEX IF NOT EXISTS idx_embeddings_{model_name.replace('-', '_')}_node_id 
        ON node_embeddings_{model_name.replace('-', '_')} (node_id);
        
        CREATE INDEX IF NOT EXISTS idx_embeddings_{model_name.replace('-', '_')}_content_type 
        ON node_embeddings_{model_name.replace('-', '_')} (content_type);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(create_table_query)
        
        # Insert embeddings
        total_stored = 0
        table_name = f"node_embeddings_{model_name.replace('-', '_')}"
        
        for content_type, data in embeddings_data.items():
            embeddings = data['embeddings']
            metadata = data['metadata']
            texts = data['texts']
            
            logger.info(f"💾 Storing {len(embeddings)} {content_type} embeddings in PostgreSQL...")
            
            # Prepare batch insert data
            insert_data = []
            for embedding, meta, text in zip(embeddings, metadata, texts):
                insert_data.append((
                    meta['node_id'],
                    content_type,
                    text,
                    embedding.tolist(),  # Convert to list for JSON storage
                    model_name
                ))
            
            # Batch insert
            insert_query = f"""
            INSERT INTO {table_name} (node_id, content_type, content_text, embedding, model_name)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (node_id, model_name) DO UPDATE SET
                content_text = EXCLUDED.content_text,
                embedding = EXCLUDED.embedding,
                created_at = CURRENT_TIMESTAMP
            """
            
            try:
                async with self.postgres_pool.acquire() as conn:
                    await conn.executemany(insert_query, insert_data)
                total_stored += len(insert_data)
                
            except Exception as e:
                logger.error(f"Failed to store {content_type} embeddings in PostgreSQL: {e}")
        
        logger.info(f"✅ Stored {total_stored} embeddings in PostgreSQL")

# Async context manager
@asynccontextmanager
async def get_physics_embedding_manager(
    neo4j_uri: str, neo4j_user: str, neo4j_password: str,
    redis_host: str = 'localhost', redis_port: int = 6379, redis_password: str = None,
    postgres_config: dict = None,
    models_to_load: List[str] = None
):
    """Context manager for physics embedding manager"""
    manager = PhysicsEmbeddingManager(
        neo4j_uri, neo4j_user, neo4j_password,
        redis_host, redis_port, redis_password,
        postgres_config
    )
    
    try:
        await manager.initialize(models_to_load)
        yield manager
    finally:
        await manager.close()

# CLI interface for generating embeddings
async def main():
    """Main function to generate and store embeddings"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate physics content embeddings')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687')
    parser.add_argument('--neo4j-user', default='neo4j')
    parser.add_argument('--neo4j-password', default='physics_graph_password_2024')
    parser.add_argument('--redis-host', default='localhost')
    parser.add_argument('--redis-port', type=int, default=6379)
    parser.add_argument('--redis-password', default='redis_secure_password_2024')
    parser.add_argument('--model', default='sentence_transformer',
                      choices=['sentence_transformer', 'physics_optimized', 'openai_model'])
    parser.add_argument('--store-neo4j', action='store_true', help='Store embeddings in Neo4j')
    parser.add_argument('--store-postgres', action='store_true', help='Store embeddings in PostgreSQL')
    
    args = parser.parse_args()
    
    postgres_config = None
    if args.store_postgres:
        postgres_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'physics_assistant',
            'user': 'physics_user',
            'password': 'physics_secure_password_2024'
        }
    
    async with get_physics_embedding_manager(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password,
        args.redis_host, args.redis_port, args.redis_password,
        postgres_config, [args.model]
    ) as manager:
        
        # Generate embeddings
        embeddings_data = await manager.generate_content_embeddings(args.model)
        
        # Store embeddings
        if args.store_neo4j:
            await manager.store_embeddings_in_neo4j(embeddings_data, args.model)
        
        if args.store_postgres:
            await manager.store_embeddings_in_postgres(embeddings_data, args.model)
        
        logger.info("🎉 Embedding generation complete!")

if __name__ == "__main__":
    asyncio.run(main())