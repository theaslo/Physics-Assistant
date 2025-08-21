#!/usr/bin/env python3
"""
Graph-Enhanced Retrieval Algorithms for Physics Content
Combines vector similarity with graph traversal for comprehensive content retrieval
"""
import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import heapq

# Third-party imports
import neo4j
from neo4j import GraphDatabase
import redis

# Local imports
from .semantic_search import SearchResult, SearchQuery, SearchType, SemanticSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphTraversalStrategy(Enum):
    """Graph traversal strategies for retrieval"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    RANDOM_WALK = "random_walk"
    PERSONALIZED_PAGERANK = "personalized_pagerank"
    SHORTEST_PATH = "shortest_path"
    CONCEPT_HIERARCHY = "concept_hierarchy"

@dataclass
class GraphNode:
    """Representation of a node in the physics knowledge graph"""
    node_id: int
    node_type: str
    name: str
    description: str = ""
    properties: Dict[str, Any] = None
    embedding: np.ndarray = None
    relationships: List[Dict] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.relationships is None:
            self.relationships = []

@dataclass
class GraphPath:
    """Represents a path through the knowledge graph"""
    nodes: List[GraphNode]
    relationships: List[Dict]
    path_weight: float
    semantic_score: float
    explanation: str = ""
    
    @property
    def length(self) -> int:
        return len(self.nodes)
    
    @property
    def total_score(self) -> float:
        return (self.semantic_score * 0.7) + (self.path_weight * 0.3)

class EducationalPathfinder:
    """Finds educational paths through physics concepts based on learning objectives"""
    
    def __init__(self, neo4j_driver: neo4j.Driver):
        self.neo4j_driver = neo4j_driver
        self.graph_cache = {}
        self.learning_sequences = {
            'mechanics': [
                'kinematics', 'forces', 'energy', 'momentum', 'rotational_motion'
            ],
            'waves_oscillations': [
                'oscillations', 'waves', 'sound', 'interference'
            ],
            'thermodynamics': [
                'temperature', 'heat', 'thermal_processes', 'entropy'
            ],
            'electromagnetism': [
                'electric_fields', 'magnetic_fields', 'electromagnetic_induction'
            ]
        }
    
    async def find_learning_path(self, start_concept: str, end_concept: str, 
                                student_level: str = 'beginner') -> List[GraphPath]:
        """Find optimal learning path between two concepts"""
        
        # Get nodes for start and end concepts
        start_nodes = await self._find_concept_nodes(start_concept)
        end_nodes = await self._find_concept_nodes(end_concept)
        
        if not start_nodes or not end_nodes:
            logger.warning(f"Could not find nodes for concepts: {start_concept} -> {end_concept}")
            return []
        
        # Find paths between all combinations
        all_paths = []
        
        for start_node in start_nodes:
            for end_node in end_nodes:
                paths = await self._find_concept_paths(
                    start_node, end_node, student_level, max_paths=3
                )
                all_paths.extend(paths)
        
        # Rank paths by educational value
        ranked_paths = self._rank_educational_paths(all_paths, student_level)
        
        return ranked_paths[:5]  # Return top 5 paths
    
    async def find_prerequisite_chain(self, concept: str, max_depth: int = 3) -> List[GraphPath]:
        """Find prerequisite concepts needed to understand a given concept"""
        
        concept_nodes = await self._find_concept_nodes(concept)
        if not concept_nodes:
            return []
        
        prerequisite_chains = []
        
        for node in concept_nodes:
            chains = await self._traverse_prerequisites(node, max_depth)
            prerequisite_chains.extend(chains)
        
        return self._rank_educational_paths(prerequisite_chains, 'beginner')
    
    async def find_related_concepts(self, concept: str, relationship_types: List[str] = None,
                                  max_hops: int = 2) -> List[GraphNode]:
        """Find concepts related to a given concept"""
        
        if relationship_types is None:
            relationship_types = ['RELATED_TO', 'APPLIES_CONCEPT', 'PREREQUISITE_FOR']
        
        concept_nodes = await self._find_concept_nodes(concept)
        if not concept_nodes:
            return []
        
        related_nodes = []
        visited = set()
        
        for start_node in concept_nodes:
            related = await self._find_related_nodes(
                start_node, relationship_types, max_hops, visited
            )
            related_nodes.extend(related)
        
        return list({node.node_id: node for node in related_nodes}.values())
    
    async def _find_concept_nodes(self, concept_name: str) -> List[GraphNode]:
        """Find all nodes matching a concept name"""
        
        query = """
        MATCH (n:Concept)
        WHERE toLower(n.name) CONTAINS toLower($concept_name)
           OR toLower(n.description) CONTAINS toLower($concept_name)
        RETURN id(n) as node_id, labels(n) as labels, 
               n.name as name, n.description as description,
               n.category as category, n.difficulty_level as difficulty,
               properties(n) as properties
        LIMIT 10
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {'concept_name': concept_name})
                return [record.data() for record in result]
        
        try:
            results = await asyncio.to_thread(run_query)
            
            nodes = []
            for record in results:
                node = GraphNode(
                    node_id=record['node_id'],
                    node_type='Concept',
                    name=record['name'],
                    description=record.get('description', ''),
                    properties=record.get('properties', {})
                )
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to find concept nodes: {e}")
            return []
    
    async def _find_concept_paths(self, start_node: GraphNode, end_node: GraphNode,
                                student_level: str, max_paths: int = 3) -> List[GraphPath]:
        """Find educational paths between two concept nodes"""
        
        # Use different strategies based on student level
        if student_level == 'beginner':
            # Prefer paths through prerequisites and simpler concepts
            relationship_priorities = ['PREREQUISITE_FOR', 'RELATED_TO', 'CONTAINS']
        elif student_level == 'advanced':
            # Allow more complex relationships
            relationship_priorities = ['APPLIES_CONCEPT', 'RELATED_TO', 'DESCRIBES', 'PREREQUISITE_FOR']
        else:
            # Intermediate - balanced approach
            relationship_priorities = ['RELATED_TO', 'PREREQUISITE_FOR', 'APPLIES_CONCEPT']
        
        paths = await self._shortest_educational_paths(
            start_node, end_node, relationship_priorities, max_paths
        )
        
        return paths
    
    async def _shortest_educational_paths(self, start_node: GraphNode, end_node: GraphNode,
                                        relationship_priorities: List[str], 
                                        max_paths: int) -> List[GraphPath]:
        """Find shortest educational paths using modified Dijkstra's algorithm"""
        
        query = """
        MATCH path = shortestPath((start:Concept)-[*1..4]-(end:Concept))
        WHERE id(start) = $start_id AND id(end) = $end_id
        RETURN [node in nodes(path) | {
            id: id(node),
            labels: labels(node),
            name: node.name,
            description: node.description,
            properties: properties(node)
        }] as nodes,
        [rel in relationships(path) | {
            type: type(rel),
            properties: properties(rel)
        }] as relationships
        LIMIT $max_paths
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {
                    'start_id': start_node.node_id,
                    'end_id': end_node.node_id,
                    'max_paths': max_paths
                })
                return [record.data() for record in result]
        
        try:
            results = await asyncio.to_thread(run_query)
            
            paths = []
            for record in results:
                # Convert to GraphPath
                nodes = []
                for node_data in record['nodes']:
                    graph_node = GraphNode(
                        node_id=node_data['id'],
                        node_type=node_data['labels'][0] if node_data['labels'] else 'Unknown',
                        name=node_data['name'],
                        description=node_data.get('description', ''),
                        properties=node_data.get('properties', {})
                    )
                    nodes.append(graph_node)
                
                relationships = record['relationships']
                
                # Calculate path weight based on educational value
                path_weight = self._calculate_educational_path_weight(
                    nodes, relationships, relationship_priorities
                )
                
                path = GraphPath(
                    nodes=nodes,
                    relationships=relationships,
                    path_weight=path_weight,
                    semantic_score=0.0,  # Will be calculated later
                    explanation=self._generate_path_explanation(nodes, relationships)
                )
                
                paths.append(path)
            
            return paths
            
        except Exception as e:
            logger.error(f"Failed to find educational paths: {e}")
            return []
    
    async def _traverse_prerequisites(self, node: GraphNode, max_depth: int) -> List[GraphPath]:
        """Traverse prerequisite relationships to build learning chains"""
        
        query = """
        MATCH path = (start:Concept)<-[:PREREQUISITE_FOR*1..$max_depth]-(prereq:Concept)
        WHERE id(start) = $node_id
        RETURN [node in nodes(path) | {
            id: id(node),
            labels: labels(node),
            name: node.name,
            description: node.description,
            difficulty_level: node.difficulty_level
        }] as nodes,
        [rel in relationships(path) | {
            type: type(rel),
            properties: properties(rel)
        }] as relationships
        ORDER BY length(path) DESC
        LIMIT 10
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {
                    'node_id': node.node_id,
                    'max_depth': max_depth
                })
                return [record.data() for record in result]
        
        try:
            results = await asyncio.to_thread(run_query)
            
            chains = []
            for record in results:
                nodes = []
                for node_data in record['nodes']:
                    graph_node = GraphNode(
                        node_id=node_data['id'],
                        node_type='Concept',
                        name=node_data['name'],
                        description=node_data.get('description', ''),
                        properties={'difficulty_level': node_data.get('difficulty_level')}
                    )
                    nodes.append(graph_node)
                
                # Reverse to get prerequisite -> target order
                nodes.reverse()
                relationships = record['relationships'][::-1]
                
                path = GraphPath(
                    nodes=nodes,
                    relationships=relationships,
                    path_weight=self._calculate_prerequisite_weight(nodes),
                    semantic_score=0.0,
                    explanation=f"Prerequisites for understanding {node.name}"
                )
                
                chains.append(path)
            
            return chains
            
        except Exception as e:
            logger.error(f"Failed to traverse prerequisites: {e}")
            return []
    
    async def _find_related_nodes(self, start_node: GraphNode, relationship_types: List[str],
                                max_hops: int, visited: Set[int]) -> List[GraphNode]:
        """Find nodes related through specified relationship types"""
        
        if start_node.node_id in visited or max_hops <= 0:
            return []
        
        visited.add(start_node.node_id)
        
        # Build relationship filter
        rel_filter = '|'.join(relationship_types)
        
        query = f"""
        MATCH (start)-[r:{rel_filter}]-(related)
        WHERE id(start) = $start_id
        RETURN id(related) as node_id, labels(related) as labels,
               related.name as name, related.description as description,
               type(r) as relationship_type,
               properties(related) as properties
        LIMIT 20
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {'start_id': start_node.node_id})
                return [record.data() for record in result]
        
        try:
            results = await asyncio.to_thread(run_query)
            
            related_nodes = []
            for record in results:
                if record['node_id'] not in visited:
                    node = GraphNode(
                        node_id=record['node_id'],
                        node_type=record['labels'][0] if record['labels'] else 'Unknown',
                        name=record['name'],
                        description=record.get('description', ''),
                        properties=record.get('properties', {}),
                        relationships=[{
                            'type': record['relationship_type'],
                            'from_node': start_node.name
                        }]
                    )
                    related_nodes.append(node)
                    
                    # Recursively find related nodes (depth-limited)
                    if max_hops > 1:
                        deeper_nodes = await self._find_related_nodes(
                            node, relationship_types, max_hops - 1, visited
                        )
                        related_nodes.extend(deeper_nodes)
            
            return related_nodes
            
        except Exception as e:
            logger.error(f"Failed to find related nodes: {e}")
            return []
    
    def _calculate_educational_path_weight(self, nodes: List[GraphNode], 
                                         relationships: List[Dict],
                                         relationship_priorities: List[str]) -> float:
        """Calculate educational value weight for a path"""
        
        base_weight = 1.0
        
        # Factor 1: Path length (shorter is often better for learning)
        length_penalty = len(nodes) * 0.1
        
        # Factor 2: Relationship types (some are more educational)
        relationship_bonus = 0.0
        for rel in relationships:
            rel_type = rel['type']
            if rel_type in relationship_priorities:
                priority_index = relationship_priorities.index(rel_type)
                relationship_bonus += (len(relationship_priorities) - priority_index) * 0.1
        
        # Factor 3: Difficulty progression (gradual increase is good)
        difficulty_bonus = self._calculate_difficulty_progression(nodes)
        
        total_weight = base_weight - length_penalty + relationship_bonus + difficulty_bonus
        return max(0.1, total_weight)  # Ensure positive weight
    
    def _calculate_prerequisite_weight(self, nodes: List[GraphNode]) -> float:
        """Calculate weight for prerequisite chains"""
        
        # Prefer chains with clear difficulty progression
        difficulty_levels = []
        for node in nodes:
            level = node.properties.get('difficulty_level', 'intermediate')
            if level == 'beginner':
                difficulty_levels.append(1)
            elif level == 'intermediate':
                difficulty_levels.append(2)
            elif level == 'advanced':
                difficulty_levels.append(3)
            else:
                difficulty_levels.append(2)
        
        # Check if difficulty increases appropriately
        progression_score = 0.0
        for i in range(1, len(difficulty_levels)):
            if difficulty_levels[i] >= difficulty_levels[i-1]:
                progression_score += 0.2
            else:
                progression_score -= 0.1
        
        return 1.0 + progression_score
    
    def _calculate_difficulty_progression(self, nodes: List[GraphNode]) -> float:
        """Calculate bonus for appropriate difficulty progression"""
        
        difficulties = []
        for node in nodes:
            diff = node.properties.get('difficulty_level', 'intermediate')
            if diff == 'beginner':
                difficulties.append(1)
            elif diff == 'intermediate':
                difficulties.append(2)
            elif diff == 'advanced':
                difficulties.append(3)
            else:
                difficulties.append(2)
        
        # Reward gradual increase in difficulty
        progression_bonus = 0.0
        for i in range(1, len(difficulties)):
            if difficulties[i] == difficulties[i-1] + 1:
                progression_bonus += 0.2
            elif difficulties[i] == difficulties[i-1]:
                progression_bonus += 0.1
            elif difficulties[i] < difficulties[i-1]:
                progression_bonus -= 0.1
        
        return progression_bonus
    
    def _rank_educational_paths(self, paths: List[GraphPath], student_level: str) -> List[GraphPath]:
        """Rank paths by educational value for given student level"""
        
        def educational_score(path: GraphPath) -> float:
            score = path.path_weight
            
            # Adjust based on student level
            avg_difficulty = self._get_average_difficulty(path.nodes)
            
            if student_level == 'beginner':
                if avg_difficulty <= 1.5:  # Prefer easier paths
                    score += 0.3
                elif avg_difficulty > 2.5:
                    score -= 0.3
            elif student_level == 'advanced':
                if avg_difficulty >= 2.5:  # Prefer harder paths
                    score += 0.3
                elif avg_difficulty < 1.5:
                    score -= 0.3
            
            # Length bonus/penalty
            if 2 <= len(path.nodes) <= 4:
                score += 0.2  # Optimal length
            elif len(path.nodes) > 6:
                score -= 0.2  # Too long
            
            return score
        
        # Sort by educational score
        paths.sort(key=educational_score, reverse=True)
        
        return paths
    
    def _get_average_difficulty(self, nodes: List[GraphNode]) -> float:
        """Calculate average difficulty of nodes in path"""
        
        total_difficulty = 0
        count = 0
        
        for node in nodes:
            diff = node.properties.get('difficulty_level', 'intermediate')
            if diff == 'beginner':
                total_difficulty += 1
            elif diff == 'intermediate':
                total_difficulty += 2
            elif diff == 'advanced':
                total_difficulty += 3
            else:
                total_difficulty += 2
            count += 1
        
        return total_difficulty / count if count > 0 else 2.0
    
    def _generate_path_explanation(self, nodes: List[GraphNode], 
                                 relationships: List[Dict]) -> str:
        """Generate human-readable explanation of learning path"""
        
        if len(nodes) <= 1:
            return "Single concept"
        
        explanation_parts = []
        
        for i in range(len(nodes) - 1):
            current_node = nodes[i].name
            next_node = nodes[i + 1].name
            
            if i < len(relationships):
                rel_type = relationships[i]['type']
                
                if rel_type == 'PREREQUISITE_FOR':
                    explanation_parts.append(f"{current_node} is prerequisite for {next_node}")
                elif rel_type == 'RELATED_TO':
                    explanation_parts.append(f"{current_node} relates to {next_node}")
                elif rel_type == 'APPLIES_CONCEPT':
                    explanation_parts.append(f"{current_node} applies {next_node}")
                else:
                    explanation_parts.append(f"{current_node} connects to {next_node}")
            else:
                explanation_parts.append(f"{current_node} leads to {next_node}")
        
        return " → ".join(explanation_parts)

class GraphEnhancedRAGRetriever:
    """Advanced retrieval system combining semantic search with graph traversal"""
    
    def __init__(self, semantic_search_engine: SemanticSearchEngine,
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 redis_host: str = 'localhost', redis_port: int = 6379, redis_password: str = None):
        
        self.search_engine = semantic_search_engine
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.pathfinder = EducationalPathfinder(self.neo4j_driver)
        
        # Redis for caching graph computations
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, password=redis_password, 
            decode_responses=True
        )
        
        # Graph algorithms configuration
        self.max_walk_length = 5
        self.random_walk_restarts = 0.15
        self.pagerank_damping = 0.85
        
    async def close(self):
        """Close database connections"""
        if self.neo4j_driver:
            await asyncio.to_thread(self.neo4j_driver.close)
        if self.redis_client:
            await asyncio.to_thread(self.redis_client.close)
    
    async def enhanced_retrieve(self, query: SearchQuery, 
                              traversal_strategy: GraphTraversalStrategy = GraphTraversalStrategy.BREADTH_FIRST,
                              include_learning_paths: bool = True,
                              student_level: str = 'intermediate') -> List[SearchResult]:
        """Enhanced retrieval combining semantic search with graph traversal"""
        
        logger.info(f"🔍 Enhanced retrieval with strategy: {traversal_strategy.value}")
        
        # Step 1: Get initial semantic search results
        initial_results = await self.search_engine.search(query)
        
        if not initial_results:
            logger.info("No initial results found")
            return []
        
        # Step 2: Extract key concepts from top results
        key_concepts = await self._extract_key_concepts(initial_results[:3])
        
        # Step 3: Expand results using graph traversal
        if traversal_strategy == GraphTraversalStrategy.RANDOM_WALK:
            expanded_results = await self._random_walk_expansion(
                initial_results, key_concepts, query.limit * 2
            )
        elif traversal_strategy == GraphTraversalStrategy.PERSONALIZED_PAGERANK:
            expanded_results = await self._personalized_pagerank_expansion(
                initial_results, key_concepts, query.limit * 2
            )
        elif traversal_strategy == GraphTraversalStrategy.CONCEPT_HIERARCHY:
            expanded_results = await self._concept_hierarchy_expansion(
                initial_results, key_concepts, student_level
            )
        else:
            expanded_results = await self._breadth_first_expansion(
                initial_results, key_concepts, query.limit * 2
            )
        
        # Step 4: Add learning paths if requested
        if include_learning_paths and len(key_concepts) >= 2:
            learning_paths = await self._find_educational_connections(
                key_concepts, student_level
            )
            expanded_results.extend(learning_paths)
        
        # Step 5: Re-rank combined results
        final_results = await self._rerank_with_graph_features(
            expanded_results, query, student_level
        )
        
        return final_results[:query.limit]
    
    async def _extract_key_concepts(self, results: List[SearchResult]) -> List[str]:
        """Extract key physics concepts from search results"""
        
        concepts = set()
        
        for result in results:
            # Extract concepts from metadata
            if result.content_type == 'concepts':
                concepts.add(result.title.lower())
            
            # Extract from content using simple pattern matching
            content_lower = result.content.lower()
            
            # Common physics terms
            physics_terms = [
                'velocity', 'acceleration', 'force', 'energy', 'momentum',
                'mass', 'displacement', 'friction', 'gravity', 'work',
                'power', 'torque', 'angular', 'oscillation', 'wave',
                'frequency', 'amplitude', 'electric', 'magnetic'
            ]
            
            for term in physics_terms:
                if term in content_lower:
                    concepts.add(term)
        
        return list(concepts)[:5]  # Limit to top 5 concepts
    
    async def _random_walk_expansion(self, initial_results: List[SearchResult],
                                   key_concepts: List[str], max_results: int) -> List[SearchResult]:
        """Expand results using random walk algorithm"""
        
        # Start random walks from initial result nodes
        start_node_ids = [result.node_id for result in initial_results[:3]]
        
        # Perform random walks
        walk_results = await self._perform_random_walks(
            start_node_ids, self.max_walk_length, num_walks=10
        )
        
        # Convert to SearchResult objects
        expanded_results = initial_results.copy()
        
        for node_data in walk_results[:max_results - len(initial_results)]:
            result = SearchResult(
                node_id=node_data['node_id'],
                content_type=node_data['content_type'],
                title=node_data['name'],
                content=node_data.get('description', ''),
                similarity_score=node_data.get('walk_score', 0.5),
                rank=len(expanded_results) + 1,
                metadata=node_data,
                explanation=f"Found via random walk from {node_data.get('source_concept', 'initial results')}"
            )
            expanded_results.append(result)
        
        return expanded_results
    
    async def _perform_random_walks(self, start_node_ids: List[int], 
                                  walk_length: int, num_walks: int) -> List[Dict]:
        """Perform multiple random walks from starting nodes"""
        
        query = """
        UNWIND $start_nodes as start_id
        MATCH (start) WHERE id(start) = start_id
        CALL {
            WITH start
            WITH start, range(1, $num_walks) as walks
            UNWIND walks as walk
            CALL apoc.path.expandConfig(start, {
                relationshipFilter: "RELATED_TO|PREREQUISITE_FOR|APPLIES_CONCEPT",
                minLevel: 1,
                maxLevel: $walk_length,
                uniqueness: "NONE",
                bfs: false,
                limit: 1
            }) YIELD path
            RETURN last(nodes(path)) as end_node, length(path) as path_length
        }
        WITH end_node, path_length, count(*) as frequency
        WHERE end_node IS NOT NULL
        RETURN id(end_node) as node_id, labels(end_node) as labels,
               end_node.name as name, end_node.description as description,
               frequency as walk_score, path_length
        ORDER BY frequency DESC, path_length ASC
        LIMIT 50
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {
                    'start_nodes': start_node_ids,
                    'num_walks': num_walks,
                    'walk_length': walk_length
                })
                return [record.data() for record in result]
        
        try:
            results = await asyncio.to_thread(run_query)
            
            # Convert to standardized format
            walk_results = []
            for record in results:
                content_type = 'concept'
                if 'Formula' in record.get('labels', []):
                    content_type = 'formula'
                elif 'Problem' in record.get('labels', []):
                    content_type = 'problem'
                elif 'Explanation' in record.get('labels', []):
                    content_type = 'explanation'
                
                walk_results.append({
                    'node_id': record['node_id'],
                    'content_type': content_type,
                    'name': record['name'],
                    'description': record.get('description', ''),
                    'walk_score': record['walk_score'] / num_walks,  # Normalize
                    'path_length': record['path_length']
                })
            
            return walk_results
            
        except Exception as e:
            logger.error(f"Failed to perform random walks: {e}")
            return []
    
    async def _personalized_pagerank_expansion(self, initial_results: List[SearchResult],
                                             key_concepts: List[str], max_results: int) -> List[SearchResult]:
        """Expand results using personalized PageRank algorithm"""
        
        # This is a simplified version - in production, you'd use graph algorithms library
        # For now, simulate by finding highly connected nodes related to initial results
        
        start_node_ids = [result.node_id for result in initial_results[:3]]
        
        query = """
        UNWIND $start_nodes as start_id
        MATCH (start)-[*1..3]-(related)
        WHERE id(start) = start_id
        WITH related, count(*) as connection_strength
        MATCH (related)-[r]-()
        WITH related, connection_strength, count(r) as total_connections
        WHERE total_connections > 2
        RETURN id(related) as node_id, labels(related) as labels,
               related.name as name, related.description as description,
               (connection_strength * 1.0 / total_connections) as pagerank_score
        ORDER BY pagerank_score DESC
        LIMIT $max_results
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {
                    'start_nodes': start_node_ids,
                    'max_results': max_results
                })
                return [record.data() for record in result]
        
        try:
            results = await asyncio.to_thread(run_query)
            
            expanded_results = initial_results.copy()
            
            for record in results:
                content_type = 'concept'
                if 'Formula' in record.get('labels', []):
                    content_type = 'formula'
                elif 'Problem' in record.get('labels', []):
                    content_type = 'problem'
                elif 'Explanation' in record.get('labels', []):
                    content_type = 'explanation'
                
                result = SearchResult(
                    node_id=record['node_id'],
                    content_type=content_type,
                    title=record['name'],
                    content=record.get('description', ''),
                    similarity_score=record['pagerank_score'],
                    rank=len(expanded_results) + 1,
                    metadata=record,
                    explanation="Found via PageRank centrality"
                )
                expanded_results.append(result)
            
            return expanded_results
            
        except Exception as e:
            logger.error(f"Failed to perform PageRank expansion: {e}")
            return initial_results
    
    async def _concept_hierarchy_expansion(self, initial_results: List[SearchResult],
                                         key_concepts: List[str], student_level: str) -> List[SearchResult]:
        """Expand results using concept hierarchy and educational structure"""
        
        expanded_results = initial_results.copy()
        
        # Find hierarchical relationships
        for concept in key_concepts:
            # Get related concepts in hierarchy
            related_nodes = await self.pathfinder.find_related_concepts(
                concept, relationship_types=['CONTAINS', 'PREREQUISITE_FOR'], max_hops=2
            )
            
            # Find prerequisite chains
            prerequisite_chains = await self.pathfinder.find_prerequisite_chain(concept, max_depth=2)
            
            # Convert to search results
            for node in related_nodes[:3]:  # Limit per concept
                result = SearchResult(
                    node_id=node.node_id,
                    content_type=node.node_type.lower(),
                    title=node.name,
                    content=node.description,
                    similarity_score=0.6,  # Fixed score for hierarchy results
                    rank=len(expanded_results) + 1,
                    metadata=asdict(node),
                    explanation=f"Related in concept hierarchy to {concept}"
                )
                expanded_results.append(result)
            
            # Add nodes from prerequisite chains
            for chain in prerequisite_chains[:2]:  # Limit chains per concept
                for node in chain.nodes:
                    if node.node_id not in [r.node_id for r in expanded_results]:
                        result = SearchResult(
                            node_id=node.node_id,
                            content_type=node.node_type.lower(),
                            title=node.name,
                            content=node.description,
                            similarity_score=0.7,
                            rank=len(expanded_results) + 1,
                            metadata=asdict(node),
                            explanation=f"Prerequisite for understanding {concept}"
                        )
                        expanded_results.append(result)
        
        return expanded_results
    
    async def _breadth_first_expansion(self, initial_results: List[SearchResult],
                                     key_concepts: List[str], max_results: int) -> List[SearchResult]:
        """Expand results using breadth-first traversal"""
        
        start_node_ids = [result.node_id for result in initial_results[:3]]
        expanded_results = initial_results.copy()
        visited = set(start_node_ids)
        
        # Perform BFS expansion
        queue = deque([(node_id, 0) for node_id in start_node_ids])  # (node_id, depth)
        max_depth = 2
        
        while queue and len(expanded_results) < max_results:
            current_node_id, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Find neighbors
            neighbors = await self._get_graph_neighbors(current_node_id)
            
            for neighbor in neighbors:
                if neighbor['node_id'] not in visited:
                    visited.add(neighbor['node_id'])
                    queue.append((neighbor['node_id'], depth + 1))
                    
                    # Convert to search result
                    result = SearchResult(
                        node_id=neighbor['node_id'],
                        content_type=neighbor['content_type'],
                        title=neighbor['name'],
                        content=neighbor.get('description', ''),
                        similarity_score=max(0.3, 0.8 - (depth * 0.2)),  # Decay with depth
                        rank=len(expanded_results) + 1,
                        metadata=neighbor,
                        explanation=f"Found at graph distance {depth + 1}"
                    )
                    expanded_results.append(result)
                    
                    if len(expanded_results) >= max_results:
                        break
        
        return expanded_results
    
    async def _get_graph_neighbors(self, node_id: int) -> List[Dict]:
        """Get direct neighbors of a node in the graph"""
        
        query = """
        MATCH (n)-[r]-(neighbor)
        WHERE id(n) = $node_id
        RETURN id(neighbor) as node_id, labels(neighbor) as labels,
               neighbor.name as name, neighbor.description as description,
               type(r) as relationship_type
        LIMIT 10
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {'node_id': node_id})
                return [record.data() for record in result]
        
        try:
            results = await asyncio.to_thread(run_query)
            
            neighbors = []
            for record in results:
                content_type = 'concept'
                if 'Formula' in record.get('labels', []):
                    content_type = 'formula'
                elif 'Problem' in record.get('labels', []):
                    content_type = 'problem'
                elif 'Explanation' in record.get('labels', []):
                    content_type = 'explanation'
                
                neighbors.append({
                    'node_id': record['node_id'],
                    'content_type': content_type,
                    'name': record['name'],
                    'description': record.get('description', ''),
                    'relationship_type': record['relationship_type']
                })
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get graph neighbors: {e}")
            return []
    
    async def _find_educational_connections(self, key_concepts: List[str], 
                                          student_level: str) -> List[SearchResult]:
        """Find educational connections between key concepts"""
        
        if len(key_concepts) < 2:
            return []
        
        educational_results = []
        
        # Find learning paths between concept pairs
        for i in range(len(key_concepts)):
            for j in range(i + 1, len(key_concepts)):
                concept1, concept2 = key_concepts[i], key_concepts[j]
                
                paths = await self.pathfinder.find_learning_path(
                    concept1, concept2, student_level
                )
                
                # Convert paths to search results
                for path in paths[:2]:  # Limit paths per concept pair
                    for node in path.nodes[1:-1]:  # Exclude start and end
                        result = SearchResult(
                            node_id=node.node_id,
                            content_type=node.node_type.lower(),
                            title=node.name,
                            content=node.description,
                            similarity_score=path.total_score,
                            rank=len(educational_results) + 1,
                            metadata=asdict(node),
                            explanation=f"Connects {concept1} and {concept2}: {path.explanation}"
                        )
                        educational_results.append(result)
        
        return educational_results
    
    async def _rerank_with_graph_features(self, results: List[SearchResult],
                                        query: SearchQuery, student_level: str) -> List[SearchResult]:
        """Re-rank results using graph-based features"""
        
        # Calculate graph-based scores for each result
        enhanced_results = []
        
        for result in results:
            # Calculate additional graph features
            centrality_score = await self._calculate_node_centrality(result.node_id)
            educational_value = await self._calculate_educational_value(
                result.node_id, student_level
            )
            
            # Combine scores
            original_score = result.similarity_score
            graph_score = (centrality_score * 0.3) + (educational_value * 0.4)
            combined_score = (original_score * 0.6) + (graph_score * 0.4)
            
            result.similarity_score = combined_score
            enhanced_results.append(result)
        
        # Sort by combined score
        enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(enhanced_results):
            result.rank = i + 1
        
        return enhanced_results
    
    async def _calculate_node_centrality(self, node_id: int) -> float:
        """Calculate centrality score for a node"""
        
        # Simple centrality based on node degree
        query = """
        MATCH (n)-[r]-()
        WHERE id(n) = $node_id
        RETURN count(r) as degree
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {'node_id': node_id})
                record = result.single()
                return record['degree'] if record else 0
        
        try:
            degree = await asyncio.to_thread(run_query)
            # Normalize degree to 0-1 range (assuming max degree ~ 50)
            return min(1.0, degree / 50.0)
        except Exception as e:
            logger.error(f"Failed to calculate centrality: {e}")
            return 0.5
    
    async def _calculate_educational_value(self, node_id: int, student_level: str) -> float:
        """Calculate educational value score for a node"""
        
        query = """
        MATCH (n) WHERE id(n) = $node_id
        RETURN n.difficulty_level as difficulty,
               n.category as category,
               labels(n) as labels
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {'node_id': node_id})
                record = result.single()
                return dict(record) if record else {}
        
        try:
            node_data = await asyncio.to_thread(run_query)
            
            educational_score = 0.5  # Base score
            
            # Difficulty alignment
            node_difficulty = node_data.get('difficulty', 'intermediate')
            if node_difficulty == student_level:
                educational_score += 0.3
            elif abs(['beginner', 'intermediate', 'advanced'].index(node_difficulty) - 
                    ['beginner', 'intermediate', 'advanced'].index(student_level)) == 1:
                educational_score += 0.1
            
            # Node type preferences
            node_labels = node_data.get('labels', [])
            if 'Explanation' in node_labels:
                educational_score += 0.2
            elif 'Problem' in node_labels:
                educational_score += 0.15
            elif 'Formula' in node_labels:
                educational_score += 0.1
            
            return min(1.0, educational_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate educational value: {e}")
            return 0.5

# Example usage and testing
async def test_graph_enhanced_retrieval():
    """Test function for graph-enhanced retrieval"""
    from .semantic_search import SemanticSearchEngine
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
    
    # Initialize embedding manager and search engine
    async with get_physics_embedding_manager(
        neo4j_uri, neo4j_user, neo4j_password,
        postgres_config=postgres_config,
        models_to_load=['sentence_transformer']
    ) as embedding_manager:
        
        search_engine = SemanticSearchEngine(
            embedding_manager, neo4j_uri, neo4j_user, neo4j_password,
            postgres_config=postgres_config
        )
        
        await search_engine.initialize()
        
        # Initialize graph-enhanced retriever
        graph_retriever = GraphEnhancedRAGRetriever(
            search_engine, neo4j_uri, neo4j_user, neo4j_password
        )
        
        try:
            # Test different retrieval strategies
            test_queries = [
                ("What is the relationship between force and motion?", GraphTraversalStrategy.CONCEPT_HIERARCHY),
                ("How does energy conservation work?", GraphTraversalStrategy.RANDOM_WALK),
                ("Explain Newton's laws", GraphTraversalStrategy.PERSONALIZED_PAGERANK),
            ]
            
            for query_text, strategy in test_queries:
                print(f"\n🔍 Testing: '{query_text}' with {strategy.value}")
                
                query = SearchQuery(
                    text=query_text,
                    search_type=SearchType.HYBRID,
                    limit=8
                )
                
                results = await graph_retriever.enhanced_retrieve(
                    query, traversal_strategy=strategy, student_level='intermediate'
                )
                
                print(f"📊 Found {len(results)} results:")
                for i, result in enumerate(results[:5]):
                    print(f"  {i+1}. {result.title} (score: {result.similarity_score:.3f})")
                    print(f"     Type: {result.content_type}")
                    if result.explanation:
                        print(f"     How found: {result.explanation}")
                
        finally:
            await search_engine.close()
            await graph_retriever.close()

if __name__ == "__main__":
    asyncio.run(test_graph_enhanced_retrieval())