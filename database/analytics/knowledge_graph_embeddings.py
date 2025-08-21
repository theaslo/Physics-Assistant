#!/usr/bin/env python3
"""
Knowledge Graph Embeddings for Physics Concepts - Phase 6
Creates and maintains embeddings of physics concepts and their relationships
for enhanced educational recommendations and adaptive learning.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransE, ComplEx
from torch_geometric.data import Data, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import uuid
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationType(Enum):
    PREREQUISITE = "prerequisite"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    APPLIES_TO = "applies_to"
    DERIVED_FROM = "derived_from"
    MEASURED_BY = "measured_by"
    EXEMPLIFIED_BY = "exemplified_by"
    DIFFICULTY_RELATED = "difficulty_related"

class ConceptType(Enum):
    FUNDAMENTAL_CONCEPT = "fundamental_concept"
    DERIVED_CONCEPT = "derived_concept"
    MATHEMATICAL_FORMULA = "mathematical_formula"
    PHYSICAL_LAW = "physical_law"
    PROBLEM_TYPE = "problem_type"
    MEASUREMENT_UNIT = "measurement_unit"
    EXPERIMENTAL_SETUP = "experimental_setup"

class EmbeddingModel(Enum):
    TRANSDUCTION = "transduction"  # Graph neural networks
    TRANSLATION = "translation"   # TransE, ComplEx
    GRAPH_SAGE = "graph_sage"     # GraphSAGE
    GRAPH_ATTENTION = "graph_attention"  # GAT

@dataclass
class PhysicsConcept:
    """Physics concept with metadata"""
    concept_id: str
    name: str
    description: str
    concept_type: ConceptType
    domain: str  # mechanics, thermodynamics, etc.
    
    # Educational metadata
    difficulty_level: float  # 0-1 scale
    grade_level: str
    learning_objectives: List[str]
    common_misconceptions: List[str]
    
    # Usage statistics
    frequency_in_problems: int = 0
    average_student_success_rate: float = 0.0
    help_request_frequency: int = 0
    
    # Relationships
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ConceptRelationship:
    """Relationship between physics concepts"""
    relationship_id: str
    source_concept_id: str
    target_concept_id: str
    relation_type: RelationType
    
    # Relationship strength and confidence
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    
    # Evidence for relationship
    evidence_type: str  # "expert_knowledge", "student_data", "curriculum"
    evidence_sources: List[str]
    
    # Learning context
    difficulty_jump: Optional[float] = None  # Difficulty increase from source to target
    success_correlation: Optional[float] = None  # How success in source predicts target
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    validated: bool = False

@dataclass
class ConceptEmbedding:
    """Concept embedding with metadata"""
    concept_id: str
    embedding_vector: np.ndarray
    embedding_dim: int
    model_type: EmbeddingModel
    
    # Quality metrics
    reconstruction_error: Optional[float] = None
    cluster_coherence: Optional[float] = None
    
    # Usage tracking
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for concept embeddings"""
    
    def __init__(self, num_concepts: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 3, 
                 model_type: str = "gcn"):
        super(GraphNeuralNetwork, self).__init__()
        
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        
        # Concept embeddings
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        if model_type == "gcn":
            self.convs.append(GCNConv(embedding_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, embedding_dim))
        elif model_type == "sage":
            self.convs.append(SAGEConv(embedding_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, embedding_dim))
        elif model_type == "gat":
            self.convs.append(GATConv(embedding_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            self.convs.append(GATConv(hidden_dim, embedding_dim))
        
        # Dropout and normalization
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.concept_embeddings.weight)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass"""
        # Initial embeddings
        if x is None:
            x = self.concept_embeddings.weight
        
        # Graph convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        x = self.layer_norm(x)
        
        return x

class KnowledgeGraphEmbedder:
    """Create and manage knowledge graph embeddings"""
    
    def __init__(self, embedding_dim: int = 128, device: str = "cpu"):
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Knowledge graph
        self.concepts = {}
        self.relationships = {}
        self.graph = nx.DiGraph()
        
        # Embeddings
        self.concept_embeddings = {}
        self.embedding_models = {}
        
        # Mappings
        self.concept_to_id = {}
        self.id_to_concept = {}
        
        # Training data
        self.training_history = []
        
        # Analysis cache
        self.similarity_cache = {}
        self.cluster_cache = {}
    
    async def initialize_physics_knowledge_graph(self):
        """Initialize knowledge graph with physics concepts"""
        try:
            logger.info("🧠 Initializing Physics Knowledge Graph")
            
            # Add fundamental physics concepts
            await self._add_fundamental_concepts()
            
            # Add derived concepts
            await self._add_derived_concepts()
            
            # Add mathematical formulas
            await self._add_mathematical_formulas()
            
            # Add relationships
            await self._add_concept_relationships()
            
            # Build NetworkX graph
            await self._build_networkx_graph()
            
            logger.info(f"✅ Knowledge graph initialized with {len(self.concepts)} concepts and {len(self.relationships)} relationships")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge graph: {e}")
            raise
    
    async def _add_fundamental_concepts(self):
        """Add fundamental physics concepts"""
        try:
            fundamental_concepts = [
                {
                    'name': 'Force',
                    'description': 'A push or pull that can change motion',
                    'domain': 'mechanics',
                    'difficulty_level': 0.3,
                    'grade_level': 'high_school',
                    'learning_objectives': [
                        'Understand force as a vector quantity',
                        'Apply Newton\'s laws of motion',
                        'Calculate net force'
                    ],
                    'common_misconceptions': [
                        'Force is needed to maintain motion',
                        'Heavier objects fall faster',
                        'Force and velocity are the same'
                    ]
                },
                {
                    'name': 'Mass',
                    'description': 'Amount of matter in an object',
                    'domain': 'mechanics',
                    'difficulty_level': 0.2,
                    'grade_level': 'middle_school',
                    'learning_objectives': [
                        'Distinguish between mass and weight',
                        'Understand mass as a measure of inertia'
                    ],
                    'common_misconceptions': [
                        'Mass and weight are the same',
                        'Mass changes with location'
                    ]
                },
                {
                    'name': 'Acceleration',
                    'description': 'Rate of change of velocity',
                    'domain': 'mechanics',
                    'difficulty_level': 0.4,
                    'grade_level': 'high_school',
                    'learning_objectives': [
                        'Calculate acceleration from velocity changes',
                        'Understand acceleration as a vector',
                        'Apply kinematic equations'
                    ],
                    'common_misconceptions': [
                        'Acceleration requires increasing speed',
                        'Zero acceleration means zero velocity'
                    ]
                },
                {
                    'name': 'Energy',
                    'description': 'Capacity to do work',
                    'domain': 'mechanics',
                    'difficulty_level': 0.5,
                    'grade_level': 'high_school',
                    'learning_objectives': [
                        'Understand energy conservation',
                        'Distinguish kinetic and potential energy',
                        'Apply work-energy theorem'
                    ],
                    'common_misconceptions': [
                        'Energy can be created or destroyed',
                        'Energy is a substance',
                        'Moving objects lose energy'
                    ]
                },
                {
                    'name': 'Momentum',
                    'description': 'Product of mass and velocity',
                    'domain': 'mechanics',
                    'difficulty_level': 0.6,
                    'grade_level': 'high_school',
                    'learning_objectives': [
                        'Calculate momentum',
                        'Apply conservation of momentum',
                        'Analyze collisions'
                    ],
                    'common_misconceptions': [
                        'Momentum is the same as force',
                        'Momentum is not conserved in all collisions'
                    ]
                }
            ]
            
            for concept_data in fundamental_concepts:
                concept_id = str(uuid.uuid4())
                concept = PhysicsConcept(
                    concept_id=concept_id,
                    name=concept_data['name'],
                    description=concept_data['description'],
                    concept_type=ConceptType.FUNDAMENTAL_CONCEPT,
                    domain=concept_data['domain'],
                    difficulty_level=concept_data['difficulty_level'],
                    grade_level=concept_data['grade_level'],
                    learning_objectives=concept_data['learning_objectives'],
                    common_misconceptions=concept_data['common_misconceptions']
                )
                
                self.concepts[concept_id] = concept
                self.concept_to_id[concept_data['name']] = concept_id
                self.id_to_concept[concept_id] = concept_data['name']
            
            logger.info(f"📚 Added {len(fundamental_concepts)} fundamental concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to add fundamental concepts: {e}")
    
    async def _add_derived_concepts(self):
        """Add derived physics concepts"""
        try:
            derived_concepts = [
                {
                    'name': 'Newton\'s Second Law',
                    'description': 'F = ma, force equals mass times acceleration',
                    'domain': 'mechanics',
                    'difficulty_level': 0.5,
                    'prerequisites': ['Force', 'Mass', 'Acceleration']
                },
                {
                    'name': 'Kinetic Energy',
                    'description': 'Energy of motion, KE = ½mv²',
                    'domain': 'mechanics',
                    'difficulty_level': 0.4,
                    'prerequisites': ['Energy', 'Mass']
                },
                {
                    'name': 'Potential Energy',
                    'description': 'Stored energy due to position',
                    'domain': 'mechanics',
                    'difficulty_level': 0.4,
                    'prerequisites': ['Energy']
                },
                {
                    'name': 'Conservation of Momentum',
                    'description': 'Total momentum before equals total momentum after',
                    'domain': 'mechanics',
                    'difficulty_level': 0.7,
                    'prerequisites': ['Momentum']
                },
                {
                    'name': 'Work-Energy Theorem',
                    'description': 'Work done equals change in kinetic energy',
                    'domain': 'mechanics',
                    'difficulty_level': 0.6,
                    'prerequisites': ['Energy', 'Kinetic Energy', 'Force']
                }
            ]
            
            for concept_data in derived_concepts:
                concept_id = str(uuid.uuid4())
                concept = PhysicsConcept(
                    concept_id=concept_id,
                    name=concept_data['name'],
                    description=concept_data['description'],
                    concept_type=ConceptType.DERIVED_CONCEPT,
                    domain=concept_data['domain'],
                    difficulty_level=concept_data['difficulty_level'],
                    grade_level='high_school',
                    learning_objectives=[f"Understand and apply {concept_data['name']}"],
                    prerequisites=[self.concept_to_id.get(prereq, '') for prereq in concept_data.get('prerequisites', [])]
                )
                
                self.concepts[concept_id] = concept
                self.concept_to_id[concept_data['name']] = concept_id
                self.id_to_concept[concept_id] = concept_data['name']
            
            logger.info(f"🔬 Added {len(derived_concepts)} derived concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to add derived concepts: {e}")
    
    async def _add_mathematical_formulas(self):
        """Add mathematical formulas as concepts"""
        try:
            formulas = [
                {
                    'name': 'F = ma',
                    'description': 'Newton\'s second law formula',
                    'applies_to': ['Newton\'s Second Law', 'Force', 'Acceleration']
                },
                {
                    'name': 'KE = ½mv²',
                    'description': 'Kinetic energy formula',
                    'applies_to': ['Kinetic Energy', 'Energy']
                },
                {
                    'name': 'PE = mgh',
                    'description': 'Gravitational potential energy formula',
                    'applies_to': ['Potential Energy', 'Energy']
                },
                {
                    'name': 'p = mv',
                    'description': 'Momentum formula',
                    'applies_to': ['Momentum']
                }
            ]
            
            for formula_data in formulas:
                concept_id = str(uuid.uuid4())
                concept = PhysicsConcept(
                    concept_id=concept_id,
                    name=formula_data['name'],
                    description=formula_data['description'],
                    concept_type=ConceptType.MATHEMATICAL_FORMULA,
                    domain='mathematics',
                    difficulty_level=0.6,
                    grade_level='high_school',
                    learning_objectives=[f"Apply {formula_data['name']} correctly"],
                    related_concepts=[self.concept_to_id.get(concept, '') for concept in formula_data.get('applies_to', [])]
                )
                
                self.concepts[concept_id] = concept
                self.concept_to_id[formula_data['name']] = concept_id
                self.id_to_concept[concept_id] = formula_data['name']
            
            logger.info(f"🧮 Added {len(formulas)} mathematical formulas")
            
        except Exception as e:
            logger.error(f"❌ Failed to add mathematical formulas: {e}")
    
    async def _add_concept_relationships(self):
        """Add relationships between concepts"""
        try:
            # Define relationships based on physics knowledge
            relationships = [
                # Prerequisite relationships
                ('Mass', 'Newton\'s Second Law', RelationType.PREREQUISITE, 0.9),
                ('Force', 'Newton\'s Second Law', RelationType.PREREQUISITE, 0.9),
                ('Acceleration', 'Newton\'s Second Law', RelationType.PREREQUISITE, 0.9),
                ('Energy', 'Kinetic Energy', RelationType.PREREQUISITE, 0.8),
                ('Energy', 'Potential Energy', RelationType.PREREQUISITE, 0.8),
                ('Momentum', 'Conservation of Momentum', RelationType.PREREQUISITE, 0.9),
                
                # Formula applications
                ('Newton\'s Second Law', 'F = ma', RelationType.EXEMPLIFIED_BY, 0.95),
                ('Kinetic Energy', 'KE = ½mv²', RelationType.EXEMPLIFIED_BY, 0.95),
                ('Potential Energy', 'PE = mgh', RelationType.EXEMPLIFIED_BY, 0.95),
                ('Momentum', 'p = mv', RelationType.EXEMPLIFIED_BY, 0.95),
                
                # Conceptual similarities
                ('Kinetic Energy', 'Potential Energy', RelationType.SIMILAR_TO, 0.7),
                ('Force', 'Momentum', RelationType.SIMILAR_TO, 0.6),
                
                # Derived relationships
                ('Force', 'Work-Energy Theorem', RelationType.APPLIES_TO, 0.8),
                ('Energy', 'Work-Energy Theorem', RelationType.APPLIES_TO, 0.8),
                ('Kinetic Energy', 'Work-Energy Theorem', RelationType.PART_OF, 0.9)
            ]
            
            for source_name, target_name, relation_type, strength in relationships:
                source_id = self.concept_to_id.get(source_name)
                target_id = self.concept_to_id.get(target_name)
                
                if source_id and target_id:
                    relationship_id = str(uuid.uuid4())
                    relationship = ConceptRelationship(
                        relationship_id=relationship_id,
                        source_concept_id=source_id,
                        target_concept_id=target_id,
                        relation_type=relation_type,
                        strength=strength,
                        confidence=0.9,  # High confidence for expert knowledge
                        evidence_type="expert_knowledge",
                        evidence_sources=["physics_curriculum", "textbook_analysis"],
                        validated=True
                    )
                    
                    self.relationships[relationship_id] = relationship
            
            logger.info(f"🔗 Added {len(relationships)} concept relationships")
            
        except Exception as e:
            logger.error(f"❌ Failed to add concept relationships: {e}")
    
    async def _build_networkx_graph(self):
        """Build NetworkX graph from concepts and relationships"""
        try:
            # Add nodes
            for concept_id, concept in self.concepts.items():
                self.graph.add_node(
                    concept_id,
                    name=concept.name,
                    concept_type=concept.concept_type.value,
                    domain=concept.domain,
                    difficulty=concept.difficulty_level
                )
            
            # Add edges
            for relationship in self.relationships.values():
                self.graph.add_edge(
                    relationship.source_concept_id,
                    relationship.target_concept_id,
                    relation_type=relationship.relation_type.value,
                    strength=relationship.strength,
                    confidence=relationship.confidence
                )
            
            logger.info(f"🕸️ Built NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"❌ Failed to build NetworkX graph: {e}")
    
    async def train_embeddings(self, model_type: EmbeddingModel = EmbeddingModel.TRANSDUCTION,
                             epochs: int = 200, learning_rate: float = 0.01) -> Dict[str, Any]:
        """Train concept embeddings using graph neural networks"""
        try:
            logger.info(f"🔥 Training {model_type.value} embeddings for {epochs} epochs")
            
            # Prepare PyTorch Geometric data
            graph_data = await self._prepare_pytorch_geometric_data()
            
            # Initialize model
            if model_type == EmbeddingModel.TRANSDUCTION:
                model = GraphNeuralNetwork(
                    num_concepts=len(self.concepts),
                    embedding_dim=self.embedding_dim,
                    model_type="gcn"
                )
            elif model_type == EmbeddingModel.GRAPH_SAGE:
                model = GraphNeuralNetwork(
                    num_concepts=len(self.concepts),
                    embedding_dim=self.embedding_dim,
                    model_type="sage"
                )
            elif model_type == EmbeddingModel.GRAPH_ATTENTION:
                model = GraphNeuralNetwork(
                    num_concepts=len(self.concepts),
                    embedding_dim=self.embedding_dim,
                    model_type="gat"
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            model = model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            model.train()
            training_losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # Forward pass
                embeddings = model(None, graph_data.edge_index)
                
                # Self-supervised loss (graph reconstruction)
                loss = self._calculate_reconstruction_loss(embeddings, graph_data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                training_losses.append(loss.item())
                
                if (epoch + 1) % 50 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            
            # Extract final embeddings
            model.eval()
            with torch.no_grad():
                final_embeddings = model(None, graph_data.edge_index)
                final_embeddings = final_embeddings.cpu().numpy()
            
            # Store embeddings
            for i, (concept_id, concept) in enumerate(self.concepts.items()):
                embedding = ConceptEmbedding(
                    concept_id=concept_id,
                    embedding_vector=final_embeddings[i],
                    embedding_dim=self.embedding_dim,
                    model_type=model_type
                )
                self.concept_embeddings[concept_id] = embedding
            
            # Store model
            self.embedding_models[model_type.value] = model
            
            training_results = {
                'model_type': model_type.value,
                'final_loss': training_losses[-1],
                'training_losses': training_losses,
                'num_epochs': epochs,
                'embedding_dim': self.embedding_dim,
                'num_concepts': len(self.concepts)
            }
            
            self.training_history.append(training_results)
            
            logger.info(f"✅ Training completed - Final loss: {training_losses[-1]:.4f}")
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Embedding training failed: {e}")
            raise
    
    async def _prepare_pytorch_geometric_data(self) -> Data:
        """Prepare data for PyTorch Geometric"""
        try:
            # Create concept ID mapping
            concept_ids = list(self.concepts.keys())
            id_to_index = {concept_id: i for i, concept_id in enumerate(concept_ids)}
            
            # Create edge index
            edge_list = []
            edge_weights = []
            
            for relationship in self.relationships.values():
                source_idx = id_to_index[relationship.source_concept_id]
                target_idx = id_to_index[relationship.target_concept_id]
                
                edge_list.append([source_idx, target_idx])
                edge_weights.append(relationship.strength)
                
                # Add reverse edge for undirected relationships
                if relationship.relation_type in [RelationType.SIMILAR_TO]:
                    edge_list.append([target_idx, source_idx])
                    edge_weights.append(relationship.strength)
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
            
            # Create node features (concept metadata)
            node_features = []
            for concept_id in concept_ids:
                concept = self.concepts[concept_id]
                features = [
                    concept.difficulty_level,
                    1.0 if concept.concept_type == ConceptType.FUNDAMENTAL_CONCEPT else 0.0,
                    1.0 if concept.concept_type == ConceptType.DERIVED_CONCEPT else 0.0,
                    1.0 if concept.concept_type == ConceptType.MATHEMATICAL_FORMULA else 0.0,
                    len(concept.learning_objectives),
                    len(concept.common_misconceptions)
                ]
                node_features.append(features)
            
            x = torch.tensor(node_features, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            logger.error(f"❌ PyTorch Geometric data preparation failed: {e}")
            raise
    
    def _calculate_reconstruction_loss(self, embeddings: torch.Tensor, 
                                     graph_data: Data) -> torch.Tensor:
        """Calculate graph reconstruction loss"""
        try:
            # Positive edges (actual edges in graph)
            edge_index = graph_data.edge_index
            pos_edge_embeddings = embeddings[edge_index[0]] * embeddings[edge_index[1]]
            pos_scores = torch.sum(pos_edge_embeddings, dim=1)
            
            # Negative edges (random non-existing edges)
            num_pos_edges = edge_index.shape[1]
            num_nodes = embeddings.shape[0]
            
            neg_edges = torch.randint(0, num_nodes, (2, num_pos_edges), device=embeddings.device)
            neg_edge_embeddings = embeddings[neg_edges[0]] * embeddings[neg_edges[1]]
            neg_scores = torch.sum(neg_edge_embeddings, dim=1)
            
            # Binary cross-entropy loss
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            
            return pos_loss + neg_loss
            
        except Exception as e:
            logger.error(f"❌ Reconstruction loss calculation failed: {e}")
            return torch.tensor(0.0, requires_grad=True)
    
    async def find_similar_concepts(self, concept_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find concepts similar to the given concept"""
        try:
            concept_id = self.concept_to_id.get(concept_name)
            if not concept_id or concept_id not in self.concept_embeddings:
                return []
            
            target_embedding = self.concept_embeddings[concept_id].embedding_vector
            similarities = []
            
            for other_id, other_embedding in self.concept_embeddings.items():
                if other_id != concept_id:
                    similarity = cosine_similarity(
                        target_embedding.reshape(1, -1),
                        other_embedding.embedding_vector.reshape(1, -1)
                    )[0][0]
                    
                    other_name = self.id_to_concept[other_id]
                    similarities.append((other_name, float(similarity)))
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Similar concepts search failed: {e}")
            return []
    
    async def find_concept_prerequisites(self, concept_name: str) -> List[Tuple[str, float]]:
        """Find prerequisite concepts with difficulty progression"""
        try:
            concept_id = self.concept_to_id.get(concept_name)
            if not concept_id:
                return []
            
            prerequisites = []
            target_concept = self.concepts[concept_id]
            
            # Find direct prerequisites from relationships
            for relationship in self.relationships.values():
                if (relationship.target_concept_id == concept_id and 
                    relationship.relation_type == RelationType.PREREQUISITE):
                    
                    prereq_id = relationship.source_concept_id
                    prereq_concept = self.concepts[prereq_id]
                    prereq_name = self.id_to_concept[prereq_id]
                    
                    # Calculate difficulty progression
                    difficulty_jump = target_concept.difficulty_level - prereq_concept.difficulty_level
                    
                    prerequisites.append((prereq_name, relationship.strength))
            
            # Also find similar concepts with lower difficulty
            if concept_id in self.concept_embeddings:
                target_embedding = self.concept_embeddings[concept_id].embedding_vector
                
                for other_id, other_embedding in self.concept_embeddings.items():
                    if other_id != concept_id:
                        other_concept = self.concepts[other_id]
                        
                        # Only consider easier concepts
                        if other_concept.difficulty_level < target_concept.difficulty_level:
                            similarity = cosine_similarity(
                                target_embedding.reshape(1, -1),
                                other_embedding.embedding_vector.reshape(1, -1)
                            )[0][0]
                            
                            # High similarity + easier = likely prerequisite
                            if similarity > 0.7:
                                other_name = self.id_to_concept[other_id]
                                prerequisites.append((other_name, similarity * 0.8))  # Weight down inferred prereqs
            
            # Remove duplicates and sort
            unique_prerequisites = list(set(prerequisites))
            unique_prerequisites.sort(key=lambda x: x[1], reverse=True)
            
            return unique_prerequisites[:5]  # Top 5 prerequisites
            
        except Exception as e:
            logger.error(f"❌ Prerequisites search failed: {e}")
            return []
    
    async def recommend_learning_path(self, target_concept: str, 
                                    student_mastered_concepts: List[str]) -> List[str]:
        """Recommend learning path to target concept"""
        try:
            target_id = self.concept_to_id.get(target_concept)
            if not target_id:
                return []
            
            mastered_ids = [self.concept_to_id.get(concept) for concept in student_mastered_concepts]
            mastered_ids = [id for id in mastered_ids if id is not None]
            
            # Use graph shortest path with difficulty consideration
            learning_path = []
            
            # Find prerequisite path
            prerequisites = await self.find_concept_prerequisites(target_concept)
            
            for prereq_name, strength in prerequisites:
                prereq_id = self.concept_to_id.get(prereq_name)
                
                # If not mastered, add to path
                if prereq_id not in mastered_ids:
                    learning_path.append(prereq_name)
                    
                    # Recursively find prerequisites of prerequisites
                    sub_prerequisites = await self.find_concept_prerequisites(prereq_name)
                    for sub_prereq_name, _ in sub_prerequisites:
                        sub_prereq_id = self.concept_to_id.get(sub_prereq_name)
                        if sub_prereq_id not in mastered_ids and sub_prereq_name not in learning_path:
                            learning_path.insert(0, sub_prereq_name)  # Insert at beginning
            
            # Add target concept at the end
            learning_path.append(target_concept)
            
            # Remove duplicates while preserving order
            seen = set()
            ordered_path = []
            for concept in learning_path:
                if concept not in seen:
                    seen.add(concept)
                    ordered_path.append(concept)
            
            return ordered_path
            
        except Exception as e:
            logger.error(f"❌ Learning path recommendation failed: {e}")
            return []
    
    async def cluster_concepts(self, n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster concepts based on embeddings"""
        try:
            if not self.concept_embeddings:
                raise ValueError("No embeddings available for clustering")
            
            # Prepare embedding matrix
            concept_ids = list(self.concept_embeddings.keys())
            embeddings_matrix = np.array([
                self.concept_embeddings[concept_id].embedding_vector 
                for concept_id in concept_ids
            ])
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_matrix)
            
            # Organize results
            clusters = defaultdict(list)
            for i, concept_id in enumerate(concept_ids):
                concept_name = self.id_to_concept[concept_id]
                cluster_id = cluster_labels[i]
                clusters[f"cluster_{cluster_id}"].append({
                    'concept_name': concept_name,
                    'concept_id': concept_id,
                    'difficulty': self.concepts[concept_id].difficulty_level,
                    'domain': self.concepts[concept_id].domain
                })
            
            # Calculate cluster characteristics
            cluster_analysis = {}
            for cluster_name, concepts in clusters.items():
                domains = [c['domain'] for c in concepts]
                difficulties = [c['difficulty'] for c in concepts]
                
                cluster_analysis[cluster_name] = {
                    'size': len(concepts),
                    'concepts': [c['concept_name'] for c in concepts],
                    'avg_difficulty': np.mean(difficulties),
                    'primary_domain': Counter(domains).most_common(1)[0][0] if domains else 'unknown',
                    'domain_distribution': dict(Counter(domains))
                }
            
            # Cache results
            self.cluster_cache[n_clusters] = cluster_analysis
            
            logger.info(f"🎯 Clustered {len(concept_ids)} concepts into {n_clusters} clusters")
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"❌ Concept clustering failed: {e}")
            return {}
    
    async def visualize_embeddings(self, method: str = "tsne") -> Dict[str, Any]:
        """Create 2D visualization of concept embeddings"""
        try:
            if not self.concept_embeddings:
                raise ValueError("No embeddings available for visualization")
            
            # Prepare data
            concept_ids = list(self.concept_embeddings.keys())
            concept_names = [self.id_to_concept[concept_id] for concept_id in concept_ids]
            embeddings_matrix = np.array([
                self.concept_embeddings[concept_id].embedding_vector 
                for concept_id in concept_ids
            ])
            
            # Dimensionality reduction
            if method == "tsne":
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(concept_ids)-1))
            elif method == "pca":
                reducer = PCA(n_components=2, random_state=42)
            else:
                raise ValueError(f"Unsupported visualization method: {method}")
            
            reduced_embeddings = reducer.fit_transform(embeddings_matrix)
            
            # Prepare visualization data
            visualization_data = {
                'method': method,
                'coordinates': reduced_embeddings.tolist(),
                'concept_names': concept_names,
                'concept_metadata': []
            }
            
            for concept_id in concept_ids:
                concept = self.concepts[concept_id]
                visualization_data['concept_metadata'].append({
                    'name': concept.name,
                    'domain': concept.domain,
                    'difficulty': concept.difficulty_level,
                    'type': concept.concept_type.value
                })
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"❌ Embedding visualization failed: {e}")
            return {}
    
    async def get_concept_analytics(self, concept_name: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a concept"""
        try:
            concept_id = self.concept_to_id.get(concept_name)
            if not concept_id:
                return {}
            
            concept = self.concepts[concept_id]
            
            # Basic concept information
            analytics = {
                'concept_info': {
                    'name': concept.name,
                    'description': concept.description,
                    'type': concept.concept_type.value,
                    'domain': concept.domain,
                    'difficulty_level': concept.difficulty_level,
                    'grade_level': concept.grade_level
                },
                'educational_metadata': {
                    'learning_objectives': concept.learning_objectives,
                    'common_misconceptions': concept.common_misconceptions,
                    'prerequisites': concept.prerequisites
                }
            }
            
            # Relationship analysis
            incoming_relationships = [r for r in self.relationships.values() 
                                    if r.target_concept_id == concept_id]
            outgoing_relationships = [r for r in self.relationships.values() 
                                    if r.source_concept_id == concept_id]
            
            analytics['relationships'] = {
                'incoming_count': len(incoming_relationships),
                'outgoing_count': len(outgoing_relationships),
                'relationship_types': {
                    'incoming': Counter(r.relation_type.value for r in incoming_relationships),
                    'outgoing': Counter(r.relation_type.value for r in outgoing_relationships)
                }
            }
            
            # Embedding analysis
            if concept_id in self.concept_embeddings:
                embedding = self.concept_embeddings[concept_id]
                similar_concepts = await self.find_similar_concepts(concept_name, 5)
                
                analytics['embedding_analysis'] = {
                    'has_embedding': True,
                    'embedding_dim': embedding.embedding_dim,
                    'model_type': embedding.model_type.value,
                    'similar_concepts': similar_concepts,
                    'last_updated': embedding.last_updated
                }
            else:
                analytics['embedding_analysis'] = {'has_embedding': False}
            
            # Graph centrality measures
            if concept_id in self.graph:
                analytics['graph_metrics'] = {
                    'degree_centrality': nx.degree_centrality(self.graph)[concept_id],
                    'betweenness_centrality': nx.betweenness_centrality(self.graph)[concept_id],
                    'closeness_centrality': nx.closeness_centrality(self.graph)[concept_id],
                    'in_degree': self.graph.in_degree(concept_id),
                    'out_degree': self.graph.out_degree(concept_id)
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Concept analytics failed: {e}")
            return {}
    
    async def save_embeddings(self, filepath: str):
        """Save embeddings to file"""
        try:
            save_data = {
                'concepts': {k: {
                    'name': v.name,
                    'description': v.description,
                    'concept_type': v.concept_type.value,
                    'domain': v.domain,
                    'difficulty_level': v.difficulty_level,
                    'grade_level': v.grade_level
                } for k, v in self.concepts.items()},
                'relationships': {k: {
                    'source_concept_id': v.source_concept_id,
                    'target_concept_id': v.target_concept_id,
                    'relation_type': v.relation_type.value,
                    'strength': v.strength,
                    'confidence': v.confidence
                } for k, v in self.relationships.items()},
                'embeddings': {k: {
                    'vector': v.embedding_vector.tolist(),
                    'model_type': v.model_type.value,
                    'embedding_dim': v.embedding_dim
                } for k, v in self.concept_embeddings.items()},
                'mappings': {
                    'concept_to_id': self.concept_to_id,
                    'id_to_concept': self.id_to_concept
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"💾 Saved embeddings to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings: {e}")
    
    async def load_embeddings(self, filepath: str):
        """Load embeddings from file"""
        try:
            with open(filepath, 'r') as f:
                load_data = json.load(f)
            
            # Reconstruct concepts
            self.concepts = {}
            for concept_id, concept_data in load_data['concepts'].items():
                concept = PhysicsConcept(
                    concept_id=concept_id,
                    name=concept_data['name'],
                    description=concept_data['description'],
                    concept_type=ConceptType(concept_data['concept_type']),
                    domain=concept_data['domain'],
                    difficulty_level=concept_data['difficulty_level'],
                    grade_level=concept_data['grade_level'],
                    learning_objectives=[],
                    common_misconceptions=[]
                )
                self.concepts[concept_id] = concept
            
            # Reconstruct relationships
            self.relationships = {}
            for rel_id, rel_data in load_data['relationships'].items():
                relationship = ConceptRelationship(
                    relationship_id=rel_id,
                    source_concept_id=rel_data['source_concept_id'],
                    target_concept_id=rel_data['target_concept_id'],
                    relation_type=RelationType(rel_data['relation_type']),
                    strength=rel_data['strength'],
                    confidence=rel_data['confidence'],
                    evidence_type="expert_knowledge",
                    evidence_sources=[]
                )
                self.relationships[rel_id] = relationship
            
            # Reconstruct embeddings
            self.concept_embeddings = {}
            for concept_id, emb_data in load_data['embeddings'].items():
                embedding = ConceptEmbedding(
                    concept_id=concept_id,
                    embedding_vector=np.array(emb_data['vector']),
                    embedding_dim=emb_data['embedding_dim'],
                    model_type=EmbeddingModel(emb_data['model_type'])
                )
                self.concept_embeddings[concept_id] = embedding
            
            # Restore mappings
            self.concept_to_id = load_data['mappings']['concept_to_id']
            self.id_to_concept = load_data['mappings']['id_to_concept']
            
            # Rebuild graph
            await self._build_networkx_graph()
            
            logger.info(f"📂 Loaded embeddings from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")

# Testing function
async def test_knowledge_graph_embeddings():
    """Test knowledge graph embeddings system"""
    try:
        logger.info("🧪 Testing Knowledge Graph Embeddings")
        
        # Initialize embedder
        embedder = KnowledgeGraphEmbedder(embedding_dim=64)
        
        # Initialize knowledge graph
        await embedder.initialize_physics_knowledge_graph()
        
        # Train embeddings
        training_results = await embedder.train_embeddings(
            model_type=EmbeddingModel.TRANSDUCTION,
            epochs=100,
            learning_rate=0.01
        )
        
        logger.info(f"🔥 Training completed - Final loss: {training_results['final_loss']:.4f}")
        
        # Test similarity search
        similar_concepts = await embedder.find_similar_concepts("Force", top_k=3)
        logger.info(f"🔍 Concepts similar to 'Force': {similar_concepts}")
        
        # Test prerequisite finding
        prerequisites = await embedder.find_concept_prerequisites("Newton's Second Law")
        logger.info(f"📚 Prerequisites for 'Newton's Second Law': {prerequisites}")
        
        # Test learning path recommendation
        learning_path = await embedder.recommend_learning_path(
            "Conservation of Momentum", 
            ["Mass", "Force"]  # Student has mastered these
        )
        logger.info(f"🎯 Learning path to 'Conservation of Momentum': {learning_path}")
        
        # Test clustering
        clusters = await embedder.cluster_concepts(n_clusters=3)
        logger.info(f"🎯 Concept clusters:")
        for cluster_name, cluster_info in clusters.items():
            logger.info(f"  {cluster_name}: {cluster_info['concepts']} (domain: {cluster_info['primary_domain']})")
        
        # Test concept analytics
        analytics = await embedder.get_concept_analytics("Energy")
        logger.info(f"📊 Analytics for 'Energy': {analytics['concept_info']['difficulty_level']} difficulty")
        
        # Test visualization
        viz_data = await embedder.visualize_embeddings("pca")
        logger.info(f"📈 Visualization: {len(viz_data['coordinates'])} concepts in 2D space")
        
        logger.info("✅ Knowledge Graph Embeddings test completed")
        
    except Exception as e:
        logger.error(f"❌ Knowledge Graph Embeddings test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_knowledge_graph_embeddings())