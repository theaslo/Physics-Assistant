#!/usr/bin/env python3
"""
Knowledge Graph Integration Service for Document Processing
Maps processed educational content to existing concepts in the Neo4j physics knowledge graph.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json
from datetime import datetime
import asyncio
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Import our processors
from multimodal_processor import ProcessedDocument, ExtractedContent
from latex_processor import LatexEquation
from diagram_analyzer import PhysicsDiagram

# Load environment variables
load_dotenv('.env.example')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConceptMapping:
    """Represents a mapping between document content and knowledge graph concepts"""
    content_id: str
    content_type: str  # equation, diagram, text_section, problem, solution
    graph_concept_id: str
    graph_concept_name: str
    confidence_score: float
    mapping_type: str  # direct_match, semantic_similarity, contextual_inference
    evidence: Dict[str, Any]

@dataclass
class GraphIntegrationResult:
    """Results of integrating a document with the knowledge graph"""
    document_hash: str
    document_node_id: str
    content_mappings: List[ConceptMapping]
    new_nodes_created: List[str]
    new_relationships_created: List[str]
    integration_metadata: Dict[str, Any]

class KnowledgeGraphIntegrator:
    """Service to integrate processed documents with the physics knowledge graph"""
    
    def __init__(self):
        # Neo4j connection
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'physics_graph_password_2024')
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Concept matching thresholds
        self.exact_match_threshold = 0.95
        self.semantic_match_threshold = 0.75
        self.contextual_match_threshold = 0.60
        
        # Physics concept mappings for better matching
        self.concept_synonyms = {
            'velocity': ['speed', 'v', 'vel'],
            'acceleration': ['a', 'acc'],
            'force': ['F', 'forces'],
            'momentum': ['p', 'linear momentum'],
            'energy': ['E', 'kinetic energy', 'potential energy'],
            'work': ['W'],
            'power': ['P'],
            'frequency': ['f', 'freq'],
            'wavelength': ['lambda', 'λ'],
            'amplitude': ['A', 'amp'],
            'period': ['T'],
            'mass': ['m'],
            'time': ['t']
        }
        
        # Formula to concept mappings
        self.formula_concepts = {
            'F = ma': ['force', 'mass', 'acceleration', 'newton_second_law'],
            'v = v0 + at': ['velocity', 'acceleration', 'kinematics'],
            'E = mc²': ['energy', 'mass', 'relativity'],
            'KE = ½mv²': ['kinetic_energy', 'mass', 'velocity'],
            'PE = mgh': ['potential_energy', 'mass', 'gravity', 'height'],
            'W = Fd': ['work', 'force', 'displacement'],
            'P = W/t': ['power', 'work', 'time'],
            'v = fλ': ['wave_speed', 'frequency', 'wavelength']
        }
        
        logger.info("Knowledge Graph Integrator initialized")
    
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def integrate_document(self, processed_doc: ProcessedDocument) -> GraphIntegrationResult:
        """Main method to integrate a processed document with the knowledge graph"""
        try:
            logger.info(f"Integrating document {processed_doc.file_path} with knowledge graph")
            
            with self.driver.session() as session:
                # Step 1: Create document node in graph
                doc_node_id = self._create_document_node(session, processed_doc)
                
                # Step 2: Map content to existing concepts
                content_mappings = self._map_content_to_concepts(session, processed_doc)
                
                # Step 3: Create new nodes for unmapped content
                new_nodes = self._create_new_content_nodes(session, processed_doc, content_mappings)
                
                # Step 4: Create relationships
                new_relationships = self._create_content_relationships(session, doc_node_id, content_mappings, new_nodes)
                
                # Step 5: Update learning paths and prerequisites
                self._update_learning_structures(session, processed_doc, content_mappings)
                
                # Create integration metadata
                integration_metadata = {
                    'integration_timestamp': datetime.now().isoformat(),
                    'total_content_items': len(processed_doc.content.equations) + len(processed_doc.content.diagrams) + len(processed_doc.content.sections),
                    'mapped_items': len(content_mappings),
                    'new_nodes': len(new_nodes),
                    'new_relationships': len(new_relationships),
                    'integration_success': True
                }
                
                return GraphIntegrationResult(
                    document_hash=processed_doc.document_hash,
                    document_node_id=doc_node_id,
                    content_mappings=content_mappings,
                    new_nodes_created=new_nodes,
                    new_relationships_created=new_relationships,
                    integration_metadata=integration_metadata
                )
                
        except Exception as e:
            logger.error(f"Error integrating document {processed_doc.file_path}: {str(e)}")
            raise
    
    def _create_document_node(self, session, processed_doc: ProcessedDocument) -> str:
        """Create a document node in the knowledge graph"""
        doc_node_id = f"doc_{processed_doc.document_hash[:12]}"
        
        query = """
        CREATE (d:Document {
            id: $doc_id,
            file_path: $file_path,
            document_hash: $doc_hash,
            difficulty_level: $difficulty,
            educational_type: $edu_type,
            physics_concepts: $concepts,
            text_length: $text_length,
            equations_count: $eq_count,
            diagrams_count: $diag_count,
            processing_timestamp: $timestamp,
            created_at: datetime()
        })
        RETURN d.id as doc_id
        """
        
        result = session.run(query, 
            doc_id=doc_node_id,
            file_path=processed_doc.file_path,
            doc_hash=processed_doc.document_hash,
            difficulty=processed_doc.difficulty_level,
            edu_type=processed_doc.educational_classification.get('primary_type', 'unknown'),
            concepts=processed_doc.physics_concepts,
            text_length=len(processed_doc.content.text_content),
            eq_count=len(processed_doc.content.equations),
            diag_count=len(processed_doc.content.diagrams),
            timestamp=processed_doc.processing_timestamp.isoformat()
        )
        
        logger.info(f"Created document node: {doc_node_id}")
        return doc_node_id
    
    def _map_content_to_concepts(self, session, processed_doc: ProcessedDocument) -> List[ConceptMapping]:
        """Map document content to existing concepts in the knowledge graph"""
        mappings = []
        
        # Map equations to concepts
        for i, equation in enumerate(processed_doc.content.equations):
            equation_mappings = self._map_equation_to_concepts(session, equation, f"eq_{i}")
            mappings.extend(equation_mappings)
        
        # Map diagrams to concepts
        for i, diagram in enumerate(processed_doc.content.diagrams):
            diagram_mappings = self._map_diagram_to_concepts(session, diagram, f"diag_{i}")
            mappings.extend(diagram_mappings)
        
        # Map text sections to concepts
        for section in processed_doc.content.sections:
            if section.section_type == "text":
                section_mappings = self._map_text_to_concepts(session, section)
                mappings.extend(section_mappings)
        
        logger.info(f"Created {len(mappings)} content mappings")
        return mappings
    
    def _map_equation_to_concepts(self, session, equation: LatexEquation, content_id: str) -> List[ConceptMapping]:
        """Map a specific equation to concepts in the graph"""
        mappings = []
        
        # Direct formula matching
        formula_mappings = self._match_formula_directly(session, equation, content_id)
        mappings.extend(formula_mappings)
        
        # Variable-based concept matching
        variable_mappings = self._match_equation_variables(session, equation, content_id)
        mappings.extend(variable_mappings)
        
        # Domain-based matching
        if equation.physics_domain:
            domain_mappings = self._match_physics_domain(session, equation.physics_domain, content_id, "equation")
            mappings.extend(domain_mappings)
        
        return mappings
    
    def _match_formula_directly(self, session, equation: LatexEquation, content_id: str) -> List[ConceptMapping]:
        """Match equation directly against known formulas in the graph"""
        mappings = []
        
        # Get all formulas from graph
        query = """
        MATCH (f:Formula)
        RETURN f.id as formula_id, f.expression as expression, f.name as name
        """
        
        results = session.run(query)
        
        for record in results:
            formula_expr = record['expression']
            
            # Calculate similarity between equations
            similarity = self._calculate_formula_similarity(equation.cleaned_latex, formula_expr)
            
            if similarity >= self.exact_match_threshold:
                # Find concepts connected to this formula
                concept_query = """
                MATCH (f:Formula {id: $formula_id})-[:DESCRIBES]->(c:Concept)
                RETURN c.name as concept_name, c.id as concept_id
                """
                
                concept_results = session.run(concept_query, formula_id=record['formula_id'])
                
                for concept_record in concept_results:
                    mappings.append(ConceptMapping(
                        content_id=content_id,
                        content_type="equation",
                        graph_concept_id=concept_record['concept_id'] if concept_record['concept_id'] else concept_record['concept_name'],
                        graph_concept_name=concept_record['concept_name'],
                        confidence_score=similarity,
                        mapping_type="direct_match",
                        evidence={
                            "matched_formula": record['expression'],
                            "similarity_score": similarity,
                            "equation_latex": equation.original_latex
                        }
                    ))
        
        return mappings
    
    def _match_equation_variables(self, session, equation: LatexEquation, content_id: str) -> List[ConceptMapping]:
        """Match equation based on variables to concepts"""
        mappings = []
        
        for variable in equation.variables:
            # Find concepts that commonly use this variable
            concepts = self._find_concepts_for_variable(session, variable)
            
            for concept_name, concept_id in concepts:
                confidence = self._calculate_variable_concept_confidence(variable, concept_name, equation)
                
                if confidence >= self.contextual_match_threshold:
                    mappings.append(ConceptMapping(
                        content_id=content_id,
                        content_type="equation",
                        graph_concept_id=concept_id,
                        graph_concept_name=concept_name,
                        confidence_score=confidence,
                        mapping_type="contextual_inference",
                        evidence={
                            "variable": variable,
                            "all_variables": equation.variables,
                            "physics_domain": equation.physics_domain
                        }
                    ))
        
        return mappings
    
    def _map_diagram_to_concepts(self, session, diagram: PhysicsDiagram, content_id: str) -> List[ConceptMapping]:
        """Map diagram to concepts based on its analysis"""
        mappings = []
        
        # Map based on diagram type
        if diagram.diagram_type != 'unknown':
            type_mappings = self._match_diagram_type(session, diagram.diagram_type, content_id)
            mappings.extend(type_mappings)
        
        # Map based on identified physics concepts
        for concept in diagram.physics_concepts:
            concept_mappings = self._match_concept_by_name(session, concept, content_id, "diagram")
            mappings.extend(concept_mappings)
        
        # Map based on objects in diagram
        for obj in diagram.objects:
            object_mappings = self._match_physics_object(session, obj, content_id)
            mappings.extend(object_mappings)
        
        return mappings
    
    def _map_text_to_concepts(self, session, section) -> List[ConceptMapping]:
        """Map text sections to concepts using NLP-like approach"""
        mappings = []
        
        # Simple keyword-based matching for now
        text_lower = section.content.lower()
        
        # Get all concepts from graph
        query = """
        MATCH (c:Concept)
        RETURN c.name as name, c.description as description, c.id as id
        """
        
        results = session.run(query)
        
        for record in results:
            concept_name = record['name'].lower()
            
            # Check for exact mention
            if concept_name in text_lower:
                confidence = 0.8
                mapping_type = "direct_match"
            else:
                # Check synonyms
                confidence = self._calculate_text_concept_similarity(text_lower, concept_name)
                mapping_type = "semantic_similarity" if confidence >= self.semantic_match_threshold else None
            
            if mapping_type and confidence >= self.contextual_match_threshold:
                mappings.append(ConceptMapping(
                    content_id=section.section_id,
                    content_type="text_section",
                    graph_concept_id=record['id'] if record['id'] else record['name'],
                    graph_concept_name=record['name'],
                    confidence_score=confidence,
                    mapping_type=mapping_type,
                    evidence={
                        "text_preview": section.content[:200],
                        "section_type": section.section_type
                    }
                ))
        
        return mappings
    
    def _calculate_formula_similarity(self, formula1: str, formula2: str) -> float:
        """Calculate similarity between two formulas"""
        # Normalize formulas for comparison
        norm1 = self._normalize_formula(formula1)
        norm2 = self._normalize_formula(formula2)
        
        # Simple string similarity for now
        if norm1 == norm2:
            return 1.0
        
        # Check for key components
        common_components = 0
        total_components = 0
        
        components1 = set(norm1.split())
        components2 = set(norm2.split())
        
        if components1 and components2:
            common_components = len(components1.intersection(components2))
            total_components = len(components1.union(components2))
            return common_components / total_components if total_components > 0 else 0.0
        
        return 0.0
    
    def _normalize_formula(self, formula: str) -> str:
        """Normalize formula for comparison"""
        import re
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', formula.strip())
        
        # Standardize common variations
        replacements = {
            '**': '^',
            '·': '*',
            '×': '*',
            '÷': '/',
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized.lower()
    
    def _find_concepts_for_variable(self, session, variable: str) -> List[Tuple[str, str]]:
        """Find concepts that commonly use a specific variable"""
        # Check our physics symbol mappings
        concept_hints = []
        
        if variable in self.concept_synonyms:
            for concept in self.concept_synonyms[variable]:
                concept_hints.append(concept)
        
        # Add direct mapping
        variable_mappings = {
            'v': 'velocity',
            'a': 'acceleration',
            'F': 'force',
            'm': 'mass',
            't': 'time',
            'p': 'momentum',
            'E': 'energy',
            'W': 'work',
            'P': 'power'
        }
        
        if variable in variable_mappings:
            concept_hints.append(variable_mappings[variable])
        
        # Query graph for these concepts
        concepts = []
        for hint in concept_hints:
            query = """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($hint)
            RETURN c.name as name, c.id as id
            LIMIT 3
            """
            
            results = session.run(query, hint=hint)
            for record in results:
                concepts.append((record['name'], record['id'] if record['id'] else record['name']))
        
        return concepts
    
    def _calculate_variable_concept_confidence(self, variable: str, concept_name: str, equation: LatexEquation) -> float:
        """Calculate confidence that a variable relates to a concept"""
        confidence = 0.0
        
        # Direct variable-concept mapping
        if variable.lower() in concept_name.lower():
            confidence += 0.6
        
        # Physics domain alignment
        if equation.physics_domain:
            domain_concepts = {
                'mechanics': ['force', 'velocity', 'acceleration', 'momentum'],
                'energy': ['energy', 'work', 'power'],
                'waves': ['frequency', 'wavelength', 'amplitude'],
            }
            
            if equation.physics_domain in domain_concepts:
                if any(term in concept_name.lower() for term in domain_concepts[equation.physics_domain]):
                    confidence += 0.3
        
        # Equation complexity bonus
        if equation.complexity_score > 5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _match_concept_by_name(self, session, concept_name: str, content_id: str, content_type: str) -> List[ConceptMapping]:
        """Match a concept by name in the graph"""
        mappings = []
        
        query = """
        MATCH (c:Concept)
        WHERE toLower(c.name) = toLower($concept_name)
        OR toLower(c.name) CONTAINS toLower($concept_name)
        RETURN c.name as name, c.id as id
        """
        
        results = session.run(query, concept_name=concept_name)
        
        for record in results:
            confidence = 0.9 if record['name'].lower() == concept_name.lower() else 0.7
            
            mappings.append(ConceptMapping(
                content_id=content_id,
                content_type=content_type,
                graph_concept_id=record['id'] if record['id'] else record['name'],
                graph_concept_name=record['name'],
                confidence_score=confidence,
                mapping_type="direct_match" if confidence > 0.8 else "semantic_similarity",
                evidence={"matched_concept": concept_name}
            ))
        
        return mappings
    
    def _match_physics_domain(self, session, domain: str, content_id: str, content_type: str) -> List[ConceptMapping]:
        """Match content to concepts based on physics domain"""
        mappings = []
        
        query = """
        MATCH (c:Concept)
        WHERE toLower(c.domain) = toLower($domain)
        OR toLower(c.category) CONTAINS toLower($domain)
        RETURN c.name as name, c.id as id
        LIMIT 5
        """
        
        results = session.run(query, domain=domain)
        
        for record in results:
            mappings.append(ConceptMapping(
                content_id=content_id,
                content_type=content_type,
                graph_concept_id=record['id'] if record['id'] else record['name'],
                graph_concept_name=record['name'],
                confidence_score=0.6,  # Lower confidence for domain-based matching
                mapping_type="contextual_inference",
                evidence={"physics_domain": domain}
            ))
        
        return mappings
    
    def _match_diagram_type(self, session, diagram_type: str, content_id: str) -> List[ConceptMapping]:
        """Match diagram type to relevant concepts"""
        diagram_concept_map = {
            'free_body': ['force', 'equilibrium', 'newton_laws'],
            'circuit': ['current', 'voltage', 'resistance'],
            'wave': ['frequency', 'wavelength', 'amplitude'],
            'field': ['electric_field', 'magnetic_field'],
            'kinematics': ['velocity', 'acceleration', 'motion']
        }
        
        mappings = []
        concepts = diagram_concept_map.get(diagram_type, [])
        
        for concept in concepts:
            concept_mappings = self._match_concept_by_name(session, concept, content_id, "diagram")
            mappings.extend(concept_mappings)
        
        return mappings
    
    def _match_physics_object(self, session, obj: str, content_id: str) -> List[ConceptMapping]:
        """Match physics objects to relevant concepts"""
        object_concept_map = {
            'block': ['force', 'friction', 'newton_laws'],
            'sphere': ['circular_motion', 'moment_of_inertia'],
            'incline': ['inclined_plane', 'friction', 'components'],
            'pulley': ['tension', 'mechanical_advantage'],
            'spring': ['hooke_law', 'elastic_potential_energy']
        }
        
        mappings = []
        concepts = object_concept_map.get(obj, [])
        
        for concept in concepts:
            concept_mappings = self._match_concept_by_name(session, concept, content_id, "diagram")
            mappings.extend(concept_mappings)
        
        return mappings
    
    def _calculate_text_concept_similarity(self, text: str, concept_name: str) -> float:
        """Calculate similarity between text and concept"""
        # Simple keyword-based similarity
        concept_words = concept_name.lower().split()
        
        similarity = 0.0
        for word in concept_words:
            if word in text:
                similarity += 1.0 / len(concept_words)
        
        # Check synonyms
        if concept_name in self.concept_synonyms:
            for synonym in self.concept_synonyms[concept_name]:
                if synonym.lower() in text:
                    similarity += 0.5 / len(concept_words)
        
        return min(similarity, 1.0)
    
    def _create_new_content_nodes(self, session, processed_doc: ProcessedDocument, existing_mappings: List[ConceptMapping]) -> List[str]:
        """Create new nodes for content that wasn't mapped to existing concepts"""
        new_nodes = []
        mapped_content_ids = {mapping.content_id for mapping in existing_mappings}
        
        # Create nodes for unmapped equations
        for i, equation in enumerate(processed_doc.content.equations):
            content_id = f"eq_{i}"
            if content_id not in mapped_content_ids:
                node_id = self._create_equation_node(session, equation, content_id, processed_doc.document_hash)
                new_nodes.append(node_id)
        
        # Create nodes for unmapped diagrams
        for i, diagram in enumerate(processed_doc.content.diagrams):
            content_id = f"diag_{i}"
            if content_id not in mapped_content_ids:
                node_id = self._create_diagram_node(session, diagram, content_id, processed_doc.document_hash)
                new_nodes.append(node_id)
        
        return new_nodes
    
    def _create_equation_node(self, session, equation: LatexEquation, content_id: str, doc_hash: str) -> str:
        """Create a new equation node in the graph"""
        node_id = f"eq_{doc_hash[:8]}_{content_id}"
        
        query = """
        CREATE (eq:DocumentEquation {
            id: $node_id,
            original_latex: $original,
            cleaned_latex: $cleaned,
            variables: $variables,
            equation_type: $eq_type,
            physics_domain: $domain,
            complexity_score: $complexity,
            is_valid: $valid,
            document_hash: $doc_hash,
            created_at: datetime()
        })
        RETURN eq.id as id
        """
        
        session.run(query,
            node_id=node_id,
            original=equation.original_latex,
            cleaned=equation.cleaned_latex,
            variables=equation.variables,
            eq_type=equation.equation_type,
            domain=equation.physics_domain,
            complexity=equation.complexity_score,
            valid=equation.is_valid,
            doc_hash=doc_hash
        )
        
        return node_id
    
    def _create_diagram_node(self, session, diagram: PhysicsDiagram, content_id: str, doc_hash: str) -> str:
        """Create a new diagram node in the graph"""
        node_id = f"diag_{doc_hash[:8]}_{content_id}"
        
        query = """
        CREATE (diag:DocumentDiagram {
            id: $node_id,
            image_path: $image_path,
            diagram_type: $diag_type,
            objects: $objects,
            physics_concepts: $concepts,
            complexity_score: $complexity,
            vector_count: $vector_count,
            document_hash: $doc_hash,
            created_at: datetime()
        })
        RETURN diag.id as id
        """
        
        session.run(query,
            node_id=node_id,
            image_path=diagram.image_path,
            diag_type=diagram.diagram_type,
            objects=diagram.objects,
            concepts=diagram.physics_concepts,
            complexity=diagram.complexity_score,
            vector_count=len(diagram.vectors),
            doc_hash=doc_hash
        )
        
        return node_id
    
    def _create_content_relationships(self, session, doc_node_id: str, mappings: List[ConceptMapping], new_nodes: List[str]) -> List[str]:
        """Create relationships between document content and concepts"""
        relationships = []
        
        # Create relationships for mapped content
        for mapping in mappings:
            rel_id = self._create_content_concept_relationship(session, doc_node_id, mapping)
            relationships.append(rel_id)
        
        # Create relationships for new nodes
        for node_id in new_nodes:
            rel_id = self._create_document_content_relationship(session, doc_node_id, node_id)
            relationships.append(rel_id)
        
        return relationships
    
    def _create_content_concept_relationship(self, session, doc_node_id: str, mapping: ConceptMapping) -> str:
        """Create relationship between document content and existing concept"""
        if mapping.content_type == "equation":
            rel_type = "CONTAINS_EQUATION"
        elif mapping.content_type == "diagram":
            rel_type = "CONTAINS_DIAGRAM" 
        else:
            rel_type = "CONTAINS_CONTENT"
        
        query = f"""
        MATCH (d:Document {{id: $doc_id}})
        MATCH (c:Concept {{name: $concept_name}})
        CREATE (d)-[r:{rel_type} {{
            content_id: $content_id,
            confidence_score: $confidence,
            mapping_type: $mapping_type,
            evidence: $evidence,
            created_at: datetime()
        }}]->(c)
        RETURN id(r) as rel_id
        """
        
        result = session.run(query,
            doc_id=doc_node_id,
            concept_name=mapping.graph_concept_name,
            content_id=mapping.content_id,
            confidence=mapping.confidence_score,
            mapping_type=mapping.mapping_type,
            evidence=json.dumps(mapping.evidence)
        )
        
        return str(result.single()['rel_id'])
    
    def _create_document_content_relationship(self, session, doc_node_id: str, content_node_id: str) -> str:
        """Create relationship between document and its content nodes"""
        query = """
        MATCH (d:Document {id: $doc_id})
        MATCH (c) WHERE c.id = $content_id
        CREATE (d)-[r:CONTAINS {created_at: datetime()}]->(c)
        RETURN id(r) as rel_id
        """
        
        result = session.run(query, doc_id=doc_node_id, content_id=content_node_id)
        return str(result.single()['rel_id'])
    
    def _update_learning_structures(self, session, processed_doc: ProcessedDocument, mappings: List[ConceptMapping]):
        """Update learning paths and prerequisites based on document content"""
        # This would implement more sophisticated learning structure updates
        # For now, we'll create basic connections
        
        # Group mappings by concept
        concept_mappings = {}
        for mapping in mappings:
            concept = mapping.graph_concept_name
            if concept not in concept_mappings:
                concept_mappings[concept] = []
            concept_mappings[concept].append(mapping)
        
        # Create relationships between concepts that appear together
        concepts = list(concept_mappings.keys())
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                self._create_concept_cooccurrence(session, concept1, concept2, processed_doc.document_hash)
    
    def _create_concept_cooccurrence(self, session, concept1: str, concept2: str, doc_hash: str):
        """Create co-occurrence relationship between concepts"""
        query = """
        MATCH (c1:Concept {name: $concept1})
        MATCH (c2:Concept {name: $concept2})
        MERGE (c1)-[r:CO_OCCURS_WITH]-(c2)
        ON CREATE SET r.count = 1, r.documents = [$doc_hash]
        ON MATCH SET r.count = r.count + 1, r.documents = r.documents + $doc_hash
        """
        
        session.run(query, concept1=concept1, concept2=concept2, doc_hash=doc_hash)
    
    def get_integration_statistics(self, session) -> Dict[str, Any]:
        """Get statistics about document integration in the graph"""
        stats_queries = {
            'total_documents': "MATCH (d:Document) RETURN count(d) as count",
            'total_equations': "MATCH (eq:DocumentEquation) RETURN count(eq) as count",
            'total_diagrams': "MATCH (diag:DocumentDiagram) RETURN count(diag) as count",
            'concept_connections': "MATCH ()-[r:CONTAINS_EQUATION|CONTAINS_DIAGRAM|CONTAINS_CONTENT]->() RETURN count(r) as count"
        }
        
        stats = {}
        for stat_name, query in stats_queries.items():
            result = session.run(query)
            stats[stat_name] = result.single()['count']
        
        return stats

# Example usage
if __name__ == "__main__":
    integrator = KnowledgeGraphIntegrator()
    
    try:
        with integrator.driver.session() as session:
            stats = integrator.get_integration_statistics(session)
            print("Knowledge Graph Integration Service initialized")
            print("Current statistics:", stats)
    except Exception as e:
        print(f"Error connecting to knowledge graph: {e}")
    finally:
        integrator.close()