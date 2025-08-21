#!/usr/bin/env python3
"""
Cypher Query Patterns for Graph RAG Implementation
Provides common query patterns for educational content retrieval from the physics knowledge graph.
"""
import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv('.env.example')

class PhysicsRAGQueries:
    """Collection of Cypher queries for Graph RAG content retrieval"""
    
    def __init__(self):
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'physics_graph_password_2024')
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        self.driver.close()
    
    def find_concept_by_name(self, concept_name: str) -> Dict[str, Any]:
        """Find a specific concept and its related information"""
        query = """
        MATCH (c:Concept {name: $concept_name})
        OPTIONAL MATCH (c)-[:HAS_FORMULA]->(f:Formula)
        OPTIONAL MATCH (c)-[:HAS_PROBLEM]->(p:Problem)
        OPTIONAL MATCH (c)-[:HAS_EXPLANATION]->(e:Explanation)
        OPTIONAL MATCH (c)<-[:PREREQUISITE_FOR]-(prereq:Concept)
        OPTIONAL MATCH (c)-[:PREREQUISITE_FOR]->(next:Concept)
        OPTIONAL MATCH (c)-[:RELATED_TO]-(related:Concept)
        RETURN c,
               collect(DISTINCT f) as formulas,
               collect(DISTINCT p) as problems,
               collect(DISTINCT e) as explanations,
               collect(DISTINCT prereq) as prerequisites,
               collect(DISTINCT next) as next_concepts,
               collect(DISTINCT related) as related_concepts
        """
        
        with self.driver.session() as session:
            result = session.run(query, concept_name=concept_name)
            record = result.single()
            
            if not record:
                return {}
            
            return {
                "concept": dict(record["c"]),
                "formulas": [dict(f) for f in record["formulas"] if f is not None],
                "problems": [dict(p) for p in record["problems"] if p is not None],
                "explanations": [dict(e) for e in record["explanations"] if e is not None],
                "prerequisites": [dict(p) for p in record["prerequisites"] if p is not None],
                "next_concepts": [dict(n) for n in record["next_concepts"] if n is not None],
                "related_concepts": [dict(r) for r in record["related_concepts"] if r is not None]
            }
    
    def find_concepts_by_difficulty(self, difficulty: str, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find concepts by difficulty level, optionally filtered by domain"""
        query = """
        MATCH (c:Concept {difficulty_level: $difficulty})
        WHERE $domain IS NULL OR c.domain = $domain
        OPTIONAL MATCH (c)-[:HAS_FORMULA]->(f:Formula)
        OPTIONAL MATCH (c)-[:HAS_EXPLANATION]->(e:Explanation)
        RETURN c, 
               collect(DISTINCT f.expression) as formulas,
               collect(DISTINCT e.title) as explanations
        ORDER BY c.name
        """
        
        with self.driver.session() as session:
            result = session.run(query, difficulty=difficulty, domain=domain)
            return [
                {
                    "concept": dict(record["c"]),
                    "formulas": [f for f in record["formulas"] if f],
                    "explanations": [e for e in record["explanations"] if e]
                }
                for record in result
            ]
    
    def get_learning_path(self, path_name: str) -> Dict[str, Any]:
        """Get a complete learning path with ordered concepts"""
        query = """
        MATCH (lp:LearningPath {name: $path_name})-[inc:INCLUDES]->(c:Concept)
        OPTIONAL MATCH (c)-[:HAS_FORMULA]->(f:Formula)
        OPTIONAL MATCH (c)-[:HAS_EXPLANATION]->(e:Explanation)
        OPTIONAL MATCH (c)-[:HAS_PROBLEM]->(p:Problem)
        RETURN lp, c, inc.order as order,
               collect(DISTINCT f) as formulas,
               collect(DISTINCT e) as explanations,
               collect(DISTINCT p) as problems
        ORDER BY inc.order
        """
        
        with self.driver.session() as session:
            result = session.run(query, path_name=path_name)
            records = list(result)
            
            if not records:
                return {}
            
            learning_path = dict(records[0]["lp"])
            concepts = []
            
            for record in records:
                concepts.append({
                    "concept": dict(record["c"]),
                    "order": record["order"],
                    "formulas": [dict(f) for f in record["formulas"] if f is not None],
                    "explanations": [dict(e) for e in record["explanations"] if e is not None],
                    "problems": [dict(p) for p in record["problems"] if p is not None]
                })
            
            return {
                "learning_path": learning_path,
                "concepts": concepts
            }
    
    def find_related_content_by_topic(self, topic_keywords: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        """Find concepts, problems, and formulas related to specific topic keywords"""
        query = """
        MATCH (c:Concept)
        WHERE ANY(keyword IN $keywords WHERE 
                  toLower(c.name) CONTAINS toLower(keyword) OR 
                  toLower(c.description) CONTAINS toLower(keyword))
        
        OPTIONAL MATCH (c)-[:HAS_FORMULA]->(f:Formula)
        OPTIONAL MATCH (c)-[:HAS_PROBLEM]->(p:Problem)
        OPTIONAL MATCH (c)-[:HAS_EXPLANATION]->(e:Explanation)
        OPTIONAL MATCH (c)-[:RELATED_TO]-(related:Concept)
        
        RETURN c,
               collect(DISTINCT f) as formulas,
               collect(DISTINCT p) as problems,
               collect(DISTINCT e) as explanations,
               collect(DISTINCT related.name) as related_concepts
        LIMIT $max_results
        """
        
        with self.driver.session() as session:
            result = session.run(query, keywords=topic_keywords, max_results=max_results)
            return [
                {
                    "concept": dict(record["c"]),
                    "formulas": [dict(f) for f in record["formulas"] if f is not None],
                    "problems": [dict(p) for p in record["problems"] if p is not None],
                    "explanations": [dict(e) for e in record["explanations"] if e is not None],
                    "related_concepts": [name for name in record["related_concepts"] if name]
                }
                for record in result
            ]
    
    def get_prerequisite_chain(self, concept_name: str) -> List[Dict[str, Any]]:
        """Get the full prerequisite chain for a concept (what you need to learn first)"""
        query = """
        MATCH path = (start:Concept)<-[:PREREQUISITE_FOR*]-(c:Concept {name: $concept_name})
        WHERE NOT (start)<-[:PREREQUISITE_FOR]-()
        WITH nodes(path) as concept_chain
        UNWIND concept_chain as concept
        RETURN DISTINCT concept
        ORDER BY LENGTH(()-[:PREREQUISITE_FOR*]->(concept))
        """
        
        with self.driver.session() as session:
            result = session.run(query, concept_name=concept_name)
            return [dict(record["concept"]) for record in result]
    
    def find_problems_by_difficulty_and_concept(self, concept_name: str, difficulty: str) -> List[Dict[str, Any]]:
        """Find problems for a specific concept and difficulty level"""
        query = """
        MATCH (c:Concept {name: $concept_name})-[:HAS_PROBLEM]->(p:Problem {difficulty_level: $difficulty})
        RETURN p, c.name as concept_name
        ORDER BY p.title
        """
        
        with self.driver.session() as session:
            result = session.run(query, concept_name=concept_name, difficulty=difficulty)
            return [
                {
                    "problem": dict(record["p"]),
                    "concept": record["concept_name"]
                }
                for record in result
            ]
    
    def get_formula_applications(self, formula_id: str) -> Dict[str, Any]:
        """Get concepts and problems that use a specific formula"""
        query = """
        MATCH (f:Formula {id: $formula_id})
        OPTIONAL MATCH (f)-[:DESCRIBES]->(c:Concept)
        OPTIONAL MATCH (c)-[:HAS_PROBLEM]->(p:Problem)
        WHERE ANY(step IN p.solution_steps WHERE toLower(step) CONTAINS toLower(f.expression) OR toLower(step) CONTAINS toLower(f.name))
        RETURN f,
               collect(DISTINCT c) as concepts,
               collect(DISTINCT p) as related_problems
        """
        
        with self.driver.session() as session:
            result = session.run(query, formula_id=formula_id)
            record = result.single()
            
            if not record:
                return {}
            
            return {
                "formula": dict(record["f"]),
                "concepts": [dict(c) for c in record["concepts"] if c is not None],
                "related_problems": [dict(p) for p in record["related_problems"] if p is not None]
            }
    
    def semantic_concept_search(self, query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for concepts using text similarity (simplified version for demonstration)"""
        query = """
        MATCH (c:Concept)
        WHERE toLower(c.name) CONTAINS toLower($query_text) OR 
              toLower(c.description) CONTAINS toLower($query_text) OR
              toLower(c.category) CONTAINS toLower($query_text)
        
        OPTIONAL MATCH (c)-[:HAS_EXPLANATION]->(e:Explanation)
        WHERE toLower(e.content) CONTAINS toLower($query_text)
        
        OPTIONAL MATCH (c)-[:HAS_FORMULA]->(f:Formula)
        
        WITH c, 
             CASE 
                WHEN toLower(c.name) = toLower($query_text) THEN 10
                WHEN toLower(c.name) CONTAINS toLower($query_text) THEN 8
                WHEN toLower(c.description) CONTAINS toLower($query_text) THEN 6
                WHEN toLower(c.category) CONTAINS toLower($query_text) THEN 4
                ELSE 2
             END as relevance_score,
             collect(DISTINCT e) as explanations,
             collect(DISTINCT f) as formulas
        
        RETURN c, explanations, formulas, relevance_score
        ORDER BY relevance_score DESC
        LIMIT $max_results
        """
        
        with self.driver.session() as session:
            result = session.run(query, query_text=query_text, max_results=max_results)
            return [
                {
                    "concept": dict(record["c"]),
                    "explanations": [dict(e) for e in record["explanations"] if e is not None],
                    "formulas": [dict(f) for f in record["formulas"] if f is not None],
                    "relevance_score": record["relevance_score"]
                }
                for record in result
            ]
    
    def get_concept_context_for_rag(self, concept_name: str, context_depth: int = 2) -> Dict[str, Any]:
        """Get comprehensive context for a concept suitable for RAG prompts"""
        query = """
        MATCH (c:Concept {name: $concept_name})
        
        // Get direct relationships
        OPTIONAL MATCH (c)-[:HAS_FORMULA]->(f:Formula)
        OPTIONAL MATCH (c)-[:HAS_EXPLANATION]->(e:Explanation)
        OPTIONAL MATCH (c)-[:HAS_PROBLEM]->(p:Problem)
        
        // Get prerequisite chain
        OPTIONAL MATCH (c)<-[:REQUIRES]-(prereq:Concept)
        
        // Get next concepts
        OPTIONAL MATCH (c)-[:PREREQUISITE_FOR]->(next:Concept)
        
        // Get related concepts
        OPTIONAL MATCH (c)-[:RELATED_TO]-(related:Concept)
        
        // Get domain context
        OPTIONAL MATCH (c)<-[:CONTAINS]-(sd:Subdomain)<-[:CONTAINS]-(d:Domain)
        
        RETURN c,
               collect(DISTINCT f) as formulas,
               collect(DISTINCT e) as explanations,
               collect(DISTINCT p) as problems,
               collect(DISTINCT prereq) as prerequisites,
               collect(DISTINCT next) as next_concepts,
               collect(DISTINCT related) as related_concepts,
               d.name as domain,
               sd.name as subdomain
        """
        
        with self.driver.session() as session:
            result = session.run(query, concept_name=concept_name)
            record = result.single()
            
            if not record:
                return {}
            
            # Format for RAG context
            context = {
                "primary_concept": dict(record["c"]),
                "domain": record["domain"],
                "subdomain": record["subdomain"],
                "educational_content": {
                    "formulas": [dict(f) for f in record["formulas"] if f is not None],
                    "explanations": [dict(e) for e in record["explanations"] if e is not None],
                    "problems": [dict(p) for p in record["problems"] if p is not None]
                },
                "learning_dependencies": {
                    "prerequisites": [dict(p) for p in record["prerequisites"] if p is not None],
                    "enables": [dict(n) for n in record["next_concepts"] if n is not None],
                    "related": [dict(r) for r in record["related_concepts"] if r is not None]
                }
            }
            
            return context
    
    def get_personalized_learning_suggestions(self, student_level: str, current_topic: str) -> List[Dict[str, Any]]:
        """Get personalized learning suggestions based on student level and current topic"""
        query = """
        // Find current concept
        MATCH (current:Concept)
        WHERE toLower(current.name) CONTAINS toLower($current_topic) OR 
              toLower(current.description) CONTAINS toLower($current_topic)
        
        // Find appropriate next concepts
        OPTIONAL MATCH (current)-[:PREREQUISITE_FOR]->(next:Concept)
        WHERE next.difficulty_level = $student_level OR 
              (next.difficulty_level = 'intermediate' AND $student_level = 'beginner') OR
              (next.difficulty_level = 'advanced' AND $student_level = 'intermediate')
        
        // Find related concepts at appropriate level
        OPTIONAL MATCH (current)-[:RELATED_TO]-(related:Concept)
        WHERE related.difficulty_level = $student_level
        
        // Get problems at student level
        OPTIONAL MATCH (current)-[:HAS_PROBLEM]->(p:Problem)
        WHERE p.difficulty_level = $student_level
        
        RETURN current,
               collect(DISTINCT next) as next_concepts,
               collect(DISTINCT related) as related_concepts,
               collect(DISTINCT p) as practice_problems
        """
        
        with self.driver.session() as session:
            result = session.run(query, current_topic=current_topic, student_level=student_level)
            return [
                {
                    "current_concept": dict(record["current"]),
                    "next_concepts": [dict(n) for n in record["next_concepts"] if n is not None],
                    "related_concepts": [dict(r) for r in record["related_concepts"] if r is not None],
                    "practice_problems": [dict(p) for p in record["practice_problems"] if p is not None]
                }
                for record in result
            ]

# Example usage and testing functions
def demonstrate_rag_queries():
    """Demonstrate the RAG query patterns"""
    print("Physics RAG Query Patterns Demo")
    print("=" * 40)
    
    rag_queries = PhysicsRAGQueries()
    
    try:
        # 1. Find concept by name
        print("\n1. Finding concept 'Force' with all related content:")
        force_context = rag_queries.find_concept_by_name("Force")
        print(f"   Found {len(force_context.get('formulas', []))} formulas, {len(force_context.get('problems', []))} problems")
        
        # 2. Search by difficulty
        print("\n2. Finding beginner-level concepts:")
        beginner_concepts = rag_queries.find_concepts_by_difficulty("beginner")
        print(f"   Found {len(beginner_concepts)} beginner concepts")
        
        # 3. Get learning path
        print("\n3. Getting 'Mechanics Fundamentals' learning path:")
        learning_path = rag_queries.get_learning_path("Mechanics Fundamentals")
        if learning_path:
            print(f"   Path has {len(learning_path.get('concepts', []))} concepts")
        
        # 4. Semantic search
        print("\n4. Searching for 'motion' concepts:")
        motion_results = rag_queries.semantic_concept_search("motion")
        print(f"   Found {len(motion_results)} related concepts")
        
        # 5. Get RAG context
        print("\n5. Getting RAG context for 'Velocity':")
        velocity_context = rag_queries.get_concept_context_for_rag("Velocity")
        if velocity_context:
            formulas_count = len(velocity_context.get('educational_content', {}).get('formulas', []))
            prereqs_count = len(velocity_context.get('learning_dependencies', {}).get('prerequisites', []))
            print(f"   Context includes {formulas_count} formulas, {prereqs_count} prerequisites")
        
        print("\n✅ All RAG query patterns working successfully!")
        
    except Exception as e:
        print(f"❌ Error demonstrating queries: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        rag_queries.close()

if __name__ == "__main__":
    demonstrate_rag_queries()