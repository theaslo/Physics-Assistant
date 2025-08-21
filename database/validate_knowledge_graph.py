#!/usr/bin/env python3
"""
Knowledge Graph Validation Scripts
Validates the integrity, completeness, and quality of the physics knowledge graph.
"""
import os
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
from dotenv import load_dotenv
from collections import defaultdict
import json

load_dotenv('.env.example')

class KnowledgeGraphValidator:
    """Validates the physics knowledge graph for integrity and quality"""
    
    def __init__(self):
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'physics_graph_password_2024')
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.validation_results = defaultdict(list)
    
    def close(self):
        self.driver.close()
    
    def validate_node_integrity(self) -> Dict[str, Any]:
        """Validate that all nodes have required properties and valid data"""
        print("Validating node integrity...")
        
        with self.driver.session() as session:
            # Check concepts have required properties
            concepts_missing_props = session.run("""
                MATCH (c:Concept)
                WHERE c.name IS NULL OR c.description IS NULL OR c.difficulty_level IS NULL
                RETURN c.name as name, properties(c) as props
            """).data()
            
            if concepts_missing_props:
                self.validation_results["concept_integrity_errors"].extend(concepts_missing_props)
            
            # Check formulas have required properties
            formulas_missing_props = session.run("""
                MATCH (f:Formula)
                WHERE f.id IS NULL OR f.expression IS NULL OR f.name IS NULL
                RETURN f.id as id, properties(f) as props
            """).data()
            
            if formulas_missing_props:
                self.validation_results["formula_integrity_errors"].extend(formulas_missing_props)
            
            # Check problems have required properties
            problems_missing_props = session.run("""
                MATCH (p:Problem)
                WHERE p.id IS NULL OR p.title IS NULL OR p.difficulty_level IS NULL
                RETURN p.id as id, properties(p) as props
            """).data()
            
            if problems_missing_props:
                self.validation_results["problem_integrity_errors"].extend(problems_missing_props)
            
            # Check for valid difficulty levels
            invalid_difficulties = session.run("""
                MATCH (n)
                WHERE n.difficulty_level IS NOT NULL 
                  AND NOT n.difficulty_level IN ['beginner', 'intermediate', 'advanced']
                RETURN labels(n) as node_type, 
                       COALESCE(n.name, n.id, n.title) as identifier,
                       n.difficulty_level as invalid_difficulty
            """).data()
            
            if invalid_difficulties:
                self.validation_results["invalid_difficulty_levels"].extend(invalid_difficulties)
        
        return {
            "concept_errors": len(self.validation_results["concept_integrity_errors"]),
            "formula_errors": len(self.validation_results["formula_integrity_errors"]),
            "problem_errors": len(self.validation_results["problem_integrity_errors"]),
            "invalid_difficulties": len(self.validation_results["invalid_difficulty_levels"])
        }
    
    def validate_relationship_integrity(self) -> Dict[str, Any]:
        """Validate relationships are logical and complete"""
        print("Validating relationship integrity...")
        
        with self.driver.session() as session:
            # Check for orphaned concepts (no incoming CONTAINS relationship)
            orphaned_concepts = session.run("""
                MATCH (c:Concept)
                WHERE NOT (c)<-[:CONTAINS]-()
                RETURN c.name as concept_name
            """).data()
            
            if orphaned_concepts:
                self.validation_results["orphaned_concepts"].extend(orphaned_concepts)
            
            # Check for circular prerequisites
            circular_prerequisites = session.run("""
                MATCH path = (c:Concept)-[:PREREQUISITE_FOR*]->(c)
                RETURN c.name as concept_name, length(path) as cycle_length
            """).data()
            
            if circular_prerequisites:
                self.validation_results["circular_prerequisites"].extend(circular_prerequisites)
            
            # Check for concepts without any educational content
            concepts_no_content = session.run("""
                MATCH (c:Concept)
                WHERE NOT (c)-[:HAS_FORMULA|HAS_PROBLEM|HAS_EXPLANATION]->()
                RETURN c.name as concept_name, c.domain as domain
            """).data()
            
            if concepts_no_content:
                self.validation_results["concepts_without_content"].extend(concepts_no_content)
            
            # Check for formulas not linked to concepts
            unlinked_formulas = session.run("""
                MATCH (f:Formula)
                WHERE NOT (f)-[:DESCRIBES]->()
                RETURN f.id as formula_id, f.name as formula_name
            """).data()
            
            if unlinked_formulas:
                self.validation_results["unlinked_formulas"].extend(unlinked_formulas)
            
            # Check for problems not linked to concepts
            unlinked_problems = session.run("""
                MATCH (p:Problem)
                WHERE NOT (p)-[:APPLIES_CONCEPT]->()
                RETURN p.id as problem_id, p.title as problem_title
            """).data()
            
            if unlinked_problems:
                self.validation_results["unlinked_problems"].extend(unlinked_problems)
        
        return {
            "orphaned_concepts": len(self.validation_results["orphaned_concepts"]),
            "circular_prerequisites": len(self.validation_results["circular_prerequisites"]),
            "concepts_without_content": len(self.validation_results["concepts_without_content"]),
            "unlinked_formulas": len(self.validation_results["unlinked_formulas"]),
            "unlinked_problems": len(self.validation_results["unlinked_problems"])
        }
    
    def validate_educational_quality(self) -> Dict[str, Any]:
        """Validate the educational quality and completeness of content"""
        print("Validating educational quality...")
        
        with self.driver.session() as session:
            # Check difficulty progression in learning paths
            invalid_progressions = session.run("""
                MATCH (lp:LearningPath)-[inc:INCLUDES]->(c:Concept)
                WITH lp, c, inc.order as order
                ORDER BY lp.name, order
                WITH lp, collect({concept: c, order: order}) as concepts
                UNWIND range(0, size(concepts)-2) as i
                WITH lp, concepts[i] as curr, concepts[i+1] as next
                WHERE (curr.concept.difficulty_level = 'advanced' AND next.concept.difficulty_level = 'beginner') OR
                      (curr.concept.difficulty_level = 'intermediate' AND next.concept.difficulty_level = 'beginner')
                RETURN lp.name as path_name, 
                       curr.concept.name as current_concept,
                       curr.concept.difficulty_level as current_difficulty,
                       next.concept.name as next_concept,
                       next.concept.difficulty_level as next_difficulty
            """).data()
            
            if invalid_progressions:
                self.validation_results["invalid_difficulty_progressions"].extend(invalid_progressions)
            
            # Check for domains with insufficient content
            domain_content_counts = session.run("""
                MATCH (c:Concept)
                WHERE c.domain IS NOT NULL
                WITH c.domain as domain, count(c) as concept_count
                OPTIONAL MATCH (c2:Concept {domain: domain})-[:HAS_PROBLEM]->(p:Problem)
                WITH domain, concept_count, count(p) as problem_count
                OPTIONAL MATCH (c3:Concept {domain: domain})-[:HAS_FORMULA]->(f:Formula)
                WITH domain, concept_count, problem_count, count(f) as formula_count
                WHERE concept_count < 5 OR problem_count < 2 OR formula_count < 2
                RETURN domain, concept_count, problem_count, formula_count
            """).data()
            
            if domain_content_counts:
                self.validation_results["insufficient_domain_content"].extend(domain_content_counts)
            
            # Check for missing prerequisite relationships (simple heuristic)
            potential_missing_prereqs = session.run("""
                MATCH (basic:Concept {difficulty_level: 'beginner'})
                MATCH (advanced:Concept {difficulty_level: 'advanced'})
                WHERE basic.domain = advanced.domain AND basic.subdomain = advanced.subdomain
                  AND NOT (basic)-[:PREREQUISITE_FOR*]->(advanced)
                RETURN basic.name as basic_concept, 
                       advanced.name as advanced_concept,
                       basic.domain as domain
                LIMIT 10
            """).data()
            
            if potential_missing_prereqs:
                self.validation_results["potential_missing_prerequisites"].extend(potential_missing_prereqs)
            
            # Check formula expression validity (simple syntax check)
            invalid_formula_expressions = session.run("""
                MATCH (f:Formula)
                WHERE NOT f.expression =~ '.*[a-zA-Z].*' OR f.expression = ''
                RETURN f.id as formula_id, f.name as formula_name, f.expression as expression
            """).data()
            
            if invalid_formula_expressions:
                self.validation_results["invalid_formula_expressions"].extend(invalid_formula_expressions)
        
        return {
            "invalid_progressions": len(self.validation_results["invalid_difficulty_progressions"]),
            "insufficient_content_domains": len(self.validation_results["insufficient_domain_content"]),
            "potential_missing_prereqs": len(self.validation_results["potential_missing_prerequisites"]),
            "invalid_formulas": len(self.validation_results["invalid_formula_expressions"])
        }
    
    def validate_graph_completeness(self) -> Dict[str, Any]:
        """Validate that the graph meets target completeness requirements"""
        print("Validating graph completeness...")
        
        with self.driver.session() as session:
            # Count total nodes and relationships
            stats = session.run("""
                MATCH (n) 
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT n) as total_nodes, 
                       count(r) as total_relationships
            """).single()
            
            # Count by node type
            node_counts = session.run("""
                MATCH (n)
                WITH labels(n) as node_labels, count(n) as count
                RETURN node_labels[0] as node_type, count
                ORDER BY count DESC
            """).data()
            
            # Count concepts by domain
            domain_distribution = session.run("""
                MATCH (c:Concept)
                WITH c.domain as domain, count(c) as concept_count
                RETURN domain, concept_count
                ORDER BY concept_count DESC
            """).data()
            
            # Count relationships by type
            relationship_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            # Validate target requirements
            target_nodes = 200
            target_relationships = 500
            
            nodes_met = stats["total_nodes"] >= target_nodes
            relationships_met = stats["total_relationships"] >= target_relationships
            
            return {
                "total_nodes": stats["total_nodes"],
                "total_relationships": stats["total_relationships"],
                "targets_met": {
                    "nodes": nodes_met,
                    "relationships": relationships_met
                },
                "node_distribution": {item["node_type"]: item["count"] for item in node_counts},
                "domain_distribution": {item["domain"]: item["concept_count"] for item in domain_distribution if item["domain"]},
                "relationship_distribution": {item["relationship_type"]: item["count"] for item in relationship_counts}
            }
    
    def validate_rag_readiness(self) -> Dict[str, Any]:
        """Validate that the graph is ready for RAG implementations"""
        print("Validating RAG readiness...")
        
        with self.driver.session() as session:
            # Check that concepts have sufficient descriptive content
            concepts_insufficient_content = session.run("""
                MATCH (c:Concept)
                WHERE length(c.description) < 20 OR c.description IS NULL
                RETURN c.name as concept_name, length(c.description) as description_length
            """).data()
            
            if concepts_insufficient_content:
                self.validation_results["insufficient_concept_descriptions"].extend(concepts_insufficient_content)
            
            # Check for explanations with substantial content
            short_explanations = session.run("""
                MATCH (e:Explanation)
                WHERE length(e.content) < 50 OR e.content IS NULL
                RETURN e.id as explanation_id, e.title as title, length(e.content) as content_length
            """).data()
            
            if short_explanations:
                self.validation_results["short_explanations"].extend(short_explanations)
            
            # Check that problems have solution steps
            problems_no_solutions = session.run("""
                MATCH (p:Problem)
                WHERE p.solution_steps IS NULL OR size(p.solution_steps) = 0
                RETURN p.id as problem_id, p.title as title
            """).data()
            
            if problems_no_solutions:
                self.validation_results["problems_without_solutions"].extend(problems_no_solutions)
            
            # Check connectivity for RAG traversal
            weakly_connected_concepts = session.run("""
                MATCH (c:Concept)
                WHERE NOT (c)-[:RELATED_TO|PREREQUISITE_FOR|REQUIRES]-(c2:Concept)
                RETURN c.name as concept_name, c.domain as domain
            """).data()
            
            if weakly_connected_concepts:
                self.validation_results["weakly_connected_concepts"].extend(weakly_connected_concepts)
        
        return {
            "insufficient_descriptions": len(self.validation_results["insufficient_concept_descriptions"]),
            "short_explanations": len(self.validation_results["short_explanations"]),
            "problems_no_solutions": len(self.validation_results["problems_without_solutions"]),
            "weakly_connected_concepts": len(self.validation_results["weakly_connected_concepts"])
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive report"""
        print("Running comprehensive knowledge graph validation...")
        print("=" * 60)
        
        validation_report = {
            "timestamp": str(session.run("RETURN datetime()").single()[0]) if hasattr(self, 'driver') else "unknown",
            "node_integrity": self.validate_node_integrity(),
            "relationship_integrity": self.validate_relationship_integrity(),
            "educational_quality": self.validate_educational_quality(),
            "graph_completeness": self.validate_graph_completeness(),
            "rag_readiness": self.validate_rag_readiness(),
            "detailed_errors": dict(self.validation_results)
        }
        
        return validation_report
    
    def generate_validation_summary(self, validation_report: Dict[str, Any]) -> str:
        """Generate a human-readable validation summary"""
        summary = []
        summary.append("PHYSICS KNOWLEDGE GRAPH VALIDATION REPORT")
        summary.append("=" * 50)
        
        # Overall statistics
        completeness = validation_report["graph_completeness"]
        summary.append(f"\nGRAPH STATISTICS:")
        summary.append(f"  Total Nodes: {completeness['total_nodes']}")
        summary.append(f"  Total Relationships: {completeness['total_relationships']}")
        summary.append(f"  Target Nodes (200+): {'✅ PASSED' if completeness['targets_met']['nodes'] else '❌ FAILED'}")
        summary.append(f"  Target Relationships (500+): {'✅ PASSED' if completeness['targets_met']['relationships'] else '❌ FAILED'}")
        
        # Node distribution
        summary.append(f"\nNODE DISTRIBUTION:")
        for node_type, count in completeness["node_distribution"].items():
            summary.append(f"  {node_type}: {count}")
        
        # Validation results
        categories = [
            ("NODE INTEGRITY", validation_report["node_integrity"]),
            ("RELATIONSHIP INTEGRITY", validation_report["relationship_integrity"]),
            ("EDUCATIONAL QUALITY", validation_report["educational_quality"]),
            ("RAG READINESS", validation_report["rag_readiness"])
        ]
        
        total_errors = 0
        for category_name, category_results in categories:
            summary.append(f"\n{category_name}:")
            category_errors = 0
            for check_name, error_count in category_results.items():
                if isinstance(error_count, int):
                    status = "✅ PASSED" if error_count == 0 else f"❌ {error_count} errors"
                    summary.append(f"  {check_name.replace('_', ' ').title()}: {status}")
                    category_errors += error_count
            total_errors += category_errors
        
        # Overall assessment
        summary.append(f"\nOVERALL ASSESSMENT:")
        if total_errors == 0:
            summary.append("✅ EXCELLENT: Knowledge graph passed all validation checks")
        elif total_errors < 10:
            summary.append(f"⚠️ GOOD: Knowledge graph has {total_errors} minor issues")
        elif total_errors < 25:
            summary.append(f"⚠️ FAIR: Knowledge graph has {total_errors} issues that should be addressed")
        else:
            summary.append(f"❌ POOR: Knowledge graph has {total_errors} significant issues")
        
        # Recommendations
        summary.append(f"\nRECOMMENDATIONS:")
        if validation_report["relationship_integrity"]["orphaned_concepts"] > 0:
            summary.append("  - Link orphaned concepts to appropriate subdomains")
        if validation_report["educational_quality"]["insufficient_content_domains"] > 0:
            summary.append("  - Add more educational content to domains with insufficient material")
        if validation_report["rag_readiness"]["weakly_connected_concepts"] > 0:
            summary.append("  - Improve concept connectivity for better RAG traversal")
        if not completeness["targets_met"]["nodes"]:
            summary.append("  - Add more concept nodes to reach target of 200+")
        if not completeness["targets_met"]["relationships"]:
            summary.append("  - Create more relationships to reach target of 500+")
        
        return "\n".join(summary)

def main():
    """Run knowledge graph validation"""
    validator = None
    try:
        validator = KnowledgeGraphValidator()
        
        # Run comprehensive validation
        report = validator.run_comprehensive_validation()
        
        # Generate and display summary
        summary = validator.generate_validation_summary(report)
        print(summary)
        
        # Save detailed report
        with open("/home/atk21004admin/Physics-Assistant/database/validation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed validation report saved to: /home/atk21004admin/Physics-Assistant/database/validation_report.json")
        
        # Return success if no critical errors
        total_critical_errors = (
            report["node_integrity"]["concept_errors"] +
            report["node_integrity"]["formula_errors"] + 
            report["relationship_integrity"]["circular_prerequisites"]
        )
        
        return total_critical_errors == 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if validator:
            validator.close()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)