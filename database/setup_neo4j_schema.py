#!/usr/bin/env python3
"""
Setup Neo4j graph database schema for Physics Assistant
"""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.example')

class Neo4jSetup:
    def __init__(self):
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'physics_graph_password_2024')
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        self.driver.close()
    
    def create_physics_ontology(self):
        """Create basic physics concept ontology"""
        
        with self.driver.session() as session:
            # Create core physics concept nodes
            concepts_query = """
            CREATE 
                (:Concept {name: 'Kinematics', description: '1D and 2D motion problems', category: 'mechanics'}),
                (:Concept {name: 'Forces', description: 'Newton\\'s laws and force analysis', category: 'mechanics'}),
                (:Concept {name: 'Energy', description: 'Work-energy theorem and conservation', category: 'mechanics'}),
                (:Concept {name: 'Momentum', description: 'Linear momentum and collisions', category: 'mechanics'}),
                (:Concept {name: 'Angular Motion', description: 'Rotational dynamics and kinematics', category: 'mechanics'}),
                (:Concept {name: 'Velocity', description: 'Rate of change of position', category: 'kinematics'}),
                (:Concept {name: 'Acceleration', description: 'Rate of change of velocity', category: 'kinematics'}),
                (:Concept {name: 'Work', description: 'Force applied over distance', category: 'energy'}),
                (:Concept {name: 'Power', description: 'Rate of doing work', category: 'energy'}),
                (:Concept {name: 'Friction', description: 'Force opposing motion', category: 'forces'}),
                (:Concept {name: 'Tension', description: 'Force transmitted through strings', category: 'forces'}),
                (:Concept {name: 'Collision', description: 'Interaction between objects', category: 'momentum'}),
                (:Concept {name: 'Torque', description: 'Rotational force', category: 'angular_motion'}),
                (:Concept {name: 'Inertia', description: 'Resistance to change in motion', category: 'angular_motion'})
            """
            
            session.run(concepts_query)
            print("✅ Created physics concept nodes")
            
            # Create relationships between concepts
            relationships_query = """
            MATCH (k:Concept {name: 'Kinematics'}), (v:Concept {name: 'Velocity'}), (a:Concept {name: 'Acceleration'})
            CREATE (k)-[:CONTAINS]->(v), (k)-[:CONTAINS]->(a), (v)-[:RELATED_TO]->(a)
            
            WITH k, v, a
            MATCH (f:Concept {name: 'Forces'}), (fr:Concept {name: 'Friction'}), (t:Concept {name: 'Tension'})
            CREATE (f)-[:CONTAINS]->(fr), (f)-[:CONTAINS]->(t), (f)-[:AFFECTS]->(a)
            
            WITH k, v, a, f
            MATCH (e:Concept {name: 'Energy'}), (w:Concept {name: 'Work'}), (p:Concept {name: 'Power'})
            CREATE (e)-[:CONTAINS]->(w), (e)-[:CONTAINS]->(p), (w)-[:RELATED_TO]->(f)
            
            WITH k, v, a, f, e
            MATCH (m:Concept {name: 'Momentum'}), (c:Concept {name: 'Collision'})
            CREATE (m)-[:CONTAINS]->(c), (m)-[:RELATED_TO]->(v)
            
            WITH k, v, a, f, e, m
            MATCH (am:Concept {name: 'Angular Motion'}), (to:Concept {name: 'Torque'}), (i:Concept {name: 'Inertia'})
            CREATE (am)-[:CONTAINS]->(to), (am)-[:CONTAINS]->(i), (to)-[:RELATED_TO]->(f)
            """
            
            session.run(relationships_query)
            print("✅ Created concept relationships")
            
    def create_indexes(self):
        """Create indexes for better performance"""
        with self.driver.session() as session:
            # Create indexes
            indexes = [
                "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",
                "CREATE INDEX concept_category_index IF NOT EXISTS FOR (c:Concept) ON (c.category)",
                "CREATE INDEX student_id_index IF NOT EXISTS FOR (s:Student) ON (s.user_id)",
                "CREATE INDEX problem_type_index IF NOT EXISTS FOR (p:Problem) ON (p.type)"
            ]
            
            for index_query in indexes:
                session.run(index_query)
            
            print("✅ Created database indexes")
    
    def create_user_nodes(self):
        """Create node types for students and interactions"""
        with self.driver.session() as session:
            try:
                # Drop existing index if it conflicts with constraint
                session.run("DROP INDEX student_id_index IF EXISTS")
                # Create constraint for unique student IDs
                session.run("CREATE CONSTRAINT student_id_unique IF NOT EXISTS FOR (s:Student) REQUIRE s.user_id IS UNIQUE")
            except Exception as e:
                if "already exists" not in str(e):
                    raise e
                print("⚠️ Constraint already exists, continuing...")
            
            # Create sample student node structure
            try:
                session.run("""
                CREATE (:Student {
                    user_id: 'sample_student_001', 
                    name: 'Sample Student', 
                    level: 'beginner',
                    created_at: datetime()
                })
                """)
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"⚠️ Sample student may already exist: {e}")
                    
            print("✅ Created student node structure")
    
    def verify_graph(self):
        """Verify the graph was created correctly"""
        with self.driver.session() as session:
            # Count nodes by type
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            
            print("\n📊 Graph database summary:")
            for record in result:
                labels = ", ".join(record["labels"])
                count = record["count"]
                print(f"  - {labels}: {count} nodes")
            
            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
            
            print("\n🔗 Relationships:")
            for record in rel_result:
                rel_type = record["rel_type"]
                count = record["count"]
                print(f"  - {rel_type}: {count} relationships")

def main():
    print("Physics Assistant Neo4j Schema Setup")
    print("====================================")
    
    neo4j_setup = None
    try:
        neo4j_setup = Neo4jSetup()
        print("✅ Connected to Neo4j")
        
        # Clear existing data (for development)
        with neo4j_setup.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🧹 Cleared existing data")
        
        # Set up schema
        neo4j_setup.create_indexes()
        neo4j_setup.create_physics_ontology()
        neo4j_setup.create_user_nodes()
        neo4j_setup.verify_graph()
        
        print("\n✅ Neo4j schema setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Neo4j: {str(e)}")
        return False
    
    finally:
        if neo4j_setup:
            neo4j_setup.close()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)