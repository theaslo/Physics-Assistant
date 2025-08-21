#!/usr/bin/env python3
"""
Comprehensive Physics Knowledge Graph Builder for Graph RAG
Creates a detailed educational physics ontology in Neo4j for content retrieval.
"""
import os
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv('.env.example')

class PhysicsKnowledgeGraph:
    def __init__(self):
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'physics_graph_password_2024')
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        print("Connected to Neo4j for Physics Knowledge Graph")
    
    def close(self):
        self.driver.close()
    
    def clear_and_setup_schema(self):
        """Clear existing data and create enhanced schema"""
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            print("Cleared existing graph data")
            
            # Create constraints and indexes for performance
            constraints = [
                "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT formula_id_unique IF NOT EXISTS FOR (f:Formula) REQUIRE f.id IS UNIQUE",
                "CREATE CONSTRAINT problem_id_unique IF NOT EXISTS FOR (p:Problem) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT unit_name_unique IF NOT EXISTS FOR (u:Unit) REQUIRE u.name IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint may already exist: {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX concept_category_idx IF NOT EXISTS FOR (c:Concept) ON (c.category)",
                "CREATE INDEX concept_difficulty_idx IF NOT EXISTS FOR (c:Concept) ON (c.difficulty_level)",
                "CREATE INDEX problem_type_idx IF NOT EXISTS FOR (p:Problem) ON (p.problem_type)",
                "CREATE INDEX formula_domain_idx IF NOT EXISTS FOR (f:Formula) ON (f.domain)",
                "CREATE INDEX learning_path_level_idx IF NOT EXISTS FOR (l:LearningPath) ON (l.level)",
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    print(f"Index may already exist: {e}")
                    
            print("Created database constraints and indexes")
    
    def get_comprehensive_physics_ontology(self) -> Dict[str, Any]:
        """Define comprehensive physics ontology for educational content"""
        
        return {
            "mechanics": {
                "kinematics": {
                    "concepts": [
                        {"name": "Position", "description": "Location of an object in space", "difficulty": "beginner"},
                        {"name": "Displacement", "description": "Change in position vector", "difficulty": "beginner"},
                        {"name": "Velocity", "description": "Rate of change of position", "difficulty": "beginner"},
                        {"name": "Acceleration", "description": "Rate of change of velocity", "difficulty": "intermediate"},
                        {"name": "Instantaneous Velocity", "description": "Velocity at a specific instant", "difficulty": "intermediate"},
                        {"name": "Average Velocity", "description": "Total displacement over time", "difficulty": "beginner"},
                        {"name": "Projectile Motion", "description": "Motion under gravity in 2D", "difficulty": "intermediate"},
                        {"name": "Circular Motion", "description": "Motion in a circular path", "difficulty": "intermediate"},
                        {"name": "Relative Motion", "description": "Motion relative to different reference frames", "difficulty": "advanced"},
                    ],
                    "formulas": [
                        {"id": "kinematics_1", "expression": "v = v₀ + at", "name": "First kinematic equation", "variables": "v: final velocity, v₀: initial velocity, a: acceleration, t: time"},
                        {"id": "kinematics_2", "expression": "x = x₀ + v₀t + ½at²", "name": "Second kinematic equation", "variables": "x: position, x₀: initial position"},
                        {"id": "kinematics_3", "expression": "v² = v₀² + 2a(x - x₀)", "name": "Third kinematic equation", "variables": "Independent of time"},
                        {"id": "projectile_range", "expression": "R = v₀²sin(2θ)/g", "name": "Projectile range", "variables": "R: range, θ: launch angle, g: gravity"},
                    ]
                },
                "forces": {
                    "concepts": [
                        {"name": "Force", "description": "Push or pull acting on an object", "difficulty": "beginner"},
                        {"name": "Newton First Law", "description": "Law of inertia", "difficulty": "beginner"},
                        {"name": "Newton Second Law", "description": "F = ma relationship", "difficulty": "intermediate"},
                        {"name": "Newton Third Law", "description": "Action-reaction pairs", "difficulty": "intermediate"},
                        {"name": "Friction", "description": "Force opposing motion", "difficulty": "intermediate"},
                        {"name": "Static Friction", "description": "Friction preventing motion", "difficulty": "intermediate"},
                        {"name": "Kinetic Friction", "description": "Friction during motion", "difficulty": "intermediate"},
                        {"name": "Normal Force", "description": "Perpendicular contact force", "difficulty": "beginner"},
                        {"name": "Tension", "description": "Force through strings/cables", "difficulty": "intermediate"},
                        {"name": "Weight", "description": "Gravitational force", "difficulty": "beginner"},
                        {"name": "Free Body Diagram", "description": "Visual representation of forces", "difficulty": "intermediate"},
                        {"name": "Equilibrium", "description": "Net force equals zero", "difficulty": "intermediate"},
                    ],
                    "formulas": [
                        {"id": "newton_second", "expression": "F = ma", "name": "Newton's second law", "variables": "F: net force, m: mass, a: acceleration"},
                        {"id": "weight", "expression": "W = mg", "name": "Weight formula", "variables": "W: weight, g: gravitational acceleration"},
                        {"id": "friction_static", "expression": "fs ≤ μsN", "name": "Static friction", "variables": "fs: static friction, μs: coefficient, N: normal force"},
                        {"id": "friction_kinetic", "expression": "fk = μkN", "name": "Kinetic friction", "variables": "fk: kinetic friction, μk: coefficient"},
                    ]
                },
                "energy": {
                    "concepts": [
                        {"name": "Work", "description": "Energy transfer through force", "difficulty": "intermediate"},
                        {"name": "Kinetic Energy", "description": "Energy of motion", "difficulty": "intermediate"},
                        {"name": "Potential Energy", "description": "Stored energy due to position", "difficulty": "intermediate"},
                        {"name": "Gravitational Potential Energy", "description": "Energy due to height", "difficulty": "intermediate"},
                        {"name": "Elastic Potential Energy", "description": "Energy in springs", "difficulty": "intermediate"},
                        {"name": "Conservation of Energy", "description": "Total energy remains constant", "difficulty": "advanced"},
                        {"name": "Work-Energy Theorem", "description": "Work equals change in kinetic energy", "difficulty": "advanced"},
                        {"name": "Power", "description": "Rate of doing work", "difficulty": "intermediate"},
                        {"name": "Mechanical Energy", "description": "Sum of kinetic and potential energy", "difficulty": "advanced"},
                    ],
                    "formulas": [
                        {"id": "work", "expression": "W = F·d·cos(θ)", "name": "Work formula", "variables": "W: work, F: force, d: displacement, θ: angle"},
                        {"id": "kinetic_energy", "expression": "KE = ½mv²", "name": "Kinetic energy", "variables": "KE: kinetic energy, m: mass, v: velocity"},
                        {"id": "gravitational_pe", "expression": "PE = mgh", "name": "Gravitational potential energy", "variables": "PE: potential energy, h: height"},
                        {"id": "elastic_pe", "expression": "PE = ½kx²", "name": "Elastic potential energy", "variables": "k: spring constant, x: displacement"},
                        {"id": "power", "expression": "P = W/t = F·v", "name": "Power formula", "variables": "P: power, t: time, v: velocity"},
                    ]
                },
                "momentum": {
                    "concepts": [
                        {"name": "Momentum", "description": "Product of mass and velocity", "difficulty": "intermediate"},
                        {"name": "Impulse", "description": "Change in momentum", "difficulty": "intermediate"},
                        {"name": "Conservation of Momentum", "description": "Total momentum remains constant", "difficulty": "advanced"},
                        {"name": "Collision", "description": "Interaction between objects", "difficulty": "intermediate"},
                        {"name": "Elastic Collision", "description": "Collision conserving kinetic energy", "difficulty": "advanced"},
                        {"name": "Inelastic Collision", "description": "Collision not conserving kinetic energy", "difficulty": "advanced"},
                        {"name": "Center of Mass", "description": "Average position of mass distribution", "difficulty": "advanced"},
                    ],
                    "formulas": [
                        {"id": "momentum", "expression": "p = mv", "name": "Momentum formula", "variables": "p: momentum, m: mass, v: velocity"},
                        {"id": "impulse", "expression": "J = Δp = FΔt", "name": "Impulse-momentum theorem", "variables": "J: impulse, Δp: change in momentum"},
                        {"id": "elastic_collision_1d", "expression": "v₁f = ((m₁-m₂)v₁i + 2m₂v₂i)/(m₁+m₂)", "name": "Elastic collision formula", "variables": "1D elastic collision velocities"},
                    ]
                },
                "rotational_motion": {
                    "concepts": [
                        {"name": "Angular Position", "description": "Rotational analog of position", "difficulty": "intermediate"},
                        {"name": "Angular Velocity", "description": "Rate of change of angular position", "difficulty": "intermediate"},
                        {"name": "Angular Acceleration", "description": "Rate of change of angular velocity", "difficulty": "intermediate"},
                        {"name": "Torque", "description": "Rotational force", "difficulty": "intermediate"},
                        {"name": "Moment of Inertia", "description": "Rotational analog of mass", "difficulty": "advanced"},
                        {"name": "Angular Momentum", "description": "Rotational analog of linear momentum", "difficulty": "advanced"},
                        {"name": "Rolling Motion", "description": "Combined translation and rotation", "difficulty": "advanced"},
                        {"name": "Rotational Kinetic Energy", "description": "Energy of rotational motion", "difficulty": "advanced"},
                    ],
                    "formulas": [
                        {"id": "torque", "expression": "τ = r × F = rF sin(θ)", "name": "Torque formula", "variables": "τ: torque, r: radius vector, F: force"},
                        {"id": "angular_momentum", "expression": "L = Iω", "name": "Angular momentum", "variables": "L: angular momentum, I: moment of inertia, ω: angular velocity"},
                        {"id": "rotational_ke", "expression": "KE = ½Iω²", "name": "Rotational kinetic energy", "variables": "Rotational energy formula"},
                        {"id": "rolling_condition", "expression": "v = rω", "name": "Rolling without slipping", "variables": "v: linear velocity, r: radius, ω: angular velocity"},
                    ]
                }
            },
            "waves_oscillations": {
                "oscillations": {
                    "concepts": [
                        {"name": "Simple Harmonic Motion", "description": "Periodic motion with restoring force", "difficulty": "intermediate"},
                        {"name": "Amplitude", "description": "Maximum displacement from equilibrium", "difficulty": "beginner"},
                        {"name": "Period", "description": "Time for one complete oscillation", "difficulty": "beginner"},
                        {"name": "Frequency", "description": "Number of oscillations per unit time", "difficulty": "beginner"},
                        {"name": "Phase", "description": "Position in oscillation cycle", "difficulty": "intermediate"},
                        {"name": "Damped Oscillations", "description": "Oscillations with energy loss", "difficulty": "advanced"},
                        {"name": "Forced Oscillations", "description": "Oscillations driven by external force", "difficulty": "advanced"},
                        {"name": "Resonance", "description": "Maximum amplitude at natural frequency", "difficulty": "advanced"},
                    ],
                    "formulas": [
                        {"id": "shm_position", "expression": "x(t) = A cos(ωt + φ)", "name": "SHM position", "variables": "A: amplitude, ω: angular frequency, φ: phase"},
                        {"id": "frequency_period", "expression": "f = 1/T", "name": "Frequency-period relation", "variables": "f: frequency, T: period"},
                        {"id": "spring_period", "expression": "T = 2π√(m/k)", "name": "Spring oscillator period", "variables": "m: mass, k: spring constant"},
                    ]
                },
                "waves": {
                    "concepts": [
                        {"name": "Wave", "description": "Disturbance that transfers energy", "difficulty": "intermediate"},
                        {"name": "Wavelength", "description": "Distance between wave peaks", "difficulty": "beginner"},
                        {"name": "Wave Speed", "description": "Speed of wave propagation", "difficulty": "intermediate"},
                        {"name": "Transverse Wave", "description": "Oscillation perpendicular to direction", "difficulty": "intermediate"},
                        {"name": "Longitudinal Wave", "description": "Oscillation parallel to direction", "difficulty": "intermediate"},
                        {"name": "Interference", "description": "Superposition of waves", "difficulty": "advanced"},
                        {"name": "Standing Wave", "description": "Wave pattern from interference", "difficulty": "advanced"},
                        {"name": "Doppler Effect", "description": "Frequency shift due to relative motion", "difficulty": "advanced"},
                    ],
                    "formulas": [
                        {"id": "wave_speed", "expression": "v = fλ", "name": "Wave speed formula", "variables": "v: wave speed, f: frequency, λ: wavelength"},
                        {"id": "wave_equation", "expression": "y(x,t) = A sin(kx - ωt + φ)", "name": "Sinusoidal wave equation", "variables": "k: wave number, ω: angular frequency"},
                    ]
                }
            },
            "thermodynamics": {
                "concepts": [
                    {"name": "Temperature", "description": "Measure of average kinetic energy", "difficulty": "beginner"},
                    {"name": "Heat", "description": "Energy transfer due to temperature difference", "difficulty": "beginner"},
                    {"name": "Internal Energy", "description": "Total energy of system particles", "difficulty": "intermediate"},
                    {"name": "First Law of Thermodynamics", "description": "Energy conservation for thermal systems", "difficulty": "intermediate"},
                    {"name": "Second Law of Thermodynamics", "description": "Entropy always increases", "difficulty": "advanced"},
                    {"name": "Entropy", "description": "Measure of disorder", "difficulty": "advanced"},
                    {"name": "Heat Capacity", "description": "Energy needed to raise temperature", "difficulty": "intermediate"},
                    {"name": "Phase Transitions", "description": "Changes between solid, liquid, gas", "difficulty": "intermediate"},
                ],
                "formulas": [
                    {"id": "first_law", "expression": "ΔU = Q - W", "name": "First law of thermodynamics", "variables": "ΔU: change in internal energy, Q: heat added, W: work done"},
                    {"id": "heat_capacity", "expression": "Q = mcΔT", "name": "Heat capacity formula", "variables": "c: specific heat capacity, ΔT: temperature change"},
                ]
            },
            "electromagnetism": {
                "electrostatics": {
                    "concepts": [
                        {"name": "Electric Charge", "description": "Fundamental property of matter", "difficulty": "beginner"},
                        {"name": "Electric Force", "description": "Force between charged objects", "difficulty": "beginner"},
                        {"name": "Electric Field", "description": "Force per unit charge", "difficulty": "intermediate"},
                        {"name": "Electric Potential", "description": "Electric potential energy per unit charge", "difficulty": "intermediate"},
                        {"name": "Capacitance", "description": "Ability to store electric charge", "difficulty": "intermediate"},
                        {"name": "Gauss Law", "description": "Relation between electric field and charge", "difficulty": "advanced"},
                    ],
                    "formulas": [
                        {"id": "coulomb_law", "expression": "F = kq₁q₂/r²", "name": "Coulomb's law", "variables": "F: force, k: Coulomb constant, q: charges, r: distance"},
                        {"id": "electric_field", "expression": "E = F/q", "name": "Electric field definition", "variables": "E: electric field, F: force, q: test charge"},
                        {"id": "electric_potential", "expression": "V = kQ/r", "name": "Electric potential", "variables": "V: potential, Q: source charge"},
                    ]
                },
                "magnetism": {
                    "concepts": [
                        {"name": "Magnetic Field", "description": "Field around magnetic objects", "difficulty": "intermediate"},
                        {"name": "Magnetic Force", "description": "Force on moving charges in magnetic field", "difficulty": "intermediate"},
                        {"name": "Electromagnetic Induction", "description": "Electric field from changing magnetic field", "difficulty": "advanced"},
                        {"name": "Lenz Law", "description": "Direction of induced current", "difficulty": "advanced"},
                    ],
                    "formulas": [
                        {"id": "magnetic_force", "expression": "F = q(v × B)", "name": "Magnetic force on charge", "variables": "q: charge, v: velocity, B: magnetic field"},
                        {"id": "faraday_law", "expression": "ε = -dΦ/dt", "name": "Faraday's law", "variables": "ε: induced EMF, Φ: magnetic flux"},
                    ]
                }
            }
        }
    
    def create_knowledge_graph(self):
        """Create comprehensive physics knowledge graph"""
        ontology = self.get_comprehensive_physics_ontology()
        
        with self.driver.session() as session:
            # Create domain nodes (top level categories)
            domains = ["mechanics", "waves_oscillations", "thermodynamics", "electromagnetism"]
            for domain in domains:
                session.run(
                    """
                    CREATE (:Domain {
                        name: $name,
                        description: $description,
                        created_at: datetime()
                    })
                    """,
                    name=domain,
                    description=f"Physics domain: {domain.replace('_', ' ').title()}"
                )
            
            # Create subdomain nodes and concepts
            for domain_name, domain_data in ontology.items():
                for subdomain_name, subdomain_data in domain_data.items():
                    # Create subdomain node
                    session.run(
                        """
                        MATCH (d:Domain {name: $domain})
                        CREATE (sd:Subdomain {
                            name: $subdomain,
                            domain: $domain,
                            description: $description,
                            created_at: datetime()
                        })
                        CREATE (d)-[:CONTAINS]->(sd)
                        """,
                        domain=domain_name,
                        subdomain=subdomain_name,
                        description=f"{domain_name.title()} subdomain: {subdomain_name.replace('_', ' ').title()}"
                    )
                    
                    # Create concept nodes
                    if "concepts" in subdomain_data:
                        for concept in subdomain_data["concepts"]:
                            session.run(
                                """
                                MATCH (sd:Subdomain {name: $subdomain, domain: $domain})
                                CREATE (c:Concept {
                                    name: $name,
                                    description: $description,
                                    difficulty_level: $difficulty,
                                    domain: $domain,
                                    subdomain: $subdomain,
                                    category: $category,
                                    learning_objectives: $objectives,
                                    common_misconceptions: $misconceptions,
                                    created_at: datetime()
                                })
                                CREATE (sd)-[:CONTAINS]->(c)
                                """,
                                subdomain=subdomain_name,
                                domain=domain_name,
                                name=concept["name"],
                                description=concept["description"],
                                difficulty=concept["difficulty"],
                                category=f"{domain_name}_{subdomain_name}",
                                objectives=f"Understand {concept['name'].lower()} and its applications in physics",
                                misconceptions=f"Common misconceptions about {concept['name'].lower()}"
                            )
                    
                    # Create formula nodes
                    if "formulas" in subdomain_data:
                        for formula in subdomain_data["formulas"]:
                            session.run(
                                """
                                MATCH (sd:Subdomain {name: $subdomain, domain: $domain})
                                CREATE (f:Formula {
                                    id: $id,
                                    name: $name,
                                    expression: $expression,
                                    variables: $variables,
                                    domain: $domain,
                                    subdomain: $subdomain,
                                    difficulty_level: 'intermediate',
                                    created_at: datetime()
                                })
                                CREATE (sd)-[:CONTAINS]->(f)
                                """,
                                subdomain=subdomain_name,
                                domain=domain_name,
                                id=formula["id"],
                                name=formula["name"],
                                expression=formula["expression"],
                                variables=formula["variables"]
                            )
            
            print(f"Created comprehensive physics knowledge graph with domains and concepts")
    
    def create_educational_content_nodes(self):
        """Create educational content like problems, examples, and explanations"""
        
        with self.driver.session() as session:
            # Create sample problems for different concepts
            problems = [
                {
                    "id": "kinematics_p001",
                    "title": "Car Acceleration Problem",
                    "description": "A car accelerates from rest to 30 m/s in 10 seconds. Find the acceleration and distance traveled.",
                    "problem_type": "calculation",
                    "difficulty": "beginner",
                    "concept": "Acceleration",
                    "solution_steps": ["Given: v₀=0, v=30m/s, t=10s", "Find acceleration: a = (v-v₀)/t", "Find distance: x = v₀t + ½at²"],
                    "answer": "a = 3 m/s², x = 150 m"
                },
                {
                    "id": "forces_p001", 
                    "title": "Friction on Incline",
                    "description": "A 10 kg block slides down a 30° incline with coefficient of friction 0.3. Find the acceleration.",
                    "problem_type": "application",
                    "difficulty": "intermediate",
                    "concept": "Friction",
                    "solution_steps": ["Draw free body diagram", "Resolve weight components", "Apply Newton's second law", "Include friction force"],
                    "answer": "a = 2.4 m/s²"
                },
                {
                    "id": "energy_p001",
                    "title": "Roller Coaster Energy",
                    "description": "A roller coaster car of mass 500 kg starts from rest at height 50 m. Find speed at bottom.",
                    "problem_type": "application", 
                    "difficulty": "intermediate",
                    "concept": "Conservation of Energy",
                    "solution_steps": ["Apply conservation of energy", "PE_initial = KE_final", "mgh = ½mv²"],
                    "answer": "v = 31.3 m/s"
                },
                {
                    "id": "waves_p001",
                    "title": "Wave Speed Calculation",
                    "description": "A wave has frequency 50 Hz and wavelength 2 m. Calculate the wave speed.",
                    "problem_type": "calculation",
                    "difficulty": "beginner", 
                    "concept": "Wave Speed",
                    "solution_steps": ["Use v = fλ", "Substitute values", "Calculate result"],
                    "answer": "v = 100 m/s"
                }
            ]
            
            for problem in problems:
                session.run(
                    """
                    CREATE (p:Problem {
                        id: $id,
                        title: $title,
                        description: $description,
                        problem_type: $type,
                        difficulty_level: $difficulty,
                        solution_steps: $steps,
                        answer: $answer,
                        created_at: datetime()
                    })
                    """,
                    id=problem["id"],
                    title=problem["title"],
                    description=problem["description"],
                    type=problem["problem_type"],
                    difficulty=problem["difficulty"],
                    steps=problem["solution_steps"],
                    answer=problem["answer"]
                )
                
                # Link problem to concept
                session.run(
                    """
                    MATCH (p:Problem {id: $problem_id})
                    MATCH (c:Concept {name: $concept_name})
                    CREATE (c)-[:HAS_PROBLEM]->(p)
                    CREATE (p)-[:APPLIES_CONCEPT]->(c)
                    """,
                    problem_id=problem["id"],
                    concept_name=problem["concept"]
                )
            
            # Create learning explanations
            explanations = [
                {
                    "id": "velocity_explanation",
                    "title": "Understanding Velocity",
                    "content": "Velocity is a vector quantity that describes the rate of change of position. Unlike speed, velocity includes direction.",
                    "explanation_type": "conceptual",
                    "concept": "Velocity"
                },
                {
                    "id": "force_explanation", 
                    "title": "What is Force?",
                    "content": "Force is a push or pull that can change an object's motion. Forces are vectors with magnitude and direction.",
                    "explanation_type": "conceptual",
                    "concept": "Force"
                },
                {
                    "id": "energy_conservation_explanation",
                    "title": "Energy Conservation Principle", 
                    "content": "Energy cannot be created or destroyed, only transformed from one form to another. Total energy remains constant.",
                    "explanation_type": "principle",
                    "concept": "Conservation of Energy"
                }
            ]
            
            for explanation in explanations:
                session.run(
                    """
                    CREATE (e:Explanation {
                        id: $id,
                        title: $title,
                        content: $content,
                        explanation_type: $type,
                        created_at: datetime()
                    })
                    """,
                    id=explanation["id"],
                    title=explanation["title"],
                    content=explanation["content"],
                    type=explanation["explanation_type"]
                )
                
                # Link to concept
                session.run(
                    """
                    MATCH (e:Explanation {id: $explanation_id})
                    MATCH (c:Concept {name: $concept_name})
                    CREATE (c)-[:HAS_EXPLANATION]->(e)
                    """,
                    explanation_id=explanation["id"],
                    concept_name=explanation["concept"]
                )
            
            # Create units
            units = [
                {"name": "meter", "symbol": "m", "quantity": "length", "si_base": True},
                {"name": "second", "symbol": "s", "quantity": "time", "si_base": True},
                {"name": "kilogram", "symbol": "kg", "quantity": "mass", "si_base": True},
                {"name": "newton", "symbol": "N", "quantity": "force", "si_base": False, "definition": "kg⋅m/s²"},
                {"name": "joule", "symbol": "J", "quantity": "energy", "si_base": False, "definition": "N⋅m"},
                {"name": "watt", "symbol": "W", "quantity": "power", "si_base": False, "definition": "J/s"},
                {"name": "hertz", "symbol": "Hz", "quantity": "frequency", "si_base": False, "definition": "1/s"},
            ]
            
            for unit in units:
                session.run(
                    """
                    CREATE (u:Unit {
                        name: $name,
                        symbol: $symbol,
                        quantity: $quantity,
                        si_base: $si_base,
                        definition: $definition,
                        created_at: datetime()
                    })
                    """,
                    name=unit["name"],
                    symbol=unit["symbol"],
                    quantity=unit["quantity"],
                    si_base=unit["si_base"],
                    definition=unit.get("definition", "")
                )
            
            print("Created educational content nodes: problems, explanations, and units")
    
    def create_learning_relationships(self):
        """Create educational relationships and learning paths"""
        
        with self.driver.session() as session:
            # Create prerequisite relationships
            prerequisites = [
                ("Velocity", "Acceleration"),
                ("Acceleration", "Force"),
                ("Force", "Work"),
                ("Work", "Energy"),
                ("Position", "Velocity"),
                ("Velocity", "Momentum"),
                ("Force", "Momentum"),
                ("Kinetic Energy", "Conservation of Energy"),
                ("Simple Harmonic Motion", "Wave"),
                ("Electric Field", "Electric Potential"),
                ("Magnetic Field", "Electromagnetic Induction"),
            ]
            
            for prereq, concept in prerequisites:
                session.run(
                    """
                    MATCH (prereq:Concept {name: $prereq_name})
                    MATCH (concept:Concept {name: $concept_name})
                    CREATE (prereq)-[:PREREQUISITE_FOR]->(concept)
                    CREATE (concept)-[:REQUIRES]->(prereq)
                    """,
                    prereq_name=prereq,
                    concept_name=concept
                )
            
            # Create concept similarity relationships
            related_concepts = [
                ("Velocity", "Acceleration", "both describe motion characteristics"),
                ("Work", "Energy", "work transfers energy"),
                ("Force", "Momentum", "both relate to Newton's laws"),
                ("Kinetic Energy", "Momentum", "both depend on mass and velocity"),
                ("Electric Field", "Magnetic Field", "both are field concepts"),
                ("Frequency", "Period", "inverse relationship"),
                ("Amplitude", "Energy", "amplitude affects wave energy"),
            ]
            
            for concept1, concept2, reason in related_concepts:
                session.run(
                    """
                    MATCH (c1:Concept {name: $concept1})
                    MATCH (c2:Concept {name: $concept2})
                    CREATE (c1)-[:RELATED_TO {reason: $reason}]->(c2)
                    CREATE (c2)-[:RELATED_TO {reason: $reason}]->(c1)
                    """,
                    concept1=concept1,
                    concept2=concept2,
                    reason=reason
                )
            
            # Link formulas to concepts they use
            formula_concept_links = [
                ("kinematics_1", "Velocity"), ("kinematics_1", "Acceleration"),
                ("kinematics_2", "Position"), ("kinematics_2", "Acceleration"),
                ("newton_second", "Force"), ("newton_second", "Acceleration"),
                ("work", "Force"), ("work", "Work"),
                ("kinetic_energy", "Kinetic Energy"), ("kinetic_energy", "Velocity"),
                ("momentum", "Momentum"), ("momentum", "Velocity"),
                ("torque", "Torque"), ("torque", "Force"),
                ("wave_speed", "Wave Speed"), ("wave_speed", "Frequency"),
            ]
            
            for formula_id, concept_name in formula_concept_links:
                session.run(
                    """
                    MATCH (f:Formula {id: $formula_id})
                    MATCH (c:Concept {name: $concept_name})
                    CREATE (f)-[:DESCRIBES]->(c)
                    CREATE (c)-[:HAS_FORMULA]->(f)
                    """,
                    formula_id=formula_id,
                    concept_name=concept_name
                )
            
            # Create learning paths
            learning_paths = [
                {
                    "name": "Mechanics Fundamentals",
                    "level": "beginner",
                    "description": "Basic mechanics concepts for introductory physics",
                    "concepts": ["Position", "Velocity", "Acceleration", "Force", "Newton First Law", "Newton Second Law"]
                },
                {
                    "name": "Energy and Motion",
                    "level": "intermediate", 
                    "description": "Energy concepts and conservation laws",
                    "concepts": ["Work", "Kinetic Energy", "Potential Energy", "Conservation of Energy", "Power"]
                },
                {
                    "name": "Advanced Mechanics",
                    "level": "advanced",
                    "description": "Complex mechanics including rotations and collisions",
                    "concepts": ["Torque", "Angular Momentum", "Moment of Inertia", "Elastic Collision", "Center of Mass"]
                },
                {
                    "name": "Waves and Oscillations",
                    "level": "intermediate",
                    "description": "Periodic motion and wave phenomena",
                    "concepts": ["Simple Harmonic Motion", "Period", "Frequency", "Wave", "Interference"]
                }
            ]
            
            for path in learning_paths:
                session.run(
                    """
                    CREATE (lp:LearningPath {
                        name: $name,
                        level: $level,
                        description: $description,
                        created_at: datetime()
                    })
                    """,
                    name=path["name"],
                    level=path["level"],
                    description=path["description"]
                )
                
                # Link concepts to learning path
                for i, concept_name in enumerate(path["concepts"]):
                    session.run(
                        """
                        MATCH (lp:LearningPath {name: $path_name})
                        MATCH (c:Concept {name: $concept_name})
                        CREATE (lp)-[:INCLUDES {order: $order}]->(c)
                        """,
                        path_name=path["name"],
                        concept_name=concept_name,
                        order=i+1
                    )
            
            print("Created learning relationships and paths")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        
        with self.driver.session() as session:
            # Count nodes by type
            node_counts = {}
            node_result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            for record in node_result:
                labels = ", ".join(record["labels"]) 
                node_counts[labels] = record["count"]
            
            # Count relationships by type
            rel_counts = {}
            rel_result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
            for record in rel_result:
                rel_counts[record["rel_type"]] = record["count"]
            
            # Get total counts
            total_nodes = session.run("MATCH (n) RETURN count(n) as total").single()["total"]
            total_relationships = session.run("MATCH ()-[r]->() RETURN count(r) as total").single()["total"]
            
            # Get difficulty distribution
            difficulty_dist = {}
            diff_result = session.run(
                "MATCH (c:Concept) RETURN c.difficulty_level as difficulty, count(c) as count"
            )
            for record in diff_result:
                difficulty_dist[record["difficulty"]] = record["count"]
            
            # Get domain distribution
            domain_dist = {}
            domain_result = session.run(
                "MATCH (c:Concept) RETURN c.domain as domain, count(c) as count"
            )
            for record in domain_result:
                if record["domain"]:
                    domain_dist[record["domain"]] = record["count"]
            
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "difficulty_distribution": difficulty_dist,
                "domain_distribution": domain_dist
            }

def main():
    print("Physics Knowledge Graph Builder - Phase 3.1")
    print("=" * 50)
    
    graph_builder = None
    try:
        graph_builder = PhysicsKnowledgeGraph()
        
        # Setup and populate knowledge graph
        print("\n🧹 Clearing existing data and setting up schema...")
        graph_builder.clear_and_setup_schema()
        
        print("\n🏗️ Creating comprehensive physics knowledge graph...")
        graph_builder.create_knowledge_graph()
        
        print("\n📚 Adding educational content nodes...")
        graph_builder.create_educational_content_nodes()
        
        print("\n🔗 Creating learning relationships and paths...")
        graph_builder.create_learning_relationships()
        
        # Get statistics
        stats = graph_builder.get_graph_statistics()
        
        print(f"\n📊 Knowledge Graph Statistics:")
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Relationships: {stats['total_relationships']}")
        
        print(f"\nNode Distribution:")
        for node_type, count in stats['node_counts'].items():
            print(f"  - {node_type}: {count}")
        
        print(f"\nRelationship Distribution:")
        for rel_type, count in stats['relationship_counts'].items():
            print(f"  - {rel_type}: {count}")
        
        print(f"\nDifficulty Level Distribution:")
        for difficulty, count in stats['difficulty_distribution'].items():
            print(f"  - {difficulty}: {count}")
        
        print(f"\nPhysics Domain Distribution:")
        for domain, count in stats['domain_distribution'].items():
            print(f"  - {domain}: {count}")
        
        # Check if we met the target
        if stats['total_nodes'] >= 200:
            print(f"\n✅ SUCCESS: Created {stats['total_nodes']} nodes (target: 200+)")
        else:
            print(f"\n⚠️ PARTIAL: Created {stats['total_nodes']} nodes (target: 200+)")
            
        if stats['total_relationships'] >= 500:
            print(f"✅ SUCCESS: Created {stats['total_relationships']} relationships (target: 500+)")
        else:
            print(f"⚠️ PARTIAL: Created {stats['total_relationships']} relationships (target: 500+)")
        
        print(f"\n✅ Physics knowledge graph setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating knowledge graph: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if graph_builder:
            graph_builder.close()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)