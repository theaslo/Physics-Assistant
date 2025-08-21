#!/usr/bin/env python3
"""
Test Knowledge Graph Structure
Creates a comprehensive test and demonstration of the physics knowledge graph structure
without requiring external Neo4j dependencies.
"""
import json
from typing import Dict, List, Any
from datetime import datetime

class MockKnowledgeGraph:
    """Mock knowledge graph to demonstrate structure and validate design"""
    
    def __init__(self):
        self.nodes = {}
        self.relationships = []
        self.node_id_counter = 1
        
    def add_node(self, node_type: str, properties: Dict[str, Any]) -> str:
        """Add a node to the mock graph"""
        node_id = f"{node_type.lower()}_{self.node_id_counter}"
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "properties": properties
        }
        self.node_id_counter += 1
        return node_id
    
    def add_relationship(self, from_id: str, relationship_type: str, to_id: str, properties: Dict[str, Any] = None):
        """Add a relationship to the mock graph"""
        self.relationships.append({
            "from": from_id,
            "type": relationship_type,
            "to": to_id,
            "properties": properties or {}
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        node_types = {}
        for node in self.nodes.values():
            node_type = node["type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        rel_types = {}
        for rel in self.relationships:
            rel_type = rel["type"]
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_relationships": len(self.relationships),
            "node_types": node_types,
            "relationship_types": rel_types
        }

def create_comprehensive_physics_knowledge_graph() -> MockKnowledgeGraph:
    """Create a comprehensive physics knowledge graph structure"""
    
    print("Creating comprehensive physics knowledge graph...")
    kg = MockKnowledgeGraph()
    
    # === DOMAINS ===
    mechanics_id = kg.add_node("Domain", {
        "name": "mechanics",
        "description": "Physics domain: Mechanics",
        "created_at": str(datetime.now())
    })
    
    waves_id = kg.add_node("Domain", {
        "name": "waves_oscillations", 
        "description": "Physics domain: Waves and Oscillations",
        "created_at": str(datetime.now())
    })
    
    thermo_id = kg.add_node("Domain", {
        "name": "thermodynamics",
        "description": "Physics domain: Thermodynamics", 
        "created_at": str(datetime.now())
    })
    
    electromag_id = kg.add_node("Domain", {
        "name": "electromagnetism",
        "description": "Physics domain: Electromagnetism",
        "created_at": str(datetime.now())
    })
    
    # === SUBDOMAINS ===
    subdomains_data = [
        ("kinematics", mechanics_id, "Kinematics: Motion description"),
        ("forces", mechanics_id, "Forces: Newton's laws and force analysis"),
        ("energy", mechanics_id, "Energy: Work-energy and conservation"),
        ("momentum", mechanics_id, "Momentum: Linear momentum and collisions"),
        ("rotational_motion", mechanics_id, "Rotational Motion: Angular dynamics"),
        ("oscillations", waves_id, "Oscillations: Simple harmonic motion"),
        ("waves", waves_id, "Waves: Wave propagation and properties"),
        ("heat", thermo_id, "Heat and temperature concepts"),
        ("electrostatics", electromag_id, "Electrostatics: Electric fields and charges"),
        ("magnetism", electromag_id, "Magnetism: Magnetic fields and forces")
    ]
    
    subdomain_ids = {}
    for subdomain_name, domain_id, description in subdomains_data:
        subdomain_id = kg.add_node("Subdomain", {
            "name": subdomain_name,
            "description": description,
            "created_at": str(datetime.now())
        })
        subdomain_ids[subdomain_name] = subdomain_id
        kg.add_relationship(domain_id, "CONTAINS", subdomain_id)
    
    # === CONCEPTS ===
    concepts_data = [
        # Kinematics concepts
        ("Position", "Location of an object in space", "beginner", "kinematics", "Understand spatial coordinates and reference frames"),
        ("Displacement", "Change in position vector", "beginner", "kinematics", "Distinguish between distance and displacement"),
        ("Velocity", "Rate of change of position", "beginner", "kinematics", "Calculate average and instantaneous velocity"),
        ("Acceleration", "Rate of change of velocity", "intermediate", "kinematics", "Apply kinematic equations for constant acceleration"),
        ("Instantaneous Velocity", "Velocity at a specific instant", "intermediate", "kinematics", "Use calculus to find instantaneous rates"),
        ("Average Velocity", "Total displacement over time", "beginner", "kinematics", "Calculate average motion quantities"),
        ("Projectile Motion", "Motion under gravity in 2D", "intermediate", "kinematics", "Analyze parabolic trajectories"),
        ("Circular Motion", "Motion in a circular path", "intermediate", "kinematics", "Apply centripetal acceleration concepts"),
        ("Relative Motion", "Motion relative to different reference frames", "advanced", "kinematics", "Transform between reference frames"),
        
        # Forces concepts
        ("Force", "Push or pull acting on an object", "beginner", "forces", "Identify forces and their effects"),
        ("Newton First Law", "Law of inertia", "beginner", "forces", "Apply principle of inertia"),
        ("Newton Second Law", "F = ma relationship", "intermediate", "forces", "Use F=ma for problem solving"),
        ("Newton Third Law", "Action-reaction pairs", "intermediate", "forces", "Identify action-reaction force pairs"),
        ("Friction", "Force opposing motion", "intermediate", "forces", "Calculate static and kinetic friction"),
        ("Static Friction", "Friction preventing motion", "intermediate", "forces", "Determine maximum static friction"),
        ("Kinetic Friction", "Friction during motion", "intermediate", "forces", "Calculate sliding friction forces"),
        ("Normal Force", "Perpendicular contact force", "beginner", "forces", "Find normal forces on surfaces"),
        ("Tension", "Force through strings/cables", "intermediate", "forces", "Analyze tension in rope systems"),
        ("Weight", "Gravitational force", "beginner", "forces", "Calculate gravitational weight"),
        ("Free Body Diagram", "Visual representation of forces", "intermediate", "forces", "Draw and analyze force diagrams"),
        ("Equilibrium", "Net force equals zero", "intermediate", "forces", "Apply equilibrium conditions"),
        
        # Energy concepts  
        ("Work", "Energy transfer through force", "intermediate", "energy", "Calculate work done by forces"),
        ("Kinetic Energy", "Energy of motion", "intermediate", "energy", "Apply kinetic energy formula"),
        ("Potential Energy", "Stored energy due to position", "intermediate", "energy", "Understand energy storage mechanisms"),
        ("Gravitational Potential Energy", "Energy due to height", "intermediate", "energy", "Calculate gravitational PE"),
        ("Elastic Potential Energy", "Energy in springs", "intermediate", "energy", "Apply Hooke's law for elastic PE"),
        ("Conservation of Energy", "Total energy remains constant", "advanced", "energy", "Apply energy conservation principles"),
        ("Work-Energy Theorem", "Work equals change in kinetic energy", "advanced", "energy", "Connect work and energy changes"),
        ("Power", "Rate of doing work", "intermediate", "energy", "Calculate power in mechanical systems"),
        ("Mechanical Energy", "Sum of kinetic and potential energy", "advanced", "energy", "Apply mechanical energy conservation"),
        
        # Momentum concepts
        ("Momentum", "Product of mass and velocity", "intermediate", "momentum", "Calculate linear momentum"),
        ("Impulse", "Change in momentum", "intermediate", "momentum", "Apply impulse-momentum theorem"),
        ("Conservation of Momentum", "Total momentum remains constant", "advanced", "momentum", "Solve collision problems"),
        ("Collision", "Interaction between objects", "intermediate", "momentum", "Analyze collision types"),
        ("Elastic Collision", "Collision conserving kinetic energy", "advanced", "momentum", "Solve elastic collision problems"),
        ("Inelastic Collision", "Collision not conserving kinetic energy", "advanced", "momentum", "Analyze energy loss in collisions"),
        ("Center of Mass", "Average position of mass distribution", "advanced", "momentum", "Calculate center of mass motion"),
        
        # Rotational motion concepts
        ("Angular Position", "Rotational analog of position", "intermediate", "rotational_motion", "Measure rotational displacement"),
        ("Angular Velocity", "Rate of change of angular position", "intermediate", "rotational_motion", "Calculate rotational speed"),
        ("Angular Acceleration", "Rate of change of angular velocity", "intermediate", "rotational_motion", "Analyze rotational acceleration"),
        ("Torque", "Rotational force", "intermediate", "rotational_motion", "Calculate torque and lever arms"),
        ("Moment of Inertia", "Rotational analog of mass", "advanced", "rotational_motion", "Calculate rotational inertia"),
        ("Angular Momentum", "Rotational analog of linear momentum", "advanced", "rotational_motion", "Apply angular momentum conservation"),
        ("Rolling Motion", "Combined translation and rotation", "advanced", "rotational_motion", "Analyze rolling without slipping"),
        ("Rotational Kinetic Energy", "Energy of rotational motion", "advanced", "rotational_motion", "Calculate rotational energy"),
        
        # Oscillations concepts
        ("Simple Harmonic Motion", "Periodic motion with restoring force", "intermediate", "oscillations", "Analyze SHM systems"),
        ("Amplitude", "Maximum displacement from equilibrium", "beginner", "oscillations", "Identify oscillation parameters"),
        ("Period", "Time for one complete oscillation", "beginner", "oscillations", "Calculate oscillation period"),
        ("Frequency", "Number of oscillations per unit time", "beginner", "oscillations", "Relate frequency and period"),
        ("Phase", "Position in oscillation cycle", "intermediate", "oscillations", "Understand phase relationships"),
        ("Damped Oscillations", "Oscillations with energy loss", "advanced", "oscillations", "Analyze damping effects"),
        ("Forced Oscillations", "Oscillations driven by external force", "advanced", "oscillations", "Study driven oscillator response"),
        ("Resonance", "Maximum amplitude at natural frequency", "advanced", "oscillations", "Identify resonance conditions"),
        
        # Wave concepts
        ("Wave", "Disturbance that transfers energy", "intermediate", "waves", "Understand wave properties"),
        ("Wavelength", "Distance between wave peaks", "beginner", "waves", "Measure wave spatial characteristics"),
        ("Wave Speed", "Speed of wave propagation", "intermediate", "waves", "Calculate wave propagation speed"),
        ("Transverse Wave", "Oscillation perpendicular to direction", "intermediate", "waves", "Identify wave types"),
        ("Longitudinal Wave", "Oscillation parallel to direction", "intermediate", "waves", "Distinguish wave orientations"),
        ("Interference", "Superposition of waves", "advanced", "waves", "Analyze wave interference patterns"),
        ("Standing Wave", "Wave pattern from interference", "advanced", "waves", "Study standing wave formation"),
        ("Doppler Effect", "Frequency shift due to relative motion", "advanced", "waves", "Calculate Doppler frequency shifts"),
        
        # Thermodynamics concepts
        ("Temperature", "Measure of average kinetic energy", "beginner", "heat", "Understand temperature scales"),
        ("Heat", "Energy transfer due to temperature difference", "beginner", "heat", "Calculate heat transfer"),
        ("Internal Energy", "Total energy of system particles", "intermediate", "heat", "Apply first law of thermodynamics"),
        ("First Law of Thermodynamics", "Energy conservation for thermal systems", "intermediate", "heat", "Solve thermodynamic processes"),
        ("Second Law of Thermodynamics", "Entropy always increases", "advanced", "heat", "Understand entropy and irreversibility"),
        ("Entropy", "Measure of disorder", "advanced", "heat", "Calculate entropy changes"),
        ("Heat Capacity", "Energy needed to raise temperature", "intermediate", "heat", "Calculate thermal energy changes"),
        ("Phase Transitions", "Changes between solid, liquid, gas", "intermediate", "heat", "Analyze phase change energy"),
        
        # Electrostatics concepts
        ("Electric Charge", "Fundamental property of matter", "beginner", "electrostatics", "Understand charge conservation"),
        ("Electric Force", "Force between charged objects", "beginner", "electrostatics", "Apply Coulomb's law"),
        ("Electric Field", "Force per unit charge", "intermediate", "electrostatics", "Calculate electric field strength"),
        ("Electric Potential", "Electric potential energy per unit charge", "intermediate", "electrostatics", "Relate potential and field"),
        ("Capacitance", "Ability to store electric charge", "intermediate", "electrostatics", "Calculate capacitor properties"),
        ("Gauss Law", "Relation between electric field and charge", "advanced", "electrostatics", "Apply Gauss's law to symmetric cases"),
        
        # Magnetism concepts
        ("Magnetic Field", "Field around magnetic objects", "intermediate", "magnetism", "Understand magnetic field sources"),
        ("Magnetic Force", "Force on moving charges in magnetic field", "intermediate", "magnetism", "Calculate magnetic forces"),
        ("Electromagnetic Induction", "Electric field from changing magnetic field", "advanced", "magnetism", "Apply Faraday's law"),
        ("Lenz Law", "Direction of induced current", "advanced", "magnetism", "Determine induced current directions"),
        
        # Additional advanced concepts for completeness
        ("Centripetal Force", "Force directed toward center of circular path", "intermediate", "forces", "Apply centripetal force concepts"),
        ("Centripetal Acceleration", "Acceleration directed toward center", "intermediate", "kinematics", "Calculate centripetal acceleration"),
        ("Universal Gravitation", "Gravitational force between masses", "intermediate", "forces", "Apply Newton's law of gravitation"),
        ("Escape Velocity", "Minimum velocity to escape gravitational field", "advanced", "energy", "Calculate escape velocities"),
        ("Orbital Motion", "Motion of objects in gravitational fields", "advanced", "kinematics", "Analyze satellite and planetary motion"),
        ("Fluid Statics", "Behavior of fluids at rest", "intermediate", "forces", "Apply pressure and buoyancy concepts"),
        ("Fluid Dynamics", "Motion of fluids", "advanced", "energy", "Apply Bernoulli's equation"),
        ("Pressure", "Force per unit area in fluids", "beginner", "forces", "Calculate pressure in various situations"),
        ("Buoyancy", "Upward force on objects in fluids", "intermediate", "forces", "Apply Archimedes' principle"),
        ("Surface Tension", "Force at liquid surfaces", "intermediate", "forces", "Understand surface effects"),
        ("Viscosity", "Resistance to fluid flow", "intermediate", "forces", "Analyze fluid friction"),
        ("Stress and Strain", "Material response to forces", "intermediate", "forces", "Calculate material deformation"),
        ("Young's Modulus", "Material stiffness property", "intermediate", "forces", "Apply elastic modulus concepts"),
        ("Shear Modulus", "Response to shear stress", "advanced", "forces", "Analyze shear deformation"),
        ("Bulk Modulus", "Response to volume stress", "advanced", "forces", "Calculate volume changes"),
        ("Thermal Conductivity", "Heat transfer through materials", "intermediate", "heat", "Apply heat conduction principles"),
        ("Convection", "Heat transfer by fluid motion", "intermediate", "heat", "Understand convective heat transfer"),
        ("Radiation", "Heat transfer by electromagnetic waves", "intermediate", "heat", "Apply Stefan-Boltzmann law"),
        ("Black Body Radiation", "Ideal thermal radiator", "advanced", "heat", "Understand thermal radiation spectra"),
        ("Heat Engine", "Device that converts heat to work", "advanced", "heat", "Apply thermodynamic cycle analysis"),
        ("Carnot Engine", "Ideal heat engine", "advanced", "heat", "Calculate maximum efficiency"),
        ("Refrigerator", "Device that removes heat", "advanced", "heat", "Analyze refrigeration cycles"),
        ("Adiabatic Process", "Process with no heat exchange", "intermediate", "heat", "Apply adiabatic conditions"),
        ("Isothermal Process", "Process at constant temperature", "intermediate", "heat", "Analyze constant temperature changes"),
        ("Isochoric Process", "Process at constant volume", "intermediate", "heat", "Apply constant volume conditions"),
        ("Isobaric Process", "Process at constant pressure", "intermediate", "heat", "Analyze constant pressure changes"),
        ("Electric Current", "Flow of electric charge", "beginner", "electrostatics", "Understand current and charge flow"),
        ("Resistance", "Opposition to electric current", "beginner", "electrostatics", "Apply Ohm's law"),
        ("Ohm's Law", "Relationship between voltage, current, and resistance", "beginner", "electrostatics", "Calculate circuit quantities"),
        ("Electric Power", "Rate of electrical energy transfer", "intermediate", "electrostatics", "Calculate electrical power"),
        ("Kirchhoff's Laws", "Rules for circuit analysis", "intermediate", "electrostatics", "Analyze complex circuits"),
        ("Series Circuits", "Components connected end-to-end", "beginner", "electrostatics", "Analyze series circuit behavior"),
        ("Parallel Circuits", "Components connected across same voltage", "beginner", "electrostatics", "Analyze parallel circuit behavior"),
        ("RC Circuit", "Circuit with resistor and capacitor", "intermediate", "electrostatics", "Analyze charging and discharging"),
        ("RL Circuit", "Circuit with resistor and inductor", "advanced", "magnetism", "Analyze inductive circuit behavior"),
        ("LC Circuit", "Circuit with inductor and capacitor", "advanced", "magnetism", "Analyze oscillatory circuits"),
        ("RLC Circuit", "Circuit with all three components", "advanced", "magnetism", "Analyze complex AC behavior"),
        ("AC Circuits", "Circuits with alternating current", "intermediate", "magnetism", "Apply phasor analysis"),
        ("Transformer", "Device that changes AC voltage", "intermediate", "magnetism", "Apply transformer principles"),
        ("Motor", "Device that converts electricity to motion", "advanced", "magnetism", "Understand electromagnetic motors"),
        ("Generator", "Device that converts motion to electricity", "advanced", "magnetism", "Apply electromagnetic induction"),
        ("Inductor", "Component that stores magnetic energy", "intermediate", "magnetism", "Calculate inductance effects"),
        ("Mutual Inductance", "Induction between coils", "advanced", "magnetism", "Analyze coupled circuits"),
        ("Self Inductance", "Induction within a coil", "advanced", "magnetism", "Calculate back EMF effects"),
        ("Electromagnetic Waves", "Waves of electric and magnetic fields", "advanced", "waves", "Understand wave propagation"),
        ("Light", "Electromagnetic radiation visible to humans", "intermediate", "waves", "Apply wave properties to optics"),
        ("Reflection", "Bouncing of waves at boundaries", "beginner", "waves", "Apply law of reflection"),
        ("Refraction", "Bending of waves at boundaries", "intermediate", "waves", "Apply Snell's law"),
        ("Diffraction", "Bending of waves around obstacles", "intermediate", "waves", "Understand wave spreading"),
        ("Polarization", "Orientation of wave oscillations", "intermediate", "waves", "Apply polarization concepts"),
        ("Lens", "Optical device that focuses light", "intermediate", "waves", "Apply thin lens equation"),
        ("Mirror", "Surface that reflects light", "beginner", "waves", "Analyze image formation"),
        ("Prism", "Transparent object that disperses light", "intermediate", "waves", "Understand dispersion"),
        ("Fiber Optics", "Guiding light through total internal reflection", "advanced", "waves", "Apply waveguide principles")
    ]
    
    concept_ids = {}
    for concept_name, description, difficulty, subdomain, objectives in concepts_data:
        concept_id = kg.add_node("Concept", {
            "name": concept_name,
            "description": description,
            "difficulty_level": difficulty,
            "domain": [sd for sd, _ in [(s, d) for s, d, _ in subdomains_data if s == subdomain]][0] if subdomain in [s for s, _, _ in subdomains_data] else "unknown",
            "subdomain": subdomain,
            "category": f"physics_{subdomain}",
            "learning_objectives": objectives,
            "common_misconceptions": f"Common misconceptions about {concept_name.lower()}",
            "created_at": str(datetime.now())
        })
        concept_ids[concept_name] = concept_id
        
        # Link to subdomain
        if subdomain in subdomain_ids:
            kg.add_relationship(subdomain_ids[subdomain], "CONTAINS", concept_id)
    
    # === FORMULAS ===
    formulas_data = [
        ("kinematics_1", "First kinematic equation", "v = v₀ + at", "v: final velocity, v₀: initial velocity, a: acceleration, t: time", "kinematics"),
        ("kinematics_2", "Second kinematic equation", "x = x₀ + v₀t + ½at²", "x: position, x₀: initial position", "kinematics"),
        ("kinematics_3", "Third kinematic equation", "v² = v₀² + 2a(x - x₀)", "Independent of time", "kinematics"),
        ("projectile_range", "Projectile range", "R = v₀²sin(2θ)/g", "R: range, θ: launch angle, g: gravity", "kinematics"),
        ("newton_second", "Newton's second law", "F = ma", "F: net force, m: mass, a: acceleration", "forces"),
        ("weight", "Weight formula", "W = mg", "W: weight, g: gravitational acceleration", "forces"),
        ("friction_static", "Static friction", "fs ≤ μsN", "fs: static friction, μs: coefficient, N: normal force", "forces"),
        ("friction_kinetic", "Kinetic friction", "fk = μkN", "fk: kinetic friction, μk: coefficient", "forces"),
        ("work", "Work formula", "W = F·d·cos(θ)", "W: work, F: force, d: displacement, θ: angle", "energy"),
        ("kinetic_energy", "Kinetic energy", "KE = ½mv²", "KE: kinetic energy, m: mass, v: velocity", "energy"),
        ("gravitational_pe", "Gravitational potential energy", "PE = mgh", "PE: potential energy, h: height", "energy"),
        ("elastic_pe", "Elastic potential energy", "PE = ½kx²", "k: spring constant, x: displacement", "energy"),
        ("power", "Power formula", "P = W/t = F·v", "P: power, t: time, v: velocity", "energy"),
        ("momentum", "Momentum formula", "p = mv", "p: momentum, m: mass, v: velocity", "momentum"),
        ("impulse", "Impulse-momentum theorem", "J = Δp = FΔt", "J: impulse, Δp: change in momentum", "momentum"),
        ("elastic_collision_1d", "Elastic collision formula", "v₁f = ((m₁-m₂)v₁i + 2m₂v₂i)/(m₁+m₂)", "1D elastic collision velocities", "momentum"),
        ("torque", "Torque formula", "τ = r × F = rF sin(θ)", "τ: torque, r: radius vector, F: force", "rotational_motion"),
        ("angular_momentum", "Angular momentum", "L = Iω", "L: angular momentum, I: moment of inertia, ω: angular velocity", "rotational_motion"),
        ("rotational_ke", "Rotational kinetic energy", "KE = ½Iω²", "Rotational energy formula", "rotational_motion"),
        ("rolling_condition", "Rolling without slipping", "v = rω", "v: linear velocity, r: radius, ω: angular velocity", "rotational_motion"),
        ("shm_position", "SHM position", "x(t) = A cos(ωt + φ)", "A: amplitude, ω: angular frequency, φ: phase", "oscillations"),
        ("frequency_period", "Frequency-period relation", "f = 1/T", "f: frequency, T: period", "oscillations"),
        ("spring_period", "Spring oscillator period", "T = 2π√(m/k)", "m: mass, k: spring constant", "oscillations"),
        ("wave_speed", "Wave speed formula", "v = fλ", "v: wave speed, f: frequency, λ: wavelength", "waves"),
        ("wave_equation", "Sinusoidal wave equation", "y(x,t) = A sin(kx - ωt + φ)", "k: wave number, ω: angular frequency", "waves"),
        ("first_law", "First law of thermodynamics", "ΔU = Q - W", "ΔU: change in internal energy, Q: heat added, W: work done", "heat"),
        ("heat_capacity", "Heat capacity formula", "Q = mcΔT", "c: specific heat capacity, ΔT: temperature change", "heat"),
        ("coulomb_law", "Coulomb's law", "F = kq₁q₂/r²", "F: force, k: Coulomb constant, q: charges, r: distance", "electrostatics"),
        ("electric_field", "Electric field definition", "E = F/q", "E: electric field, F: force, q: test charge", "electrostatics"),
        ("electric_potential", "Electric potential", "V = kQ/r", "V: potential, Q: source charge", "electrostatics"),
        ("magnetic_force", "Magnetic force on charge", "F = q(v × B)", "q: charge, v: velocity, B: magnetic field", "magnetism"),
        ("faraday_law", "Faraday's law", "ε = -dΦ/dt", "ε: induced EMF, Φ: magnetic flux", "magnetism")
    ]
    
    formula_ids = {}
    for formula_id, name, expression, variables, subdomain in formulas_data:
        fid = kg.add_node("Formula", {
            "id": formula_id,
            "name": name,
            "expression": expression,
            "variables": variables,
            "domain": [sd for sd, _ in [(s, d) for s, d, _ in subdomains_data if s == subdomain]][0] if subdomain in [s for s, _, _ in subdomains_data] else "unknown",
            "subdomain": subdomain,
            "difficulty_level": "intermediate",
            "created_at": str(datetime.now())
        })
        formula_ids[formula_id] = fid
        
        # Link to subdomain
        if subdomain in subdomain_ids:
            kg.add_relationship(subdomain_ids[subdomain], "CONTAINS", fid)
    
    # === PROBLEMS ===
    problems_data = [
        # Kinematics problems
        ("kinematics_p001", "Car Acceleration Problem", "A car accelerates from rest to 30 m/s in 10 seconds. Find the acceleration and distance traveled.", "calculation", "beginner", "Acceleration", ["Given: v₀=0, v=30m/s, t=10s", "Find acceleration: a = (v-v₀)/t", "Find distance: x = v₀t + ½at²"], "a = 3 m/s², x = 150 m"),
        ("kinematics_p002", "Projectile Motion", "A ball is thrown horizontally at 15 m/s from a height of 20 m. Find the time of flight and range.", "application", "intermediate", "Projectile Motion", ["Analyze vertical motion", "Find time using y = ½gt²", "Calculate horizontal range"], "t = 2.02 s, R = 30.3 m"),
        ("kinematics_p003", "Free Fall Problem", "A stone is dropped from a 45 m tall building. Find the velocity just before hitting the ground.", "calculation", "beginner", "Acceleration", ["Use v² = v₀² + 2as", "v₀ = 0, a = g, s = 45 m", "Calculate final velocity"], "v = 29.7 m/s"),
        ("kinematics_p004", "Relative Motion", "Car A travels at 60 km/h east, Car B at 40 km/h west. Find relative velocity.", "application", "intermediate", "Relative Motion", ["Define reference frame", "Add velocities vectorially", "Consider opposite directions"], "v_rel = 100 km/h"),
        ("kinematics_p005", "Circular Motion", "A car goes around a circular track of radius 50 m at 20 m/s. Find centripetal acceleration.", "calculation", "intermediate", "Circular Motion", ["Use a_c = v²/r", "Substitute values", "Calculate acceleration"], "a_c = 8 m/s²"),
        
        # Forces problems
        ("forces_p001", "Friction on Incline", "A 10 kg block slides down a 30° incline with coefficient of friction 0.3. Find the acceleration.", "application", "intermediate", "Friction", ["Draw free body diagram", "Resolve weight components", "Apply Newton's second law", "Include friction force"], "a = 2.4 m/s²"),
        ("forces_p002", "Pulley System", "Two masses 5 kg and 3 kg are connected over a pulley. Find the acceleration of the system.", "application", "advanced", "Tension", ["Draw FBDs for both masses", "Write Newton's second law", "Solve simultaneous equations"], "a = 2.45 m/s²"),
        ("forces_p003", "Spring Force", "A 2 kg mass stretches a spring by 0.1 m. Find the spring constant.", "calculation", "beginner", "Force", ["Use Hooke's law F = kx", "Weight provides restoring force", "mg = kx"], "k = 196 N/m"),
        ("forces_p004", "Elevator Problem", "A 70 kg person in an elevator accelerating upward at 2 m/s². Find apparent weight.", "application", "intermediate", "Newton Second Law", ["Apply F = ma", "Include both weight and normal force", "N - mg = ma"], "N = 826 N"),
        ("forces_p005", "Banked Curve", "A car takes a banked curve at 30 m/s. Bank angle is 15°. Find minimum radius.", "application", "advanced", "Circular Motion", ["Resolve forces on banked curve", "Apply centripetal force condition", "Consider friction"], "r = 310 m"),
        
        # Energy problems
        ("energy_p001", "Roller Coaster Energy", "A roller coaster car of mass 500 kg starts from rest at height 50 m. Find speed at bottom.", "application", "intermediate", "Conservation of Energy", ["Apply conservation of energy", "PE_initial = KE_final", "mgh = ½mv²"], "v = 31.3 m/s"),
        ("energy_p002", "Work Against Friction", "A 10 kg box is pulled 5 m across a floor with μ = 0.3. How much work is done against friction?", "calculation", "intermediate", "Work", ["Calculate friction force", "W = F·d", "Work = friction × distance"], "W = 147 J"),
        ("energy_p003", "Spring Energy", "A spring with k = 200 N/m is compressed 0.15 m. Find stored elastic energy.", "calculation", "beginner", "Elastic Potential Energy", ["Use PE = ½kx²", "Substitute values", "Calculate energy"], "PE = 2.25 J"),
        ("energy_p004", "Power Calculation", "A motor lifts a 500 kg load 10 m in 20 s. Find the power output.", "calculation", "intermediate", "Power", ["Calculate work done", "W = mgh", "Power = Work/time"], "P = 2450 W"),
        ("energy_p005", "Pendulum Energy", "A pendulum bob of mass 2 kg is released from 60° from vertical. Find speed at bottom.", "application", "advanced", "Conservation of Energy", ["Calculate height change", "Apply energy conservation", "mgh = ½mv²"], "v = 3.13 m/s"),
        
        # Momentum problems
        ("momentum_p001", "Collision Problem", "Two cars collide: 1000 kg at 20 m/s hits 800 kg at rest. Find velocities after inelastic collision.", "application", "advanced", "Inelastic Collision", ["Apply conservation of momentum", "Calculate combined mass", "Find final velocity"], "v_final = 11.1 m/s"),
        ("momentum_p002", "Elastic Collision", "Two balls: 2 kg at 10 m/s hits 3 kg at rest elastically. Find final velocities.", "application", "advanced", "Elastic Collision", ["Conserve momentum and energy", "Use collision formulas", "Solve for both velocities"], "v₁ = -2 m/s, v₂ = 8 m/s"),
        ("momentum_p003", "Impulse Problem", "A 0.15 kg baseball is pitched at 40 m/s and caught. Find impulse if contact time is 0.05 s.", "calculation", "intermediate", "Impulse", ["Calculate change in momentum", "J = Δp = mΔv", "Consider direction change"], "J = 12 N·s"),
        ("momentum_p004", "Explosion Problem", "A 10 kg object explodes into two pieces: 6 kg and 4 kg. If 6 kg moves at 8 m/s east, find velocity of 4 kg piece.", "application", "intermediate", "Conservation of Momentum", ["Initial momentum is zero", "Apply momentum conservation", "p₁ + p₂ = 0"], "v₂ = 12 m/s west"),
        ("momentum_p005", "Rocket Propulsion", "A rocket of mass 1000 kg ejects 10 kg of fuel at 500 m/s. Find rocket's recoil velocity.", "application", "advanced", "Conservation of Momentum", ["Apply momentum conservation", "Consider mass change", "Calculate recoil"], "v = 5.05 m/s"),
        
        # Rotational motion problems
        ("rotation_p001", "Torque Calculation", "A force of 50 N is applied 2 m from pivot at 60°. Calculate torque.", "calculation", "beginner", "Torque", ["Use τ = rF sin θ", "Substitute values", "Calculate torque"], "τ = 86.6 N·m"),
        ("rotation_p002", "Rolling Motion", "A solid sphere of radius 0.1 m rolls down a 30° incline. Find acceleration.", "application", "advanced", "Rolling Motion", ["Apply energy methods", "Consider rotational inertia", "Use rolling condition"], "a = 3.5 m/s²"),
        ("rotation_p003", "Angular Kinematics", "A wheel accelerates from rest at 2 rad/s² for 5 s. Find final angular velocity and displacement.", "calculation", "intermediate", "Angular Acceleration", ["Use ω = ω₀ + αt", "θ = ω₀t + ½αt²", "Substitute values"], "ω = 10 rad/s, θ = 25 rad"),
        ("rotation_p004", "Conservation of Angular Momentum", "A figure skater spins at 2 rev/s with arms out (I = 4 kg·m²). Find spin rate with arms in (I = 1 kg·m²).", "application", "advanced", "Angular Momentum", ["Apply L = Iω", "L₁ = L₂", "Solve for final ω"], "ω₂ = 8 rev/s"),
        
        # Wave and oscillation problems
        ("waves_p001", "Wave Speed Calculation", "A wave has frequency 50 Hz and wavelength 2 m. Calculate the wave speed.", "calculation", "beginner", "Wave Speed", ["Use v = fλ", "Substitute values", "Calculate result"], "v = 100 m/s"),
        ("waves_p002", "Doppler Effect", "An ambulance siren at 1000 Hz approaches at 30 m/s. Find frequency heard by observer.", "application", "advanced", "Doppler Effect", ["Use Doppler formula", "f' = f(v + v₀)/(v - v_s)", "Substitute values"], "f' = 1096 Hz"),
        ("oscillations_p001", "Simple Pendulum", "A 2 m pendulum oscillates on Earth. Find the period.", "calculation", "beginner", "Period", ["Use T = 2π√(L/g)", "Substitute values", "Calculate period"], "T = 2.84 s"),
        ("oscillations_p002", "Mass-Spring System", "A 0.5 kg mass on a spring with k = 200 N/m. Find oscillation frequency.", "calculation", "intermediate", "Frequency", ["Use f = (1/2π)√(k/m)", "Substitute values", "Calculate frequency"], "f = 3.18 Hz"),
        ("oscillations_p003", "Energy in SHM", "A 1 kg mass oscillates with amplitude 0.1 m on a spring k = 400 N/m. Find maximum velocity.", "application", "intermediate", "Simple Harmonic Motion", ["Use energy conservation", "E = ½kA² = ½mv²", "Solve for v_max"], "v_max = 2 m/s"),
        
        # Thermodynamics problems
        ("thermo_p001", "Heat Capacity", "How much heat is needed to raise temperature of 2 kg water from 20°C to 80°C? (c = 4186 J/kg·K)", "calculation", "beginner", "Heat Capacity", ["Use Q = mcΔT", "Substitute values", "Calculate heat"], "Q = 502,320 J"),
        ("thermo_p002", "First Law of Thermodynamics", "A gas absorbs 500 J of heat and does 300 J of work. Find change in internal energy.", "application", "intermediate", "First Law of Thermodynamics", ["Apply ΔU = Q - W", "Q = +500 J (absorbed)", "W = +300 J (done by system)"], "ΔU = 200 J"),
        ("thermo_p003", "Thermal Expansion", "A steel rod of length 2 m is heated from 20°C to 120°C. Find the change in length. (α = 12×10⁻⁶/°C)", "calculation", "intermediate", "Temperature", ["Use ΔL = αLΔT", "Substitute values", "Calculate expansion"], "ΔL = 2.4 mm"),
        
        # Electromagnetism problems
        ("electrostatics_p001", "Electric Force Problem", "Two charges +2μC and -3μC are 0.1 m apart. Calculate the electric force.", "calculation", "intermediate", "Electric Force", ["Apply Coulomb's law", "F = kq₁q₂/r²", "Substitute values"], "F = 5.4 N attractive"),
        ("electrostatics_p002", "Electric Field", "Find the electric field at distance 0.05 m from a +10 μC charge.", "calculation", "beginner", "Electric Field", ["Use E = kQ/r²", "Substitute values", "Calculate field strength"], "E = 3.6×10⁷ N/C"),
        ("electrostatics_p003", "Electric Potential", "Calculate electric potential at 0.1 m from a +5 μC charge.", "calculation", "intermediate", "Electric Potential", ["Use V = kQ/r", "Substitute values", "Calculate potential"], "V = 450,000 V"),
        ("magnetism_p001", "Magnetic Force", "An electron moves at 10⁶ m/s perpendicular to a 0.1 T magnetic field. Find the magnetic force.", "calculation", "intermediate", "Magnetic Force", ["Use F = qvB", "q = -1.6×10⁻¹⁹ C", "Calculate force"], "F = 1.6×10⁻¹⁴ N"),
        ("magnetism_p002", "Electromagnetic Induction", "A coil with 100 turns has flux changing at 0.01 Wb/s. Find induced EMF.", "calculation", "intermediate", "Electromagnetic Induction", ["Use Faraday's law", "ε = -NdΦ/dt", "Substitute values"], "ε = -1.0 V")
    ]
    
    problem_ids = {}
    for problem_id, title, description, problem_type, difficulty, concept_name, solution_steps, answer in problems_data:
        pid = kg.add_node("Problem", {
            "id": problem_id,
            "title": title,
            "description": description,
            "problem_type": problem_type,
            "difficulty_level": difficulty,
            "solution_steps": solution_steps,
            "answer": answer,
            "created_at": str(datetime.now())
        })
        problem_ids[problem_id] = pid
        
        # Link to concept
        if concept_name in concept_ids:
            kg.add_relationship(concept_ids[concept_name], "HAS_PROBLEM", pid)
            kg.add_relationship(pid, "APPLIES_CONCEPT", concept_ids[concept_name])
    
    # === EXPLANATIONS ===
    explanations_data = [
        # Basic concept explanations
        ("velocity_explanation", "Understanding Velocity", "Velocity is a vector quantity that describes the rate of change of position. Unlike speed, velocity includes direction. Average velocity is total displacement divided by time, while instantaneous velocity is the rate of change at a specific moment.", "conceptual", "Velocity"),
        ("acceleration_explanation", "Understanding Acceleration", "Acceleration is the rate of change of velocity. It can represent speeding up, slowing down, or changing direction. Constant acceleration leads to the kinematic equations used in motion problems.", "conceptual", "Acceleration"),
        ("force_explanation", "What is Force?", "Force is a push or pull that can change an object's motion. Forces are vectors with magnitude and direction. Net force determines acceleration according to Newton's second law.", "conceptual", "Force"),
        ("newton_laws_explanation", "Newton's Laws of Motion", "Newton's three laws form the foundation of classical mechanics: (1) Objects at rest stay at rest unless acted upon by a force (inertia), (2) F = ma relates force to acceleration, (3) Forces come in action-reaction pairs.", "principle", "Newton Second Law"),
        ("friction_explanation", "Understanding Friction", "Friction is a force that opposes motion between surfaces in contact. Static friction prevents motion, while kinetic friction acts during motion. Friction depends on the normal force and surface properties.", "conceptual", "Friction"),
        
        # Energy and work explanations
        ("work_explanation", "Work and Energy", "Work is the transfer of energy through force acting over a distance. Work equals force times displacement times the cosine of the angle between them. Work can be positive (energy added) or negative (energy removed).", "conceptual", "Work"),
        ("kinetic_energy_explanation", "Kinetic Energy", "Kinetic energy is the energy of motion, given by KE = ½mv². All moving objects have kinetic energy. The work-energy theorem states that work done equals the change in kinetic energy.", "conceptual", "Kinetic Energy"),
        ("potential_energy_explanation", "Potential Energy", "Potential energy is stored energy due to position or configuration. Gravitational potential energy depends on height, while elastic potential energy depends on deformation. Conservative forces can store and release potential energy.", "conceptual", "Potential Energy"),
        ("energy_conservation_explanation", "Energy Conservation Principle", "Energy cannot be created or destroyed, only transformed from one form to another. In isolated systems, total mechanical energy (kinetic plus potential) remains constant when only conservative forces act.", "principle", "Conservation of Energy"),
        ("power_explanation", "Power in Physics", "Power is the rate of doing work or transferring energy. Power equals work divided by time, or force times velocity. Higher power means energy is transferred more quickly.", "conceptual", "Power"),
        
        # Momentum explanations
        ("momentum_explanation", "Understanding Momentum", "Momentum is the product of mass and velocity (p = mv). It's a vector quantity that describes the motion of objects. Momentum is conserved in isolated systems, making it crucial for analyzing collisions.", "conceptual", "Momentum"),
        ("impulse_explanation", "Impulse and Momentum Change", "Impulse is the product of force and time, equal to the change in momentum. Large forces acting for short times or small forces for long times can produce the same impulse. This explains why airbags and crumple zones work.", "conceptual", "Impulse"),
        ("collision_explanation", "Types of Collisions", "Collisions are classified as elastic or inelastic. Elastic collisions conserve both momentum and kinetic energy, while inelastic collisions conserve only momentum. Real collisions are often partially elastic.", "conceptual", "Collision"),
        
        # Rotational motion explanations
        ("torque_explanation", "Understanding Torque", "Torque is the rotational equivalent of force, causing objects to rotate. It equals the force times the lever arm (perpendicular distance from axis). Greater torque produces greater angular acceleration.", "conceptual", "Torque"),
        ("angular_momentum_explanation", "Angular Momentum", "Angular momentum is the rotational analog of linear momentum, equal to moment of inertia times angular velocity. It's conserved when no external torques act, explaining phenomena like ice skater spins.", "conceptual", "Angular Momentum"),
        ("rotational_kinetic_energy_explanation", "Rotational Kinetic Energy", "Rotating objects have kinetic energy due to their rotation, given by KE_rot = ½Iω². This is separate from translational kinetic energy. Rolling objects have both translational and rotational kinetic energy.", "conceptual", "Rotational Kinetic Energy"),
        
        # Wave and oscillation explanations
        ("shm_explanation", "Simple Harmonic Motion", "Simple harmonic motion occurs when the restoring force is proportional to displacement. Examples include springs and pendulums. SHM produces sinusoidal motion with characteristic period and frequency.", "conceptual", "Simple Harmonic Motion"),
        ("wave_properties_explanation", "Properties of Waves", "Waves transfer energy without transferring matter. Key properties include wavelength (distance between peaks), frequency (oscillations per second), amplitude (maximum displacement), and speed (v = fλ).", "conceptual", "Wave"),
        ("interference_explanation", "Wave Interference", "When waves overlap, they interfere. Constructive interference occurs when waves are in phase (peaks align), while destructive interference occurs when waves are out of phase. This creates complex wave patterns.", "conceptual", "Interference"),
        ("doppler_explanation", "Doppler Effect", "The Doppler effect is the change in frequency when source or observer moves. Frequency increases when approaching and decreases when receding. This explains why sirens change pitch as they pass by.", "conceptual", "Doppler Effect"),
        
        # Thermodynamics explanations
        ("temperature_explanation", "Temperature and Heat", "Temperature measures the average kinetic energy of particles, while heat is energy transferred due to temperature differences. Heat flows from hot to cold until thermal equilibrium is reached.", "conceptual", "Temperature"),
        ("first_law_explanation", "First Law of Thermodynamics", "The first law states that energy is conserved: the change in internal energy equals heat added minus work done by the system (ΔU = Q - W). This connects thermodynamics to energy conservation.", "principle", "First Law of Thermodynamics"),
        ("entropy_explanation", "Entropy and the Second Law", "Entropy measures disorder in a system. The second law states that entropy of isolated systems always increases. This explains why some processes are irreversible and why heat engines have limited efficiency.", "conceptual", "Entropy"),
        
        # Electromagnetism explanations
        ("electric_field_explanation", "Electric Fields", "Electric fields represent the force per unit charge around charged objects. Field lines show direction and strength - closer lines mean stronger fields. Electric fields can accelerate charged particles.", "conceptual", "Electric Field"),
        ("electric_potential_explanation", "Electric Potential", "Electric potential is potential energy per unit charge. Potential difference (voltage) drives current flow. Higher potential means more energy available per unit charge. Work is done moving charges through potential differences.", "conceptual", "Electric Potential"),
        ("magnetic_field_explanation", "Magnetic Fields", "Magnetic fields exert forces on moving charges and current-carrying wires. Field lines form closed loops from north to south poles. Magnetic fields can change the direction but not the speed of charged particles.", "conceptual", "Magnetic Field"),
        ("electromagnetic_induction_explanation", "Electromagnetic Induction", "Changing magnetic fields induce electric fields and currents (Faraday's law). This principle underlies generators, transformers, and motors. Lenz's law determines the direction of induced currents.", "principle", "Electromagnetic Induction"),
        
        # Mathematical and conceptual explanations
        ("vector_explanation", "Vectors in Physics", "Vectors have both magnitude and direction, unlike scalars which have only magnitude. Vector addition follows geometric rules. Many physics quantities are vectors: velocity, acceleration, force, momentum, and fields.", "mathematical", "Force"),
        ("reference_frame_explanation", "Reference Frames", "Physics laws look the same in all inertial reference frames. Relative motion depends on the chosen frame. Galilean transformations connect different frames at low speeds, while Lorentz transformations apply at high speeds.", "conceptual", "Relative Motion"),
        ("conservation_laws_explanation", "Conservation Laws", "Conservation laws state that certain quantities remain constant in isolated systems. Energy, momentum, angular momentum, and charge are conserved. These laws are fundamental to understanding physics.", "principle", "Conservation of Energy"),
        ("dimensional_analysis_explanation", "Dimensional Analysis", "Dimensional analysis checks equation validity and derives relationships. Physical quantities have dimensions (length, time, mass). Equations must be dimensionally consistent. This technique helps solve problems and check answers.", "mathematical", "Force")
    ]
    
    explanation_ids = {}
    for explanation_id, title, content, explanation_type, concept_name in explanations_data:
        eid = kg.add_node("Explanation", {
            "id": explanation_id,
            "title": title,
            "content": content,
            "explanation_type": explanation_type,
            "created_at": str(datetime.now())
        })
        explanation_ids[explanation_id] = eid
        
        # Link to concept
        if concept_name in concept_ids:
            kg.add_relationship(concept_ids[concept_name], "HAS_EXPLANATION", eid)
    
    # === UNITS ===
    units_data = [
        ("meter", "m", "length", True, ""),
        ("second", "s", "time", True, ""),
        ("kilogram", "kg", "mass", True, ""),
        ("newton", "N", "force", False, "kg⋅m/s²"),
        ("joule", "J", "energy", False, "N⋅m"),
        ("watt", "W", "power", False, "J/s"),
        ("hertz", "Hz", "frequency", False, "1/s"),
        ("coulomb", "C", "electric charge", False, "A⋅s"),
        ("volt", "V", "electric potential", False, "J/C"),
        ("tesla", "T", "magnetic field", False, "Wb/m²")
    ]
    
    unit_ids = {}
    for name, symbol, quantity, si_base, definition in units_data:
        uid = kg.add_node("Unit", {
            "name": name,
            "symbol": symbol,
            "quantity": quantity,
            "si_base": si_base,
            "definition": definition,
            "created_at": str(datetime.now())
        })
        unit_ids[name] = uid
    
    # === CREATE EDUCATIONAL RELATIONSHIPS ===
    
    # Prerequisites relationships
    prerequisites = [
        # Basic mechanics prerequisites
        ("Position", "Displacement"),
        ("Displacement", "Velocity"),
        ("Velocity", "Acceleration"),
        ("Acceleration", "Force"),
        ("Force", "Work"),
        ("Work", "Energy"),
        ("Work", "Kinetic Energy"),
        ("Kinetic Energy", "Conservation of Energy"),
        ("Force", "Momentum"),
        ("Velocity", "Momentum"),
        ("Momentum", "Impulse"),
        ("Force", "Torque"),
        ("Velocity", "Angular Velocity"),
        ("Acceleration", "Angular Acceleration"),
        
        # Advanced mechanics prerequisites
        ("Newton First Law", "Newton Second Law"),
        ("Newton Second Law", "Newton Third Law"),
        ("Newton Second Law", "Friction"),
        ("Force", "Normal Force"),
        ("Force", "Tension"),
        ("Weight", "Gravitational Potential Energy"),
        ("Force", "Spring Force"),
        ("Elastic Potential Energy", "Simple Harmonic Motion"),
        ("Circular Motion", "Centripetal Force"),
        ("Centripetal Force", "Centripetal Acceleration"),
        ("Universal Gravitation", "Orbital Motion"),
        ("Energy", "Escape Velocity"),
        
        # Rotational motion prerequisites
        ("Angular Velocity", "Angular Acceleration"),
        ("Torque", "Angular Momentum"),
        ("Moment of Inertia", "Rotational Kinetic Energy"),
        ("Rolling Motion", "Angular Momentum"),
        
        # Fluid mechanics prerequisites
        ("Pressure", "Buoyancy"),
        ("Pressure", "Fluid Statics"),
        ("Fluid Statics", "Fluid Dynamics"),
        ("Energy", "Bernoulli's Equation"),
        
        # Wave and oscillation prerequisites
        ("Simple Harmonic Motion", "Period"),
        ("Period", "Frequency"),
        ("Frequency", "Wave Speed"),
        ("Amplitude", "Simple Harmonic Motion"),
        ("Simple Harmonic Motion", "Wave"),
        ("Wave", "Wavelength"),
        ("Wave", "Interference"),
        ("Wave", "Standing Wave"),
        ("Wave", "Doppler Effect"),
        ("Oscillations", "Damped Oscillations"),
        ("Oscillations", "Forced Oscillations"),
        ("Forced Oscillations", "Resonance"),
        
        # Thermodynamics prerequisites
        ("Temperature", "Heat"),
        ("Heat", "Heat Capacity"),
        ("Internal Energy", "First Law of Thermodynamics"),
        ("First Law of Thermodynamics", "Heat Engine"),
        ("Heat Engine", "Carnot Engine"),
        ("Entropy", "Second Law of Thermodynamics"),
        ("Heat", "Thermal Conductivity"),
        ("Heat", "Convection"),
        ("Heat", "Radiation"),
        
        # Electromagnetism prerequisites
        ("Electric Charge", "Electric Force"),
        ("Electric Force", "Electric Field"),
        ("Electric Field", "Electric Potential"),
        ("Electric Potential", "Capacitance"),
        ("Electric Current", "Resistance"),
        ("Resistance", "Ohm's Law"),
        ("Ohm's Law", "Electric Power"),
        ("Electric Power", "Kirchhoff's Laws"),
        ("Kirchhoff's Laws", "Series Circuits"),
        ("Series Circuits", "Parallel Circuits"),
        ("Capacitance", "RC Circuit"),
        ("Magnetic Field", "Magnetic Force"),
        ("Magnetic Force", "Electromagnetic Induction"),
        ("Electromagnetic Induction", "Lenz Law"),
        ("Inductor", "RL Circuit"),
        ("Inductor", "LC Circuit"),
        ("LC Circuit", "RLC Circuit"),
        ("AC Circuits", "Transformer"),
        ("Electromagnetic Induction", "Generator"),
        ("Electromagnetic Induction", "Motor"),
        ("Electromagnetic Waves", "Light"),
        ("Light", "Reflection"),
        ("Light", "Refraction"),
        ("Light", "Diffraction"),
        ("Light", "Polarization")
    ]
    
    for prereq, concept in prerequisites:
        if prereq in concept_ids and concept in concept_ids:
            kg.add_relationship(concept_ids[prereq], "PREREQUISITE_FOR", concept_ids[concept])
            kg.add_relationship(concept_ids[concept], "REQUIRES", concept_ids[prereq])
    
    # Related concepts
    related_concepts = [
        # Kinematics relationships
        ("Velocity", "Acceleration", "both describe motion characteristics"),
        ("Displacement", "Distance", "both measure spatial change"),
        ("Average Velocity", "Instantaneous Velocity", "different velocity measures"),
        ("Projectile Motion", "Circular Motion", "both are 2D motion types"),
        ("Relative Motion", "Reference Frame", "motion depends on frame"),
        
        # Force relationships  
        ("Force", "Momentum", "both relate to Newton's laws"),
        ("Static Friction", "Kinetic Friction", "two types of friction"),
        ("Normal Force", "Weight", "perpendicular contact forces"),
        ("Tension", "Compression", "opposite force types"),
        ("Centripetal Force", "Centrifugal Force", "action-reaction in circular motion"),
        ("Gravitational Force", "Electric Force", "both follow inverse square laws"),
        ("Spring Force", "Elastic Potential Energy", "spring stores energy"),
        
        # Energy relationships
        ("Work", "Energy", "work transfers energy"),
        ("Kinetic Energy", "Momentum", "both depend on mass and velocity"),
        ("Kinetic Energy", "Potential Energy", "energy can transform between forms"),
        ("Gravitational Potential Energy", "Elastic Potential Energy", "both are stored energy"),
        ("Conservation of Energy", "Conservation of Momentum", "both are conservation laws"),
        ("Power", "Energy", "power is rate of energy transfer"),
        ("Work-Energy Theorem", "Conservation of Energy", "both relate work and energy"),
        ("Mechanical Energy", "Total Energy", "mechanical is subset of total"),
        
        # Rotational motion relationships
        ("Linear Motion", "Rotational Motion", "analogous motion types"),
        ("Force", "Torque", "linear vs rotational force analogs"),
        ("Mass", "Moment of Inertia", "resistance to motion"),
        ("Linear Momentum", "Angular Momentum", "momentum analogs"),
        ("Kinetic Energy", "Rotational Kinetic Energy", "translational vs rotational energy"),
        ("Rolling Motion", "Translational Motion", "combined motion types"),
        
        # Oscillations and waves
        ("Frequency", "Period", "inverse relationship"),
        ("Amplitude", "Energy", "amplitude affects oscillation energy"),
        ("Simple Harmonic Motion", "Circular Motion", "SHM is projection of circular motion"),
        ("Oscillations", "Waves", "waves are traveling oscillations"),
        ("Wavelength", "Frequency", "wave properties"),
        ("Wave Speed", "Medium Properties", "speed depends on medium"),
        ("Transverse Wave", "Longitudinal Wave", "perpendicular wave orientations"),
        ("Interference", "Superposition", "wave combination principles"),
        ("Standing Wave", "Traveling Wave", "different wave behaviors"),
        ("Doppler Effect", "Relative Motion", "frequency shift due to motion"),
        ("Resonance", "Natural Frequency", "maximum response at natural frequency"),
        
        # Thermodynamics relationships
        ("Temperature", "Heat", "thermal concepts"),
        ("Heat", "Work", "both are energy transfer methods"),
        ("Internal Energy", "Temperature", "energy relates to temperature"),
        ("First Law of Thermodynamics", "Conservation of Energy", "energy conservation in thermal systems"),
        ("Second Law of Thermodynamics", "Entropy", "entropy always increases"),
        ("Heat Engine", "Refrigerator", "opposite thermodynamic cycles"),
        ("Carnot Engine", "Ideal Gas", "theoretical maximum efficiency"),
        ("Adiabatic Process", "Isothermal Process", "different thermodynamic processes"),
        ("Thermal Conductivity", "Heat Transfer", "conduction mechanism"),
        ("Convection", "Fluid Motion", "heat transfer by bulk motion"),
        ("Radiation", "Electromagnetic Waves", "heat transfer by EM waves"),
        
        # Electromagnetism relationships
        ("Electric Field", "Magnetic Field", "both are field concepts"),
        ("Electric Charge", "Electric Force", "charge creates force"),
        ("Electric Field", "Electric Potential", "field and potential are related"),
        ("Electric Current", "Magnetic Field", "current creates magnetic field"),
        ("Changing Electric Field", "Magnetic Field", "Maxwell's equations"),
        ("Changing Magnetic Field", "Electric Field", "Faraday's law"),
        ("Capacitor", "Electric Field", "capacitor stores electric field energy"),
        ("Inductor", "Magnetic Field", "inductor stores magnetic field energy"),
        ("Resistance", "Electric Current", "opposition to current flow"),
        ("Series Circuits", "Parallel Circuits", "different circuit topologies"),
        ("AC Circuits", "DC Circuits", "alternating vs direct current"),
        ("Transformer", "Mutual Inductance", "transformer uses mutual induction"),
        ("Motor", "Generator", "opposite electromagnetic conversions"),
        ("Electromagnetic Waves", "Light", "light is electromagnetic radiation"),
        
        # Optics relationships
        ("Reflection", "Refraction", "wave behavior at boundaries"),
        ("Reflection", "Mirror", "mirrors use reflection"),
        ("Refraction", "Lens", "lenses use refraction"),
        ("Diffraction", "Interference", "wave interference effects"),
        ("Polarization", "Transverse Waves", "polarization applies to transverse waves"),
        ("Dispersion", "Refraction", "frequency-dependent refraction"),
        
        # Fluid mechanics relationships
        ("Pressure", "Force", "pressure is force per area"),
        ("Buoyancy", "Fluid Statics", "buoyant force in static fluids"),
        ("Fluid Dynamics", "Bernoulli's Equation", "energy conservation in fluids"),
        ("Viscosity", "Fluid Flow", "resistance to flow"),
        ("Surface Tension", "Intermolecular Forces", "molecular attraction at surfaces"),
        
        # Material properties relationships
        ("Stress", "Strain", "force causes deformation"),
        ("Young's Modulus", "Elasticity", "measure of stiffness"),
        ("Elastic Deformation", "Plastic Deformation", "reversible vs permanent deformation"),
        ("Shear Modulus", "Shear Stress", "response to shearing forces"),
        ("Bulk Modulus", "Volume Change", "resistance to volume change"),
        
        # Mathematical relationships
        ("Vector", "Scalar", "quantities with and without direction"),
        ("Derivative", "Rate of Change", "mathematical rate concepts"),
        ("Integration", "Area Under Curve", "mathematical accumulation"),
        ("Sine Function", "Simple Harmonic Motion", "sinusoidal motion patterns"),
        ("Exponential Decay", "Damped Oscillations", "decreasing oscillations"),
        
        # Cross-domain relationships
        ("Mechanical Waves", "Electromagnetic Waves", "different wave types"),
        ("Gravitational Field", "Electric Field", "field concept analogy"),
        ("Gravitational Potential", "Electric Potential", "potential concept analogy"),
        ("Thermal Energy", "Kinetic Energy", "random vs ordered motion"),
        ("Chemical Energy", "Electrical Energy", "energy conversion in batteries"),
        ("Nuclear Energy", "Mass-Energy Equivalence", "E=mc² relationship"),
    ]
    
    for concept1, concept2, reason in related_concepts:
        if concept1 in concept_ids and concept2 in concept_ids:
            kg.add_relationship(concept_ids[concept1], "RELATED_TO", concept_ids[concept2], {"reason": reason})
            kg.add_relationship(concept_ids[concept2], "RELATED_TO", concept_ids[concept1], {"reason": reason})
    
    # Link formulas to concepts
    formula_concept_links = [
        # Kinematics formulas
        ("kinematics_1", "Velocity"), ("kinematics_1", "Acceleration"), ("kinematics_1", "Time"),
        ("kinematics_2", "Position"), ("kinematics_2", "Acceleration"), ("kinematics_2", "Velocity"),
        ("kinematics_3", "Velocity"), ("kinematics_3", "Acceleration"), ("kinematics_3", "Position"),
        ("projectile_range", "Projectile Motion"), ("projectile_range", "Velocity"), ("projectile_range", "Gravity"),
        
        # Force formulas
        ("newton_second", "Force"), ("newton_second", "Acceleration"), ("newton_second", "Mass"),
        ("weight", "Weight"), ("weight", "Mass"), ("weight", "Gravity"),
        ("friction_static", "Static Friction"), ("friction_static", "Normal Force"),
        ("friction_kinetic", "Kinetic Friction"), ("friction_kinetic", "Normal Force"),
        
        # Energy formulas
        ("work", "Work"), ("work", "Force"), ("work", "Displacement"),
        ("kinetic_energy", "Kinetic Energy"), ("kinetic_energy", "Mass"), ("kinetic_energy", "Velocity"),
        ("gravitational_pe", "Gravitational Potential Energy"), ("gravitational_pe", "Mass"), ("gravitational_pe", "Height"),
        ("elastic_pe", "Elastic Potential Energy"), ("elastic_pe", "Spring Constant"), ("elastic_pe", "Displacement"),
        ("power", "Power"), ("power", "Work"), ("power", "Force"), ("power", "Velocity"),
        
        # Momentum formulas
        ("momentum", "Momentum"), ("momentum", "Mass"), ("momentum", "Velocity"),
        ("impulse", "Impulse"), ("impulse", "Force"), ("impulse", "Time"), ("impulse", "Momentum"),
        ("elastic_collision_1d", "Elastic Collision"), ("elastic_collision_1d", "Velocity"), ("elastic_collision_1d", "Mass"),
        
        # Rotational motion formulas
        ("torque", "Torque"), ("torque", "Force"), ("torque", "Radius"), ("torque", "Angle"),
        ("angular_momentum", "Angular Momentum"), ("angular_momentum", "Moment of Inertia"), ("angular_momentum", "Angular Velocity"),
        ("rotational_ke", "Rotational Kinetic Energy"), ("rotational_ke", "Moment of Inertia"), ("rotational_ke", "Angular Velocity"),
        ("rolling_condition", "Rolling Motion"), ("rolling_condition", "Velocity"), ("rolling_condition", "Angular Velocity"),
        
        # Oscillation formulas
        ("shm_position", "Simple Harmonic Motion"), ("shm_position", "Amplitude"), ("shm_position", "Frequency"), ("shm_position", "Phase"),
        ("frequency_period", "Frequency"), ("frequency_period", "Period"),
        ("spring_period", "Period"), ("spring_period", "Mass"), ("spring_period", "Spring Constant"),
        
        # Wave formulas
        ("wave_speed", "Wave Speed"), ("wave_speed", "Frequency"), ("wave_speed", "Wavelength"),
        ("wave_equation", "Wave"), ("wave_equation", "Amplitude"), ("wave_equation", "Wavelength"), ("wave_equation", "Frequency"),
        
        # Thermodynamics formulas
        ("first_law", "First Law of Thermodynamics"), ("first_law", "Internal Energy"), ("first_law", "Heat"), ("first_law", "Work"),
        ("heat_capacity", "Heat Capacity"), ("heat_capacity", "Heat"), ("heat_capacity", "Temperature"), ("heat_capacity", "Mass"),
        
        # Electromagnetism formulas
        ("coulomb_law", "Electric Force"), ("coulomb_law", "Electric Charge"), ("coulomb_law", "Distance"),
        ("electric_field", "Electric Field"), ("electric_field", "Electric Force"), ("electric_field", "Charge"),
        ("electric_potential", "Electric Potential"), ("electric_potential", "Electric Charge"), ("electric_potential", "Distance"),
        ("magnetic_force", "Magnetic Force"), ("magnetic_force", "Charge"), ("magnetic_force", "Velocity"), ("magnetic_force", "Magnetic Field"),
        ("faraday_law", "Electromagnetic Induction"), ("faraday_law", "Electric Field"), ("faraday_law", "Magnetic Field")
    ]
    
    for formula_id, concept_name in formula_concept_links:
        if formula_id in formula_ids and concept_name in concept_ids:
            kg.add_relationship(formula_ids[formula_id], "DESCRIBES", concept_ids[concept_name])
            kg.add_relationship(concept_ids[concept_name], "HAS_FORMULA", formula_ids[formula_id])
    
    # === CREATE LEARNING PATHS ===
    learning_paths_data = [
        ("Mechanics Fundamentals", "beginner", "Basic mechanics concepts for introductory physics", 
         ["Position", "Velocity", "Acceleration", "Force", "Newton First Law", "Newton Second Law"]),
        ("Energy and Motion", "intermediate", "Energy concepts and conservation laws",
         ["Work", "Kinetic Energy", "Potential Energy", "Conservation of Energy", "Power"]),
        ("Advanced Mechanics", "advanced", "Complex mechanics including rotations and collisions",
         ["Torque", "Angular Momentum", "Moment of Inertia", "Elastic Collision", "Center of Mass"]),
        ("Waves and Oscillations", "intermediate", "Periodic motion and wave phenomena",
         ["Simple Harmonic Motion", "Period", "Frequency", "Wave", "Interference"]),
        ("Electromagnetism Basics", "intermediate", "Introduction to electric and magnetic fields",
         ["Electric Charge", "Electric Force", "Electric Field", "Magnetic Field", "Magnetic Force"])
    ]
    
    for path_name, level, description, concepts in learning_paths_data:
        lp_id = kg.add_node("LearningPath", {
            "name": path_name,
            "level": level,
            "description": description,
            "created_at": str(datetime.now())
        })
        
        for i, concept_name in enumerate(concepts):
            if concept_name in concept_ids:
                kg.add_relationship(lp_id, "INCLUDES", concept_ids[concept_name], {"order": i+1})
    
    return kg

def validate_knowledge_graph_structure(kg: MockKnowledgeGraph) -> Dict[str, Any]:
    """Validate the mock knowledge graph structure"""
    
    stats = kg.get_statistics()
    validation_results = {
        "total_nodes": stats["total_nodes"],
        "total_relationships": stats["total_relationships"],
        "node_distribution": stats["node_types"],
        "relationship_distribution": stats["relationship_types"]
    }
    
    # Check targets
    validation_results["targets_met"] = {
        "nodes_200_plus": stats["total_nodes"] >= 200,
        "relationships_500_plus": stats["total_relationships"] >= 500
    }
    
    # Check for required node types
    required_node_types = ["Domain", "Subdomain", "Concept", "Formula", "Problem", "Explanation", "Unit", "LearningPath"]
    validation_results["required_node_types_present"] = {}
    for node_type in required_node_types:
        validation_results["required_node_types_present"][node_type] = node_type in stats["node_types"]
    
    # Check for required relationship types
    required_rel_types = ["CONTAINS", "PREREQUISITE_FOR", "REQUIRES", "RELATED_TO", "HAS_PROBLEM", "HAS_FORMULA", "HAS_EXPLANATION", "APPLIES_CONCEPT", "DESCRIBES", "INCLUDES"]
    validation_results["required_relationship_types_present"] = {}
    for rel_type in required_rel_types:
        validation_results["required_relationship_types_present"][rel_type] = rel_type in stats["relationship_types"]
    
    return validation_results

def demonstrate_rag_query_patterns(kg: MockKnowledgeGraph):
    """Demonstrate how RAG queries would work on this structure"""
    
    print("\n" + "="*60)
    print("RAG QUERY PATTERNS DEMONSTRATION")
    print("="*60)
    
    # 1. Find concept by name with related content
    print("\n1. CONCEPT LOOKUP WITH CONTEXT:")
    print("   Query: Find 'Force' concept with all educational content")
    
    force_nodes = [node for node in kg.nodes.values() if node["properties"].get("name") == "Force"]
    if force_nodes:
        force_id = force_nodes[0]["id"]
        print(f"   Found concept: {force_nodes[0]['properties']['description']}")
        
        # Find related formulas, problems, explanations
        related_formulas = [rel for rel in kg.relationships if rel["from"] == force_id and rel["type"] == "HAS_FORMULA"]
        related_problems = [rel for rel in kg.relationships if rel["from"] == force_id and rel["type"] == "HAS_PROBLEM"]
        related_explanations = [rel for rel in kg.relationships if rel["from"] == force_id and rel["type"] == "HAS_EXPLANATION"]
        
        print(f"   Related formulas: {len(related_formulas)}")
        print(f"   Related problems: {len(related_problems)}")
        print(f"   Related explanations: {len(related_explanations)}")
    
    # 2. Learning path traversal
    print("\n2. LEARNING PATH TRAVERSAL:")
    print("   Query: Get 'Mechanics Fundamentals' learning path")
    
    mechanics_paths = [node for node in kg.nodes.values() if node["properties"].get("name") == "Mechanics Fundamentals"]
    if mechanics_paths:
        path_id = mechanics_paths[0]["id"]
        path_concepts = [rel for rel in kg.relationships if rel["from"] == path_id and rel["type"] == "INCLUDES"]
        path_concepts.sort(key=lambda x: x["properties"]["order"])
        print(f"   Found learning path with {len(path_concepts)} concepts in order")
    
    # 3. Prerequisite chain analysis
    print("\n3. PREREQUISITE CHAIN ANALYSIS:")
    print("   Query: Find prerequisite chain for 'Conservation of Energy'")
    
    energy_concepts = [node for node in kg.nodes.values() if node["properties"].get("name") == "Conservation of Energy"]
    if energy_concepts:
        energy_id = energy_concepts[0]["id"]
        prereqs = [rel for rel in kg.relationships if rel["to"] == energy_id and rel["type"] == "REQUIRES"]
        print(f"   Found {len(prereqs)} direct prerequisites")
        
        # Could traverse deeper for full chain
        prereq_names = []
        for prereq_rel in prereqs:
            prereq_node = kg.nodes[prereq_rel["from"]]
            prereq_names.append(prereq_node["properties"]["name"])
        print(f"   Direct prerequisites: {', '.join(prereq_names)}")
    
    # 4. Domain-based content retrieval
    print("\n4. DOMAIN-BASED RETRIEVAL:")
    print("   Query: Find all intermediate-level mechanics concepts")
    
    mechanics_concepts = [
        node for node in kg.nodes.values() 
        if (node["type"] == "Concept" and 
            node["properties"].get("difficulty_level") == "intermediate" and 
            node["properties"].get("domain") == "mechanics")
    ]
    print(f"   Found {len(mechanics_concepts)} intermediate mechanics concepts")
    
    # 5. Formula application context
    print("\n5. FORMULA APPLICATION CONTEXT:")
    print("   Query: Find concepts and problems using 'F = ma'")
    
    newton_formulas = [node for node in kg.nodes.values() if node["properties"].get("expression") == "F = ma"]
    if newton_formulas:
        formula_id = newton_formulas[0]["id"]
        describes_rels = [rel for rel in kg.relationships if rel["from"] == formula_id and rel["type"] == "DESCRIBES"]
        print(f"   Formula describes {len(describes_rels)} concepts")

def main():
    """Main function to create and validate the comprehensive physics knowledge graph"""
    
    print("COMPREHENSIVE PHYSICS KNOWLEDGE GRAPH - PHASE 3.1")
    print("=" * 60)
    
    # Create the comprehensive knowledge graph
    kg = create_comprehensive_physics_knowledge_graph()
    
    # Validate the structure
    validation_results = validate_knowledge_graph_structure(kg)
    
    # Display results
    print(f"\n📊 KNOWLEDGE GRAPH STATISTICS:")
    print(f"   Total Nodes: {validation_results['total_nodes']}")
    print(f"   Total Relationships: {validation_results['total_relationships']}")
    
    print(f"\n📈 TARGET ACHIEVEMENT:")
    targets = validation_results['targets_met']
    print(f"   Nodes ≥200: {'✅ PASSED' if targets['nodes_200_plus'] else '❌ FAILED'} ({validation_results['total_nodes']})")
    print(f"   Relationships ≥500: {'✅ PASSED' if targets['relationships_500_plus'] else '❌ FAILED'} ({validation_results['total_relationships']})")
    
    print(f"\n🏗️ NODE DISTRIBUTION:")
    for node_type, count in validation_results['node_distribution'].items():
        print(f"   {node_type}: {count}")
    
    print(f"\n🔗 RELATIONSHIP DISTRIBUTION:")
    for rel_type, count in validation_results['relationship_distribution'].items():
        print(f"   {rel_type}: {count}")
    
    print(f"\n✅ REQUIRED COMPONENTS CHECK:")
    node_checks = validation_results['required_node_types_present']
    for node_type, present in node_checks.items():
        status = "✅" if present else "❌"
        print(f"   {node_type}: {status}")
    
    rel_checks = validation_results['required_relationship_types_present']
    missing_rels = [rel for rel, present in rel_checks.items() if not present]
    if missing_rels:
        print(f"\n⚠️ Missing relationship types: {', '.join(missing_rels)}")
    else:
        print(f"\n✅ All required relationship types present")
    
    # Demonstrate RAG patterns
    demonstrate_rag_query_patterns(kg)
    
    # Save structure to JSON for reference
    output_data = {
        "statistics": validation_results,
        "sample_nodes": {node_id: node for node_id, node in list(kg.nodes.items())[:10]},
        "sample_relationships": kg.relationships[:20]
    }
    
    with open("/home/atk21004admin/Physics-Assistant/database/knowledge_graph_structure.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n📄 Knowledge graph structure saved to: /home/atk21004admin/Physics-Assistant/database/knowledge_graph_structure.json")
    
    # Overall assessment
    total_nodes = validation_results['total_nodes']
    total_rels = validation_results['total_relationships']
    node_target = targets['nodes_200_plus']
    rel_target = targets['relationships_500_plus']
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if node_target and rel_target:
        print("✅ EXCELLENT: Knowledge graph exceeds all target requirements")
        print("   Ready for Graph RAG implementation with comprehensive physics content")
    elif node_target:
        print("⚠️ GOOD: Node target met, but need more relationships for optimal RAG performance")
        print("   Consider adding more concept interconnections and educational content links")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Targets not fully met")
        print(f"   Need {200 - total_nodes} more nodes and {500 - total_rels} more relationships")
    
    print(f"\n🚀 NEXT STEPS FOR RAG INTEGRATION:")
    print("   1. Implement semantic embeddings for concepts and content")
    print("   2. Add graph traversal algorithms for context retrieval")
    print("   3. Create RAG prompt templates using graph context")
    print("   4. Build learning personalization based on student progress")
    print("   5. Add real-time content recommendation algorithms")
    
    return node_target and rel_target

if __name__ == "__main__":
    success = main()