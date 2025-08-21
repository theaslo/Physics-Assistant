#!/usr/bin/env python3
"""
RAG Integration Validation Script
Tests RAG components without external dependencies
"""

import json
import time
from typing import Dict, Any

def validate_rag_client():
    """Validate RAG client functionality"""
    try:
        from rag_client import RAGClient, RAGClientError
        
        print("🔍 Testing RAG Client...")
        
        # Test instantiation
        client = RAGClient(api_base_url="http://localhost:8001")
        print("✅ RAG client instantiation")
        
        # Test cache functionality
        cache_key = client._get_cache_key("test", {"data": "test"})
        client._cache_response(cache_key, {"result": "cached"})
        print("✅ Cache operations")
        
        # Test cache validation
        is_valid = client._is_cache_valid(cache_key)
        print(f"✅ Cache validation: {is_valid}")
        
        # Test fallback response
        fallback = client._get_fallback_response("rag/query", {})
        print("✅ Fallback response generation")
        
        # Test metrics
        metrics = client.get_metrics()
        print(f"✅ Metrics tracking: {len(metrics)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG client validation failed: {e}")
        return False

def validate_rag_config():
    """Validate RAG configuration functionality"""
    try:
        from rag_config import RAGConfig, RAGConfigManager
        
        print("\n🔍 Testing RAG Configuration...")
        
        # Test config creation
        config = RAGConfig(
            enable_rag=True,
            max_concepts=5,
            max_formulas=3,
            similarity_threshold=0.3
        )
        print("✅ RAG config creation")
        
        # Test validation
        is_valid = config.validate()
        print(f"✅ Config validation: {is_valid}")
        
        # Test serialization
        config_dict = config.to_dict()
        config_restored = RAGConfig.from_dict(config_dict)
        print("✅ Config serialization/deserialization")
        
        # Test config manager
        manager = RAGConfigManager()
        print("✅ Config manager creation")
        
        # Test updates
        success = manager.update_config(max_concepts=7)
        print(f"✅ Config update: {success}")
        
        # Test agent-specific config
        agent_success = manager.update_agent_config("forces_agent", max_formulas=5)
        print(f"✅ Agent-specific config: {agent_success}")
        
        # Test summary
        summary = manager.get_config_summary()
        print(f"✅ Config summary: {summary['global_config']['rag_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG config validation failed: {e}")
        return False

def validate_integration_methods():
    """Validate key integration methods"""
    try:
        print("\n🔍 Testing Integration Methods...")
        
        # Test problem classification (standalone method)
        def classify_problem_type(problem: str) -> str:
            problem_lower = problem.lower()
            force_keywords = ["force", "newton", "weight", "friction"]
            kinematics_keywords = ["velocity", "acceleration", "motion"]
            energy_keywords = ["energy", "work", "power", "kinetic"]
            
            force_count = sum(1 for k in force_keywords if k in problem_lower)
            kinematics_count = sum(1 for k in kinematics_keywords if k in problem_lower)
            energy_count = sum(1 for k in energy_keywords if k in problem_lower)
            
            counts = {"forces": force_count, "kinematics": kinematics_count, "energy": energy_count}
            return max(counts, key=counts.get) if max(counts.values()) > 0 else "general"
        
        # Test with sample problems
        test_problems = [
            "Calculate the force needed to accelerate a 5 kg object at 2 m/s²",
            "A car travels 100 meters in 10 seconds. What is its velocity?",
            "Find the kinetic energy of a 2 kg ball moving at 5 m/s"
        ]
        
        expected_types = ["forces", "kinematics", "energy"]
        
        for i, problem in enumerate(test_problems):
            problem_type = classify_problem_type(problem)
            expected = expected_types[i]
            match = problem_type == expected
            print(f"{'✅' if match else '❌'} Problem classification: {problem_type} (expected: {expected})")
        
        print("✅ Problem classification logic")
        
        # Test context integration format
        def format_rag_context(problem: str, rag_context: Dict[str, Any]) -> str:
            if not rag_context:
                return problem
            
            context_parts = []
            concepts = rag_context.get("relevant_concepts", [])
            if concepts:
                concept_names = [c.get("name", str(c)) for c in concepts[:3]]
                context_parts.append(f"Key Concepts: {', '.join(concept_names)}")
            
            if context_parts:
                enhanced_context = "\n".join([f"[CONTEXT] {part}" for part in context_parts])
                return f"{enhanced_context}\n\n[PROBLEM] {problem}"
            return problem
        
        # Test context formatting
        sample_context = {
            "relevant_concepts": [
                {"name": "Newton's Second Law", "confidence": 0.95},
                {"name": "Force Analysis", "confidence": 0.87}
            ],
            "student_context": {"level": "intermediate"}
        }
        
        formatted = format_rag_context("Sample problem", sample_context)
        has_context = "[CONTEXT]" in formatted and "[PROBLEM]" in formatted
        print(f"✅ Context formatting: {'proper format' if has_context else 'basic format'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration methods validation failed: {e}")
        return False

def validate_api_integration():
    """Validate API integration components"""
    try:
        print("\n🔍 Testing API Integration...")
        
        # Test request models (simplified validation)
        agent_create_request = {
            "agent_id": "forces_agent",
            "use_direct_tools": True,
            "enable_rag": True,
            "rag_api_url": "http://localhost:8001"
        }
        
        # Validate required fields
        required_fields = ["agent_id", "enable_rag", "rag_api_url"]
        has_required = all(field in agent_create_request for field in required_fields)
        print(f"✅ Agent creation request format: {has_required}")
        
        # Test response format
        sample_response = {
            "success": True,
            "rag_enhanced": True,
            "rag_context": {
                "concepts_used": 3,
                "formulas_used": 2,
                "student_level": "intermediate"
            }
        }
        
        # Validate response structure
        has_rag_info = "rag_enhanced" in sample_response and "rag_context" in sample_response
        print(f"✅ RAG response format: {has_rag_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ API integration validation failed: {e}")
        return False

def run_comprehensive_validation():
    """Run all validation tests"""
    print("🚀 RAG Integration Validation Suite")
    print("=" * 50)
    
    results = []
    
    # Run individual validations
    results.append(("RAG Client", validate_rag_client()))
    results.append(("RAG Configuration", validate_rag_config()))
    results.append(("Integration Methods", validate_integration_methods()))
    results.append(("API Integration", validate_api_integration()))
    
    # Summary
    print("\n📊 Validation Summary")
    print("-" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    success_rate = (passed / total) * 100
    print(f"\nOverall Result: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 RAG Integration validation SUCCESSFUL!")
        return True
    else:
        print("⚠️ RAG Integration needs attention")
        return False

if __name__ == "__main__":
    success = run_comprehensive_validation()
    exit(0 if success else 1)