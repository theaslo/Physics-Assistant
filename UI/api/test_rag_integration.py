#!/usr/bin/env python3
"""
Comprehensive Test Suite for RAG Integration with Physics Agents
Tests all aspects of the RAG-enhanced agent system
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any
import requests
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGIntegrationTester:
    """
    Comprehensive tester for RAG integration with Physics Agents
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000", rag_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url.rstrip('/')
        self.rag_base_url = rag_base_url.rstrip('/')
        self.test_results = {}
        
    def test_rag_api_availability(self) -> bool:
        """Test if RAG API is available"""
        try:
            response = requests.get(f"{self.rag_base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_physics_api_availability(self) -> bool:
        """Test if Physics API is available"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_rag_status_endpoint(self) -> Dict[str, Any]:
        """Test RAG status endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/rag/status")
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_agent_creation_with_rag(self) -> Dict[str, Any]:
        """Test creating agents with RAG enabled"""
        agent_types = ["forces_agent", "kinematics_agent", "energy_agent"]
        results = {}
        
        for agent_type in agent_types:
            try:
                payload = {
                    "agent_id": agent_type,
                    "use_direct_tools": True,
                    "enable_rag": True,
                    "rag_api_url": self.rag_base_url
                }
                
                response = requests.post(
                    f"{self.api_base_url}/agent/create",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results[agent_type] = {
                        "success": True,
                        "agent_id": data.get("agent_id"),
                        "capabilities": len(data.get("capabilities", {}).get("available_tools", []))
                    }
                else:
                    results[agent_type] = {
                        "success": False,
                        "error": f"Status code: {response.status_code}",
                        "response": response.text[:200]
                    }
                    
            except Exception as e:
                results[agent_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def test_problem_solving_with_rag(self) -> Dict[str, Any]:
        """Test problem solving with RAG enhancement"""
        test_problems = {
            "forces_agent": {
                "problem": "A 5 kg block sits on a table. Calculate the normal force acting on the block.",
                "expected_concepts": ["normal force", "weight", "equilibrium"]
            },
            "kinematics_agent": {
                "problem": "A car accelerates from 0 to 30 m/s in 10 seconds. What is its acceleration?",
                "expected_concepts": ["acceleration", "velocity", "kinematics equations"]
            },
            "energy_agent": {
                "problem": "A ball is dropped from 10 meters. Calculate its kinetic energy just before hitting the ground.",
                "expected_concepts": ["kinetic energy", "potential energy", "conservation of energy"]
            }
        }
        
        results = {}
        
        for agent_type, test_data in test_problems.items():
            try:
                payload = {
                    "problem": test_data["problem"],
                    "user_id": "test_user",
                    "context": {"test_mode": True}
                }
                
                # Test with RAG enabled
                response = requests.post(
                    f"{self.api_base_url}/agent/{agent_type}/solve?enable_rag=true",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results[agent_type] = {
                        "success": True,
                        "rag_enhanced": data.get("rag_enhanced", False),
                        "concepts_used": data.get("rag_context", {}).get("concepts_used", 0),
                        "formulas_used": data.get("rag_context", {}).get("formulas_used", 0),
                        "execution_time_ms": data.get("execution_time_ms", 0),
                        "solution_length": len(data.get("solution", "")),
                        "has_solution": bool(data.get("solution", "").strip())
                    }
                else:
                    results[agent_type] = {
                        "success": False,
                        "error": f"Status code: {response.status_code}",
                        "response": response.text[:200]
                    }
                    
            except Exception as e:
                results[agent_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def test_rag_vs_no_rag_comparison(self) -> Dict[str, Any]:
        """Compare agent performance with and without RAG"""
        test_problem = "A 2 kg object falls from a height of 5 meters. Calculate the velocity when it hits the ground."
        agent_type = "energy_agent"
        
        results = {}
        
        # Test without RAG
        try:
            payload = {
                "problem": test_problem,
                "user_id": "test_user_no_rag"
            }
            
            response = requests.post(
                f"{self.api_base_url}/agent/{agent_type}/solve?enable_rag=false",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                results["without_rag"] = {
                    "success": True,
                    "execution_time_ms": data.get("execution_time_ms", 0),
                    "solution_length": len(data.get("solution", "")),
                    "rag_enhanced": data.get("rag_enhanced", False)
                }
            else:
                results["without_rag"] = {"success": False, "error": f"Status: {response.status_code}"}
                
        except Exception as e:
            results["without_rag"] = {"success": False, "error": str(e)}
        
        # Test with RAG
        try:
            payload = {
                "problem": test_problem,
                "user_id": "test_user_with_rag"
            }
            
            response = requests.post(
                f"{self.api_base_url}/agent/{agent_type}/solve?enable_rag=true",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                results["with_rag"] = {
                    "success": True,
                    "execution_time_ms": data.get("execution_time_ms", 0),
                    "solution_length": len(data.get("solution", "")),
                    "rag_enhanced": data.get("rag_enhanced", False),
                    "concepts_used": data.get("rag_context", {}).get("concepts_used", 0),
                    "formulas_used": data.get("rag_context", {}).get("formulas_used", 0)
                }
            else:
                results["with_rag"] = {"success": False, "error": f"Status: {response.status_code}"}
                
        except Exception as e:
            results["with_rag"] = {"success": False, "error": str(e)}
        
        return results
    
    def test_rag_cache_functionality(self) -> Dict[str, Any]:
        """Test RAG caching functionality"""
        try:
            # Clear cache first
            response = requests.post(f"{self.api_base_url}/rag/clear-cache")
            cache_clear_success = response.status_code == 200
            
            # Make the same request twice to test caching
            problem = "Calculate the force needed to accelerate a 10 kg object at 2 m/s²"
            payload = {"problem": problem, "user_id": "cache_test"}
            
            # First request (should populate cache)
            start_time = time.time()
            response1 = requests.post(
                f"{self.api_base_url}/agent/forces_agent/solve?enable_rag=true",
                json=payload,
                timeout=60
            )
            first_time = time.time() - start_time
            
            # Second request (should use cache)
            start_time = time.time()
            response2 = requests.post(
                f"{self.api_base_url}/agent/forces_agent/solve?enable_rag=true",
                json=payload,
                timeout=60
            )
            second_time = time.time() - start_time
            
            return {
                "cache_clear_success": cache_clear_success,
                "first_request_success": response1.status_code == 200,
                "second_request_success": response2.status_code == 200,
                "first_request_time": first_time,
                "second_request_time": second_time,
                "potential_cache_speedup": first_time > second_time
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_student_progress_tracking(self) -> Dict[str, Any]:
        """Test student progress tracking in RAG system"""
        try:
            # Solve multiple problems with the same user
            problems = [
                "Calculate force for 5 kg object with 3 m/s² acceleration",
                "Find acceleration when 20 N force acts on 4 kg mass",
                "Determine mass when 15 N force causes 5 m/s² acceleration"
            ]
            
            results = {}
            user_id = "progress_test_user"
            
            for i, problem in enumerate(problems):
                payload = {
                    "problem": problem,
                    "user_id": user_id
                }
                
                response = requests.post(
                    f"{self.api_base_url}/agent/forces_agent/solve?enable_rag=true",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results[f"problem_{i+1}"] = {
                        "success": True,
                        "rag_enhanced": data.get("rag_enhanced", False),
                        "execution_time": data.get("execution_time_ms", 0)
                    }
                else:
                    results[f"problem_{i+1}"] = {
                        "success": False,
                        "error": f"Status: {response.status_code}"
                    }
            
            # Check if we can get user profile (would require RAG API endpoint)
            try:
                profile_response = requests.get(f"{self.rag_base_url}/rag/student-profile/{user_id}")
                results["profile_check"] = {
                    "success": profile_response.status_code == 200,
                    "has_profile": profile_response.status_code == 200
                }
            except Exception:
                results["profile_check"] = {"success": False, "error": "Profile endpoint not accessible"}
            
            return results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all RAG integration tests"""
        logger.info("🚀 Starting comprehensive RAG integration tests...")
        
        # Pre-flight checks
        logger.info("📡 Checking API availability...")
        rag_available = self.test_rag_api_availability()
        physics_available = self.test_physics_api_availability()
        
        self.test_results["pre_flight"] = {
            "rag_api_available": rag_available,
            "physics_api_available": physics_available
        }
        
        if not physics_available:
            logger.error("❌ Physics API not available - stopping tests")
            return self.test_results
        
        # RAG status check
        logger.info("📊 Testing RAG status endpoint...")
        self.test_results["rag_status"] = self.test_rag_status_endpoint()
        
        # Agent creation tests
        logger.info("🤖 Testing agent creation with RAG...")
        self.test_results["agent_creation"] = self.test_agent_creation_with_rag()
        
        # Problem solving tests
        logger.info("🧮 Testing problem solving with RAG enhancement...")
        self.test_results["problem_solving"] = self.test_problem_solving_with_rag()
        
        # RAG vs No-RAG comparison
        logger.info("⚖️ Testing RAG vs No-RAG comparison...")
        self.test_results["rag_comparison"] = self.test_rag_vs_no_rag_comparison()
        
        # Cache functionality tests
        logger.info("💾 Testing RAG cache functionality...")
        self.test_results["cache_tests"] = self.test_rag_cache_functionality()
        
        # Student progress tracking tests
        logger.info("📚 Testing student progress tracking...")
        self.test_results["progress_tracking"] = self.test_student_progress_tracking()
        
        logger.info("✅ All tests completed!")
        return self.test_results
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report"""
        if not self.test_results:
            return "No test results available. Run tests first."
        
        report = []
        report.append("=" * 60)
        report.append("RAG INTEGRATION TEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        total_tests = 0
        passed_tests = 0
        
        # Pre-flight results
        pre_flight = self.test_results.get("pre_flight", {})
        report.append("🛫 PRE-FLIGHT CHECKS:")
        report.append(f"  RAG API Available: {'✅' if pre_flight.get('rag_api_available') else '❌'}")
        report.append(f"  Physics API Available: {'✅' if pre_flight.get('physics_api_available') else '❌'}")
        report.append("")
        
        # RAG Status
        rag_status = self.test_results.get("rag_status", {})
        report.append("📊 RAG STATUS ENDPOINT:")
        report.append(f"  Status Check: {'✅' if rag_status.get('success') else '❌'}")
        if rag_status.get('success') and rag_status.get('data'):
            data = rag_status['data']
            report.append(f"  RAG Enabled Agents: {data.get('rag_enabled_agents', 0)}")
            report.append(f"  Total Agents: {data.get('total_agents', 0)}")
        report.append("")
        
        # Agent Creation
        agent_creation = self.test_results.get("agent_creation", {})
        report.append("🤖 AGENT CREATION WITH RAG:")
        for agent, result in agent_creation.items():
            status = "✅" if result.get("success") else "❌"
            report.append(f"  {agent}: {status}")
            if result.get("success"):
                report.append(f"    Tools: {result.get('capabilities', 0)}")
            total_tests += 1
            if result.get("success"):
                passed_tests += 1
        report.append("")
        
        # Problem Solving
        problem_solving = self.test_results.get("problem_solving", {})
        report.append("🧮 PROBLEM SOLVING WITH RAG:")
        for agent, result in problem_solving.items():
            status = "✅" if result.get("success") else "❌"
            report.append(f"  {agent}: {status}")
            if result.get("success"):
                report.append(f"    RAG Enhanced: {'✅' if result.get('rag_enhanced') else '❌'}")
                report.append(f"    Concepts Used: {result.get('concepts_used', 0)}")
                report.append(f"    Execution Time: {result.get('execution_time_ms', 0)}ms")
            total_tests += 1
            if result.get("success"):
                passed_tests += 1
        report.append("")
        
        # RAG Comparison
        comparison = self.test_results.get("rag_comparison", {})
        report.append("⚖️ RAG vs NO-RAG COMPARISON:")
        without_rag = comparison.get("without_rag", {})
        with_rag = comparison.get("with_rag", {})
        
        report.append(f"  Without RAG: {'✅' if without_rag.get('success') else '❌'}")
        report.append(f"  With RAG: {'✅' if with_rag.get('success') else '❌'}")
        
        if without_rag.get("success") and with_rag.get("success"):
            report.append(f"    Performance Comparison:")
            report.append(f"      No RAG Time: {without_rag.get('execution_time_ms', 0)}ms")
            report.append(f"      RAG Time: {with_rag.get('execution_time_ms', 0)}ms")
            report.append(f"      RAG Enhanced: {'✅' if with_rag.get('rag_enhanced') else '❌'}")
        report.append("")
        
        # Summary
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        report.append("📈 SUMMARY:")
        report.append(f"  Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        report.append(f"  Overall Status: {'✅ PASS' if success_rate >= 80 else '❌ NEEDS ATTENTION'}")
        
        return "\n".join(report)
    
    def save_detailed_results(self, filename: str = "rag_test_results.json"):
        """Save detailed test results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            logger.info(f"Detailed results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main test execution"""
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = "http://localhost:8000"
    
    if len(sys.argv) > 2:
        rag_url = sys.argv[2]
    else:
        rag_url = "http://localhost:8001"
    
    tester = RAGIntegrationTester(api_url, rag_url)
    
    # Run tests
    results = tester.run_comprehensive_tests()
    
    # Generate and display report
    report = tester.generate_test_report()
    print(report)
    
    # Save detailed results
    tester.save_detailed_results()
    
    # Return appropriate exit code
    pre_flight = results.get("pre_flight", {})
    if not pre_flight.get("physics_api_available"):
        sys.exit(1)
    
    # Calculate overall success rate
    total_successes = 0
    total_tests = 0
    
    for test_category, test_data in results.items():
        if test_category == "pre_flight":
            continue
            
        if isinstance(test_data, dict):
            for test_name, test_result in test_data.items():
                if isinstance(test_result, dict) and "success" in test_result:
                    total_tests += 1
                    if test_result["success"]:
                        total_successes += 1
    
    success_rate = (total_successes / total_tests * 100) if total_tests > 0 else 0
    
    if success_rate < 80:
        sys.exit(1)


if __name__ == "__main__":
    main()