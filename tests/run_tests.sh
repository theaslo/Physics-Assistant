#!/bin/bash
#
# Physics Assistant Integration Tests
# ====================================
# Run from project root: ./tests/run_tests.sh
#

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

pass() { echo -e "  ${GREEN}✅${NC} $1"; PASSED=$((PASSED+1)); }
fail() { echo -e "  ${RED}❌${NC} $1"; FAILED=$((FAILED+1)); }

echo ""
echo "🔬 Physics Assistant Integration Tests"
echo "======================================="
echo ""

# Check Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running${NC}"
    exit 1
fi

# Check container is running
if ! docker ps --format '{{.Names}}' | grep -q 'physics-agents-api'; then
    echo -e "${RED}physics-agents-api container not running${NC}"
    exit 1
fi

echo "📡 Service Health"
echo "-----------------"

# Health checks via docker exec
STATUS=$(docker exec physics-agents-api curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
if [ "$STATUS" = "200" ]; then pass "Physics API"; else fail "Physics API (HTTP $STATUS)"; fi

STATUS=$(docker exec physics-agents-api curl -s -o /dev/null -w "%{http_code}" http://database-api:8001/health 2>/dev/null || echo "000")
if [ "$STATUS" = "200" ]; then pass "Database API"; else fail "Database API (HTTP $STATUS)"; fi

STATUS=$(docker exec physics-agents-api curl -s -o /dev/null -w "%{http_code}" http://database-api:8001/rag/system-status 2>/dev/null || echo "000")
if [ "$STATUS" = "200" ]; then pass "RAG System"; else fail "RAG System (HTTP $STATUS)"; fi

echo ""
echo "📋 API Endpoints"
echo "----------------"

AGENTS=$(docker exec physics-agents-api curl -s http://localhost:8000/agents/list 2>/dev/null | grep -o '"agent_id"' | wc -l | tr -d ' ')
if [ "$AGENTS" -ge 11 ]; then pass "Agent List ($AGENTS agents)"; else fail "Agent List (expected 11, got $AGENTS)"; fi

# Quick mode
if [ "$1" = "--quick" ]; then
    echo ""
    echo "======================================="
    echo "Passed: $PASSED | Failed: $FAILED"
    exit $FAILED
fi

echo ""
echo "🧪 Agent Tool Tests"
echo "-------------------"

# Test each agent with a problem
test_agent() {
    local AGENT=$1
    local PROBLEM=$2
    local DESC=$3

    RESPONSE=$(docker exec physics-agents-api curl -s -X POST \
        http://localhost:8000/agent/$AGENT/solve \
        -H "Content-Type: application/json" \
        -d "{\"problem\": \"$PROBLEM\", \"user_id\": \"test\"}" \
        --max-time 60 2>/dev/null)

    if echo "$RESPONSE" | grep -q '"success":true' && echo "$RESPONSE" | grep -q "MCP tools"; then
        pass "$DESC"
    else
        fail "$DESC"
    fi
}

# Physics 101 agents (LangChain-based)
test_agent "kinematics_agent" "Ball dropped from 20m. Time to hit ground?" "Kinematics: Free fall"
test_agent "forces_agent" "Spring k=200 N/m compressed 0.1m. Spring force?" "Forces: Spring force"
test_agent "momentum_agent" "5kg ball at 10 m/s. Calculate momentum." "Momentum: Basic"
test_agent "energy_agent" "2kg ball dropped 10m. KE at bottom?" "Energy: Conservation"
test_agent "angular_motion_agent" "Wheel r=0.5m at 10 rad/s. Tangential velocity?" "Angular: Kinematics"
test_agent "math_agent" "Solve x^2 - 5x + 6 = 0" "Math: Quadratic"

echo ""
echo "🧪 Physics 102-202 Agent Tests (Strands)"
echo "-----------------------------------------"

# Test function for Strands agents (they use different response format)
test_strands_agent() {
    local AGENT=$1
    local PROBLEM=$2
    local DESC=$3

    RESPONSE=$(docker exec physics-agents-api curl -s -X POST \
        http://localhost:8000/agent/$AGENT/solve \
        -H "Content-Type: application/json" \
        -d "{\"problem\": \"$PROBLEM\", \"user_id\": \"test\"}" \
        --max-time 90 2>/dev/null)

    if echo "$RESPONSE" | grep -q '"success":true'; then
        pass "$DESC"
    else
        fail "$DESC"
    fi
}

# Physics 102 agents (Strands-based)
test_strands_agent "thermodynamics_agent" "1 mole of ideal gas at 101325 Pa and 0.0224 m3. What is the temperature?" "Thermo: Ideal Gas"
test_strands_agent "waves_agent" "Sound wave with frequency 440 Hz and wavelength 0.78 m. What is the wave speed?" "Waves: Wave Equation"

# Physics 201 agents (Strands-based)
test_strands_agent "electromagnetism_agent" "Two charges of 1 uC and 2 uC separated by 0.1 m. Calculate the force." "E&M: Coulombs Law"

# Physics 202 agents (Strands-based)
test_strands_agent "optics_agent" "Light goes from air (n=1.0) to glass (n=1.5) at 30 degrees. Find refraction angle." "Optics: Snells Law"
test_strands_agent "modern_physics_agent" "A spaceship travels at 0.8c. If 1 second passes on the ship, how much time passes on Earth?" "Modern: Time Dilation"

echo ""
echo "🔄 Fallback Behavior"
echo "--------------------"

# Test non-physics question
RESPONSE=$(docker exec physics-agents-api curl -s -X POST \
    http://localhost:8000/agent/kinematics_agent/solve \
    -H "Content-Type: application/json" \
    -d '{"problem": "What is the capital of France?", "user_id": "test"}' \
    --max-time 60 2>/dev/null)

SOLUTION=$(echo "$RESPONSE" | grep -o '"solution":"[^"]*' | head -1)

if echo "$SOLUTION" | grep -qi "cannot\|no tool\|not physics\|unable"; then
    pass "Non-physics: Indicates no tool"
else
    fail "Non-physics: Should indicate no tool available"
    echo -e "       ${YELLOW}Current behavior: AI tries physics tools anyway${NC}"
fi

echo ""
echo "======================================="
echo "Passed: $PASSED | Failed: $FAILED"

[ $FAILED -eq 0 ] && echo -e "${GREEN}All tests passed!${NC}" || echo -e "${YELLOW}Some tests failed${NC}"
exit $FAILED
