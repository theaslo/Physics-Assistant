#!/bin/bash

# Physics Assistant System Health Check Script
# Comprehensive health monitoring and validation

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Health check configuration
HEALTH_CHECK_TIMEOUT=30
CRITICAL_SERVICES="postgres-primary redis-cluster neo4j-cluster nginx-loadbalancer"
API_SERVICES="database-api-1 database-api-2 dashboard-api physics-agents-api-1 physics-agents-api-2"
MCP_SERVICES="mcp-forces mcp-kinematics mcp-math mcp-energy mcp-momentum mcp-angular-motion"
FRONTEND_SERVICES="streamlit-ui-1 streamlit-ui-2 react-dashboard"
MONITORING_SERVICES="prometheus grafana alertmanager"

# Global health status
OVERALL_HEALTH=0
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Parse command line arguments
VERBOSE=false
JSON_OUTPUT=false
ENVIRONMENT="production"
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --verbose, -v      Enable verbose output"
            echo "  --json             Output in JSON format"
            echo "  --env ENVIRONMENT  Environment to check (production, staging)"
            echo "  --output, -o FILE  Write output to file"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Update environment-specific files
if [[ "$ENVIRONMENT" == "staging" ]]; then
    ENV_FILE="$PROJECT_ROOT/.env.staging"
elif [[ "$ENVIRONMENT" == "development" ]]; then
    ENV_FILE="$PROJECT_ROOT/.env.development"
fi

# Load environment variables
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
fi

# Health check results storage
declare -A HEALTH_RESULTS
declare -A SERVICE_DETAILS

# Record health check result
record_check() {
    local check_name="$1"
    local status="$2"
    local details="${3:-}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    HEALTH_RESULTS["$check_name"]="$status"
    SERVICE_DETAILS["$check_name"]="$details"
    
    case "$status" in
        "PASS")
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
            ;;
        "FAIL")
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            OVERALL_HEALTH=1
            ;;
        "WARN")
            WARNINGS=$((WARNINGS + 1))
            ;;
    esac
}

# Check Docker daemon
check_docker() {
    log_info "Checking Docker daemon..."
    
    if docker info >/dev/null 2>&1; then
        local docker_version=$(docker version --format '{{.Server.Version}}')
        record_check "docker_daemon" "PASS" "Docker version: $docker_version"
        [[ "$VERBOSE" == true ]] && log_success "Docker daemon is running (version: $docker_version)"
    else
        record_check "docker_daemon" "FAIL" "Docker daemon is not running"
        log_error "Docker daemon is not running"
    fi
}

# Check Docker Compose
check_docker_compose() {
    log_info "Checking Docker Compose..."
    
    if command -v docker-compose >/dev/null 2>&1; then
        local compose_version=$(docker-compose version --short)
        record_check "docker_compose" "PASS" "Docker Compose version: $compose_version"
        [[ "$VERBOSE" == true ]] && log_success "Docker Compose is available (version: $compose_version)"
    elif docker compose version >/dev/null 2>&1; then
        local compose_version=$(docker compose version --short)
        record_check "docker_compose" "PASS" "Docker Compose v2 version: $compose_version"
        [[ "$VERBOSE" == true ]] && log_success "Docker Compose v2 is available (version: $compose_version)"
    else
        record_check "docker_compose" "FAIL" "Docker Compose is not available"
        log_error "Docker Compose is not available"
    fi
}

# Check system resources
check_system_resources() {
    log_info "Checking system resources..."
    
    # Check available memory
    local mem_total=$(free -m | awk '/^Mem:/{print $2}')
    local mem_available=$(free -m | awk '/^Mem:/{print $7}')
    local mem_usage_percent=$(( (mem_total - mem_available) * 100 / mem_total ))
    
    if [[ $mem_usage_percent -lt 80 ]]; then
        record_check "memory_usage" "PASS" "Memory usage: ${mem_usage_percent}% (${mem_available}MB available)"
        [[ "$VERBOSE" == true ]] && log_success "Memory usage is acceptable: ${mem_usage_percent}%"
    elif [[ $mem_usage_percent -lt 90 ]]; then
        record_check "memory_usage" "WARN" "Memory usage: ${mem_usage_percent}% (${mem_available}MB available)"
        log_warning "High memory usage: ${mem_usage_percent}%"
    else
        record_check "memory_usage" "FAIL" "Memory usage: ${mem_usage_percent}% (${mem_available}MB available)"
        log_error "Critical memory usage: ${mem_usage_percent}%"
    fi
    
    # Check disk space
    local disk_usage=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
    local disk_available=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    
    if [[ $disk_usage -lt 80 ]]; then
        record_check "disk_usage" "PASS" "Disk usage: ${disk_usage}% (${disk_available} available)"
        [[ "$VERBOSE" == true ]] && log_success "Disk usage is acceptable: ${disk_usage}%"
    elif [[ $disk_usage -lt 90 ]]; then
        record_check "disk_usage" "WARN" "Disk usage: ${disk_usage}% (${disk_available} available)"
        log_warning "High disk usage: ${disk_usage}%"
    else
        record_check "disk_usage" "FAIL" "Disk usage: ${disk_usage}% (${disk_available} available)"
        log_error "Critical disk usage: ${disk_usage}%"
    fi
    
    # Check CPU load
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_cores=$(nproc)
    local load_percent=$(echo "$load_avg * 100 / $cpu_cores" | bc -l | cut -d. -f1)
    
    if [[ $load_percent -lt 70 ]]; then
        record_check "cpu_load" "PASS" "CPU load: ${load_avg} (${load_percent}% of ${cpu_cores} cores)"
        [[ "$VERBOSE" == true ]] && log_success "CPU load is acceptable: ${load_percent}%"
    elif [[ $load_percent -lt 90 ]]; then
        record_check "cpu_load" "WARN" "CPU load: ${load_avg} (${load_percent}% of ${cpu_cores} cores)"
        log_warning "High CPU load: ${load_percent}%"
    else
        record_check "cpu_load" "FAIL" "CPU load: ${load_avg} (${load_percent}% of ${cpu_cores} cores)"
        log_error "Critical CPU load: ${load_percent}%"
    fi
}

# Check service health
check_service_health() {
    local service="$1"
    local category="$2"
    
    if ! docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" >/dev/null 2>&1; then
        record_check "${category}_${service}" "FAIL" "Service not found"
        [[ "$VERBOSE" == true ]] && log_error "Service $service not found"
        return 1
    fi
    
    local status=$(docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" --format "table {{.State}}" | tail -n 1)
    local health=$(docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" --format "table {{.Health}}" | tail -n 1)
    
    if [[ "$status" == "Up" ]]; then
        if [[ "$health" == "healthy" || "$health" == "(healthy)" ]]; then
            record_check "${category}_${service}" "PASS" "Status: $status, Health: $health"
            [[ "$VERBOSE" == true ]] && log_success "Service $service is healthy"
        elif [[ "$health" == "unhealthy" || "$health" == "(unhealthy)" ]]; then
            record_check "${category}_${service}" "FAIL" "Status: $status, Health: $health"
            log_error "Service $service is unhealthy"
        else
            record_check "${category}_${service}" "WARN" "Status: $status, Health: $health"
            [[ "$VERBOSE" == true ]] && log_warning "Service $service health unknown"
        fi
    else
        record_check "${category}_${service}" "FAIL" "Status: $status"
        log_error "Service $service is not running"
    fi
}

# Check all services
check_all_services() {
    log_info "Checking service health..."
    
    # Check critical services
    for service in $CRITICAL_SERVICES; do
        check_service_health "$service" "critical"
    done
    
    # Check API services
    for service in $API_SERVICES; do
        check_service_health "$service" "api"
    done
    
    # Check MCP services
    for service in $MCP_SERVICES; do
        check_service_health "$service" "mcp"
    done
    
    # Check frontend services
    for service in $FRONTEND_SERVICES; do
        check_service_health "$service" "frontend"
    done
    
    # Check monitoring services
    for service in $MONITORING_SERVICES; do
        check_service_health "$service" "monitoring"
    done
}

# Check network connectivity
check_network_connectivity() {
    log_info "Checking network connectivity..."
    
    # Check if containers can communicate
    local network_name="physics-network"
    if docker network ls | grep -q "$network_name"; then
        record_check "docker_network" "PASS" "Network $network_name exists"
        [[ "$VERBOSE" == true ]] && log_success "Docker network $network_name exists"
    else
        record_check "docker_network" "FAIL" "Network $network_name not found"
        log_error "Docker network $network_name not found"
    fi
    
    # Check if load balancer is accessible
    if curl -sf "http://localhost/health" >/dev/null 2>&1; then
        record_check "loadbalancer_http" "PASS" "Load balancer HTTP accessible"
        [[ "$VERBOSE" == true ]] && log_success "Load balancer HTTP endpoint accessible"
    else
        record_check "loadbalancer_http" "FAIL" "Load balancer HTTP not accessible"
        log_error "Load balancer HTTP endpoint not accessible"
    fi
    
    # Check HTTPS if enabled
    if [[ "${SSL_ENABLED:-false}" == "true" ]]; then
        if curl -sf "https://localhost/health" >/dev/null 2>&1; then
            record_check "loadbalancer_https" "PASS" "Load balancer HTTPS accessible"
            [[ "$VERBOSE" == true ]] && log_success "Load balancer HTTPS endpoint accessible"
        else
            record_check "loadbalancer_https" "FAIL" "Load balancer HTTPS not accessible"
            log_error "Load balancer HTTPS endpoint not accessible"
        fi
    fi
}

# Check API endpoints
check_api_endpoints() {
    log_info "Checking API endpoints..."
    
    local base_url="http://localhost"
    if [[ "${SSL_ENABLED:-false}" == "true" ]]; then
        base_url="https://${DOMAIN_NAME:-localhost}"
    fi
    
    # Check database API
    if curl -sf "${base_url}/api/database/health" >/dev/null 2>&1; then
        record_check "api_database" "PASS" "Database API endpoint accessible"
        [[ "$VERBOSE" == true ]] && log_success "Database API endpoint accessible"
    else
        record_check "api_database" "FAIL" "Database API endpoint not accessible"
        log_error "Database API endpoint not accessible"
    fi
    
    # Check physics agents API
    if curl -sf "${base_url}/api/agents/health" >/dev/null 2>&1; then
        record_check "api_agents" "PASS" "Physics agents API endpoint accessible"
        [[ "$VERBOSE" == true ]] && log_success "Physics agents API endpoint accessible"
    else
        record_check "api_agents" "FAIL" "Physics agents API endpoint not accessible"
        log_error "Physics agents API endpoint not accessible"
    fi
    
    # Check dashboard API
    if curl -sf "${base_url}/api/dashboard/health" >/dev/null 2>&1; then
        record_check "api_dashboard" "PASS" "Dashboard API endpoint accessible"
        [[ "$VERBOSE" == true ]] && log_success "Dashboard API endpoint accessible"
    else
        record_check "api_dashboard" "FAIL" "Dashboard API endpoint not accessible"
        log_error "Dashboard API endpoint not accessible"
    fi
}

# Check monitoring endpoints
check_monitoring_endpoints() {
    log_info "Checking monitoring endpoints..."
    
    # Check Prometheus
    if curl -sf "http://localhost:9090/-/healthy" >/dev/null 2>&1; then
        record_check "prometheus_health" "PASS" "Prometheus health endpoint accessible"
        [[ "$VERBOSE" == true ]] && log_success "Prometheus health endpoint accessible"
    else
        record_check "prometheus_health" "FAIL" "Prometheus health endpoint not accessible"
        log_error "Prometheus health endpoint not accessible"
    fi
    
    # Check Grafana
    if curl -sf "http://localhost:3000/api/health" >/dev/null 2>&1; then
        record_check "grafana_health" "PASS" "Grafana health endpoint accessible"
        [[ "$VERBOSE" == true ]] && log_success "Grafana health endpoint accessible"
    else
        record_check "grafana_health" "FAIL" "Grafana health endpoint not accessible"
        log_error "Grafana health endpoint not accessible"
    fi
    
    # Check Alertmanager
    if curl -sf "http://localhost:9093/-/healthy" >/dev/null 2>&1; then
        record_check "alertmanager_health" "PASS" "Alertmanager health endpoint accessible"
        [[ "$VERBOSE" == true ]] && log_success "Alertmanager health endpoint accessible"
    else
        record_check "alertmanager_health" "FAIL" "Alertmanager health endpoint not accessible"
        log_error "Alertmanager health endpoint not accessible"
    fi
}

# Generate health report
generate_health_report() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    if [[ "$JSON_OUTPUT" == true ]]; then
        # Generate JSON report
        echo "{"
        echo "  \"timestamp\": \"$timestamp\","
        echo "  \"environment\": \"$ENVIRONMENT\","
        echo "  \"overall_health\": $([ $OVERALL_HEALTH -eq 0 ] && echo "\"HEALTHY\"" || echo "\"UNHEALTHY\""),"
        echo "  \"summary\": {"
        echo "    \"total_checks\": $TOTAL_CHECKS,"
        echo "    \"passed\": $PASSED_CHECKS,"
        echo "    \"failed\": $FAILED_CHECKS,"
        echo "    \"warnings\": $WARNINGS"
        echo "  },"
        echo "  \"checks\": {"
        
        local first=true
        for check in "${!HEALTH_RESULTS[@]}"; do
            if [[ "$first" == false ]]; then
                echo ","
            fi
            echo "    \"$check\": {"
            echo "      \"status\": \"${HEALTH_RESULTS[$check]}\","
            echo "      \"details\": \"${SERVICE_DETAILS[$check]}\""
            echo -n "    }"
            first=false
        done
        echo ""
        echo "  }"
        echo "}"
    else
        # Generate human-readable report
        echo ""
        echo "=============================================="
        echo "Physics Assistant Health Check Report"
        echo "=============================================="
        echo "Timestamp: $timestamp"
        echo "Environment: $ENVIRONMENT"
        echo "Overall Health: $([ $OVERALL_HEALTH -eq 0 ] && echo "HEALTHY" || echo "UNHEALTHY")"
        echo ""
        echo "Summary:"
        echo "  Total Checks: $TOTAL_CHECKS"
        echo "  Passed: $PASSED_CHECKS"
        echo "  Failed: $FAILED_CHECKS"
        echo "  Warnings: $WARNINGS"
        echo ""
        echo "Detailed Results:"
        echo "=================="
        
        for check in "${!HEALTH_RESULTS[@]}"; do
            local status="${HEALTH_RESULTS[$check]}"
            local details="${SERVICE_DETAILS[$check]}"
            
            case "$status" in
                "PASS")
                    echo -e "${GREEN}✓${NC} $check: $status"
                    ;;
                "FAIL")
                    echo -e "${RED}✗${NC} $check: $status"
                    ;;
                "WARN")
                    echo -e "${YELLOW}⚠${NC} $check: $status"
                    ;;
            esac
            
            if [[ -n "$details" && "$VERBOSE" == true ]]; then
                echo "    Details: $details"
            fi
        done
        
        if [[ $OVERALL_HEALTH -eq 0 ]]; then
            echo ""
            log_success "All critical systems are healthy!"
        else
            echo ""
            log_error "System health issues detected. Please review failed checks."
        fi
    fi
}

# Main health check function
main() {
    log_info "Starting Physics Assistant Health Check"
    log_info "Environment: $ENVIRONMENT"
    log_info "Timestamp: $(date)"
    
    # Redirect output if specified
    if [[ -n "$OUTPUT_FILE" ]]; then
        exec > >(tee "$OUTPUT_FILE")
    fi
    
    # Run all health checks
    check_docker
    check_docker_compose
    check_system_resources
    check_all_services
    check_network_connectivity
    check_api_endpoints
    check_monitoring_endpoints
    
    # Generate and display report
    generate_health_report
    
    # Exit with appropriate code
    exit $OVERALL_HEALTH
}

# Run main function
main "$@"