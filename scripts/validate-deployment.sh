#!/bin/bash

# Physics Assistant Deployment Validation Script
# Comprehensive post-deployment testing and validation

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

# Validation configuration
VALIDATION_TIMEOUT=300
RETRY_ATTEMPTS=3
RETRY_DELAY=10

# Global validation status
VALIDATION_RESULTS=()
FAILED_VALIDATIONS=0
TOTAL_VALIDATIONS=0

# Parse command line arguments
ENVIRONMENT="production"
VERBOSE=false
REPORT_FILE=""
QUICK_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --report|-r)
            REPORT_FILE="$2"
            shift 2
            ;;
        --quick)
            QUICK_CHECK=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env ENVIRONMENT  Environment to validate (production, staging)"
            echo "  --verbose, -v      Enable verbose output"
            echo "  --report, -r FILE  Write validation report to file"
            echo "  --quick            Run quick validation checks only"
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

# Record validation result
record_validation() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    local details="${4:-}"
    
    TOTAL_VALIDATIONS=$((TOTAL_VALIDATIONS + 1))
    
    if [[ "$status" == "PASS" ]]; then
        VALIDATION_RESULTS+=("✅ $test_name: $message")
        [[ "$VERBOSE" == true ]] && log_success "$test_name: $message"
    else
        VALIDATION_RESULTS+=("❌ $test_name: $message")
        FAILED_VALIDATIONS=$((FAILED_VALIDATIONS + 1))
        log_error "$test_name: $message"
    fi
    
    if [[ -n "$details" && "$VERBOSE" == true ]]; then
        echo "   Details: $details"
    fi
}

# Retry function for flaky tests
retry_test() {
    local test_command="$1"
    local test_name="$2"
    local attempt=1
    
    while [[ $attempt -le $RETRY_ATTEMPTS ]]; do
        if eval "$test_command" >/dev/null 2>&1; then
            return 0
        fi
        
        if [[ $attempt -lt $RETRY_ATTEMPTS ]]; then
            [[ "$VERBOSE" == true ]] && log_info "Retrying $test_name (attempt $((attempt + 1))/$RETRY_ATTEMPTS)"
            sleep $RETRY_DELAY
        fi
        
        attempt=$((attempt + 1))
    done
    
    return 1
}

# Validate service availability
validate_service_availability() {
    log_info "Validating service availability..."
    
    # Check if all critical services are running
    local critical_services="postgres-primary redis-cluster neo4j-cluster nginx-loadbalancer"
    
    for service in $critical_services; do
        if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" | grep -q "Up"; then
            record_validation "service_$service" "PASS" "Service is running"
        else
            record_validation "service_$service" "FAIL" "Service is not running"
        fi
    done
}

# Validate HTTP endpoints
validate_http_endpoints() {
    log_info "Validating HTTP endpoints..."
    
    local base_url="http://localhost"
    if [[ "${SSL_ENABLED:-false}" == "true" ]]; then
        base_url="https://${DOMAIN_NAME:-localhost}"
    fi
    
    # Main application health
    if retry_test "curl -sf '$base_url/health'" "main_app_health"; then
        record_validation "main_app_health" "PASS" "Main application health endpoint accessible"
    else
        record_validation "main_app_health" "FAIL" "Main application health endpoint not accessible"
    fi
    
    # Database API
    if retry_test "curl -sf '$base_url/api/database/health'" "database_api_health"; then
        record_validation "database_api_health" "PASS" "Database API health endpoint accessible"
    else
        record_validation "database_api_health" "FAIL" "Database API health endpoint not accessible"
    fi
    
    # Physics agents API
    if retry_test "curl -sf '$base_url/api/agents/health'" "agents_api_health"; then
        record_validation "agents_api_health" "PASS" "Physics agents API health endpoint accessible"
    else
        record_validation "agents_api_health" "FAIL" "Physics agents API health endpoint not accessible"
    fi
    
    # Dashboard API
    if retry_test "curl -sf '$base_url/api/dashboard/health'" "dashboard_api_health"; then
        record_validation "dashboard_api_health" "PASS" "Dashboard API health endpoint accessible"
    else
        record_validation "dashboard_api_health" "FAIL" "Dashboard API health endpoint not accessible"
    fi
}

# Validate database connectivity
validate_database_connectivity() {
    log_info "Validating database connectivity..."
    
    # PostgreSQL connectivity
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres-primary pg_isready -U postgres >/dev/null 2>&1; then
        record_validation "postgres_connectivity" "PASS" "PostgreSQL is accepting connections"
    else
        record_validation "postgres_connectivity" "FAIL" "PostgreSQL is not accepting connections"
    fi
    
    # Redis connectivity
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T redis-cluster redis-cli ping | grep -q "PONG"; then
        record_validation "redis_connectivity" "PASS" "Redis is responding to ping"
    else
        record_validation "redis_connectivity" "FAIL" "Redis is not responding to ping"
    fi
    
    # Neo4j connectivity
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T neo4j-cluster cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
        record_validation "neo4j_connectivity" "PASS" "Neo4j is accepting connections"
    else
        record_validation "neo4j_connectivity" "FAIL" "Neo4j is not accepting connections"
    fi
}

# Validate MCP services
validate_mcp_services() {
    log_info "Validating MCP services..."
    
    local mcp_services="mcp-forces mcp-kinematics mcp-math mcp-energy mcp-momentum mcp-angular-motion"
    
    for service in $mcp_services; do
        local container_name="physics-$service"
        if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
            # Check if MCP service is responding (assuming they have health endpoints)
            if docker exec "$container_name" curl -sf "http://localhost:10100/health" >/dev/null 2>&1; then
                record_validation "mcp_$service" "PASS" "MCP service is responding"
            else
                record_validation "mcp_$service" "FAIL" "MCP service is not responding"
            fi
        else
            record_validation "mcp_$service" "FAIL" "MCP service container not found"
        fi
    done
}

# Validate monitoring stack
validate_monitoring_stack() {
    log_info "Validating monitoring stack..."
    
    # Prometheus
    if retry_test "curl -sf 'http://localhost:9090/-/healthy'" "prometheus_health"; then
        record_validation "prometheus" "PASS" "Prometheus is healthy"
        
        # Check if Prometheus is scraping targets
        local targets_up=$(curl -s "http://localhost:9090/api/v1/query?query=up" | jq -r '.data.result | length')
        if [[ "$targets_up" -gt 0 ]]; then
            record_validation "prometheus_targets" "PASS" "Prometheus has $targets_up active targets"
        else
            record_validation "prometheus_targets" "FAIL" "Prometheus has no active targets"
        fi
    else
        record_validation "prometheus" "FAIL" "Prometheus is not healthy"
    fi
    
    # Grafana
    if retry_test "curl -sf 'http://localhost:3000/api/health'" "grafana_health"; then
        record_validation "grafana" "PASS" "Grafana is healthy"
    else
        record_validation "grafana" "FAIL" "Grafana is not healthy"
    fi
    
    # Alertmanager
    if retry_test "curl -sf 'http://localhost:9093/-/healthy'" "alertmanager_health"; then
        record_validation "alertmanager" "PASS" "Alertmanager is healthy"
    else
        record_validation "alertmanager" "FAIL" "Alertmanager is not healthy"
    fi
    
    # Loki (if not quick check)
    if [[ "$QUICK_CHECK" != true ]]; then
        if retry_test "curl -sf 'http://localhost:3100/ready'" "loki_ready"; then
            record_validation "loki" "PASS" "Loki is ready"
        else
            record_validation "loki" "FAIL" "Loki is not ready"
        fi
    fi
}

# Validate load balancing
validate_load_balancing() {
    log_info "Validating load balancing..."
    
    # Check if multiple backend instances are available
    local streamlit_backends=0
    local api_backends=0
    
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps streamlit-ui-1 | grep -q "Up"; then
        streamlit_backends=$((streamlit_backends + 1))
    fi
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps streamlit-ui-2 | grep -q "Up"; then
        streamlit_backends=$((streamlit_backends + 1))
    fi
    
    if [[ $streamlit_backends -gt 1 ]]; then
        record_validation "streamlit_load_balancing" "PASS" "$streamlit_backends Streamlit instances available"
    else
        record_validation "streamlit_load_balancing" "FAIL" "Only $streamlit_backends Streamlit instance available"
    fi
    
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps physics-agents-api-1 | grep -q "Up"; then
        api_backends=$((api_backends + 1))
    fi
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps physics-agents-api-2 | grep -q "Up"; then
        api_backends=$((api_backends + 1))
    fi
    
    if [[ $api_backends -gt 1 ]]; then
        record_validation "api_load_balancing" "PASS" "$api_backends API instances available"
    else
        record_validation "api_load_balancing" "FAIL" "Only $api_backends API instance available"
    fi
}

# Validate SSL/TLS configuration
validate_ssl_configuration() {
    if [[ "${SSL_ENABLED:-false}" != "true" ]]; then
        log_info "SSL/TLS not enabled, skipping SSL validation"
        return 0
    fi
    
    log_info "Validating SSL/TLS configuration..."
    
    local domain="${DOMAIN_NAME:-localhost}"
    
    # Check SSL certificate
    if echo | timeout 10 openssl s_client -connect "$domain:443" -servername "$domain" 2>/dev/null | openssl x509 -noout -text | grep -q "Subject:"; then
        record_validation "ssl_certificate" "PASS" "SSL certificate is valid"
        
        # Check certificate expiry
        local expiry_date=$(echo | timeout 10 openssl s_client -connect "$domain:443" -servername "$domain" 2>/dev/null | openssl x509 -noout -enddate | cut -d= -f2)
        local expiry_timestamp=$(date -d "$expiry_date" +%s)
        local current_timestamp=$(date +%s)
        local days_until_expiry=$(( (expiry_timestamp - current_timestamp) / 86400 ))
        
        if [[ $days_until_expiry -gt 30 ]]; then
            record_validation "ssl_expiry" "PASS" "SSL certificate expires in $days_until_expiry days"
        else
            record_validation "ssl_expiry" "FAIL" "SSL certificate expires in $days_until_expiry days (renewal needed)"
        fi
    else
        record_validation "ssl_certificate" "FAIL" "SSL certificate is not valid or accessible"
    fi
    
    # Check HTTPS redirect
    if curl -sf -I "http://$domain" | grep -q "Location: https://"; then
        record_validation "https_redirect" "PASS" "HTTP to HTTPS redirect is working"
    else
        record_validation "https_redirect" "FAIL" "HTTP to HTTPS redirect is not working"
    fi
}

# Validate backup system
validate_backup_system() {
    if [[ "$QUICK_CHECK" == true ]]; then
        log_info "Skipping backup validation in quick check mode"
        return 0
    fi
    
    log_info "Validating backup system..."
    
    # Check if backup service is running
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps backup-service | grep -q "Up"; then
        record_validation "backup_service" "PASS" "Backup service is running"
        
        # Check backup directories
        local backup_path="${BACKUP_PATH:-./backups}"
        if [[ -d "$backup_path" && -w "$backup_path" ]]; then
            record_validation "backup_directory" "PASS" "Backup directory is writable"
        else
            record_validation "backup_directory" "FAIL" "Backup directory is not accessible or writable"
        fi
    else
        record_validation "backup_service" "FAIL" "Backup service is not running"
    fi
}

# Validate performance metrics
validate_performance_metrics() {
    if [[ "$QUICK_CHECK" == true ]]; then
        log_info "Skipping performance validation in quick check mode"
        return 0
    fi
    
    log_info "Validating performance metrics..."
    
    # Check response times
    local base_url="http://localhost"
    if [[ "${SSL_ENABLED:-false}" == "true" ]]; then
        base_url="https://${DOMAIN_NAME:-localhost}"
    fi
    
    local response_time=$(curl -o /dev/null -s -w '%{time_total}' "$base_url/health")
    local response_time_ms=$(echo "$response_time * 1000" | bc | cut -d. -f1)
    
    if [[ $response_time_ms -lt 1000 ]]; then
        record_validation "response_time" "PASS" "Response time: ${response_time_ms}ms"
    elif [[ $response_time_ms -lt 5000 ]]; then
        record_validation "response_time" "PASS" "Response time: ${response_time_ms}ms (acceptable)"
    else
        record_validation "response_time" "FAIL" "Response time: ${response_time_ms}ms (too slow)"
    fi
    
    # Check memory usage
    local total_memory=$(docker stats --no-stream --format "table {{.MemUsage}}" | grep -v "MEM USAGE" | awk -F'/' '{sum += $1} END {print sum}' | sed 's/[^0-9.]//g')
    if [[ -n "$total_memory" && $(echo "$total_memory < 8000" | bc) -eq 1 ]]; then
        record_validation "memory_usage" "PASS" "Total memory usage: ${total_memory}MB"
    else
        record_validation "memory_usage" "FAIL" "High memory usage: ${total_memory}MB"
    fi
}

# Generate validation report
generate_validation_report() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local success_rate=$(( (TOTAL_VALIDATIONS - FAILED_VALIDATIONS) * 100 / TOTAL_VALIDATIONS ))
    
    local report_content=""
    report_content+="Physics Assistant Deployment Validation Report\n"
    report_content+="=============================================\n"
    report_content+="Timestamp: $timestamp\n"
    report_content+="Environment: $ENVIRONMENT\n"
    report_content+="Total Validations: $TOTAL_VALIDATIONS\n"
    report_content+="Failed Validations: $FAILED_VALIDATIONS\n"
    report_content+="Success Rate: $success_rate%\n"
    report_content+="\n"
    report_content+="Validation Results:\n"
    report_content+="==================\n"
    
    for result in "${VALIDATION_RESULTS[@]}"; do
        report_content+="$result\n"
    done
    
    if [[ -n "$REPORT_FILE" ]]; then
        echo -e "$report_content" > "$REPORT_FILE"
        log_info "Validation report written to: $REPORT_FILE"
    else
        echo -e "$report_content"
    fi
    
    # Summary
    if [[ $FAILED_VALIDATIONS -eq 0 ]]; then
        log_success "All validations passed! Deployment is healthy."
    else
        log_error "$FAILED_VALIDATIONS validation(s) failed. Please review the issues."
    fi
}

# Main validation function
main() {
    log_info "Starting Physics Assistant Deployment Validation"
    log_info "Environment: $ENVIRONMENT"
    log_info "Quick Check: $([ "$QUICK_CHECK" == true ] && echo "Enabled" || echo "Disabled")"
    log_info "Timestamp: $(date)"
    
    # Run validation checks
    validate_service_availability
    validate_http_endpoints
    validate_database_connectivity
    validate_mcp_services
    validate_monitoring_stack
    validate_load_balancing
    validate_ssl_configuration
    validate_backup_system
    validate_performance_metrics
    
    # Generate report
    generate_validation_report
    
    # Exit with appropriate code
    exit $FAILED_VALIDATIONS
}

# Run main function
main "$@"