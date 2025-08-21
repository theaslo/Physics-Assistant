#!/bin/bash
# Start OpenTelemetry Collector with advanced configuration

set -euo pipefail

OTEL_CONFIG_DIR="/etc/otelcol/config"
OTEL_CONFIG_FILE="${OTEL_CONFIG_DIR}/otel-collector.yaml"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Function to validate configuration
validate_config() {
    log "Validating OpenTelemetry Collector configuration..."
    
    if [[ ! -f "${OTEL_CONFIG_FILE}" ]]; then
        log "ERROR: Configuration file not found: ${OTEL_CONFIG_FILE}"
        exit 1
    fi
    
    # Basic YAML syntax validation
    if ! /otelcol --config "${OTEL_CONFIG_FILE}" --dry-run 2>&1; then
        log "ERROR: Configuration validation failed"
        exit 1
    fi
    
    log "Configuration validation passed"
}

# Function to wait for dependencies
wait_for_dependencies() {
    log "Waiting for dependencies to be ready..."
    
    local dependencies=(
        "jaeger:14250"
        "prometheus:9090"
        "elasticsearch:9200"
        "loki:3100"
    )
    
    for dep in "${dependencies[@]}"; do
        local host="${dep%%:*}"
        local port="${dep##*:}"
        
        log "Checking dependency: ${host}:${port}"
        
        local retry_count=0
        local max_retries=30
        
        while ! nc -z "${host}" "${port}" 2>/dev/null; do
            if [[ ${retry_count} -ge ${max_retries} ]]; then
                log "WARNING: Dependency ${host}:${port} not available, continuing anyway..."
                break
            fi
            
            sleep 5
            ((retry_count++))
        done
        
        if [[ ${retry_count} -lt ${max_retries} ]]; then
            log "✓ ${host}:${port} is ready"
        fi
    done
    
    log "Dependency check completed"
}

# Function to setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    # Set default values if not provided
    export ELASTICSEARCH_USERNAME="${ELASTICSEARCH_USERNAME:-elastic}"
    export ELASTICSEARCH_PASSWORD="${ELASTICSEARCH_PASSWORD:-physics_elastic_2024}"
    export PROMETHEUS_REMOTE_WRITE_URL="${PROMETHEUS_REMOTE_WRITE_URL:-http://prometheus:9090/api/v1/write}"
    export JAEGER_ENDPOINT="${JAEGER_ENDPOINT:-jaeger:14250}"
    export LOKI_ENDPOINT="${LOKI_ENDPOINT:-http://loki:3100/loki/api/v1/push}"
    
    # Resource limits
    export GOMEMLIMIT="${GOMEMLIMIT:-512MiB}"
    export GOMAXPROCS="${GOMAXPROCS:-2}"
    
    log "Environment variables configured"
}

# Function to setup telemetry
setup_telemetry() {
    log "Setting up telemetry configuration..."
    
    # Configure OpenTelemetry SDK for the collector itself
    export OTEL_SERVICE_NAME="otel-collector"
    export OTEL_SERVICE_VERSION="0.90.1"
    export OTEL_SERVICE_NAMESPACE="physics-assistant"
    export OTEL_RESOURCE_ATTRIBUTES="service.name=${OTEL_SERVICE_NAME},service.version=${OTEL_SERVICE_VERSION},service.namespace=${OTEL_SERVICE_NAMESPACE},deployment.environment=production"
    
    # Configure logging
    export OTEL_LOG_LEVEL="${OTEL_LOG_LEVEL:-info}"
    
    log "Telemetry configuration completed"
}

# Function to create dynamic configuration
create_dynamic_config() {
    log "Creating dynamic configuration..."
    
    # Substitute environment variables in configuration
    envsubst < "${OTEL_CONFIG_FILE}" > "${OTEL_CONFIG_FILE}.tmp"
    mv "${OTEL_CONFIG_FILE}.tmp" "${OTEL_CONFIG_FILE}"
    
    log "Dynamic configuration created"
}

# Function to monitor collector health
monitor_health() {
    log "Starting health monitoring..."
    
    while true; do
        sleep 30
        
        # Check health endpoint
        if curl -f http://localhost:13133/health >/dev/null 2>&1; then
            log "Health check passed"
        else
            log "WARNING: Health check failed"
        fi
        
        # Check memory usage
        if command -v ps >/dev/null 2>&1; then
            local memory_usage
            memory_usage=$(ps -o pid,ppid,%mem,cmd -p $$ | tail -1 | awk '{print $3}')
            
            if (( $(echo "${memory_usage} > 80" | bc -l) )); then
                log "WARNING: High memory usage: ${memory_usage}%"
            fi
        fi
    done
}

# Function to handle graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    log "Cleanup completed"
}

# Main execution
main() {
    log "Starting OpenTelemetry Collector for Physics Assistant"
    
    # Setup signal handlers
    trap cleanup SIGTERM SIGINT
    
    # Initialize
    setup_environment
    setup_telemetry
    validate_config
    create_dynamic_config
    wait_for_dependencies
    
    # Start health monitoring in background
    monitor_health &
    
    log "Starting OpenTelemetry Collector..."
    
    # Start the collector with the configuration
    exec /otelcol \
        --config="${OTEL_CONFIG_FILE}" \
        --log-level="${OTEL_LOG_LEVEL}" \
        --feature-gates=+component.UseLocalHostAsDefaultHost
}

# Execute main function
main "$@"