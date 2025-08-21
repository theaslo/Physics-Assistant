#!/bin/bash
# Start Jaeger with advanced configuration for Physics Assistant

set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Function to wait for Elasticsearch
wait_for_elasticsearch() {
    log "Waiting for Elasticsearch to be ready..."
    
    local retry_count=0
    local max_retries=30
    
    while [[ ${retry_count} -lt ${max_retries} ]]; do
        if curl -s -f "${ES_SERVER_URLS}/_health" >/dev/null 2>&1; then
            log "Elasticsearch is ready"
            return 0
        fi
        
        log "Elasticsearch not ready, retrying in 5 seconds... (${retry_count}/${max_retries})"
        sleep 5
        ((retry_count++))
    done
    
    log "ERROR: Elasticsearch failed to become ready within timeout"
    return 1
}

# Function to create Elasticsearch indices
create_es_indices() {
    log "Creating Elasticsearch indices for Jaeger..."
    
    # Create span index template
    curl -X PUT "${ES_SERVER_URLS}/_index_template/jaeger-spans" \
        -H "Content-Type: application/json" \
        -d '{
            "index_patterns": ["jaeger-span-*"],
            "template": {
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 1,
                    "index.lifecycle.name": "jaeger-span-policy",
                    "index.lifecycle.rollover_alias": "jaeger-span"
                },
                "mappings": {
                    "properties": {
                        "traceID": { "type": "keyword" },
                        "spanID": { "type": "keyword" },
                        "parentSpanID": { "type": "keyword" },
                        "operationName": { "type": "keyword" },
                        "startTime": { "type": "date" },
                        "duration": { "type": "long" },
                        "tags": { "type": "nested" },
                        "process": { "type": "object" },
                        "logs": { "type": "nested" }
                    }
                }
            }
        }' || log "Failed to create span index template"
    
    # Create service index template
    curl -X PUT "${ES_SERVER_URLS}/_index_template/jaeger-services" \
        -H "Content-Type: application/json" \
        -d '{
            "index_patterns": ["jaeger-service-*"],
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1
                },
                "mappings": {
                    "properties": {
                        "serviceName": { "type": "keyword" },
                        "operationName": { "type": "keyword" },
                        "spanKind": { "type": "keyword" }
                    }
                }
            }
        }' || log "Failed to create service index template"
    
    log "Elasticsearch indices created"
}

# Function to setup ILM policies
setup_ilm_policies() {
    log "Setting up Index Lifecycle Management policies..."
    
    # Create ILM policy for spans
    curl -X PUT "${ES_SERVER_URLS}/_ilm/policy/jaeger-span-policy" \
        -H "Content-Type: application/json" \
        -d '{
            "policy": {
                "phases": {
                    "hot": {
                        "actions": {
                            "rollover": {
                                "max_size": "10GB",
                                "max_age": "7d"
                            }
                        }
                    },
                    "warm": {
                        "min_age": "7d",
                        "actions": {
                            "allocate": {
                                "number_of_replicas": 0
                            }
                        }
                    },
                    "delete": {
                        "min_age": "30d"
                    }
                }
            }
        }' || log "Failed to create ILM policy"
    
    log "ILM policies configured"
}

# Function to start Jaeger services
start_jaeger() {
    log "Starting Jaeger services..."
    
    # Set Jaeger configuration
    export SPAN_STORAGE_TYPE=elasticsearch
    export ES_SERVER_URLS=${ES_SERVER_URLS}
    export ES_USERNAME=${ES_USERNAME:-elastic}
    export ES_PASSWORD=${ES_PASSWORD}
    export ES_INDEX_PREFIX=${ES_INDEX_PREFIX:-jaeger}
    export ES_CREATE_INDEX_TEMPLATES=${ES_CREATE_INDEX_TEMPLATES:-true}
    export ES_VERSION=${ES_VERSION:-8}
    
    # Collector configuration
    export COLLECTOR_OTLP_ENABLED=true
    export COLLECTOR_ZIPKIN_HOST_PORT=:9411
    export COLLECTOR_GRPC_TLS_ENABLED=false
    export COLLECTOR_HTTP_TLS_ENABLED=false
    
    # Query service configuration
    export QUERY_BASE_PATH=/jaeger
    export QUERY_LOG_LEVEL=info
    export QUERY_ADDITIONAL_HEADERS="Access-Control-Allow-Origin: *"
    
    # Sampling configuration
    export SAMPLING_STRATEGIES_FILE=/opt/jaeger/config/sampling.json
    
    # Agent configuration
    export AGENT_LOG_LEVEL=info
    
    log "Jaeger configuration set"
    
    # Start Jaeger all-in-one
    exec /go/bin/all-in-one-linux \
        --collector.otlp.enabled=true \
        --collector.otlp.grpc.host-port=0.0.0.0:14250 \
        --collector.otlp.http.host-port=0.0.0.0:14268 \
        --query.ui-config=/opt/jaeger/config/ui.json \
        --prometheus.server-url=http://prometheus:9090 \
        --prometheus.query.support-spanmetrics-connector=true
}

# Main execution
main() {
    log "Starting Jaeger for Physics Assistant platform"
    
    # Wait for dependencies
    if [[ "${SPAN_STORAGE_TYPE}" == "elasticsearch" ]]; then
        wait_for_elasticsearch
        create_es_indices
        setup_ilm_policies
    fi
    
    # Start Jaeger
    start_jaeger
}

# Execute main function
main "$@"