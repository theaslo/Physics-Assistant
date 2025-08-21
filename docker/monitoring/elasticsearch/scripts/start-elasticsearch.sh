#!/bin/bash
# Start Elasticsearch with APM configuration for Physics Assistant

set -euo pipefail

ES_HOME="/usr/share/elasticsearch"
ES_CONFIG="${ES_HOME}/config"
ES_DATA="${ES_HOME}/data"
ES_LOGS="${ES_HOME}/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${ES_LOGS}/startup.log"
}

# Function to setup security
setup_security() {
    log "Setting up Elasticsearch security..."
    
    # Set passwords for built-in users
    if [[ ! -f "${ES_DATA}/.security_configured" ]]; then
        log "Configuring built-in user passwords..."
        
        # Start Elasticsearch temporarily to set passwords
        ${ES_HOME}/bin/elasticsearch -d -p /tmp/es.pid
        
        # Wait for Elasticsearch to start
        local retry_count=0
        while ! curl -s http://localhost:9200 >/dev/null && [[ ${retry_count} -lt 30 ]]; do
            sleep 2
            ((retry_count++))
        done
        
        if [[ ${retry_count} -eq 30 ]]; then
            log "ERROR: Elasticsearch failed to start for security setup"
            return 1
        fi
        
        # Set passwords
        echo "y" | ${ES_HOME}/bin/elasticsearch-setup-passwords auto > "${ES_DATA}/passwords.txt" 2>&1 || {
            # If auto setup fails, set manual passwords
            ${ES_HOME}/bin/elasticsearch-setup-passwords interactive <<EOF
${ELASTIC_PASSWORD:-physics_elastic_2024}
${ELASTIC_PASSWORD:-physics_elastic_2024}
${KIBANA_PASSWORD:-physics_kibana_2024}
${KIBANA_PASSWORD:-physics_kibana_2024}
${LOGSTASH_PASSWORD:-physics_logstash_2024}
${LOGSTASH_PASSWORD:-physics_logstash_2024}
${BEATS_PASSWORD:-physics_beats_2024}
${BEATS_PASSWORD:-physics_beats_2024}
${APM_PASSWORD:-physics_apm_2024}
${APM_PASSWORD:-physics_apm_2024}
${REMOTE_MONITORING_PASSWORD:-physics_monitoring_2024}
${REMOTE_MONITORING_PASSWORD:-physics_monitoring_2024}
EOF
        }
        
        # Stop temporary instance
        kill $(cat /tmp/es.pid) || true
        sleep 5
        
        touch "${ES_DATA}/.security_configured"
        log "Security configured"
    else
        log "Security already configured"
    fi
}

# Function to setup index templates
setup_index_templates() {
    log "Setting up index templates..."
    
    # Wait for Elasticsearch to be ready
    local retry_count=0
    while ! curl -s -u "elastic:${ELASTIC_PASSWORD}" http://localhost:9200/_cluster/health >/dev/null && [[ ${retry_count} -lt 30 ]]; do
        sleep 5
        ((retry_count++))
    done
    
    # APM index template
    curl -X PUT "localhost:9200/_index_template/apm-template" \
        -u "elastic:${ELASTIC_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d @"${ES_CONFIG}/templates/apm-template.json" || log "Failed to create APM template"
    
    # Metrics index template
    curl -X PUT "localhost:9200/_index_template/metrics-template" \
        -u "elastic:${ELASTIC_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d @"${ES_CONFIG}/templates/metrics-template.json" || log "Failed to create metrics template"
    
    # Logs index template
    curl -X PUT "localhost:9200/_index_template/logs-template" \
        -u "elastic:${ELASTIC_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d @"${ES_CONFIG}/templates/logs-template.json" || log "Failed to create logs template"
    
    log "Index templates configured"
}

# Function to setup ILM policies
setup_ilm_policies() {
    log "Setting up Index Lifecycle Management policies..."
    
    # APM ILM policy
    curl -X PUT "localhost:9200/_ilm/policy/apm-policy" \
        -u "elastic:${ELASTIC_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d '{
            "policy": {
                "phases": {
                    "hot": {
                        "actions": {
                            "rollover": {
                                "max_size": "5GB",
                                "max_age": "1d"
                            },
                            "set_priority": {
                                "priority": 100
                            }
                        }
                    },
                    "warm": {
                        "min_age": "2d",
                        "actions": {
                            "set_priority": {
                                "priority": 50
                            },
                            "allocate": {
                                "number_of_replicas": 0
                            }
                        }
                    },
                    "cold": {
                        "min_age": "7d",
                        "actions": {
                            "set_priority": {
                                "priority": 0
                            }
                        }
                    },
                    "delete": {
                        "min_age": "30d"
                    }
                }
            }
        }' || log "Failed to create APM ILM policy"
    
    # Metrics ILM policy
    curl -X PUT "localhost:9200/_ilm/policy/metrics-policy" \
        -u "elastic:${ELASTIC_PASSWORD}" \
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
                        "min_age": "90d"
                    }
                }
            }
        }' || log "Failed to create metrics ILM policy"
    
    log "ILM policies configured"
}

# Function to create users and roles
create_users_and_roles() {
    log "Creating users and roles..."
    
    # Create Physics Assistant monitoring role
    curl -X POST "localhost:9200/_security/role/physics-assistant-monitoring" \
        -u "elastic:${ELASTIC_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d '{
            "cluster": ["monitor", "manage_ilm"],
            "indices": [
                {
                    "names": ["apm-*", "metrics-*", "logs-*", ".monitoring-*"],
                    "privileges": ["read", "write", "create_index", "manage"]
                }
            ]
        }' || log "Failed to create monitoring role"
    
    # Create Physics Assistant APM user
    curl -X POST "localhost:9200/_security/user/physics-apm-user" \
        -u "elastic:${ELASTIC_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d '{
            "password": "'${APM_PASSWORD:-physics_apm_2024}'",
            "roles": ["physics-assistant-monitoring"],
            "full_name": "Physics Assistant APM User"
        }' || log "Failed to create APM user"
    
    log "Users and roles created"
}

# Function to setup machine learning
setup_ml() {
    log "Setting up machine learning..."
    
    # Enable ML for anomaly detection
    curl -X PUT "localhost:9200/_cluster/settings" \
        -u "elastic:${ELASTIC_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d '{
            "persistent": {
                "xpack.ml.enabled": true,
                "xpack.ml.max_machine_memory_percent": 30
            }
        }' || log "Failed to configure ML"
    
    log "Machine learning configured"
}

# Function to optimize performance
optimize_performance() {
    log "Optimizing Elasticsearch performance..."
    
    # Set cluster settings for performance
    curl -X PUT "localhost:9200/_cluster/settings" \
        -u "elastic:${ELASTIC_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d '{
            "persistent": {
                "indices.memory.index_buffer_size": "20%",
                "indices.memory.min_index_buffer_size": "96mb",
                "thread_pool.write.queue_size": "1000",
                "thread_pool.search.queue_size": "1000",
                "cluster.routing.allocation.disk.watermark.low": "85%",
                "cluster.routing.allocation.disk.watermark.high": "90%",
                "cluster.routing.allocation.disk.watermark.flood_stage": "95%"
            }
        }' || log "Failed to set performance settings"
    
    log "Performance optimized"
}

# Main execution
main() {
    log "Starting Elasticsearch for Physics Assistant APM"
    
    # Ensure data directory exists
    mkdir -p "${ES_DATA}" "${ES_LOGS}"
    
    # Setup security if enabled
    if [[ "${xpack.security.enabled:-true}" == "true" ]]; then
        setup_security
    fi
    
    # Start Elasticsearch in background for configuration
    log "Starting Elasticsearch..."
    ${ES_HOME}/bin/elasticsearch &
    ES_PID=$!
    
    # Wait for Elasticsearch to be ready
    log "Waiting for Elasticsearch to be ready..."
    local retry_count=0
    while ! curl -s http://localhost:9200/_cluster/health >/dev/null && [[ ${retry_count} -lt 60 ]]; do
        if ! kill -0 ${ES_PID} 2>/dev/null; then
            log "ERROR: Elasticsearch process died"
            exit 1
        fi
        sleep 5
        ((retry_count++))
    done
    
    if [[ ${retry_count} -eq 60 ]]; then
        log "ERROR: Elasticsearch failed to start within timeout"
        exit 1
    fi
    
    log "Elasticsearch is ready, configuring..."
    
    # Configure Elasticsearch
    setup_index_templates
    setup_ilm_policies
    create_users_and_roles
    setup_ml
    optimize_performance
    
    log "Elasticsearch configuration completed"
    log "Elasticsearch is running with PID: ${ES_PID}"
    
    # Wait for Elasticsearch process
    wait ${ES_PID}
}

# Execute main function
main "$@"