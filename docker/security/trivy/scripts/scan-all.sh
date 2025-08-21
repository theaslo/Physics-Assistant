#!/bin/bash
# Comprehensive container security scanning script

set -euo pipefail

# Configuration
REPORT_DIR="/scanner/reports"
POLICIES_DIR="/scanner/policies"
SCAN_DATE=$(date +%Y%m%d-%H%M%S)
SEVERITY_THRESHOLD="HIGH,CRITICAL"

# Create report directory structure
mkdir -p "${REPORT_DIR}/"{vulnerability,config,secret,license,compliance}

# Physics Assistant container images to scan
IMAGES=(
    "physics-postgres"
    "physics-neo4j"
    "physics-redis"
    "physics-database-api"
    "physics-dashboard-api"
    "physics-agents-api"
    "physics-streamlit-ui"
    "physics-react-dashboard"
    "physics-nginx-gateway"
    "physics-mcp-forces"
    "physics-mcp-kinematics"
    "physics-mcp-math"
    "physics-mcp-energy"
    "physics-mcp-momentum"
    "physics-mcp-angular-motion"
    "physics-ml-engine"
    "physics-task-processor"
    "physics-flower-monitor"
    "physics-prometheus"
    "physics-grafana"
    "physics-alertmanager"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${REPORT_DIR}/scan.log"
}

# Function to scan image vulnerabilities
scan_vulnerabilities() {
    local image=$1
    local report_file="${REPORT_DIR}/vulnerability/${image}-vuln-${SCAN_DATE}.json"
    
    log "Scanning vulnerabilities for ${image}..."
    
    trivy image \
        --format json \
        --output "${report_file}" \
        --severity "${SEVERITY_THRESHOLD}" \
        --ignore-unfixed \
        --quiet \
        "${image}" || {
        log "WARNING: Vulnerability scan failed for ${image}"
        return 1
    }
    
    # Generate human-readable summary
    trivy image \
        --format table \
        --output "${REPORT_DIR}/vulnerability/${image}-vuln-${SCAN_DATE}.txt" \
        --severity "${SEVERITY_THRESHOLD}" \
        --ignore-unfixed \
        "${image}" 2>/dev/null || true
    
    log "Vulnerability scan completed for ${image}"
}

# Function to scan configuration issues
scan_configuration() {
    local image=$1
    local report_file="${REPORT_DIR}/config/${image}-config-${SCAN_DATE}.json"
    
    log "Scanning configuration for ${image}..."
    
    trivy config \
        --format json \
        --output "${report_file}" \
        --policy "${POLICIES_DIR}/docker-security.rego" \
        --quiet \
        "${image}" || {
        log "WARNING: Configuration scan failed for ${image}"
        return 1
    }
    
    log "Configuration scan completed for ${image}"
}

# Function to scan for secrets
scan_secrets() {
    local image=$1
    local report_file="${REPORT_DIR}/secret/${image}-secrets-${SCAN_DATE}.json"
    
    log "Scanning secrets for ${image}..."
    
    trivy fs \
        --format json \
        --output "${report_file}" \
        --scanners secret \
        --quiet \
        "${image}" || {
        log "WARNING: Secret scan failed for ${image}"
        return 1
    }
    
    log "Secret scan completed for ${image}"
}

# Function to check compliance
check_compliance() {
    local image=$1
    local report_file="${REPORT_DIR}/compliance/${image}-compliance-${SCAN_DATE}.json"
    
    log "Checking compliance for ${image}..."
    
    # Custom compliance checks based on security frameworks
    trivy image \
        --format json \
        --output "${report_file}" \
        --compliance docker-cis \
        --quiet \
        "${image}" || {
        log "WARNING: Compliance check failed for ${image}"
        return 1
    }
    
    log "Compliance check completed for ${image}"
}

# Function to generate security report summary
generate_summary() {
    local summary_file="${REPORT_DIR}/security-summary-${SCAN_DATE}.json"
    
    log "Generating security summary..."
    
    {
        echo "{"
        echo "  \"scan_date\": \"${SCAN_DATE}\","
        echo "  \"total_images\": ${#IMAGES[@]},"
        echo "  \"severity_threshold\": \"${SEVERITY_THRESHOLD}\","
        echo "  \"images\": ["
        
        for i in "${!IMAGES[@]}"; do
            local image="${IMAGES[$i]}"
            echo "    {"
            echo "      \"name\": \"${image}\","
            echo "      \"vulnerability_report\": \"vulnerability/${image}-vuln-${SCAN_DATE}.json\","
            echo "      \"config_report\": \"config/${image}-config-${SCAN_DATE}.json\","
            echo "      \"secret_report\": \"secret/${image}-secrets-${SCAN_DATE}.json\","
            echo "      \"compliance_report\": \"compliance/${image}-compliance-${SCAN_DATE}.json\""
            if [ $i -lt $((${#IMAGES[@]} - 1)) ]; then
                echo "    },"
            else
                echo "    }"
            fi
        done
        
        echo "  ]"
        echo "}"
    } > "${summary_file}"
    
    log "Security summary generated: ${summary_file}"
}

# Function to send alerts for critical findings
send_security_alerts() {
    local critical_count=0
    
    log "Checking for critical security findings..."
    
    for image in "${IMAGES[@]}"; do
        local vuln_file="${REPORT_DIR}/vulnerability/${image}-vuln-${SCAN_DATE}.json"
        
        if [[ -f "${vuln_file}" ]]; then
            local critical_vulns=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | .VulnerabilityID' "${vuln_file}" 2>/dev/null | wc -l || echo "0")
            critical_count=$((critical_count + critical_vulns))
        fi
    done
    
    if [[ ${critical_count} -gt 0 ]]; then
        log "ALERT: Found ${critical_count} critical vulnerabilities across all images"
        
        # Send webhook notification if configured
        if [[ -n "${SECURITY_WEBHOOK_URL:-}" ]]; then
            curl -X POST "${SECURITY_WEBHOOK_URL}" \
                -H "Content-Type: application/json" \
                -d "{
                    \"text\": \"Physics Assistant Security Alert: ${critical_count} critical vulnerabilities found\",
                    \"scan_date\": \"${SCAN_DATE}\",
                    \"critical_count\": ${critical_count}
                }" || log "Failed to send webhook notification"
        fi
    else
        log "No critical vulnerabilities found"
    fi
}

# Main execution
main() {
    log "Starting comprehensive security scan for Physics Assistant platform"
    log "Scanning ${#IMAGES[@]} container images with severity threshold: ${SEVERITY_THRESHOLD}"
    
    # Update Trivy database
    log "Updating Trivy vulnerability database..."
    trivy image --download-db-only --quiet
    
    # Scan all images
    for image in "${IMAGES[@]}"; do
        log "Processing image: ${image}"
        
        # Run all scan types
        scan_vulnerabilities "${image}" &
        scan_configuration "${image}" &
        scan_secrets "${image}" &
        check_compliance "${image}" &
        
        # Wait for parallel scans to complete
        wait
    done
    
    # Generate summary and alerts
    generate_summary
    send_security_alerts
    
    log "Security scan completed successfully"
    log "Reports available in: ${REPORT_DIR}"
}

# Execute main function
main "$@"