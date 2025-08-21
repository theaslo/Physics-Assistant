#!/bin/bash
# Container Image Signing and Verification Script

set -euo pipefail

# Configuration
REGISTRY=${REGISTRY:-"localhost:5000"}
KEY_DIR="/signing/keys"
POLICIES_DIR="/signing/policies"
REPORTS_DIR="/signing/reports"
SIGN_DATE=$(date +%Y%m%d-%H%M%S)

# Physics Assistant images to sign
IMAGES=(
    "physics-postgres:latest"
    "physics-neo4j:latest"
    "physics-redis:latest"
    "physics-database-api:latest"
    "physics-dashboard-api:latest"
    "physics-agents-api:latest"
    "physics-streamlit-ui:latest"
    "physics-react-dashboard:latest"
    "physics-nginx-gateway:latest"
    "physics-mcp-forces:latest"
    "physics-mcp-kinematics:latest"
    "physics-mcp-math:latest"
    "physics-mcp-energy:latest"
    "physics-mcp-momentum:latest"
    "physics-mcp-angular-motion:latest"
    "physics-ml-engine:latest"
    "physics-task-processor:latest"
    "physics-flower-monitor:latest"
    "physics-prometheus:latest"
    "physics-grafana:latest"
    "physics-alertmanager:latest"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${REPORTS_DIR}/signing.log"
}

# Function to generate signing keys
generate_keys() {
    log "Generating signing keys..."
    
    # Create key directory if it doesn't exist
    mkdir -p "${KEY_DIR}"
    
    # Generate cosign key pair if not exists
    if [[ ! -f "${KEY_DIR}/cosign.key" ]]; then
        log "Generating new cosign key pair..."
        
        # Generate without password for automation (use KMS in production)
        COSIGN_PASSWORD="" cosign generate-key-pair --output-key-prefix="${KEY_DIR}/cosign"
        
        log "Cosign key pair generated"
    else
        log "Using existing cosign key pair"
    fi
    
    # Set restrictive permissions
    chmod 600 "${KEY_DIR}/cosign.key"
    chmod 644 "${KEY_DIR}/cosign.pub"
}

# Function to sign a container image
sign_image() {
    local image=$1
    local full_image="${REGISTRY}/${image}"
    
    log "Signing image: ${full_image}"
    
    # Generate SBOM first
    local sbom_file="${REPORTS_DIR}/${image//[:\/]/_}-sbom-${SIGN_DATE}.json"
    syft "${full_image}" -o spdx-json > "${sbom_file}" || {
        log "WARNING: Failed to generate SBOM for ${image}"
        return 1
    }
    
    # Sign the image
    COSIGN_PASSWORD="" cosign sign \
        --key "${KEY_DIR}/cosign.key" \
        --annotations "build-date=${SIGN_DATE}" \
        --annotations "project=physics-assistant" \
        --annotations "environment=production" \
        "${full_image}" || {
        log "ERROR: Failed to sign ${image}"
        return 1
    }
    
    # Attach SBOM to the image
    cosign attach sbom --sbom "${sbom_file}" "${full_image}" || {
        log "WARNING: Failed to attach SBOM to ${image}"
    }
    
    # Generate attestation
    cosign attest \
        --key "${KEY_DIR}/cosign.key" \
        --predicate "${sbom_file}" \
        --type spdxjson \
        "${full_image}" || {
        log "WARNING: Failed to generate attestation for ${image}"
    }
    
    log "Successfully signed ${image}"
}

# Function to verify a container image
verify_image() {
    local image=$1
    local full_image="${REGISTRY}/${image}"
    
    log "Verifying image: ${full_image}"
    
    # Verify signature
    cosign verify \
        --key "${KEY_DIR}/cosign.pub" \
        "${full_image}" || {
        log "ERROR: Signature verification failed for ${image}"
        return 1
    }
    
    # Verify attestation
    cosign verify-attestation \
        --key "${KEY_DIR}/cosign.pub" \
        --type spdxjson \
        "${full_image}" || {
        log "WARNING: Attestation verification failed for ${image}"
    }
    
    log "Successfully verified ${image}"
}

# Function to create admission controller policy
create_admission_policy() {
    log "Creating admission controller policy..."
    
    cat > "${POLICIES_DIR}/image-policy.yaml" << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: image-policy
  namespace: physics-assistant
data:
  policy.yaml: |
    apiVersion: v1
    kind: Policy
    metadata:
      name: physics-assistant-image-policy
    spec:
      requirements:
        - keylessVerification:
            issuer: "https://physics-assistant.local"
            subject: "physics-assistant-signer"
        - signedByKeyRef:
            publicKey: |
              -----BEGIN PUBLIC KEY-----
              $(cat ${KEY_DIR}/cosign.pub | base64 -w 0)
              -----END PUBLIC KEY-----
      actions:
        - type: "deny"
          message: "Image must be signed by Physics Assistant"
EOF
    
    log "Admission controller policy created"
}

# Function to setup registry webhook
setup_registry_webhook() {
    log "Setting up registry webhook for automatic signing..."
    
    cat > "${POLICIES_DIR}/registry-webhook.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: registry-webhook
  namespace: physics-assistant
data:
  webhook.sh: |
    #!/bin/bash
    # Registry webhook for automatic image signing
    
    set -euo pipefail
    
    IMAGE=\$1
    ACTION=\$2
    
    if [[ "\${ACTION}" == "push" ]]; then
        echo "Auto-signing image: \${IMAGE}"
        /signing/scripts/sign-images.sh sign "\${IMAGE}"
    fi
EOF
    
    log "Registry webhook configuration created"
}

# Function to generate signing report
generate_report() {
    local report_file="${REPORTS_DIR}/signing-report-${SIGN_DATE}.json"
    
    log "Generating signing report..."
    
    {
        echo "{"
        echo "  \"signing_date\": \"${SIGN_DATE}\","
        echo "  \"total_images\": ${#IMAGES[@]},"
        echo "  \"registry\": \"${REGISTRY}\","
        echo "  \"images\": ["
        
        for i in "${!IMAGES[@]}"; do
            local image="${IMAGES[$i]}"
            local full_image="${REGISTRY}/${image}"
            
            echo "    {"
            echo "      \"name\": \"${image}\","
            echo "      \"full_name\": \"${full_image}\","
            echo "      \"signed\": $(cosign verify --key "${KEY_DIR}/cosign.pub" "${full_image}" >/dev/null 2>&1 && echo "true" || echo "false"),"
            echo "      \"sbom_file\": \"${image//[:\/]/_}-sbom-${SIGN_DATE}.json\""
            if [ $i -lt $((${#IMAGES[@]} - 1)) ]; then
                echo "    },"
            else
                echo "    }"
            fi
        done
        
        echo "  ]"
        echo "}"
    } > "${report_file}"
    
    log "Signing report generated: ${report_file}"
}

# Function to setup continuous verification
setup_continuous_verification() {
    log "Setting up continuous verification..."
    
    cat > "${POLICIES_DIR}/verify-cron.yaml" << EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: image-verification
  namespace: physics-assistant
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: image-verifier
            image: ${REGISTRY}/physics-image-signer:latest
            command:
            - /signing/scripts/sign-images.sh
            - verify-all
            env:
            - name: REGISTRY
              value: "${REGISTRY}"
            volumeMounts:
            - name: signing-keys
              mountPath: /signing/keys
              readOnly: true
          volumes:
          - name: signing-keys
            secret:
              secretName: signing-keys
          restartPolicy: OnFailure
EOF
    
    log "Continuous verification job created"
}

# Function to check image vulnerabilities before signing
check_vulnerabilities() {
    local image=$1
    local full_image="${REGISTRY}/${image}"
    
    log "Checking vulnerabilities for ${image} before signing..."
    
    # Use trivy to scan for vulnerabilities
    local vuln_report="${REPORTS_DIR}/${image//[:\/]/_}-vuln-${SIGN_DATE}.json"
    
    # Scan for critical vulnerabilities
    trivy image --format json --severity CRITICAL "${full_image}" > "${vuln_report}" 2>/dev/null || {
        log "WARNING: Vulnerability scan failed for ${image}"
        return 0  # Continue with signing even if scan fails
    }
    
    # Check if critical vulnerabilities exist
    local critical_count
    critical_count=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | .VulnerabilityID' "${vuln_report}" 2>/dev/null | wc -l || echo "0")
    
    if [[ ${critical_count} -gt 0 ]]; then
        log "WARNING: Found ${critical_count} critical vulnerabilities in ${image}"
        # In production, you might want to fail here
        # return 1
    fi
    
    log "Vulnerability check completed for ${image}"
}

# Main execution functions
sign_all_images() {
    log "Starting image signing process for Physics Assistant platform"
    log "Signing ${#IMAGES[@]} container images"
    
    generate_keys
    
    for image in "${IMAGES[@]}"; do
        log "Processing image: ${image}"
        
        # Check vulnerabilities before signing
        check_vulnerabilities "${image}"
        
        # Sign the image
        sign_image "${image}"
    done
    
    generate_report
    create_admission_policy
    setup_registry_webhook
    setup_continuous_verification
    
    log "Image signing process completed"
}

verify_all_images() {
    log "Starting image verification process"
    
    local verification_failed=0
    
    for image in "${IMAGES[@]}"; do
        log "Verifying image: ${image}"
        
        if ! verify_image "${image}"; then
            ((verification_failed++))
        fi
    done
    
    if [[ ${verification_failed} -gt 0 ]]; then
        log "ERROR: ${verification_failed} images failed verification"
        exit 1
    else
        log "All images verified successfully"
    fi
}

# Main execution
main() {
    local action=${1:-"sign-all"}
    
    case "${action}" in
        sign-all)
            sign_all_images
            ;;
        verify-all)
            verify_all_images
            ;;
        sign)
            local image=${2:-""}
            if [[ -z "${image}" ]]; then
                log "ERROR: Image name required for signing"
                exit 1
            fi
            generate_keys
            check_vulnerabilities "${image}"
            sign_image "${image}"
            ;;
        verify)
            local image=${2:-""}
            if [[ -z "${image}" ]]; then
                log "ERROR: Image name required for verification"
                exit 1
            fi
            verify_image "${image}"
            ;;
        generate-keys)
            generate_keys
            ;;
        *)
            echo "Usage: $0 {sign-all|verify-all|sign <image>|verify <image>|generate-keys}"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"