#!/bin/bash
# Advanced firewall configuration for Physics Assistant Platform

set -euo pipefail

# Configuration
DOCKER_NETWORK="physics-network"
DOCKER_SUBNET="172.20.0.0/16"
LOG_PREFIX="PHYSICS-FW"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Function to setup iptables rules
setup_iptables() {
    log "Setting up iptables firewall rules..."
    
    # Create custom chains
    iptables -N PHYSICS_INPUT 2>/dev/null || true
    iptables -N PHYSICS_FORWARD 2>/dev/null || true
    iptables -N PHYSICS_OUTPUT 2>/dev/null || true
    iptables -N PHYSICS_DOCKER 2>/dev/null || true
    iptables -N PHYSICS_LOG_DROP 2>/dev/null || true
    
    # Clear existing rules in custom chains
    iptables -F PHYSICS_INPUT
    iptables -F PHYSICS_FORWARD
    iptables -F PHYSICS_OUTPUT
    iptables -F PHYSICS_DOCKER
    iptables -F PHYSICS_LOG_DROP
    
    # Default policies
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT ACCEPT
    
    # Allow loopback
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A OUTPUT -o lo -j ACCEPT
    
    # Allow established connections
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    iptables -A FORWARD -m state --state ESTABLISHED,RELATED -j ACCEPT
    
    # Jump to custom chains
    iptables -A INPUT -j PHYSICS_INPUT
    iptables -A FORWARD -j PHYSICS_FORWARD
    iptables -A OUTPUT -j PHYSICS_OUTPUT
    
    log "Base iptables rules configured"
}

# Function to configure container network rules
setup_container_rules() {
    log "Setting up container network security rules..."
    
    # Allow SSH (be careful with this in production)
    iptables -A PHYSICS_INPUT -p tcp --dport 22 -m limit --limit 5/min -j ACCEPT
    
    # Allow HTTP/HTTPS from external
    iptables -A PHYSICS_INPUT -p tcp --dport 80 -j ACCEPT
    iptables -A PHYSICS_INPUT -p tcp --dport 443 -j ACCEPT
    
    # Allow monitoring ports (restrict by source if needed)
    iptables -A PHYSICS_INPUT -p tcp --dport 3000 -s 10.0.0.0/8 -j ACCEPT    # Grafana
    iptables -A PHYSICS_INPUT -p tcp --dport 9090 -s 10.0.0.0/8 -j ACCEPT    # Prometheus
    iptables -A PHYSICS_INPUT -p tcp --dport 9093 -s 10.0.0.0/8 -j ACCEPT    # Alertmanager
    iptables -A PHYSICS_INPUT -p tcp --dport 5555 -s 10.0.0.0/8 -j ACCEPT    # Flower
    
    # Docker network rules
    iptables -A PHYSICS_FORWARD -s ${DOCKER_SUBNET} -j PHYSICS_DOCKER
    iptables -A PHYSICS_FORWARD -d ${DOCKER_SUBNET} -j PHYSICS_DOCKER
    
    # Allow internal container communication within physics network
    iptables -A PHYSICS_DOCKER -s ${DOCKER_SUBNET} -d ${DOCKER_SUBNET} -j ACCEPT
    
    # Database access restrictions
    iptables -A PHYSICS_DOCKER -p tcp --dport 5432 -s ${DOCKER_SUBNET} -m comment --comment "PostgreSQL access" -j ACCEPT
    iptables -A PHYSICS_DOCKER -p tcp --dport 7687 -s ${DOCKER_SUBNET} -m comment --comment "Neo4j access" -j ACCEPT
    iptables -A PHYSICS_DOCKER -p tcp --dport 6379 -s ${DOCKER_SUBNET} -m comment --comment "Redis access" -j ACCEPT
    
    # API services access
    iptables -A PHYSICS_DOCKER -p tcp --dport 8000 -s ${DOCKER_SUBNET} -m comment --comment "Physics Agents API" -j ACCEPT
    iptables -A PHYSICS_DOCKER -p tcp --dport 8001 -s ${DOCKER_SUBNET} -m comment --comment "Database API" -j ACCEPT
    iptables -A PHYSICS_DOCKER -p tcp --dport 8002 -s ${DOCKER_SUBNET} -m comment --comment "Dashboard API" -j ACCEPT
    
    # MCP services access
    iptables -A PHYSICS_DOCKER -p tcp --dport 10100 -s ${DOCKER_SUBNET} -m comment --comment "MCP services" -j ACCEPT
    
    # Vault access
    iptables -A PHYSICS_DOCKER -p tcp --dport 8200 -s ${DOCKER_SUBNET} -m comment --comment "Vault API" -j ACCEPT
    
    log "Container network rules configured"
}

# Function to setup rate limiting
setup_rate_limiting() {
    log "Setting up rate limiting rules..."
    
    # Rate limit SSH connections
    iptables -I PHYSICS_INPUT -p tcp --dport 22 -m state --state NEW -m recent --set --name SSH
    iptables -I PHYSICS_INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j PHYSICS_LOG_DROP
    
    # Rate limit HTTP connections
    iptables -I PHYSICS_INPUT -p tcp --dport 80 -m limit --limit 100/sec --limit-burst 200 -j ACCEPT
    iptables -I PHYSICS_INPUT -p tcp --dport 443 -m limit --limit 100/sec --limit-burst 200 -j ACCEPT
    
    # Rate limit API calls
    iptables -A PHYSICS_DOCKER -p tcp --dport 8000 -m limit --limit 50/sec --limit-burst 100 -j ACCEPT
    iptables -A PHYSICS_DOCKER -p tcp --dport 8001 -m limit --limit 50/sec --limit-burst 100 -j ACCEPT
    iptables -A PHYSICS_DOCKER -p tcp --dport 8002 -m limit --limit 50/sec --limit-burst 100 -j ACCEPT
    
    log "Rate limiting configured"
}

# Function to setup logging
setup_logging() {
    log "Setting up firewall logging..."
    
    # Log and drop function
    iptables -A PHYSICS_LOG_DROP -m limit --limit 5/min -j LOG --log-prefix "${LOG_PREFIX}-DROP: " --log-level 4
    iptables -A PHYSICS_LOG_DROP -j DROP
    
    # Log suspicious activity
    iptables -A PHYSICS_INPUT -p tcp --tcp-flags ALL NONE -m limit --limit 1/min -j LOG --log-prefix "${LOG_PREFIX}-NULL: "
    iptables -A PHYSICS_INPUT -p tcp --tcp-flags ALL ALL -m limit --limit 1/min -j LOG --log-prefix "${LOG_PREFIX}-XMAS: "
    iptables -A PHYSICS_INPUT -p tcp --tcp-flags ALL SYN,RST,ACK,FIN,URG -m limit --limit 1/min -j LOG --log-prefix "${LOG_PREFIX}-SCAN: "
    
    # Drop after logging
    iptables -A PHYSICS_INPUT -p tcp --tcp-flags ALL NONE -j DROP
    iptables -A PHYSICS_INPUT -p tcp --tcp-flags ALL ALL -j DROP
    iptables -A PHYSICS_INPUT -p tcp --tcp-flags ALL SYN,RST,ACK,FIN,URG -j DROP
    
    # Log denied packets (at the end)
    iptables -A PHYSICS_INPUT -j PHYSICS_LOG_DROP
    iptables -A PHYSICS_FORWARD -j PHYSICS_LOG_DROP
    iptables -A PHYSICS_DOCKER -j PHYSICS_LOG_DROP
    
    log "Firewall logging configured"
}

# Function to setup fail2ban integration
setup_fail2ban() {
    log "Setting up fail2ban integration..."
    
    # Create fail2ban chain
    iptables -N fail2ban-ssh 2>/dev/null || true
    iptables -F fail2ban-ssh
    
    # Insert fail2ban chain before SSH rule
    iptables -I PHYSICS_INPUT -p tcp --dport 22 -j fail2ban-ssh
    
    # Return to main chain if not blocked
    iptables -A fail2ban-ssh -j RETURN
    
    log "fail2ban integration configured"
}

# Function to setup DDoS protection
setup_ddos_protection() {
    log "Setting up DDoS protection..."
    
    # SYN flood protection
    iptables -A PHYSICS_INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT
    iptables -A PHYSICS_INPUT -p tcp --syn -j PHYSICS_LOG_DROP
    
    # Ping flood protection
    iptables -A PHYSICS_INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s --limit-burst 2 -j ACCEPT
    iptables -A PHYSICS_INPUT -p icmp --icmp-type echo-request -j PHYSICS_LOG_DROP
    
    # Port scan protection
    iptables -A PHYSICS_INPUT -m recent --name portscan --rcheck --seconds 86400 -j PHYSICS_LOG_DROP
    iptables -A PHYSICS_INPUT -m recent --name portscan --remove
    iptables -A PHYSICS_INPUT -p tcp -m tcp --dport 139 -m recent --name portscan --set -j PHYSICS_LOG_DROP
    
    log "DDoS protection configured"
}

# Function to save iptables rules
save_rules() {
    log "Saving iptables rules..."
    
    # Save rules to file (location varies by distribution)
    if command -v iptables-save >/dev/null 2>&1; then
        iptables-save > /etc/iptables/rules.v4 2>/dev/null || \
        iptables-save > /etc/iptables.rules 2>/dev/null || \
        log "WARNING: Could not save iptables rules automatically"
    fi
    
    log "Firewall rules saved"
}

# Function to show current rules
show_rules() {
    log "Current iptables rules:"
    echo "=== INPUT chain ==="
    iptables -L INPUT -n -v
    echo ""
    echo "=== FORWARD chain ==="
    iptables -L FORWARD -n -v
    echo ""
    echo "=== PHYSICS_DOCKER chain ==="
    iptables -L PHYSICS_DOCKER -n -v 2>/dev/null || echo "Chain not found"
}

# Function to test connectivity
test_connectivity() {
    log "Testing basic connectivity..."
    
    # Test external connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        log "✓ External connectivity working"
    else
        log "✗ External connectivity failed"
    fi
    
    # Test Docker network connectivity
    if docker network ls | grep -q "${DOCKER_NETWORK}"; then
        log "✓ Docker network ${DOCKER_NETWORK} exists"
    else
        log "✗ Docker network ${DOCKER_NETWORK} not found"
    fi
}

# Main execution
main() {
    case "${1:-setup}" in
        setup)
            log "Setting up Physics Assistant firewall..."
            setup_iptables
            setup_container_rules
            setup_rate_limiting
            setup_logging
            setup_fail2ban
            setup_ddos_protection
            save_rules
            test_connectivity
            log "Firewall setup completed"
            ;;
        show)
            show_rules
            ;;
        test)
            test_connectivity
            ;;
        reset)
            log "Resetting firewall rules..."
            iptables -F
            iptables -X
            iptables -t nat -F
            iptables -t nat -X
            iptables -P INPUT ACCEPT
            iptables -P FORWARD ACCEPT
            iptables -P OUTPUT ACCEPT
            log "Firewall rules reset"
            ;;
        *)
            echo "Usage: $0 {setup|show|test|reset}"
            exit 1
            ;;
    esac
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log "ERROR: This script must be run as root"
    exit 1
fi

# Execute main function
main "$@"