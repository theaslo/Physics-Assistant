#!/bin/bash
# Simple Health Monitor for Physics Assistant
# No external dependencies required

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HEALTH_LOG="logs/health_monitor.log"
ALERT_LOG="logs/alerts.log"
CHECK_INTERVAL=60
MAX_RESPONSE_TIME=2

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to log with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$HEALTH_LOG"
}

# Function to log alerts
log_alert() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ALERT: $1" | tee -a "$ALERT_LOG"
    echo -e "${RED}🚨 ALERT: $1${NC}"
}

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check MCP server health
check_mcp_server() {
    local name=$1
    local port=$2
    local start_time=$(date +%s.%N)

    if curl -s -m $MAX_RESPONSE_TIME http://localhost:$port/ >/dev/null 2>&1; then
        local end_time=$(date +%s.%N)
        local response_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
        local response_ms=$(echo "$response_time * 1000" | bc 2>/dev/null || echo "0")

        echo "✅ $name (port $port): Online (${response_ms%.*}ms)"
        return 0
    else
        echo "❌ $name (port $port): Offline"
        log_alert "$name server (port $port) is offline"
        return 1
    fi
}

# Function to check container resources
check_container_resources() {
    echo ""
    echo "📊 Container Resource Usage:"

    # Try docker first, then podman
    if command -v docker >/dev/null 2>&1; then
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null | grep physics-assistant || echo "No Docker containers found"
    elif command -v podman >/dev/null 2>&1; then
        podman stats --no-stream --format "table {{.Name}}\t{{.CPU}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null | grep physics-assistant || echo "No Podman containers found"
    else
        echo "Docker/Podman not available for resource monitoring"
    fi
}

# Function to check disk space
check_disk_space() {
    echo ""
    echo "💾 Disk Space Usage:"

    df -h . | tail -1 | while read filesystem size used avail percent mount; do
        usage_num=$(echo $percent | sed 's/%//')
        echo "  Total: $size, Used: $used ($percent), Available: $avail"

        if [ "$usage_num" -gt 85 ]; then
            log_alert "High disk usage: $percent used, only $avail available"
        fi
    done
}

# Function to check system load
check_system_load() {
    echo ""
    echo "⚡ System Load:"

    if command -v uptime >/dev/null 2>&1; then
        uptime | sed 's/.*load average: /  Load Average: /'
    fi

    if command -v free >/dev/null 2>&1; then
        echo "  Memory Usage:"
        free -h | grep -E "Mem|Swap" | sed 's/^/    /'
    fi
}

# Function to run single health check
run_health_check() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo ""
    echo "🏥 Physics Assistant Health Check - $timestamp"
    echo "=" * 60

    # Check MCP servers
    echo "🔍 MCP Server Status:"
    local healthy_count=0
    local total_count=6

    # Array of server names and ports
    servers=(
        "Forces:10100"
        "Kinematics:10101"
        "Math:10103"
        "Momentum:10104"
        "Energy:10105"
        "Angular-Motion:10106"
    )

    for server_info in "${servers[@]}"; do
        IFS=':' read -r name port <<< "$server_info"
        if check_mcp_server "$name" "$port"; then
            ((healthy_count++))
        fi
    done

    echo ""
    echo "📈 Summary: $healthy_count/$total_count servers healthy"

    if [ $healthy_count -lt $total_count ]; then
        log_alert "Only $healthy_count/$total_count MCP servers are healthy"
    fi

    # Check resources
    check_container_resources
    check_disk_space
    check_system_load

    # Log health check completion
    log_message "Health check completed - $healthy_count/$total_count servers healthy"

    return $(($total_count - $healthy_count))
}

# Function to run continuous monitoring
run_continuous_monitoring() {
    echo "🔄 Starting continuous health monitoring (interval: ${CHECK_INTERVAL}s)"
    echo "Press Ctrl+C to stop monitoring"
    echo ""

    local check_count=0

    while true; do
        ((check_count++))

        echo "🔄 Health Check #$check_count"
        run_health_check

        if [ $? -eq 0 ]; then
            print_status "All systems healthy ✅"
        else
            print_warning "Some systems need attention ⚠️"
        fi

        echo ""
        echo "Next check in ${CHECK_INTERVAL} seconds..."
        echo "-" * 60

        sleep $CHECK_INTERVAL
    done
}

# Function to show recent alerts
show_recent_alerts() {
    echo "🚨 Recent Alerts (last 24 hours):"
    echo "=" * 40

    if [ -f "$ALERT_LOG" ]; then
        # Show alerts from last 24 hours
        local yesterday=$(date -d '24 hours ago' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -v-24H '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "1970-01-01 00:00:00")
        awk -v cutoff="$yesterday" '$0 >= cutoff' "$ALERT_LOG" || echo "No recent alerts"
    else
        echo "No alerts logged yet"
    fi
}

# Function to test alerting system
test_alerts() {
    echo "🧪 Testing alert system..."
    log_alert "Test alert - monitoring system functional"
    echo "Alert logged to $ALERT_LOG"
}

# Function to show health status summary
show_summary() {
    echo "📊 Physics Assistant Status Summary"
    echo "=" * 40

    # Quick server check
    echo "MCP Servers:"
    check_mcp_server "Forces" "10100" >/dev/null 2>&1 && echo "  ✅ Forces" || echo "  ❌ Forces"
    check_mcp_server "Kinematics" "10101" >/dev/null 2>&1 && echo "  ✅ Kinematics" || echo "  ❌ Kinematics"
    check_mcp_server "Math" "10103" >/dev/null 2>&1 && echo "  ✅ Math" || echo "  ❌ Math"
    check_mcp_server "Momentum" "10104" >/dev/null 2>&1 && echo "  ✅ Momentum" || echo "  ❌ Momentum"
    check_mcp_server "Energy" "10105" >/dev/null 2>&1 && echo "  ✅ Energy" || echo "  ❌ Energy"
    check_mcp_server "Angular-Motion" "10106" >/dev/null 2>&1 && echo "  ✅ Angular-Motion" || echo "  ❌ Angular-Motion"

    echo ""
    echo "Logs:"
    echo "  Health Log: $HEALTH_LOG"
    echo "  Alert Log: $ALERT_LOG"

    if [ -f "$HEALTH_LOG" ]; then
        echo "  Last Health Check: $(tail -1 "$HEALTH_LOG" 2>/dev/null | cut -d' ' -f1-2 || echo 'Never')"
    fi
}

# Main function
main() {
    case "${1:-check}" in
        "check")
            run_health_check
            ;;
        "monitor")
            run_continuous_monitoring
            ;;
        "alerts")
            show_recent_alerts
            ;;
        "test-alert")
            test_alerts
            ;;
        "summary")
            show_summary
            ;;
        "help"|"-h"|"--help")
            echo "Physics Assistant Health Monitor"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  check       Run single health check (default)"
            echo "  monitor     Run continuous monitoring"
            echo "  alerts      Show recent alerts"
            echo "  test-alert  Test alert logging"
            echo "  summary     Show quick status summary"
            echo "  help        Show this help"
            echo ""
            echo "Configuration:"
            echo "  Check interval: ${CHECK_INTERVAL}s"
            echo "  Max response time: ${MAX_RESPONSE_TIME}s"
            echo "  Health log: $HEALTH_LOG"
            echo "  Alert log: $ALERT_LOG"
            ;;
        *)
            echo "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Trap Ctrl+C for graceful exit
trap 'echo -e "\n🛑 Monitoring stopped"; exit 0' INT

# Run main function with all arguments
main "$@"