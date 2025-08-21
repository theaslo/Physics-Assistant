#!/bin/bash
set -e

# Physics Assistant Kubernetes Deployment Script
echo "=== Physics Assistant Kubernetes Deployment ==="

# Configuration
NAMESPACE="physics-assistant"
KUBECTL_CMD="kubectl"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster"
    echo "Please ensure kubectl is configured correctly"
    exit 1
fi

echo "Connected to Kubernetes cluster:"
kubectl cluster-info | head -1

# Function to wait for deployment to be ready
wait_for_deployment() {
    local deployment=$1
    local namespace=$2
    echo "Waiting for deployment $deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $namespace
}

# Function to wait for statefulset to be ready
wait_for_statefulset() {
    local statefulset=$1
    local namespace=$2
    echo "Waiting for statefulset $statefulset to be ready..."
    kubectl wait --for=jsonpath='{.status.readyReplicas}'=1 --timeout=300s statefulset/$statefulset -n $namespace
}

# Apply namespace and RBAC
echo "Creating namespace and applying RBAC..."
kubectl apply -f k8s/namespace.yaml

# Apply ConfigMaps and Secrets
echo "Applying ConfigMaps and Secrets..."
kubectl apply -f k8s/configmaps.yaml
kubectl apply -f k8s/secrets.yaml

# Apply Storage
echo "Setting up persistent storage..."
kubectl apply -f k8s/storage.yaml

# Wait for PVCs to be bound
echo "Waiting for PVCs to be bound..."
kubectl wait --for=condition=Bound --timeout=120s pvc --all -n $NAMESPACE

# Deploy databases (order matters)
echo "Deploying database services..."
kubectl apply -f k8s/databases.yaml

# Wait for databases to be ready
wait_for_statefulset "postgres" $NAMESPACE
wait_for_statefulset "neo4j" $NAMESPACE
wait_for_statefulset "redis" $NAMESPACE

# Deploy MCP services
echo "Deploying MCP services..."
kubectl apply -f k8s/mcp-services.yaml

# Wait for MCP services to be ready
MCP_SERVICES=("mcp-forces" "mcp-kinematics" "mcp-math" "mcp-energy" "mcp-momentum" "mcp-angular-motion")
for service in "${MCP_SERVICES[@]}"; do
    wait_for_deployment "$service" $NAMESPACE
done

# Deploy API services
echo "Deploying API services..."
kubectl apply -f k8s/apis.yaml

# Wait for API services to be ready
API_SERVICES=("database-api" "dashboard-api" "physics-agents-api")
for service in "${API_SERVICES[@]}"; do
    wait_for_deployment "$service" $NAMESPACE
done

# Deploy frontend services
echo "Deploying frontend services..."
kubectl apply -f k8s/frontends.yaml

# Wait for frontend services to be ready
FRONTEND_SERVICES=("streamlit-ui" "react-dashboard" "nginx-gateway")
for service in "${FRONTEND_SERVICES[@]}"; do
    wait_for_deployment "$service" $NAMESPACE
done

# Deploy analytics services
echo "Deploying analytics services..."
kubectl apply -f k8s/analytics.yaml

# Wait for analytics services to be ready
ANALYTICS_SERVICES=("ml-engine" "task-processor" "flower-monitor")
for service in "${ANALYTICS_SERVICES[@]}"; do
    wait_for_deployment "$service" $NAMESPACE
done

# Deploy monitoring services
echo "Deploying monitoring services..."
kubectl apply -f k8s/monitoring.yaml

# Wait for monitoring services to be ready
MONITORING_SERVICES=("prometheus" "grafana" "alertmanager")
for service in "${MONITORING_SERVICES[@]}"; do
    wait_for_deployment "$service" $NAMESPACE
done

# Check overall deployment status
echo ""
echo "=== Deployment Status ==="
kubectl get all -n $NAMESPACE

echo ""
echo "=== Pod Status ==="
kubectl get pods -n $NAMESPACE -o wide

echo ""
echo "=== Service Status ==="
kubectl get services -n $NAMESPACE

echo ""
echo "=== Ingress Status ==="
kubectl get ingress -n $NAMESPACE

# Get external IP for LoadBalancer service
echo ""
echo "=== Access Information ==="
EXTERNAL_IP=$(kubectl get service nginx-gateway-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$EXTERNAL_IP" ]; then
    echo "External IP: $EXTERNAL_IP"
    echo "Main Application: http://$EXTERNAL_IP"
    echo "Analytics Dashboard: http://$EXTERNAL_IP/dashboard"
else
    echo "LoadBalancer external IP not yet assigned. Use port-forward for testing:"
    echo "kubectl port-forward -n $NAMESPACE svc/nginx-gateway-service 8080:80"
    echo "Then access: http://localhost:8080"
fi

# Show useful commands
echo ""
echo "=== Useful Commands ==="
echo "View logs: kubectl logs -f deployment/[service-name] -n $NAMESPACE"
echo "Scale service: kubectl scale deployment/[service-name] --replicas=N -n $NAMESPACE"
echo "Port forward: kubectl port-forward -n $NAMESPACE svc/[service-name] [local-port]:[service-port]"
echo "Delete deployment: kubectl delete namespace $NAMESPACE"

echo ""
echo "=== Deployment Complete ==="
echo "All services are running successfully in Kubernetes!"

# Optional: Run basic health checks
echo ""
echo "Running basic health checks..."
kubectl get pods -n $NAMESPACE | grep -v Running | grep -v Completed || echo "All pods are running!"

# Show resource usage
echo ""
echo "=== Resource Usage ==="
kubectl top pods -n $NAMESPACE 2>/dev/null || echo "Metrics server not available for resource usage"