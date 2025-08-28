#!/bin/bash

# Fix Dockerfiles to use Red Hat UBI and registry.redhat.io images
# for RHEL 9 server with restricted registries

echo "🔧 Fixing Docker base images for Red Hat Enterprise Linux..."

# Python images: Use Red Hat UBI Python
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM python:3.11-slim|FROM registry.redhat.io/ubi9/python-311:latest|g' {} \;

# Alpine images: Use Red Hat UBI minimal
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM alpine:3.19|FROM registry.redhat.io/ubi9/ubi-minimal:latest|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM nginx:alpine|FROM registry.redhat.io/ubi9/nginx-120:latest|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM redis:7-alpine|FROM registry.redhat.io/rhel9/redis-7:latest|g' {} \;

# Node images: Use Red Hat UBI Node
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM node:20-alpine|FROM registry.redhat.io/ubi9/nodejs-20:latest|g' {} \;

# PostgreSQL: Use Red Hat PostgreSQL
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM postgres:16-alpine|FROM registry.redhat.io/rhel9/postgresql-16:latest|g' {} \;

# Monitoring images: Try Red Hat alternatives or use docker.io explicitly
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM prom/prometheus:latest|FROM docker.io/prom/prometheus:latest|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM prom/alertmanager:latest|FROM docker.io/prom/alertmanager:latest|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM grafana/grafana:latest|FROM docker.io/grafana/grafana:latest|g' {} \;

# Neo4j: Use docker.io explicitly 
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM neo4j:5.15-enterprise|FROM docker.io/neo4j:5.15-community|g' {} \;

# Other third-party images: Use docker.io explicitly
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM hashicorp/vault:1.15|FROM docker.io/hashicorp/vault:1.15|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM aquasec/trivy:latest|FROM docker.io/aquasec/trivy:latest|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM otel/opentelemetry-collector-contrib:0.90.1|FROM docker.io/otel/opentelemetry-collector-contrib:0.90.1|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM jaegertracing/all-in-one:1.51|FROM docker.io/jaegertracing/all-in-one:1.51|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM docker.elastic.co/elasticsearch/elasticsearch:8.11.0|FROM docker.io/elastic/elasticsearch:8.11.0|g' {} \;

echo "✅ Updated all Dockerfiles to use Red Hat compatible images"
echo ""
echo "📋 Image mapping applied:"
echo "  python:3.11-slim → registry.redhat.io/ubi9/python-311:latest"
echo "  alpine → registry.redhat.io/ubi9/ubi-minimal:latest" 
echo "  nginx:alpine → registry.redhat.io/ubi9/nginx-120:latest"
echo "  redis:7-alpine → registry.redhat.io/rhel9/redis-7:latest"
echo "  node:20-alpine → registry.redhat.io/ubi9/nodejs-20:latest"
echo "  postgres:16-alpine → registry.redhat.io/rhel9/postgresql-16:latest"
echo "  Third-party images → docker.io/[image]"
echo ""
echo "🚀 Now try building again:"
echo "  podman-compose -f docker-compose.production.yml build"