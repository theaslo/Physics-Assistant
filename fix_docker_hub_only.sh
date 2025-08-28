#!/bin/bash

# Use only Docker Hub images to avoid registry restrictions
echo "🔧 Converting all images to use docker.io registry..."

# Python images: Use official Docker Hub Python
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM registry.redhat.io/ubi9/python-311:latest|FROM docker.io/python:3.11-slim|g' {} \;

# Base system images
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM registry.redhat.io/ubi9/ubi-minimal:latest|FROM docker.io/alpine:3.19|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM registry.redhat.io/ubi9/nginx-120:latest|FROM docker.io/nginx:alpine|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM registry.redhat.io/rhel9/redis-7:latest|FROM docker.io/redis:7-alpine|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM registry.redhat.io/ubi9/nodejs-20:latest|FROM docker.io/node:20-alpine|g' {} \;
find docker -name "Dockerfile*" -type f -exec sed -i 's|FROM registry.redhat.io/rhel9/postgresql-16:latest|FROM docker.io/postgres:16-alpine|g' {} \;

echo "✅ All images now use docker.io registry"
echo "🚀 Try building: podman-compose -f docker-compose.production.yml build"