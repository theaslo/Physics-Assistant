#!/bin/sh
set -e

echo "Starting Prometheus..."

# Validate configuration
prometheus --config.file=/etc/prometheus/prometheus.yml --web.config.file= --storage.tsdb.path=/prometheus/data --web.console.libraries=/usr/share/prometheus/console_libraries --web.console.templates=/usr/share/prometheus/consoles --web.enable-lifecycle --web.enable-admin-api --config.file=/etc/prometheus/prometheus.yml --dry-run

# Start Prometheus with proper configuration
exec prometheus \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus/data \
    --storage.tsdb.retention.time=15d \
    --storage.tsdb.retention.size=10GB \
    --web.console.libraries=/usr/share/prometheus/console_libraries \
    --web.console.templates=/usr/share/prometheus/consoles \
    --web.enable-lifecycle \
    --web.enable-admin-api \
    --web.listen-address=0.0.0.0:9090 \
    --log.level=info