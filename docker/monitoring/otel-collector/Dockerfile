# OpenTelemetry Collector for Distributed Tracing
FROM otel/opentelemetry-collector-contrib:0.90.1

# Switch to root for additional setup
USER root

# Install additional tools
RUN apk add --no-cache \
    bash \
    curl \
    jq

# Create directories for configuration
RUN mkdir -p /etc/otelcol/config /etc/otelcol/scripts

# Copy configuration and scripts
COPY monitoring/otel-collector/config/ /etc/otelcol/config/
COPY monitoring/otel-collector/scripts/ /etc/otelcol/scripts/

# Set permissions
RUN chmod +x /etc/otelcol/scripts/*.sh

# Create otel user directories
RUN chown -R 10001:10001 /etc/otelcol

USER 10001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:13133/health || exit 1

# Expose collector ports
# 4317 - OTLP gRPC receiver
# 4318 - OTLP HTTP receiver
# 8888 - Prometheus metrics
# 13133 - Health check
# 14250 - Jaeger gRPC
# 14268 - Jaeger HTTP

EXPOSE 4317 4318 8888 13133 14250 14268

CMD ["/etc/otelcol/scripts/start-collector.sh"]