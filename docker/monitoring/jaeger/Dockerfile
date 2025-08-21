# Jaeger Distributed Tracing for Physics Assistant
FROM jaegertracing/all-in-one:1.51

# Switch to root to install additional tools
USER root

# Install additional monitoring tools
RUN apk add --no-cache \
    bash \
    curl \
    jq

# Create directories for configuration
RUN mkdir -p /opt/jaeger/config /opt/jaeger/scripts

# Copy configuration and scripts
COPY monitoring/jaeger/config/ /opt/jaeger/config/
COPY monitoring/jaeger/scripts/ /opt/jaeger/scripts/

# Set permissions
RUN chmod +x /opt/jaeger/scripts/*.sh

# Create jaeger user directories
RUN chown -R jaeger:jaeger /opt/jaeger

USER jaeger

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:14269/metrics || exit 1

# Expose Jaeger ports
# 14268 - jaeger.thrift over HTTP
# 14250 - gRPC 
# 16686 - Jaeger UI
# 14269 - admin port

EXPOSE 14268 14250 16686 14269

# Environment variables for configuration
ENV SPAN_STORAGE_TYPE=elasticsearch
ENV ES_SERVER_URLS=http://elasticsearch:9200
ENV ES_USERNAME=elastic
ENV ES_PASSWORD=${ELASTICSEARCH_PASSWORD}
ENV COLLECTOR_OTLP_ENABLED=true
ENV QUERY_BASE_PATH=/jaeger

CMD ["/opt/jaeger/scripts/start-jaeger.sh"]