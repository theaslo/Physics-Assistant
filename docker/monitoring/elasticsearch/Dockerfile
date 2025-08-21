# Elasticsearch for APM and Observability Data
FROM docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Switch to root for configuration
USER root

# Install additional tools
RUN yum update -y && yum install -y \
    curl \
    jq \
    && yum clean all

# Create directories for configuration
RUN mkdir -p /usr/share/elasticsearch/config/scripts \
             /usr/share/elasticsearch/config/templates \
             /usr/share/elasticsearch/data \
             /usr/share/elasticsearch/logs

# Copy configuration files
COPY monitoring/elasticsearch/config/ /usr/share/elasticsearch/config/
COPY monitoring/elasticsearch/scripts/ /usr/share/elasticsearch/config/scripts/
COPY monitoring/elasticsearch/templates/ /usr/share/elasticsearch/config/templates/

# Set permissions
RUN chmod +x /usr/share/elasticsearch/config/scripts/*.sh && \
    chown -R elasticsearch:elasticsearch /usr/share/elasticsearch

# Switch back to elasticsearch user
USER elasticsearch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:9200/_cluster/health || exit 1

# Environment variables for production
ENV discovery.type=single-node
ENV xpack.security.enabled=true
ENV xpack.security.http.ssl.enabled=false
ENV xpack.security.transport.ssl.enabled=false
ENV xpack.monitoring.collection.enabled=true
ENV xpack.ml.enabled=true
ENV cluster.name=physics-assistant-cluster
ENV node.name=physics-assistant-node-1
ENV bootstrap.memory_lock=true
ENV ES_JAVA_OPTS="-Xms2g -Xmx2g"

# Expose Elasticsearch port
EXPOSE 9200 9300

CMD ["/usr/share/elasticsearch/config/scripts/start-elasticsearch.sh"]