# Neo4j Container for Physics Assistant Knowledge Graph
FROM neo4j:5.15-enterprise

LABEL maintainer="Physics Assistant Team"
LABEL description="Neo4j knowledge graph database for Physics Assistant platform"

# Install additional tools
USER root
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    jq \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r physicsapp -g 1001 && \
    useradd -r -g physicsapp -u 1001 physicsapp

# Copy initialization scripts and data
COPY docker/database/neo4j/scripts/ /opt/physics-assistant/scripts/
COPY docker/database/neo4j/plugins/ /var/lib/neo4j/plugins/
COPY database/knowledge_graph_structure.json /opt/physics-assistant/data/
COPY database/setup_complete_knowledge_graph.py /opt/physics-assistant/scripts/

# Set proper permissions
RUN chmod +x /opt/physics-assistant/scripts/*.sh && \
    chown -R neo4j:neo4j /opt/physics-assistant/ && \
    chown -R neo4j:neo4j /var/lib/neo4j/

# Copy custom Neo4j configuration
COPY docker/database/neo4j/neo4j.conf /var/lib/neo4j/conf/neo4j.conf

# Install Python dependencies for initialization scripts
RUN pip3 install neo4j python-dotenv

# Create directories for backups and logs
RUN mkdir -p /var/lib/neo4j/backups /var/log/neo4j && \
    chown -R neo4j:neo4j /var/lib/neo4j/backups /var/log/neo4j

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7474/db/data/ || exit 1

# Switch back to neo4j user
USER neo4j

# Expose ports
EXPOSE 7474 7687

# Environment variables
ENV NEO4J_AUTH=neo4j/physics_assistant_2024
ENV NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
ENV NEO4J_PLUGINS=["apoc", "graph-data-science"]

# Volume for data persistence
VOLUME ["/data", "/logs", "/import", "/plugins"]