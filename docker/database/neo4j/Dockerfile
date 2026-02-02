FROM docker.io/neo4j:5.15-community

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7474/ || exit 1

EXPOSE 7474 7687