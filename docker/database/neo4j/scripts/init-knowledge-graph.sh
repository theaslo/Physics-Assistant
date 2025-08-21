#!/bin/bash
set -e

echo "Initializing Physics Assistant Knowledge Graph..."

# Wait for Neo4j to be ready
until curl -f -s -H "Content-Type: application/json" \
    -d '{"statements":[{"statement":"RETURN 1"}]}' \
    http://localhost:7474/db/data/transaction/commit > /dev/null; do
    echo "Waiting for Neo4j to be ready..."
    sleep 5
done

echo "Neo4j is ready. Initializing knowledge graph..."

# Run the knowledge graph setup script
cd /opt/physics-assistant/scripts/
python3 setup_complete_knowledge_graph.py

echo "Knowledge graph initialization completed."

# Create indexes for performance
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "statements": [
            {
                "statement": "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)"
            },
            {
                "statement": "CREATE INDEX concept_type_index IF NOT EXISTS FOR (c:Concept) ON (c.type)"
            },
            {
                "statement": "CREATE INDEX concept_level_index IF NOT EXISTS FOR (c:Concept) ON (c.level)"
            },
            {
                "statement": "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r]-() ON (type(r))"
            }
        ]
    }' \
    http://localhost:7474/db/data/transaction/commit

echo "Performance indexes created."