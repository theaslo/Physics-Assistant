#!/usr/bin/env python3
"""
Setup database schema for Physics Assistant.

This script is intentionally safe to run on an existing database so container
startup can apply additive schema updates (for example HITL tables) without
recreating the core schema or re-inserting sample data.
"""
import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv


def _load_environment() -> None:
    """Load local env file when available, otherwise fall back to example env."""
    database_dir = Path(__file__).parent
    env_path = database_dir / ".env"
    example_env_path = database_dir / ".env.example"

    if env_path.exists():
        load_dotenv(env_path)
    elif example_env_path.exists():
        load_dotenv(example_env_path)


_load_environment()


async def _table_exists(conn: asyncpg.Connection, table_name: str) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = $1
            )
            """,
            table_name,
        )
    )


async def _execute_sql_file(
    conn: asyncpg.Connection,
    file_path: Path,
    description: str,
) -> None:
    with file_path.open("r") as f:
        sql = f.read()

    await conn.execute(sql)
    print(f"✅ {description}")

async def setup_schema():
    """Create database schema"""
    database_dir = Path(__file__).parent
    
    # Database connection parameters
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'physics_assistant'),
        'user': os.getenv('POSTGRES_USER', 'physics_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'physics_secure_password_2024')
    }
    
    print("Physics Assistant Schema Setup")
    print("==============================")
    print(f"Connecting to {db_config['host']}:{db_config['port']}/{db_config['database']}")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**db_config)
        print("✅ Connected to PostgreSQL")

        schema_dir = database_dir / "schema"
        core_schema_path = schema_dir / "01_core_tables.sql"
        sample_data_path = schema_dir / "02_sample_data.sql"
        hitl_schema_path = schema_dir / "03_hitl_knowledge_transfer.sql"
        agent_enum_migration_path = schema_dir / "04_agent_type_enum_updates.sql"

        has_core_schema = await _table_exists(conn, "users")
        if has_core_schema:
            print("ℹ️  Core tables already exist, skipping 01_core_tables.sql")
        elif core_schema_path.exists():
            await _execute_sql_file(conn, core_schema_path, "Core tables schema created successfully")

        if hitl_schema_path.exists():
            await _execute_sql_file(conn, hitl_schema_path, "HITL knowledge transfer schema/seed applied successfully")

        if agent_enum_migration_path.exists():
            await _execute_sql_file(conn, agent_enum_migration_path, "Agent type enum updates applied successfully")

        if sample_data_path.exists():
            has_admin_user = False
            if await _table_exists(conn, "users"):
                has_admin_user = bool(
                    await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM users WHERE username = 'admin')"
                    )
                )

            if has_admin_user:
                print("ℹ️  Sample data already present, skipping 02_sample_data.sql")
            else:
                await _execute_sql_file(conn, sample_data_path, "Sample data inserted successfully")
        
        # Verify tables were created
        result = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        print(f"\n📊 Created {len(result)} tables:")
        for row in result:
            print(f"  - {row['table_name']}")
        
        await conn.close()
        print("\n✅ Schema setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error setting up schema: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(setup_schema())
    exit(0 if success else 1)
