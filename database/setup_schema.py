#!/usr/bin/env python3
"""
Setup database schema for Physics Assistant
"""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.example')

async def setup_schema():
    """Create database schema"""
    
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
        
        # Read schema file
        with open('schema/01_core_tables.sql', 'r') as f:
            schema_sql = f.read()
        
        print("📄 Loaded schema file")
        
        # Execute schema
        await conn.execute(schema_sql)
        print("✅ Core tables schema created successfully")
        
        # Read sample data if it exists
        if os.path.exists('schema/02_sample_data.sql'):
            with open('schema/02_sample_data.sql', 'r') as f:
                sample_data_sql = f.read()
            
            await conn.execute(sample_data_sql)
            print("✅ Sample data inserted successfully")
        
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