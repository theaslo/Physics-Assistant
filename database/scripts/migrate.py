#!/usr/bin/env python3
"""
Database migration script for Physics Assistant
Handles schema creation, updates, and data migrations
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))

try:
    import asyncpg
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required packages. Install with: pip install asyncpg python-dotenv")
    print(f"Error: {e}")
    sys.exit(1)


class DatabaseMigrator:
    def __init__(self):
        self.load_config()
        self.migrations_dir = Path(__file__).parent.parent / 'schema'
        self.migration_files = []
    
    def load_config(self):
        """Load database configuration from environment"""
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'physics_assistant'),
            'user': os.getenv('POSTGRES_USER', 'physics_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'physics_secure_password_2024')
        }
    
    async def get_connection(self):
        """Get database connection"""
        return await asyncpg.connect(
            host=self.postgres_config['host'],
            port=self.postgres_config['port'],
            database=self.postgres_config['database'],
            user=self.postgres_config['user'],
            password=self.postgres_config['password']
        )
    
    def discover_migration_files(self) -> List[Path]:
        """Discover SQL migration files in the schema directory"""
        if not self.migrations_dir.exists():
            print(f"❌ Schema directory not found: {self.migrations_dir}")
            return []
        
        # Find all .sql files and sort them
        sql_files = list(self.migrations_dir.glob('*.sql'))
        sql_files.sort()
        
        print(f"Found {len(sql_files)} migration files:")
        for file in sql_files:
            print(f"  - {file.name}")
        
        return sql_files
    
    async def create_migrations_table(self, conn):
        """Create migrations tracking table if it doesn't exist"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                checksum VARCHAR(64) NOT NULL,
                execution_time_ms INTEGER
            );
        """)
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file content"""
        import hashlib
        content = file_path.read_text(encoding='utf-8')
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def is_migration_applied(self, conn, filename: str) -> bool:
        """Check if migration has already been applied"""
        result = await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE filename = $1",
            filename
        )
        return result > 0
    
    async def apply_migration(self, conn, file_path: Path) -> bool:
        """Apply a single migration file"""
        filename = file_path.name
        print(f"Applying migration: {filename}")
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            checksum = self.calculate_file_checksum(file_path)
            
            start_time = datetime.now()
            
            # Execute the migration
            await conn.execute(content)
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record migration as applied
            await conn.execute("""
                INSERT INTO schema_migrations (filename, checksum, execution_time_ms)
                VALUES ($1, $2, $3)
            """, filename, checksum, execution_time)
            
            print(f"✅ Migration {filename} applied successfully ({execution_time}ms)")
            return True
            
        except Exception as e:
            print(f"❌ Migration {filename} failed: {e}")
            return False
    
    async def run_migrations(self, force: bool = False):
        """Run all pending migrations"""
        print("Starting database migration process...")
        
        # Discover migration files
        migration_files = self.discover_migration_files()
        if not migration_files:
            print("No migration files found.")
            return True
        
        try:
            conn = await self.get_connection()
            
            # Create migrations tracking table
            await self.create_migrations_table(conn)
            
            applied_count = 0
            skipped_count = 0
            
            for file_path in migration_files:
                filename = file_path.name
                
                # Skip if already applied (unless force flag is set)
                if not force and await self.is_migration_applied(conn, filename):
                    print(f"⏭️  Skipping {filename} (already applied)")
                    skipped_count += 1
                    continue
                
                # Apply migration
                if await self.apply_migration(conn, file_path):
                    applied_count += 1
                else:
                    print(f"❌ Migration process stopped due to error in {filename}")
                    await conn.close()
                    return False
            
            await conn.close()
            
            print(f"\n✅ Migration process completed:")
            print(f"   Applied: {applied_count} migrations")
            print(f"   Skipped: {skipped_count} migrations")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration process failed: {e}")
            return False
    
    async def rollback_migration(self, filename: str):
        """Rollback a specific migration (if rollback script exists)"""
        print(f"Rolling back migration: {filename}")
        
        # Look for rollback file
        rollback_file = self.migrations_dir / f"rollback_{filename}"
        if not rollback_file.exists():
            print(f"❌ Rollback file not found: {rollback_file}")
            return False
        
        try:
            conn = await self.get_connection()
            
            # Execute rollback
            content = rollback_file.read_text(encoding='utf-8')
            await conn.execute(content)
            
            # Remove from migrations table
            await conn.execute(
                "DELETE FROM schema_migrations WHERE filename = $1",
                filename
            )
            
            await conn.close()
            print(f"✅ Migration {filename} rolled back successfully")
            return True
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False
    
    async def show_migration_status(self):
        """Show status of all migrations"""
        print("Migration Status Report")
        print("=" * 50)
        
        try:
            conn = await self.get_connection()
            
            # Create migrations table if it doesn't exist
            await self.create_migrations_table(conn)
            
            # Get applied migrations
            applied_migrations = await conn.fetch("""
                SELECT filename, applied_at, execution_time_ms
                FROM schema_migrations
                ORDER BY applied_at
            """)
            
            # Get all available migrations
            available_files = self.discover_migration_files()
            applied_filenames = {row['filename'] for row in applied_migrations}
            
            print("\nApplied Migrations:")
            for row in applied_migrations:
                print(f"✅ {row['filename']} - {row['applied_at']} ({row['execution_time_ms']}ms)")
            
            print("\nPending Migrations:")
            pending_found = False
            for file_path in available_files:
                if file_path.name not in applied_filenames:
                    print(f"⏳ {file_path.name}")
                    pending_found = True
            
            if not pending_found:
                print("   No pending migrations")
            
            await conn.close()
            
        except Exception as e:
            print(f"❌ Failed to get migration status: {e}")
    
    async def reset_database(self):
        """Reset database by dropping all tables and reapplying migrations"""
        print("⚠️  DANGER: This will drop all tables and data!")
        
        # In a real scenario, you'd want user confirmation here
        confirm = input("Type 'RESET' to confirm database reset: ")
        if confirm != 'RESET':
            print("Database reset cancelled.")
            return False
        
        try:
            conn = await self.get_connection()
            
            # Drop all tables
            await conn.execute("DROP SCHEMA public CASCADE;")
            await conn.execute("CREATE SCHEMA public;")
            await conn.execute("GRANT ALL ON SCHEMA public TO physics_user;")
            await conn.execute("GRANT ALL ON SCHEMA public TO public;")
            
            await conn.close()
            
            print("✅ Database reset completed")
            
            # Reapply all migrations
            return await self.run_migrations(force=True)
            
        except Exception as e:
            print(f"❌ Database reset failed: {e}")
            return False


async def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Physics Assistant Database Migration Tool')
    parser.add_argument('action', choices=['migrate', 'status', 'rollback', 'reset'], 
                       help='Migration action to perform')
    parser.add_argument('--force', action='store_true', 
                       help='Force reapply all migrations')
    parser.add_argument('--file', type=str, 
                       help='Specific migration file for rollback')
    
    args = parser.parse_args()
    
    migrator = DatabaseMigrator()
    
    if args.action == 'migrate':
        success = await migrator.run_migrations(force=args.force)
    elif args.action == 'status':
        await migrator.show_migration_status()
        success = True
    elif args.action == 'rollback':
        if not args.file:
            print("❌ --file parameter required for rollback action")
            success = False
        else:
            success = await migrator.rollback_migration(args.file)
    elif args.action == 'reset':
        success = await migrator.reset_database()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())