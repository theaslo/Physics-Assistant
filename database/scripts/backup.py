#!/usr/bin/env python3
"""
Database backup and restore script for Physics Assistant
Supports automated backups, compression, and cleanup
"""

import asyncio
import gzip
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing python-dotenv package. Install with: pip install python-dotenv")
    sys.exit(1)


class DatabaseBackup:
    def __init__(self):
        self.load_config()
        self.backup_dir = Path(__file__).parent.parent / 'backups'
        self.backup_dir.mkdir(exist_ok=True)
    
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
        
        self.retention_days = int(os.getenv('BACKUP_RETENTION_DAYS', 30))
    
    def get_backup_filename(self, backup_type: str = 'full') -> str:
        """Generate backup filename with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"physics_assistant_{backup_type}_{timestamp}.sql"
    
    def create_backup(self, backup_type: str = 'full', compress: bool = True) -> Optional[Path]:
        """Create database backup using pg_dump"""
        print(f"Creating {backup_type} database backup...")
        
        filename = self.get_backup_filename(backup_type)
        backup_path = self.backup_dir / filename
        
        # Prepare pg_dump command
        env = os.environ.copy()
        env['PGPASSWORD'] = self.postgres_config['password']
        
        cmd = [
            'pg_dump',
            '-h', self.postgres_config['host'],
            '-p', str(self.postgres_config['port']),
            '-U', self.postgres_config['user'],
            '-d', self.postgres_config['database'],
            '--verbose',
            '--no-owner',
            '--no-privileges',
        ]
        
        if backup_type == 'schema':
            cmd.append('--schema-only')
        elif backup_type == 'data':
            cmd.append('--data-only')
        
        try:
            start_time = datetime.now()
            
            # Run pg_dump
            with open(backup_path, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                    check=True
                )
            
            backup_time = datetime.now() - start_time
            backup_size = backup_path.stat().st_size
            
            # Compress if requested
            if compress:
                compressed_path = backup_path.with_suffix('.sql.gz')
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed file
                backup_path.unlink()
                backup_path = compressed_path
                compressed_size = backup_path.stat().st_size
                compression_ratio = (1 - compressed_size / backup_size) * 100
                
                print(f"✅ Backup created: {backup_path.name}")
                print(f"   Original size: {self.format_bytes(backup_size)}")
                print(f"   Compressed size: {self.format_bytes(compressed_size)}")
                print(f"   Compression: {compression_ratio:.1f}%")
            else:
                print(f"✅ Backup created: {backup_path.name}")
                print(f"   Size: {self.format_bytes(backup_size)}")
            
            print(f"   Duration: {backup_time.total_seconds():.2f} seconds")
            
            return backup_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Backup failed: {e}")
            print(f"   Error output: {e.stderr}")
            if backup_path.exists():
                backup_path.unlink()
            return None
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            if backup_path.exists():
                backup_path.unlink()
            return None
    
    def restore_backup(self, backup_file: Path, drop_existing: bool = False) -> bool:
        """Restore database from backup file"""
        if not backup_file.exists():
            print(f"❌ Backup file not found: {backup_file}")
            return False
        
        print(f"Restoring database from: {backup_file.name}")
        
        # Prepare environment
        env = os.environ.copy()
        env['PGPASSWORD'] = self.postgres_config['password']
        
        try:
            # Drop existing database if requested
            if drop_existing:
                print("⚠️  Dropping existing database...")
                drop_cmd = [
                    'dropdb',
                    '-h', self.postgres_config['host'],
                    '-p', str(self.postgres_config['port']),
                    '-U', self.postgres_config['user'],
                    '--if-exists',
                    self.postgres_config['database']
                ]
                subprocess.run(drop_cmd, env=env, check=True, stderr=subprocess.PIPE)
                
                # Recreate database
                create_cmd = [
                    'createdb',
                    '-h', self.postgres_config['host'],
                    '-p', str(self.postgres_config['port']),
                    '-U', self.postgres_config['user'],
                    self.postgres_config['database']
                ]
                subprocess.run(create_cmd, env=env, check=True, stderr=subprocess.PIPE)
            
            # Restore from backup
            start_time = datetime.now()
            
            if backup_file.suffix == '.gz':
                # Restore from compressed backup
                with gzip.open(backup_file, 'rb') as f:
                    restore_cmd = [
                        'psql',
                        '-h', self.postgres_config['host'],
                        '-p', str(self.postgres_config['port']),
                        '-U', self.postgres_config['user'],
                        '-d', self.postgres_config['database'],
                        '--quiet'
                    ]
                    result = subprocess.run(
                        restore_cmd,
                        stdin=f,
                        stderr=subprocess.PIPE,
                        env=env,
                        check=True
                    )
            else:
                # Restore from uncompressed backup
                restore_cmd = [
                    'psql',
                    '-h', self.postgres_config['host'],
                    '-p', str(self.postgres_config['port']),
                    '-U', self.postgres_config['user'],
                    '-d', self.postgres_config['database'],
                    '-f', str(backup_file),
                    '--quiet'
                ]
                result = subprocess.run(
                    restore_cmd,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=True
                )
            
            restore_time = datetime.now() - start_time
            
            print(f"✅ Database restored successfully")
            print(f"   Duration: {restore_time.total_seconds():.2f} seconds")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Restore failed: {e}")
            print(f"   Error output: {e.stderr.decode() if e.stderr else 'No error output'}")
            return False
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False
    
    def list_backups(self) -> List[Path]:
        """List all available backup files"""
        backups = []
        
        # Find all SQL and SQL.GZ files
        for pattern in ['*.sql', '*.sql.gz']:
            backups.extend(self.backup_dir.glob(pattern))
        
        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return backups
    
    def cleanup_old_backups(self) -> int:
        """Remove backup files older than retention period"""
        print(f"Cleaning up backups older than {self.retention_days} days...")
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        backups = self.list_backups()
        
        removed_count = 0
        for backup_path in backups:
            backup_time = datetime.fromtimestamp(backup_path.stat().st_mtime)
            if backup_time < cutoff_date:
                print(f"Removing old backup: {backup_path.name}")
                backup_path.unlink()
                removed_count += 1
        
        if removed_count == 0:
            print("No old backups to remove")
        else:
            print(f"✅ Removed {removed_count} old backup(s)")
        
        return removed_count
    
    def show_backup_status(self):
        """Show backup status and statistics"""
        print("Database Backup Status")
        print("=" * 50)
        
        backups = self.list_backups()
        
        if not backups:
            print("No backups found")
            return
        
        total_size = sum(backup.stat().st_size for backup in backups)
        
        print(f"Total backups: {len(backups)}")
        print(f"Total size: {self.format_bytes(total_size)}")
        print(f"Retention period: {self.retention_days} days")
        print("\nBackup files:")
        
        for backup_path in backups:
            stat = backup_path.stat()
            size = self.format_bytes(stat.st_size)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {backup_path.name:<40} {size:>10} {modified}")
    
    @staticmethod
    def format_bytes(bytes_count: int) -> str:
        """Format bytes as human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f}{unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f}PB"


async def main():
    """Main backup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Physics Assistant Database Backup Tool')
    parser.add_argument('action', choices=['backup', 'restore', 'list', 'cleanup', 'status'],
                       help='Backup action to perform')
    parser.add_argument('--type', choices=['full', 'schema', 'data'], default='full',
                       help='Type of backup to create')
    parser.add_argument('--file', type=str,
                       help='Backup file for restore operation')
    parser.add_argument('--compress', action='store_true', default=True,
                       help='Compress backup file')
    parser.add_argument('--no-compress', action='store_true',
                       help='Do not compress backup file')
    parser.add_argument('--drop', action='store_true',
                       help='Drop existing database before restore')
    
    args = parser.parse_args()
    
    backup_tool = DatabaseBackup()
    
    if args.action == 'backup':
        compress = args.compress and not args.no_compress
        backup_path = backup_tool.create_backup(args.type, compress)
        success = backup_path is not None
    
    elif args.action == 'restore':
        if not args.file:
            print("❌ --file parameter required for restore action")
            success = False
        else:
            backup_file = Path(args.file)
            if not backup_file.is_absolute():
                backup_file = backup_tool.backup_dir / backup_file
            success = backup_tool.restore_backup(backup_file, args.drop)
    
    elif args.action == 'list':
        backups = backup_tool.list_backups()
        print(f"Found {len(backups)} backup files:")
        for backup in backups:
            print(f"  {backup.name}")
        success = True
    
    elif args.action == 'cleanup':
        backup_tool.cleanup_old_backups()
        success = True
    
    elif args.action == 'status':
        backup_tool.show_backup_status()
        success = True
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())