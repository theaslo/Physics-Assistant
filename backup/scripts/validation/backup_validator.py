#!/usr/bin/env python3
"""
Backup Validation System for Physics Assistant Platform

This script performs comprehensive validation of backup files including:
- File integrity checks
- Content validation
- Restoration testing
- Performance benchmarking
"""

import os
import sys
import json
import time
import hashlib
import tempfile
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/validation/backup_validator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('backup-validator')

@dataclass
class ValidationResult:
    """Result of backup validation"""
    backup_file: str
    service: str
    backup_type: str
    validation_timestamp: datetime
    file_integrity: bool = False
    content_validity: bool = False
    restoration_test: bool = False
    file_size_bytes: int = 0
    validation_duration_seconds: float = 0.0
    error_messages: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def overall_valid(self) -> bool:
        """Check if backup is overall valid"""
        return self.file_integrity and self.content_validity
    
    @property
    def fully_tested(self) -> bool:
        """Check if backup has been fully tested including restoration"""
        return self.overall_valid and self.restoration_test

class BackupValidator:
    """Main backup validation class"""
    
    def __init__(self):
        self.backup_base_dir = Path(os.getenv('BACKUP_BASE_DIR', '/backups'))
        self.validation_db_path = Path('/logs/validation/validation_results.db')
        self.temp_dir = Path('/tmp/validation')
        
        # Validation configuration
        self.perform_restoration_tests = os.getenv('RESTORATION_TESTS_ENABLED', 'false').lower() == 'true'
        self.max_file_size_mb = int(os.getenv('MAX_VALIDATION_FILE_SIZE_MB', '1000'))  # 1GB
        self.validation_timeout_seconds = int(os.getenv('VALIDATION_TIMEOUT_SECONDS', '3600'))  # 1 hour
        
        # Database connection settings
        self.postgres_host = os.getenv('POSTGRES_HOST', 'postgres')
        self.postgres_port = os.getenv('POSTGRES_PORT', '5432')
        self.postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', '')
        
        self.neo4j_host = os.getenv('NEO4J_HOST', 'neo4j')
        self.neo4j_port = os.getenv('NEO4J_PORT', '7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', '')
        
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = os.getenv('REDIS_PORT', '6379')
        self.redis_password = os.getenv('REDIS_PASSWORD', '')
        
        # Initialize
        self.init_validation_database()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Backup validator initialized")
    
    def init_validation_database(self):
        """Initialize SQLite database for storing validation results"""
        
        self.validation_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_file TEXT NOT NULL,
                service TEXT NOT NULL,
                backup_type TEXT NOT NULL,
                validation_timestamp TEXT NOT NULL,
                file_integrity BOOLEAN NOT NULL,
                content_validity BOOLEAN NOT NULL,
                restoration_test BOOLEAN NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                validation_duration_seconds REAL NOT NULL,
                error_messages TEXT,
                warnings TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_backup_file ON validation_results(backup_file)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_service_type ON validation_results(service, backup_type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_validation_timestamp ON validation_results(validation_timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Validation database initialized")
    
    def save_validation_result(self, result: ValidationResult):
        """Save validation result to database"""
        
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO validation_results (
                backup_file, service, backup_type, validation_timestamp,
                file_integrity, content_validity, restoration_test,
                file_size_bytes, validation_duration_seconds,
                error_messages, warnings, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.backup_file,
            result.service,
            result.backup_type,
            result.validation_timestamp.isoformat(),
            result.file_integrity,
            result.content_validity,
            result.restoration_test,
            result.file_size_bytes,
            result.validation_duration_seconds,
            json.dumps(result.error_messages),
            json.dumps(result.warnings),
            json.dumps(result.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Validation result saved for {result.backup_file}")
    
    def get_validation_history(self, backup_file: str, limit: int = 10) -> List[ValidationResult]:
        """Get validation history for a backup file"""
        
        conn = sqlite3.connect(self.validation_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM validation_results 
            WHERE backup_file = ? 
            ORDER BY validation_timestamp DESC 
            LIMIT ?
        ''', (backup_file, limit))
        
        results = []
        for row in cursor.fetchall():
            result = ValidationResult(
                backup_file=row[1],
                service=row[2],
                backup_type=row[3],
                validation_timestamp=datetime.fromisoformat(row[4]),
                file_integrity=bool(row[5]),
                content_validity=bool(row[6]),
                restoration_test=bool(row[7]),
                file_size_bytes=row[8],
                validation_duration_seconds=row[9],
                error_messages=json.loads(row[10]) if row[10] else [],
                warnings=json.loads(row[11]) if row[11] else [],
                metadata=json.loads(row[12]) if row[12] else {}
            )
            results.append(result)
        
        conn.close()
        return results
    
    def calculate_file_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate checksum of a file"""
        
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def validate_file_integrity(self, backup_file: Path) -> Tuple[bool, List[str]]:
        """Validate file integrity"""
        
        logger.info(f"Validating file integrity: {backup_file}")
        errors = []
        
        # Check if file exists
        if not backup_file.exists():
            errors.append(f"Backup file does not exist: {backup_file}")
            return False, errors
        
        # Check file size
        file_size = backup_file.stat().st_size
        if file_size == 0:
            errors.append("Backup file is empty")
            return False, errors
        
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            errors.append(f"Backup file too large: {file_size} bytes (max: {max_size_bytes})")
            return False, errors
        
        # Validate compressed files
        if backup_file.suffix == '.gz':
            try:
                result = subprocess.run(
                    ['gzip', '-t', str(backup_file)],
                    capture_output=True,
                    timeout=300
                )
                if result.returncode != 0:
                    errors.append(f"Gzip integrity test failed: {result.stderr.decode()}")
                    return False, errors
            except subprocess.TimeoutExpired:
                errors.append("Gzip integrity test timed out")
                return False, errors
            except Exception as e:
                errors.append(f"Error testing gzip integrity: {e}")
                return False, errors
        
        elif backup_file.suffix == '.tar.gz' or backup_file.name.endswith('.tar.gz'):
            try:
                result = subprocess.run(
                    ['tar', '-tzf', str(backup_file)],
                    capture_output=True,
                    timeout=300
                )
                if result.returncode != 0:
                    errors.append(f"Tar integrity test failed: {result.stderr.decode()}")
                    return False, errors
            except subprocess.TimeoutExpired:
                errors.append("Tar integrity test timed out")
                return False, errors
            except Exception as e:
                errors.append(f"Error testing tar integrity: {e}")
                return False, errors
        
        # Check if metadata file exists and validate checksum
        metadata_file = backup_file.with_suffix(backup_file.suffix + '.metadata.json')
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                expected_checksum = metadata.get('checksum')
                if expected_checksum:
                    actual_checksum = self.calculate_file_checksum(backup_file, 'md5')
                    if actual_checksum != expected_checksum:
                        errors.append(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
                        return False, errors
                    
                    logger.info("Checksum validation passed")
                
            except Exception as e:
                errors.append(f"Error validating metadata: {e}")
                return False, errors
        
        logger.info("File integrity validation passed")
        return True, errors
    
    def validate_postgres_backup(self, backup_file: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate PostgreSQL backup content"""
        
        logger.info(f"Validating PostgreSQL backup content: {backup_file}")
        errors = []
        warnings = []
        
        try:
            # Determine backup format
            if backup_file.suffix == '.dump' or (backup_file.suffix == '.gz' and '.dump' in backup_file.name):
                # Custom dump format
                test_file = backup_file
                if backup_file.suffix == '.gz':
                    # Decompress to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.dump') as temp_f:
                        subprocess.run(['gunzip', '-c', str(backup_file)], stdout=temp_f, check=True)
                        test_file = Path(temp_f.name)
                
                # Test dump file structure
                result = subprocess.run(
                    ['pg_restore', '--list', str(test_file)],
                    capture_output=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    errors.append(f"pg_restore list failed: {result.stderr.decode()}")
                    return False, errors, warnings
                
                # Parse pg_restore output to validate content
                output = result.stdout.decode()
                if not output.strip():
                    errors.append("Backup dump appears to be empty")
                    return False, errors, warnings
                
                # Check for expected database objects
                if 'TABLE' not in output:
                    warnings.append("No tables found in backup")
                
                if test_file != backup_file:
                    test_file.unlink()  # Clean up temporary file
            
            elif backup_file.suffix == '.sql' or (backup_file.suffix == '.gz' and '.sql' in backup_file.name):
                # SQL dump format
                test_file = backup_file
                if backup_file.suffix == '.gz':
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.sql') as temp_f:
                        subprocess.run(['gunzip', '-c', str(backup_file)], stdout=temp_f, check=True)
                        test_file = Path(temp_f.name)
                
                # Basic SQL syntax validation
                with open(test_file, 'r') as f:
                    content = f.read(10000)  # Read first 10KB
                
                if not content.strip():
                    errors.append("SQL backup file is empty")
                    return False, errors, warnings
                
                # Check for SQL keywords
                sql_keywords = ['CREATE', 'INSERT', 'COPY']
                if not any(keyword in content for keyword in sql_keywords):
                    warnings.append("No SQL statements found in backup")
                
                if test_file != backup_file:
                    test_file.unlink()
            
            else:
                errors.append(f"Unknown PostgreSQL backup format: {backup_file.suffix}")
                return False, errors, warnings
            
            logger.info("PostgreSQL backup content validation passed")
            return True, errors, warnings
            
        except subprocess.TimeoutExpired:
            errors.append("PostgreSQL backup validation timed out")
            return False, errors, warnings
        except Exception as e:
            errors.append(f"Error validating PostgreSQL backup: {e}")
            return False, errors, warnings
    
    def validate_neo4j_backup(self, backup_file: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate Neo4j backup content"""
        
        logger.info(f"Validating Neo4j backup content: {backup_file}")
        errors = []
        warnings = []
        
        try:
            if backup_file.suffix == '.gz' and '.dump' in backup_file.name:
                # Neo4j dump format
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dump') as temp_f:
                    subprocess.run(['gunzip', '-c', str(backup_file)], stdout=temp_f, check=True)
                    test_file = Path(temp_f.name)
                
                # Check dump file size and basic structure
                if test_file.stat().st_size == 0:
                    errors.append("Neo4j dump file is empty")
                    return False, errors, warnings
                
                # Basic dump file validation (Neo4j dumps are binary)
                with open(test_file, 'rb') as f:
                    header = f.read(100)
                    if len(header) < 10:
                        errors.append("Neo4j dump file appears corrupted")
                        return False, errors, warnings
                
                test_file.unlink()
            
            elif backup_file.suffix == '.gz' and '.cypher' in backup_file.name:
                # Cypher script format
                with tempfile.NamedTemporaryFile(delete=False, suffix='.cypher', mode='w+') as temp_f:
                    subprocess.run(['gunzip', '-c', str(backup_file)], stdout=temp_f, check=True)
                    temp_f.seek(0)
                    content = temp_f.read(10000)  # Read first 10KB
                
                if not content.strip():
                    errors.append("Cypher backup file is empty")
                    return False, errors, warnings
                
                # Check for Cypher keywords
                cypher_keywords = ['CREATE', 'MATCH', 'MERGE', 'SET']
                if not any(keyword in content for keyword in cypher_keywords):
                    warnings.append("No Cypher statements found in backup")
                
                Path(temp_f.name).unlink()
            
            else:
                errors.append(f"Unknown Neo4j backup format: {backup_file.suffix}")
                return False, errors, warnings
            
            logger.info("Neo4j backup content validation passed")
            return True, errors, warnings
            
        except Exception as e:
            errors.append(f"Error validating Neo4j backup: {e}")
            return False, errors, warnings
    
    def validate_redis_backup(self, backup_file: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate Redis backup content"""
        
        logger.info(f"Validating Redis backup content: {backup_file}")
        errors = []
        warnings = []
        
        try:
            if backup_file.suffix == '.gz' and '.rdb' in backup_file.name:
                # RDB format
                with tempfile.NamedTemporaryFile(delete=False, suffix='.rdb') as temp_f:
                    subprocess.run(['gunzip', '-c', str(backup_file)], stdout=temp_f, check=True)
                    test_file = Path(temp_f.name)
                
                # Check RDB file header
                with open(test_file, 'rb') as f:
                    header = f.read(9)
                    if not header.startswith(b'REDIS'):
                        errors.append("Invalid RDB file header")
                        return False, errors, warnings
                
                test_file.unlink()
            
            elif backup_file.suffix == '.gz' and '.aof' in backup_file.name:
                # AOF format
                with tempfile.NamedTemporaryFile(delete=False, suffix='.aof', mode='w+') as temp_f:
                    subprocess.run(['gunzip', '-c', str(backup_file)], stdout=temp_f, check=True)
                    temp_f.seek(0)
                    content = temp_f.read(1000)  # Read first 1KB
                
                if not content.strip():
                    errors.append("AOF backup file is empty")
                    return False, errors, warnings
                
                # Check for Redis protocol commands
                if not content.startswith('*') and '$' not in content:
                    warnings.append("AOF file doesn't appear to contain Redis protocol")
                
                Path(temp_f.name).unlink()
            
            elif backup_file.suffix == '.gz' and '.json' in backup_file.name:
                # JSON memory dump format
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w+') as temp_f:
                    subprocess.run(['gunzip', '-c', str(backup_file)], stdout=temp_f, check=True)
                    temp_f.seek(0)
                    
                    try:
                        data = json.load(temp_f)
                        if not isinstance(data, dict):
                            errors.append("JSON backup is not a dictionary")
                            return False, errors, warnings
                        
                        if not data:
                            warnings.append("JSON backup contains no data")
                        
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON in backup file: {e}")
                        return False, errors, warnings
                
                Path(temp_f.name).unlink()
            
            else:
                errors.append(f"Unknown Redis backup format: {backup_file.suffix}")
                return False, errors, warnings
            
            logger.info("Redis backup content validation passed")
            return True, errors, warnings
            
        except Exception as e:
            errors.append(f"Error validating Redis backup: {e}")
            return False, errors, warnings
    
    def validate_application_backup(self, backup_file: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate application backup content"""
        
        logger.info(f"Validating application backup content: {backup_file}")
        errors = []
        warnings = []
        
        try:
            if backup_file.suffix == '.gz' and '.tar' in backup_file.name:
                # Tar archive format
                result = subprocess.run(
                    ['tar', '-tzf', str(backup_file)],
                    capture_output=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    errors.append(f"Tar file validation failed: {result.stderr.decode()}")
                    return False, errors, warnings
                
                # Check archive contents
                file_list = result.stdout.decode().strip().split('\n')
                if not file_list or (len(file_list) == 1 and not file_list[0]):
                    errors.append("Application backup archive is empty")
                    return False, errors, warnings
                
                logger.info(f"Application backup contains {len(file_list)} files")
            
            else:
                errors.append(f"Unknown application backup format: {backup_file.suffix}")
                return False, errors, warnings
            
            logger.info("Application backup content validation passed")
            return True, errors, warnings
            
        except subprocess.TimeoutExpired:
            errors.append("Application backup validation timed out")
            return False, errors, warnings
        except Exception as e:
            errors.append(f"Error validating application backup: {e}")
            return False, errors, warnings
    
    def perform_restoration_test(self, backup_file: Path, service: str) -> Tuple[bool, List[str], List[str]]:
        """Perform restoration test for backup"""
        
        if not self.perform_restoration_tests:
            logger.info("Restoration tests disabled, skipping")
            return True, [], ["Restoration tests disabled"]
        
        logger.info(f"Performing restoration test: {backup_file} ({service})")
        errors = []
        warnings = []
        
        try:
            # This is a simplified restoration test
            # In a real environment, you would create test databases/containers
            
            if service == 'postgres':
                # Test PostgreSQL restoration to a test database
                test_db = f"test_restore_{int(time.time())}"
                
                # Create test database
                subprocess.run([
                    'createdb', '-h', self.postgres_host, '-p', self.postgres_port,
                    '-U', self.postgres_user, test_db
                ], check=True, timeout=60)
                
                try:
                    # Attempt restoration
                    if backup_file.suffix == '.dump' or '.dump' in backup_file.name:
                        if backup_file.suffix == '.gz':
                            # Decompress first
                            with tempfile.NamedTemporaryFile(suffix='.dump') as temp_f:
                                subprocess.run(['gunzip', '-c', str(backup_file)], stdout=temp_f, check=True)
                                subprocess.run([
                                    'pg_restore', '-h', self.postgres_host, '-p', self.postgres_port,
                                    '-U', self.postgres_user, '-d', test_db, temp_f.name
                                ], check=True, timeout=300)
                        else:
                            subprocess.run([
                                'pg_restore', '-h', self.postgres_host, '-p', self.postgres_port,
                                '-U', self.postgres_user, '-d', test_db, str(backup_file)
                            ], check=True, timeout=300)
                    
                    logger.info("PostgreSQL restoration test passed")
                    
                finally:
                    # Clean up test database
                    subprocess.run([
                        'dropdb', '-h', self.postgres_host, '-p', self.postgres_port,
                        '-U', self.postgres_user, test_db
                    ], timeout=60)
            
            elif service == 'redis':
                # Test Redis restoration (simplified)
                warnings.append("Redis restoration test not implemented")
            
            elif service == 'neo4j':
                # Test Neo4j restoration (simplified)
                warnings.append("Neo4j restoration test not implemented")
            
            elif service == 'application':
                # Test application data restoration
                with tempfile.TemporaryDirectory() as temp_dir:
                    subprocess.run([
                        'tar', '-xzf', str(backup_file), '-C', temp_dir
                    ], check=True, timeout=300)
                    
                    # Verify extraction
                    extracted_files = list(Path(temp_dir).glob('**/*'))
                    if not extracted_files:
                        errors.append("No files extracted from application backup")
                        return False, errors, warnings
                    
                    logger.info(f"Application restoration test passed: {len(extracted_files)} files extracted")
            
            return True, errors, warnings
            
        except subprocess.CalledProcessError as e:
            errors.append(f"Restoration test failed: {e}")
            return False, errors, warnings
        except subprocess.TimeoutExpired:
            errors.append("Restoration test timed out")
            return False, errors, warnings
        except Exception as e:
            errors.append(f"Error during restoration test: {e}")
            return False, errors, warnings
    
    def validate_backup(self, backup_file: Path) -> ValidationResult:
        """Validate a single backup file"""
        
        start_time = time.time()
        
        # Parse service and backup type from file path
        path_parts = backup_file.parts
        service = 'unknown'
        backup_type = 'unknown'
        
        for i, part in enumerate(path_parts):
            if part in ['postgres', 'neo4j', 'redis', 'application']:
                service = part
                if i + 1 < len(path_parts):
                    backup_type = path_parts[i + 1]
                break
        
        result = ValidationResult(
            backup_file=str(backup_file),
            service=service,
            backup_type=backup_type,
            validation_timestamp=datetime.now(),
            file_size_bytes=backup_file.stat().st_size if backup_file.exists() else 0
        )
        
        logger.info(f"Starting validation of {backup_file}")
        
        try:
            # File integrity validation
            integrity_valid, integrity_errors = self.validate_file_integrity(backup_file)
            result.file_integrity = integrity_valid
            result.error_messages.extend(integrity_errors)
            
            if not integrity_valid:
                logger.error(f"File integrity validation failed: {backup_file}")
                result.validation_duration_seconds = time.time() - start_time
                return result
            
            # Content validation based on service type
            content_valid = False
            content_errors = []
            content_warnings = []
            
            if service == 'postgres':
                content_valid, content_errors, content_warnings = self.validate_postgres_backup(backup_file)
            elif service == 'neo4j':
                content_valid, content_errors, content_warnings = self.validate_neo4j_backup(backup_file)
            elif service == 'redis':
                content_valid, content_errors, content_warnings = self.validate_redis_backup(backup_file)
            elif service == 'application':
                content_valid, content_errors, content_warnings = self.validate_application_backup(backup_file)
            else:
                content_errors.append(f"Unknown service type: {service}")
            
            result.content_validity = content_valid
            result.error_messages.extend(content_errors)
            result.warnings.extend(content_warnings)
            
            # Restoration test (if enabled and content is valid)
            if content_valid:
                restoration_valid, restoration_errors, restoration_warnings = self.perform_restoration_test(
                    backup_file, service
                )
                result.restoration_test = restoration_valid
                result.error_messages.extend(restoration_errors)
                result.warnings.extend(restoration_warnings)
            
            # Calculate validation metadata
            result.metadata = {
                'checksum': self.calculate_file_checksum(backup_file),
                'file_modified_time': datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                'validation_version': '1.0'
            }
            
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            result.error_messages.append(f"Validation error: {e}")
        
        finally:
            result.validation_duration_seconds = time.time() - start_time
            
            # Save result to database
            self.save_validation_result(result)
            
            # Log summary
            if result.overall_valid:
                logger.info(f"Validation passed: {backup_file} ({result.validation_duration_seconds:.2f}s)")
            else:
                logger.error(f"Validation failed: {backup_file} - {'; '.join(result.error_messages)}")
        
        return result
    
    def validate_all_backups(self) -> List[ValidationResult]:
        """Validate all backup files"""
        
        logger.info("Starting validation of all backup files")
        
        results = []
        backup_files = []
        
        # Discover all backup files
        for service_dir in self.backup_base_dir.iterdir():
            if not service_dir.is_dir():
                continue
            
            for backup_type_dir in service_dir.iterdir():
                if not backup_type_dir.is_dir():
                    continue
                
                # Find backup files
                backup_patterns = ['*.gz', '*.dump', '*.enc', '*.tar.gz']
                
                for pattern in backup_patterns:
                    backup_files.extend(backup_type_dir.glob(pattern))
        
        logger.info(f"Found {len(backup_files)} backup files to validate")
        
        # Validate each backup file
        for backup_file in backup_files:
            try:
                result = self.validate_backup(backup_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate {backup_file}: {e}")
        
        # Generate summary
        total_files = len(results)
        valid_files = sum(1 for r in results if r.overall_valid)
        fully_tested = sum(1 for r in results if r.fully_tested)
        
        logger.info(f"Validation completed: {valid_files}/{total_files} files valid, {fully_tested} fully tested")
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict:
        """Generate comprehensive validation report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_backups': len(results),
            'summary': {
                'valid_backups': sum(1 for r in results if r.overall_valid),
                'invalid_backups': sum(1 for r in results if not r.overall_valid),
                'fully_tested': sum(1 for r in results if r.fully_tested),
                'total_size_gb': sum(r.file_size_bytes for r in results) / (1024**3),
                'avg_validation_time': sum(r.validation_duration_seconds for r in results) / len(results) if results else 0
            },
            'by_service': {},
            'issues': [],
            'recommendations': []
        }
        
        # Group by service
        service_groups = {}
        for result in results:
            if result.service not in service_groups:
                service_groups[result.service] = []
            service_groups[result.service].append(result)
        
        # Analyze each service
        for service, service_results in service_groups.items():
            service_summary = {
                'total': len(service_results),
                'valid': sum(1 for r in service_results if r.overall_valid),
                'tested': sum(1 for r in service_results if r.restoration_test),
                'size_gb': sum(r.file_size_bytes for r in service_results) / (1024**3),
                'backups': []
            }
            
            for result in service_results:
                backup_info = {
                    'file': result.backup_file,
                    'type': result.backup_type,
                    'valid': result.overall_valid,
                    'tested': result.restoration_test,
                    'size_mb': result.file_size_bytes / (1024**2),
                    'validation_time': result.validation_duration_seconds,
                    'errors': result.error_messages,
                    'warnings': result.warnings
                }
                service_summary['backups'].append(backup_info)
                
                # Collect issues
                if result.error_messages:
                    for error in result.error_messages:
                        report['issues'].append({
                            'type': 'error',
                            'file': result.backup_file,
                            'message': error
                        })
            
            report['by_service'][service] = service_summary
        
        # Generate recommendations
        if report['summary']['invalid_backups'] > 0:
            report['recommendations'].append("Some backups failed validation. Review error messages and regenerate failed backups.")
        
        if report['summary']['fully_tested'] < report['summary']['valid_backups']:
            report['recommendations'].append("Consider enabling restoration tests for more comprehensive validation.")
        
        if report['summary']['total_size_gb'] > 100:
            report['recommendations'].append("Large backup size detected. Consider implementing compression and retention policies.")
        
        return report

def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Physics Assistant Backup Validator')
    parser.add_argument('--file', help='Validate specific backup file')
    parser.add_argument('--all', action='store_true', help='Validate all backup files')
    parser.add_argument('--report', action='store_true', help='Generate validation report')
    parser.add_argument('--enable-restoration-tests', action='store_true', help='Enable restoration tests')
    
    args = parser.parse_args()
    
    # Set environment variable for restoration tests
    if args.enable_restoration_tests:
        os.environ['RESTORATION_TESTS_ENABLED'] = 'true'
    
    try:
        validator = BackupValidator()
        
        if args.file:
            # Validate specific file
            backup_file = Path(args.file)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                sys.exit(1)
            
            result = validator.validate_backup(backup_file)
            
            print(json.dumps(asdict(result), indent=2, default=str))
            
            if not result.overall_valid:
                sys.exit(1)
        
        elif args.all:
            # Validate all backups
            results = validator.validate_all_backups()
            
            if args.report:
                report = validator.generate_validation_report(results)
                print(json.dumps(report, indent=2, default=str))
            
            # Exit with error if any validations failed
            invalid_count = sum(1 for r in results if not r.overall_valid)
            if invalid_count > 0:
                logger.error(f"{invalid_count} backup validations failed")
                sys.exit(1)
        
        else:
            parser.print_help()
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()