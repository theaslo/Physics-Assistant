#!/usr/bin/env python3
"""
Backup Monitoring System for Physics Assistant Platform

This script monitors backup operations, validates backup integrity,
and integrates with Prometheus and Alertmanager for alerting.
"""

import os
import sys
import json
import time
import logging
import requests
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/monitoring/backup_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('backup-monitor')

@dataclass
class BackupInfo:
    """Information about a backup file"""
    service: str
    backup_type: str
    file_path: str
    file_size: int
    timestamp: datetime
    checksum: Optional[str] = None
    encrypted: bool = False
    valid: bool = True
    age_hours: float = 0.0

@dataclass
class BackupMetrics:
    """Backup metrics for monitoring"""
    total_backups: int = 0
    failed_backups: int = 0
    successful_backups: int = 0
    total_size_bytes: int = 0
    oldest_backup_hours: float = 0.0
    newest_backup_hours: float = 0.0

class BackupMonitor:
    """Main backup monitoring class"""
    
    def __init__(self):
        self.backup_base_dir = Path(os.getenv('BACKUP_BASE_DIR', '/backups'))
        self.check_interval = int(os.getenv('BACKUP_CHECK_INTERVAL', '300'))  # 5 minutes
        self.max_backup_age_hours = float(os.getenv('BACKUP_MAX_AGE_HOURS', '25'))  # 25 hours
        self.min_backup_size_mb = float(os.getenv('BACKUP_MIN_SIZE_MB', '1'))  # 1 MB
        
        # Prometheus configuration
        self.metrics_port = int(os.getenv('METRICS_PORT', '8084'))
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        self.alertmanager_url = os.getenv('ALERTMANAGER_URL', 'http://alertmanager:9093')
        
        # Notification configuration
        self.webhook_url = os.getenv('WEBHOOK_URL')
        self.email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.email_to = os.getenv('EMAIL_TO')
        
        # Initialize Prometheus metrics
        self.registry = CollectorRegistry()
        self.init_prometheus_metrics()
        
        # Backup state tracking
        self.last_check_time = datetime.now()
        self.backup_states: Dict[str, BackupInfo] = {}
        self.alert_states: Dict[str, bool] = {}
        
        logger.info("Backup monitor initialized")
        logger.info(f"Monitoring directory: {self.backup_base_dir}")
        logger.info(f"Check interval: {self.check_interval} seconds")
        logger.info(f"Max backup age: {self.max_backup_age_hours} hours")
    
    def init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        
        # Backup status metrics
        self.backup_status_gauge = Gauge(
            'backup_status',
            'Backup status (1=healthy, 0=unhealthy)',
            ['service', 'backup_type'],
            registry=self.registry
        )
        
        self.backup_age_gauge = Gauge(
            'backup_age_hours',
            'Age of most recent backup in hours',
            ['service', 'backup_type'],
            registry=self.registry
        )
        
        self.backup_size_gauge = Gauge(
            'backup_size_bytes',
            'Size of most recent backup in bytes',
            ['service', 'backup_type'],
            registry=self.registry
        )
        
        self.backup_count_gauge = Gauge(
            'backup_count_total',
            'Total number of backups',
            ['service', 'backup_type'],
            registry=self.registry
        )
        
        # Monitoring metrics
        self.monitor_checks_counter = Counter(
            'backup_monitor_checks_total',
            'Total number of backup checks performed',
            registry=self.registry
        )
        
        self.monitor_errors_counter = Counter(
            'backup_monitor_errors_total',
            'Total number of monitoring errors',
            ['error_type'],
            registry=self.registry
        )
        
        self.monitor_alerts_counter = Counter(
            'backup_monitor_alerts_total',
            'Total number of alerts sent',
            ['alert_type'],
            registry=self.registry
        )
        
        # Check duration histogram
        self.check_duration_histogram = Histogram(
            'backup_monitor_check_duration_seconds',
            'Time spent performing backup checks',
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized")
    
    def discover_backups(self) -> List[BackupInfo]:
        """Discover all backup files in the backup directory"""
        backups = []
        
        try:
            if not self.backup_base_dir.exists():
                logger.warning(f"Backup directory does not exist: {self.backup_base_dir}")
                return backups
            
            # Scan for backup files
            for service_dir in self.backup_base_dir.iterdir():
                if not service_dir.is_dir():
                    continue
                
                service_name = service_dir.name
                
                for backup_type_dir in service_dir.iterdir():
                    if not backup_type_dir.is_dir():
                        continue
                    
                    backup_type = backup_type_dir.name
                    
                    # Find backup files
                    backup_patterns = ['*.gz', '*.dump', '*.enc', '*.tar.gz']
                    
                    for pattern in backup_patterns:
                        for backup_file in backup_type_dir.glob(pattern):
                            if backup_file.is_file():
                                backup_info = self.analyze_backup_file(
                                    service_name, backup_type, backup_file
                                )
                                if backup_info:
                                    backups.append(backup_info)
            
            logger.info(f"Discovered {len(backups)} backup files")
            
        except Exception as e:
            logger.error(f"Error discovering backups: {e}")
            self.monitor_errors_counter.labels(error_type='discovery').inc()
        
        return backups
    
    def analyze_backup_file(self, service: str, backup_type: str, file_path: Path) -> Optional[BackupInfo]:
        """Analyze a backup file and extract information"""
        try:
            stat = file_path.stat()
            file_size = stat.st_size
            timestamp = datetime.fromtimestamp(stat.st_mtime)
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            
            # Check if file is encrypted
            encrypted = file_path.suffix == '.enc'
            
            # Try to read checksum from metadata
            checksum = None
            metadata_file = file_path.with_suffix(file_path.suffix + '.metadata.json')
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        checksum = metadata.get('checksum')
                except Exception as e:
                    logger.warning(f"Failed to read metadata for {file_path}: {e}")
            
            # Validate backup file
            valid = self.validate_backup_file(file_path, file_size, age_hours)
            
            return BackupInfo(
                service=service,
                backup_type=backup_type,
                file_path=str(file_path),
                file_size=file_size,
                timestamp=timestamp,
                checksum=checksum,
                encrypted=encrypted,
                valid=valid,
                age_hours=age_hours
            )
            
        except Exception as e:
            logger.error(f"Error analyzing backup file {file_path}: {e}")
            return None
    
    def validate_backup_file(self, file_path: Path, file_size: int, age_hours: float) -> bool:
        """Validate a backup file"""
        
        # Check file size
        min_size_bytes = self.min_backup_size_mb * 1024 * 1024
        if file_size < min_size_bytes:
            logger.warning(f"Backup file too small: {file_path} ({file_size} bytes)")
            return False
        
        # Check file age
        if age_hours > self.max_backup_age_hours:
            logger.warning(f"Backup file too old: {file_path} ({age_hours:.1f} hours)")
            return False
        
        # Check file integrity (basic)
        if not file_path.exists():
            logger.error(f"Backup file does not exist: {file_path}")
            return False
        
        # For compressed files, try to test integrity
        if file_path.suffix == '.gz':
            try:
                result = subprocess.run(
                    ['gzip', '-t', str(file_path)],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.error(f"Gzip integrity test failed for {file_path}")
                    return False
            except Exception as e:
                logger.warning(f"Could not test gzip integrity for {file_path}: {e}")
        
        return True
    
    def update_prometheus_metrics(self, backups: List[BackupInfo]):
        """Update Prometheus metrics with backup information"""
        
        # Group backups by service and type
        backup_groups = {}
        for backup in backups:
            key = (backup.service, backup.backup_type)
            if key not in backup_groups:
                backup_groups[key] = []
            backup_groups[key].append(backup)
        
        # Update metrics for each service/type combination
        for (service, backup_type), group_backups in backup_groups.items():
            
            # Find most recent backup
            recent_backup = max(group_backups, key=lambda b: b.timestamp)
            
            # Update status (healthy if recent backup exists and is valid)
            status = 1 if recent_backup.valid and recent_backup.age_hours <= self.max_backup_age_hours else 0
            self.backup_status_gauge.labels(service=service, backup_type=backup_type).set(status)
            
            # Update age
            self.backup_age_gauge.labels(service=service, backup_type=backup_type).set(recent_backup.age_hours)
            
            # Update size
            self.backup_size_gauge.labels(service=service, backup_type=backup_type).set(recent_backup.file_size)
            
            # Update count
            self.backup_count_gauge.labels(service=service, backup_type=backup_type).set(len(group_backups))
        
        logger.debug(f"Updated Prometheus metrics for {len(backup_groups)} backup groups")
    
    def check_backup_alerts(self, backups: List[BackupInfo]):
        """Check for backup issues that require alerting"""
        
        alerts_to_send = []
        
        # Group backups by service and type
        backup_groups = {}
        for backup in backups:
            key = (backup.service, backup.backup_type)
            if key not in backup_groups:
                backup_groups[key] = []
            backup_groups[key].append(backup)
        
        # Expected backup types for each service
        expected_backups = {
            'postgres': ['daily'],
            'neo4j': ['daily'],
            'redis': ['daily'],
            'application': ['daily']
        }
        
        # Check each expected backup
        for service, backup_types in expected_backups.items():
            for backup_type in backup_types:
                key = (service, backup_type)
                alert_key = f"{service}_{backup_type}"
                
                if key not in backup_groups:
                    # Missing backup
                    alert = {
                        'type': 'missing_backup',
                        'service': service,
                        'backup_type': backup_type,
                        'message': f"No {backup_type} backups found for {service}",
                        'severity': 'critical'
                    }
                    alerts_to_send.append(alert)
                    
                    # Track alert state
                    if alert_key not in self.alert_states or not self.alert_states[alert_key]:
                        self.alert_states[alert_key] = True
                        logger.error(alert['message'])
                
                else:
                    group_backups = backup_groups[key]
                    recent_backup = max(group_backups, key=lambda b: b.timestamp)
                    
                    # Check for stale backups
                    if recent_backup.age_hours > self.max_backup_age_hours:
                        alert = {
                            'type': 'stale_backup',
                            'service': service,
                            'backup_type': backup_type,
                            'message': f"Backup for {service} ({backup_type}) is stale: {recent_backup.age_hours:.1f} hours old",
                            'severity': 'warning'
                        }
                        alerts_to_send.append(alert)
                        
                        if alert_key not in self.alert_states or not self.alert_states[alert_key]:
                            self.alert_states[alert_key] = True
                            logger.warning(alert['message'])
                    
                    # Check for invalid backups
                    elif not recent_backup.valid:
                        alert = {
                            'type': 'invalid_backup',
                            'service': service,
                            'backup_type': backup_type,
                            'message': f"Backup for {service} ({backup_type}) is invalid",
                            'severity': 'critical'
                        }
                        alerts_to_send.append(alert)
                        
                        if alert_key not in self.alert_states or not self.alert_states[alert_key]:
                            self.alert_states[alert_key] = True
                            logger.error(alert['message'])
                    
                    else:
                        # Backup is healthy, clear alert state
                        if alert_key in self.alert_states and self.alert_states[alert_key]:
                            self.alert_states[alert_key] = False
                            logger.info(f"Backup for {service} ({backup_type}) is now healthy")
        
        # Send alerts
        for alert in alerts_to_send:
            self.send_alert(alert)
    
    def send_alert(self, alert: Dict):
        """Send an alert notification"""
        
        try:
            # Send to webhook if configured
            if self.webhook_url:
                self.send_webhook_alert(alert)
            
            # Send email if configured
            if self.email_enabled and self.email_to:
                self.send_email_alert(alert)
            
            # Send to Alertmanager
            self.send_alertmanager_alert(alert)
            
            # Update metrics
            self.monitor_alerts_counter.labels(alert_type=alert['type']).inc()
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            self.monitor_errors_counter.labels(error_type='alert_sending').inc()
    
    def send_webhook_alert(self, alert: Dict):
        """Send alert to webhook"""
        
        color = 'danger' if alert['severity'] == 'critical' else 'warning'
        
        payload = {
            'text': f"Backup Alert: {alert['message']}",
            'color': color,
            'fields': [
                {'title': 'Service', 'value': alert['service'], 'short': True},
                {'title': 'Type', 'value': alert['backup_type'], 'short': True},
                {'title': 'Severity', 'value': alert['severity'], 'short': True},
                {'title': 'Time', 'value': datetime.now().isoformat(), 'short': True}
            ]
        }
        
        response = requests.post(self.webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        
        logger.info(f"Webhook alert sent for {alert['type']}")
    
    def send_email_alert(self, alert: Dict):
        """Send alert via email"""
        
        subject = f"Physics Assistant Backup Alert: {alert['service']} ({alert['severity']})"
        body = f"""
Backup Alert Report
==================

Service: {alert['service']}
Backup Type: {alert['backup_type']}
Alert Type: {alert['type']}
Severity: {alert['severity']}
Message: {alert['message']}
Timestamp: {datetime.now().isoformat()}
Host: {os.getenv('HOSTNAME', 'unknown')}

This is an automated alert from the Physics Assistant backup monitoring system.
Please investigate and resolve the issue promptly.
"""
        
        # Send email using sendmail or similar
        try:
            import subprocess
            
            proc = subprocess.Popen(
                ['mail', '-s', subject, self.email_to],
                stdin=subprocess.PIPE,
                text=True
            )
            proc.communicate(input=body)
            
            if proc.returncode == 0:
                logger.info(f"Email alert sent for {alert['type']}")
            else:
                logger.error(f"Failed to send email alert: return code {proc.returncode}")
                
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_alertmanager_alert(self, alert: Dict):
        """Send alert to Alertmanager"""
        
        alert_payload = {
            'labels': {
                'alertname': f"BackupAlert{alert['type'].title()}",
                'service': alert['service'],
                'backup_type': alert['backup_type'],
                'severity': alert['severity'],
                'instance': os.getenv('HOSTNAME', 'backup-monitor')
            },
            'annotations': {
                'summary': alert['message'],
                'description': f"Backup issue detected for {alert['service']} ({alert['backup_type']})"
            },
            'startsAt': datetime.now().isoformat() + 'Z'
        }
        
        try:
            response = requests.post(
                f"{self.alertmanager_url}/api/v1/alerts",
                json=[alert_payload],
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Alertmanager alert sent for {alert['type']}")
            
        except Exception as e:
            logger.warning(f"Failed to send Alertmanager alert: {e}")
    
    def generate_backup_report(self, backups: List[BackupInfo]) -> Dict:
        """Generate a comprehensive backup status report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_backups': len(backups),
            'services': {},
            'overall_status': 'healthy',
            'issues': []
        }
        
        # Group by service
        service_groups = {}
        for backup in backups:
            if backup.service not in service_groups:
                service_groups[backup.service] = []
            service_groups[backup.service].append(backup)
        
        # Analyze each service
        for service, service_backups in service_groups.items():
            service_info = {
                'backup_count': len(service_backups),
                'total_size_mb': sum(b.file_size for b in service_backups) / (1024 * 1024),
                'backup_types': {},
                'status': 'healthy'
            }
            
            # Group by backup type
            type_groups = {}
            for backup in service_backups:
                if backup.backup_type not in type_groups:
                    type_groups[backup.backup_type] = []
                type_groups[backup.backup_type].append(backup)
            
            # Analyze each backup type
            for backup_type, type_backups in type_groups.items():
                recent_backup = max(type_backups, key=lambda b: b.timestamp)
                
                type_info = {
                    'count': len(type_backups),
                    'most_recent': recent_backup.timestamp.isoformat(),
                    'age_hours': recent_backup.age_hours,
                    'size_mb': recent_backup.file_size / (1024 * 1024),
                    'valid': recent_backup.valid,
                    'encrypted': recent_backup.encrypted
                }
                
                # Check for issues
                if recent_backup.age_hours > self.max_backup_age_hours:
                    type_info['status'] = 'stale'
                    service_info['status'] = 'warning'
                    report['overall_status'] = 'warning'
                    report['issues'].append(f"{service} {backup_type} backup is stale")
                elif not recent_backup.valid:
                    type_info['status'] = 'invalid'
                    service_info['status'] = 'error'
                    report['overall_status'] = 'error'
                    report['issues'].append(f"{service} {backup_type} backup is invalid")
                else:
                    type_info['status'] = 'healthy'
                
                service_info['backup_types'][backup_type] = type_info
            
            report['services'][service] = service_info
        
        return report
    
    def run_check(self):
        """Run a single backup check cycle"""
        
        start_time = time.time()
        
        try:
            logger.info("Starting backup check cycle")
            
            # Discover backups
            backups = self.discover_backups()
            
            # Update Prometheus metrics
            self.update_prometheus_metrics(backups)
            
            # Check for alerts
            self.check_backup_alerts(backups)
            
            # Generate report
            report = self.generate_backup_report(backups)
            
            # Save report
            report_file = Path('/logs/monitoring') / f"backup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Update check counter
            self.monitor_checks_counter.inc()
            
            # Record check duration
            duration = time.time() - start_time
            self.check_duration_histogram.observe(duration)
            
            logger.info(f"Backup check completed in {duration:.2f} seconds")
            logger.info(f"Status: {report['overall_status']}, Backups: {report['total_backups']}, Issues: {len(report['issues'])}")
            
        except Exception as e:
            logger.error(f"Error during backup check: {e}")
            self.monitor_errors_counter.labels(error_type='check_cycle').inc()
    
    def start_monitoring(self):
        """Start the backup monitoring service"""
        
        logger.info("Starting backup monitoring service")
        
        # Start Prometheus metrics server
        start_http_server(self.metrics_port, registry=self.registry)
        logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        
        # Schedule regular checks
        schedule.every(self.check_interval).seconds.do(self.run_check)
        
        # Run initial check
        self.run_check()
        
        # Main monitoring loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds for scheduled jobs
                
        except KeyboardInterrupt:
            logger.info("Backup monitoring service stopped by user")
        except Exception as e:
            logger.error(f"Backup monitoring service error: {e}")
            raise

def main():
    """Main entry point"""
    
    try:
        monitor = BackupMonitor()
        monitor.start_monitoring()
        
    except Exception as e:
        logger.error(f"Failed to start backup monitor: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()