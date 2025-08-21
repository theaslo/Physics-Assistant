#!/usr/bin/env python3
"""
Security Incident Response and Monitoring Service
Automated threat detection, incident response, and security orchestration
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import hashlib
import ipaddress
import re

import aiohttp
import asyncpg
import redis.asyncio as redis
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import networkx as nx
import psutil
import geoip2.database
import yara
from cryptography.fernet import Fernet

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
SECURITY_INCIDENTS = Counter('security_incidents_total', 'Total security incidents', ['severity', 'type'])
THREAT_DETECTIONS = Counter('threat_detections_total', 'Total threat detections', ['threat_type'])
RESPONSE_TIME = Histogram('incident_response_time_seconds', 'Time to respond to incidents', ['severity'])
ACTIVE_THREATS = Gauge('active_threats', 'Currently active threats')
BLOCKED_IPS = Gauge('blocked_ips_total', 'Total blocked IP addresses')

class ThreatIntelligence:
    """Manages threat intelligence and IOCs"""
    
    def __init__(self):
        self.malicious_ips = set()
        self.malicious_domains = set()
        self.malicious_hashes = set()
        self.threat_feeds = []
        self.geoip_reader = None
        
    async def initialize(self):
        """Initialize threat intelligence sources"""
        logger.info("Initializing threat intelligence")
        
        # Load GeoIP database if available
        try:
            if os.path.exists('/opt/geoip/GeoLite2-City.mmdb'):
                self.geoip_reader = geoip2.database.Reader('/opt/geoip/GeoLite2-City.mmdb')
        except Exception as e:
            logger.warning("Failed to load GeoIP database", error=str(e))
        
        # Load initial threat feeds
        await self.update_threat_feeds()
        
    async def update_threat_feeds(self):
        """Update threat intelligence feeds"""
        logger.info("Updating threat intelligence feeds")
        
        feeds = [
            {
                'name': 'abuse_ch_malware_ips',
                'url': 'https://feodotracker.abuse.ch/downloads/ipblocklist.txt',
                'type': 'ip'
            },
            {
                'name': 'spamhaus_drop',
                'url': 'https://www.spamhaus.org/drop/drop.txt',
                'type': 'ip'
            },
            {
                'name': 'emergingthreats_compromised',
                'url': 'https://rules.emergingthreats.net/fwrules/emerging-Block-IPs.txt',
                'type': 'ip'
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            for feed in feeds:
                try:
                    async with session.get(feed['url'], timeout=30) as response:
                        if response.status == 200:
                            content = await response.text()
                            await self.parse_threat_feed(content, feed['type'])
                            logger.info("Updated threat feed", feed=feed['name'])
                except Exception as e:
                    logger.warning("Failed to update threat feed", feed=feed['name'], error=str(e))
    
    async def parse_threat_feed(self, content: str, feed_type: str):
        """Parse threat feed content"""
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if feed_type == 'ip':
                # Extract IP addresses or CIDR blocks
                ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?)', line)
                if ip_match:
                    ip_str = ip_match.group(1)
                    try:
                        if '/' in ip_str:
                            # CIDR block
                            network = ipaddress.ip_network(ip_str, strict=False)
                            for ip in network.hosts():
                                self.malicious_ips.add(str(ip))
                        else:
                            # Single IP
                            ipaddress.ip_address(ip_str)  # Validate
                            self.malicious_ips.add(ip_str)
                    except ValueError:
                        continue
    
    def is_malicious_ip(self, ip: str) -> bool:
        """Check if IP is known malicious"""
        return ip in self.malicious_ips
    
    def is_malicious_domain(self, domain: str) -> bool:
        """Check if domain is known malicious"""
        return domain in self.malicious_domains
    
    def get_ip_geolocation(self, ip: str) -> Optional[Dict[str, Any]]:
        """Get geolocation information for IP"""
        if not self.geoip_reader:
            return None
            
        try:
            response = self.geoip_reader.city(ip)
            return {
                'country': response.country.name,
                'city': response.city.name,
                'latitude': float(response.location.latitude) if response.location.latitude else None,
                'longitude': float(response.location.longitude) if response.location.longitude else None,
                'accuracy': response.location.accuracy_radius
            }
        except Exception:
            return None

class AnomalyDetector:
    """Detects security anomalies using machine learning"""
    
    def __init__(self):
        self.models = {}
        self.training_data = {}
        self.baseline_established = False
        
    async def train_models(self, data: Dict[str, List[Dict]]):
        """Train anomaly detection models"""
        logger.info("Training anomaly detection models")
        
        for data_type, records in data.items():
            if not records:
                continue
                
            try:
                # Convert to DataFrame
                df = pd.DataFrame(records)
                
                # Feature engineering based on data type
                if data_type == 'network_traffic':
                    features = self.extract_network_features(df)
                elif data_type == 'api_requests':
                    features = self.extract_api_features(df)
                elif data_type == 'auth_events':
                    features = self.extract_auth_features(df)
                else:
                    continue
                
                if features is not None and len(features) > 10:
                    # Train Isolation Forest for anomaly detection
                    model = IsolationForest(
                        contamination=0.1,
                        random_state=42,
                        n_estimators=100
                    )
                    model.fit(features)
                    self.models[data_type] = model
                    
                    logger.info("Trained model", data_type=data_type, samples=len(features))
                    
            except Exception as e:
                logger.error("Failed to train model", data_type=data_type, error=str(e))
        
        self.baseline_established = True
    
    def extract_network_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features from network traffic data"""
        try:
            features = []
            
            # Group by source IP and extract statistical features
            for src_ip, group in df.groupby('src_ip'):
                feature_vector = [
                    len(group),  # Number of connections
                    group['bytes_sent'].sum(),  # Total bytes sent
                    group['bytes_received'].sum(),  # Total bytes received
                    group['duration'].mean(),  # Average connection duration
                    group['duration'].std(),  # Duration variance
                    len(group['dst_port'].unique()),  # Unique destination ports
                    len(group['dst_ip'].unique()),  # Unique destination IPs
                ]
                features.append(feature_vector)
            
            return np.array(features) if features else None
        except Exception as e:
            logger.error("Failed to extract network features", error=str(e))
            return None
    
    def extract_api_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features from API request data"""
        try:
            features = []
            
            # Group by source IP and extract features
            for src_ip, group in df.groupby('src_ip'):
                feature_vector = [
                    len(group),  # Number of requests
                    group['response_time'].mean(),  # Average response time
                    group['response_time'].std(),  # Response time variance
                    len(group['endpoint'].unique()),  # Unique endpoints
                    len(group[group['status_code'] >= 400]),  # Error requests
                    group['request_size'].sum(),  # Total request size
                    group['response_size'].sum(),  # Total response size
                ]
                features.append(feature_vector)
            
            return np.array(features) if features else None
        except Exception as e:
            logger.error("Failed to extract API features", error=str(e))
            return None
    
    def extract_auth_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features from authentication events"""
        try:
            features = []
            
            # Group by user and extract features
            for user, group in df.groupby('username'):
                feature_vector = [
                    len(group),  # Number of auth attempts
                    len(group[group['success'] == True]),  # Successful auths
                    len(group[group['success'] == False]),  # Failed auths
                    len(group['src_ip'].unique()),  # Unique source IPs
                    group['session_duration'].mean(),  # Average session duration
                ]
                features.append(feature_vector)
            
            return np.array(features) if features else None
        except Exception as e:
            logger.error("Failed to extract auth features", error=str(e))
            return None
    
    async def detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in current data"""
        if not self.baseline_established:
            return []
        
        anomalies = []
        
        for data_type, records in data.items():
            if data_type not in self.models or not records:
                continue
                
            try:
                df = pd.DataFrame(records)
                
                # Extract features
                if data_type == 'network_traffic':
                    features = self.extract_network_features(df)
                elif data_type == 'api_requests':
                    features = self.extract_api_features(df)
                elif data_type == 'auth_events':
                    features = self.extract_auth_features(df)
                else:
                    continue
                
                if features is not None:
                    # Predict anomalies
                    predictions = self.models[data_type].predict(features)
                    anomaly_scores = self.models[data_type].decision_function(features)
                    
                    # Identify anomalies
                    for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):
                        if prediction == -1:  # Anomaly detected
                            anomalies.append({
                                'type': f'{data_type}_anomaly',
                                'severity': 'high' if score < -0.5 else 'medium',
                                'score': float(score),
                                'data_type': data_type,
                                'timestamp': datetime.utcnow().isoformat(),
                                'details': f"Anomaly detected in {data_type} data with score {score:.3f}"
                            })
                            
            except Exception as e:
                logger.error("Failed to detect anomalies", data_type=data_type, error=str(e))
        
        return anomalies

class IncidentManager:
    """Manages security incidents and responses"""
    
    def __init__(self, db_connections):
        self.db = db_connections
        self.active_incidents = {}
        self.incident_history = []
        self.response_playbooks = {}
        
    async def create_incident(self, incident_data: Dict[str, Any]) -> str:
        """Create a new security incident"""
        incident_id = hashlib.sha256(
            f"{incident_data['type']}_{incident_data['timestamp']}_{incident_data.get('source_ip', '')}".encode()
        ).hexdigest()[:16]
        
        incident = {
            'id': incident_id,
            'created_at': datetime.utcnow().isoformat(),
            'type': incident_data['type'],
            'severity': incident_data.get('severity', 'medium'),
            'status': 'open',
            'source_ip': incident_data.get('source_ip'),
            'details': incident_data.get('details', ''),
            'evidence': incident_data.get('evidence', []),
            'response_actions': [],
            'false_positive': False
        }
        
        self.active_incidents[incident_id] = incident
        
        # Record metrics
        SECURITY_INCIDENTS.labels(
            severity=incident['severity'],
            type=incident['type']
        ).inc()
        
        logger.info("Created security incident", incident_id=incident_id, type=incident['type'])
        
        # Trigger automated response
        await self.trigger_response(incident)
        
        return incident_id
    
    async def trigger_response(self, incident: Dict[str, Any]):
        """Trigger automated incident response"""
        start_time = time.time()
        
        try:
            incident_type = incident['type']
            severity = incident['severity']
            
            # Execute appropriate response playbook
            if incident_type in ['malicious_ip', 'suspicious_activity']:
                await self.block_malicious_ip(incident)
            elif incident_type in ['brute_force', 'auth_anomaly']:
                await self.handle_auth_incident(incident)
            elif incident_type in ['vulnerability_scan', 'port_scan']:
                await self.handle_scan_incident(incident)
            elif incident_type in ['malware_detected', 'suspicious_file']:
                await self.handle_malware_incident(incident)
            
            # Send notifications based on severity
            if severity in ['high', 'critical']:
                await self.send_immediate_alert(incident)
            elif severity == 'medium':
                await self.send_notification(incident)
            
            # Update incident with response actions
            incident['status'] = 'responding'
            incident['response_time'] = time.time() - start_time
            
            RESPONSE_TIME.labels(severity=severity).observe(incident['response_time'])
            
        except Exception as e:
            logger.error("Failed to trigger incident response", incident_id=incident['id'], error=str(e))
    
    async def block_malicious_ip(self, incident: Dict[str, Any]):
        """Block malicious IP address"""
        source_ip = incident.get('source_ip')
        if not source_ip:
            return
        
        try:
            # Add to Redis blocklist
            await self.db.redis_client.sadd('blocked_ips', source_ip)
            await self.db.redis_client.setex(f'block_reason:{source_ip}', 86400, incident['type'])
            
            # Update firewall rules (would require proper implementation)
            # This is a placeholder for actual firewall integration
            logger.info("Blocked malicious IP", ip=source_ip, incident_id=incident['id'])
            
            incident['response_actions'].append({
                'action': 'block_ip',
                'target': source_ip,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            BLOCKED_IPS.inc()
            
        except Exception as e:
            logger.error("Failed to block IP", ip=source_ip, error=str(e))
    
    async def handle_auth_incident(self, incident: Dict[str, Any]):
        """Handle authentication-related incidents"""
        try:
            # Temporarily lock account or increase security measures
            logger.info("Handling auth incident", incident_id=incident['id'])
            
            incident['response_actions'].append({
                'action': 'increase_auth_security',
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error("Failed to handle auth incident", error=str(e))
    
    async def handle_scan_incident(self, incident: Dict[str, Any]):
        """Handle scanning incidents"""
        try:
            source_ip = incident.get('source_ip')
            
            # Rate limit or temporarily block scanner
            if source_ip:
                await self.db.redis_client.setex(f'rate_limit:{source_ip}', 3600, '1')
            
            logger.info("Handling scan incident", incident_id=incident['id'])
            
            incident['response_actions'].append({
                'action': 'rate_limit_scanner',
                'target': source_ip,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error("Failed to handle scan incident", error=str(e))
    
    async def handle_malware_incident(self, incident: Dict[str, Any]):
        """Handle malware detection incidents"""
        try:
            # Isolate affected systems or files
            logger.info("Handling malware incident", incident_id=incident['id'])
            
            incident['response_actions'].append({
                'action': 'isolate_threat',
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error("Failed to handle malware incident", error=str(e))
    
    async def send_immediate_alert(self, incident: Dict[str, Any]):
        """Send immediate alert for critical incidents"""
        try:
            alert_data = {
                'incident_id': incident['id'],
                'type': incident['type'],
                'severity': incident['severity'],
                'timestamp': incident['created_at'],
                'details': incident['details']
            }
            
            # Send to multiple channels
            await self.send_slack_alert(alert_data)
            await self.send_email_alert(alert_data)
            await self.send_webhook_alert(alert_data)
            
        except Exception as e:
            logger.error("Failed to send immediate alert", error=str(e))
    
    async def send_notification(self, incident: Dict[str, Any]):
        """Send regular notification"""
        try:
            # Send to monitoring systems
            await self.send_webhook_alert({
                'incident_id': incident['id'],
                'type': incident['type'],
                'severity': incident['severity']
            })
            
        except Exception as e:
            logger.error("Failed to send notification", error=str(e))
    
    async def send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Slack"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not webhook_url:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'text': f"🚨 Security Incident Alert",
                    'attachments': [{
                        'color': 'danger' if alert_data['severity'] in ['high', 'critical'] else 'warning',
                        'fields': [
                            {'title': 'Incident ID', 'value': alert_data['incident_id'], 'short': True},
                            {'title': 'Type', 'value': alert_data['type'], 'short': True},
                            {'title': 'Severity', 'value': alert_data['severity'].upper(), 'short': True},
                            {'title': 'Time', 'value': alert_data['timestamp'], 'short': True},
                            {'title': 'Details', 'value': alert_data['details'], 'short': False}
                        ]
                    }]
                }
                
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack alert sent", incident_id=alert_data['incident_id'])
                    
        except Exception as e:
            logger.error("Failed to send Slack alert", error=str(e))
    
    async def send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Send alert to webhook"""
        webhook_url = os.getenv('ALERT_WEBHOOK_URL')
        if not webhook_url:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=alert_data) as response:
                    if response.status == 200:
                        logger.info("Webhook alert sent", incident_id=alert_data['incident_id'])
                    
        except Exception as e:
            logger.error("Failed to send webhook alert", error=str(e))
    
    async def send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert"""
        # Implementation would depend on email service provider
        logger.info("Email alert would be sent", incident_id=alert_data['incident_id'])

class SecurityMonitor:
    """Main security monitoring and response orchestrator"""
    
    def __init__(self):
        self.db_connections = None
        self.threat_intel = ThreatIntelligence()
        self.anomaly_detector = AnomalyDetector()
        self.incident_manager = None
        self.scheduler = AsyncIOScheduler()
        
    async def initialize(self):
        """Initialize the security monitoring service"""
        logger.info("Initializing security monitoring service")
        
        # Initialize components
        await self.threat_intel.initialize()
        self.incident_manager = IncidentManager(self.db_connections)
        
        # Schedule periodic tasks
        self.scheduler.add_job(
            self.collect_security_data,
            'interval',
            minutes=5,
            id='security_data_collection'
        )
        
        self.scheduler.add_job(
            self.threat_intel.update_threat_feeds,
            'interval',
            hours=6,
            id='threat_feed_update'
        )
        
        self.scheduler.add_job(
            self.analyze_security_trends,
            'interval',
            hours=1,
            id='security_trend_analysis'
        )
        
        self.scheduler.start()
        
        # Start metrics server
        start_http_server(8080)
        
        logger.info("Security monitoring service initialized")
    
    async def collect_security_data(self):
        """Collect security-related data for analysis"""
        try:
            # Collect various types of security data
            security_data = {
                'network_traffic': await self.collect_network_data(),
                'api_requests': await self.collect_api_data(),
                'auth_events': await self.collect_auth_data(),
                'system_events': await self.collect_system_data()
            }
            
            # Detect anomalies
            anomalies = await self.anomaly_detector.detect_anomalies(security_data)
            
            # Process detected anomalies
            for anomaly in anomalies:
                await self.process_security_event(anomaly)
            
        except Exception as e:
            logger.error("Failed to collect security data", error=str(e))
    
    async def collect_network_data(self) -> List[Dict[str, Any]]:
        """Collect network traffic data"""
        # Placeholder - would integrate with network monitoring tools
        return []
    
    async def collect_api_data(self) -> List[Dict[str, Any]]:
        """Collect API request data"""
        # Placeholder - would collect from API logs
        return []
    
    async def collect_auth_data(self) -> List[Dict[str, Any]]:
        """Collect authentication event data"""
        # Placeholder - would collect from auth logs
        return []
    
    async def collect_system_data(self) -> List[Dict[str, Any]]:
        """Collect system event data"""
        # Placeholder - would collect from system logs
        return []
    
    async def process_security_event(self, event: Dict[str, Any]):
        """Process a security event"""
        try:
            # Enrich event with threat intelligence
            if 'source_ip' in event:
                ip = event['source_ip']
                if self.threat_intel.is_malicious_ip(ip):
                    event['threat_intel'] = 'known_malicious_ip'
                    event['severity'] = 'high'
                
                geo_info = self.threat_intel.get_ip_geolocation(ip)
                if geo_info:
                    event['geolocation'] = geo_info
            
            # Create incident for significant events
            if event.get('severity') in ['high', 'critical']:
                await self.incident_manager.create_incident(event)
            
            THREAT_DETECTIONS.labels(threat_type=event.get('type', 'unknown')).inc()
            
        except Exception as e:
            logger.error("Failed to process security event", error=str(e))
    
    async def analyze_security_trends(self):
        """Analyze security trends and patterns"""
        try:
            # Analyze incident patterns over time
            # This would include trend analysis, pattern recognition, etc.
            logger.info("Analyzing security trends")
            
        except Exception as e:
            logger.error("Failed to analyze security trends", error=str(e))
    
    async def run(self):
        """Run the security monitoring service"""
        logger.info("Starting security monitoring service")
        
        try:
            await self.initialize()
            
            # Keep the service running
            while True:
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error("Service error", error=str(e))
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the security monitoring service"""
        logger.info("Shutting down security monitoring service")
        
        if self.scheduler:
            self.scheduler.shutdown()

if __name__ == "__main__":
    service = SecurityMonitor()
    asyncio.run(service.run())