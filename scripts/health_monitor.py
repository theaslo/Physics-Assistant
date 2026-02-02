#!/usr/bin/env python3
"""
Advanced Health Monitoring System for Physics Assistant
Monitors all components and provides alerting capabilities
"""

import asyncio
import aiohttp
import time
import json
import subprocess
import logging
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/health_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HealthMonitor:
    def __init__(self):
        self.mcp_servers = {
            'forces': {'url': 'http://localhost:10100', 'status': 'unknown'},
            'kinematics': {'url': 'http://localhost:10101', 'status': 'unknown'},
            'math': {'url': 'http://localhost:10103', 'status': 'unknown'},
            'momentum': {'url': 'http://localhost:10104', 'status': 'unknown'},
            'energy': {'url': 'http://localhost:10105', 'status': 'unknown'},
            'angular-motion': {'url': 'http://localhost:10106', 'status': 'unknown'}
        }

        self.alert_thresholds = {
            'response_time_ms': 2000,
            'memory_usage_percent': 80,
            'cpu_usage_percent': 85,
            'error_rate_percent': 5,
            'consecutive_failures': 3
        }

        self.failure_counts = {server: 0 for server in self.mcp_servers.keys()}
        self.last_alert_time = {}
        self.alert_cooldown_minutes = 15

        # Load configuration from environment
        self.load_config()

    def load_config(self):
        """Load configuration from environment variables"""
        self.smtp_host = os.getenv('SMTP_HOST', '')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.alert_email = os.getenv('ALERT_EMAIL', '')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL', '')

    async def check_mcp_server(self, server_name: str, server_info: dict) -> dict:
        """Check health of individual MCP server"""
        start_time = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(server_info['url']) as response:
                    response_time = (time.time() - start_time) * 1000

                    status = {
                        'server': server_name,
                        'url': server_info['url'],
                        'status': 'healthy' if response.status in [200, 404] else 'unhealthy',
                        'response_time_ms': round(response_time, 2),
                        'status_code': response.status,
                        'timestamp': datetime.now().isoformat()
                    }

                    # Reset failure count on success
                    if status['status'] == 'healthy':
                        self.failure_counts[server_name] = 0

                    return status

        except Exception as e:
            # Increment failure count
            self.failure_counts[server_name] += 1

            return {
                'server': server_name,
                'url': server_info['url'],
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': None,
                'failure_count': self.failure_counts[server_name],
                'timestamp': datetime.now().isoformat()
            }

    async def check_all_servers(self) -> Dict[str, dict]:
        """Check health of all MCP servers concurrently"""
        tasks = [
            self.check_mcp_server(name, info)
            for name, info in self.mcp_servers.items()
        ]

        results = await asyncio.gather(*tasks)
        return {result['server']: result for result in results}

    def get_container_stats(self) -> Dict[str, dict]:
        """Get Docker/Podman container resource usage statistics"""
        try:
            # Try docker first, then podman
            commands = [
                'docker stats --no-stream --format "{{.Container}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}"',
                'podman stats --no-stream --format "{{.Name}},{{.CPU}},{{.MemUsage}},{{.MemPerc}}"'
            ]

            for cmd in commands:
                try:
                    result = subprocess.run(
                        cmd.split(),
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        stats = {}
                        for line in result.stdout.strip().split('\n'):
                            if 'physics-assistant' in line:
                                parts = line.split(',')
                                if len(parts) >= 4:
                                    container_name = parts[0]
                                    cpu_percent = parts[1].replace('%', '')
                                    memory_usage = parts[2]
                                    memory_percent = parts[3].replace('%', '')

                                    stats[container_name] = {
                                        'cpu_percent': float(cpu_percent) if cpu_percent != '--' else 0,
                                        'memory_usage': memory_usage,
                                        'memory_percent': float(memory_percent) if memory_percent != '--' else 0,
                                        'timestamp': datetime.now().isoformat()
                                    }
                        return stats
                except subprocess.SubprocessError:
                    continue

        except Exception as e:
            logger.error(f"Error getting container stats: {e}")

        return {}

    def check_disk_space(self) -> dict:
        """Check disk space usage"""
        try:
            result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    return {
                        'total': parts[1],
                        'used': parts[2],
                        'available': parts[3],
                        'use_percent': parts[4].replace('%', ''),
                        'timestamp': datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")

        return {}

    def should_send_alert(self, server_name: str) -> bool:
        """Check if we should send an alert (respecting cooldown)"""
        now = datetime.now()
        last_alert = self.last_alert_time.get(server_name)

        if last_alert is None:
            return True

        time_since_last = now - last_alert
        return time_since_last > timedelta(minutes=self.alert_cooldown_minutes)

    async def send_email_alert(self, subject: str, body: str):
        """Send email alert"""
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.alert_email]):
            logger.warning("Email configuration incomplete, skipping email alert")
            return

        try:
            msg = MimeMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.alert_email
            msg['Subject'] = f"Physics Assistant Alert: {subject}"

            msg.attach(MimeText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def send_slack_alert(self, message: str):
        """Send Slack alert via webhook"""
        if not self.slack_webhook:
            logger.warning("Slack webhook not configured, skipping Slack alert")
            return

        try:
            payload = {
                "text": f"🚨 Physics Assistant Alert",
                "attachments": [{
                    "color": "danger",
                    "text": message,
                    "ts": int(time.time())
                }]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack alert sent successfully")
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def process_alerts(self, health_data: dict):
        """Process health data and send alerts if needed"""
        alerts = []

        # Check MCP server health
        for server_name, status in health_data['mcp_servers'].items():
            if status['status'] == 'unhealthy':
                failure_count = self.failure_counts[server_name]

                if failure_count >= self.alert_thresholds['consecutive_failures']:
                    if self.should_send_alert(server_name):
                        alert_msg = f"MCP Server '{server_name}' has failed {failure_count} consecutive health checks. Error: {status.get('error', 'Unknown')}"
                        alerts.append(alert_msg)
                        self.last_alert_time[server_name] = datetime.now()

            elif status.get('response_time_ms', 0) > self.alert_thresholds['response_time_ms']:
                alert_msg = f"MCP Server '{server_name}' has high response time: {status['response_time_ms']}ms"
                alerts.append(alert_msg)

        # Check container resource usage
        for container_name, stats in health_data['container_stats'].items():
            if stats['memory_percent'] > self.alert_thresholds['memory_usage_percent']:
                alert_msg = f"Container '{container_name}' high memory usage: {stats['memory_percent']}%"
                alerts.append(alert_msg)

            if stats['cpu_percent'] > self.alert_thresholds['cpu_usage_percent']:
                alert_msg = f"Container '{container_name}' high CPU usage: {stats['cpu_percent']}%"
                alerts.append(alert_msg)

        # Check disk space
        disk_info = health_data.get('disk_space', {})
        if disk_info.get('use_percent'):
            usage = float(disk_info['use_percent'])
            if usage > 85:
                alert_msg = f"High disk usage: {usage}% ({disk_info['available']} available)"
                alerts.append(alert_msg)

        # Send alerts
        if alerts:
            alert_summary = "\n".join(f"• {alert}" for alert in alerts)

            # Send email alert
            await self.send_email_alert(
                "System Health Issues Detected",
                f"The following issues were detected:\n\n{alert_summary}"
            )

            # Send Slack alert
            await self.send_slack_alert(alert_summary)

    async def run_health_check(self) -> dict:
        """Run complete health check"""
        logger.info("Starting health check...")

        # Check MCP servers
        mcp_status = await self.check_all_servers()

        # Get container stats
        container_stats = self.get_container_stats()

        # Check disk space
        disk_space = self.check_disk_space()

        # Compile health data
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'mcp_servers': mcp_status,
            'container_stats': container_stats,
            'disk_space': disk_space,
            'summary': {
                'healthy_servers': sum(1 for s in mcp_status.values() if s['status'] == 'healthy'),
                'total_servers': len(mcp_status),
                'containers_monitored': len(container_stats)
            }
        }

        # Process alerts
        await self.process_alerts(health_data)

        logger.info(f"Health check complete - {health_data['summary']['healthy_servers']}/{health_data['summary']['total_servers']} servers healthy")

        return health_data

    def save_health_data(self, health_data: dict):
        """Save health data to file for historical tracking"""
        try:
            os.makedirs('logs', exist_ok=True)
            filename = f"logs/health_data_{datetime.now().strftime('%Y%m%d')}.jsonl"

            with open(filename, 'a') as f:
                f.write(json.dumps(health_data) + '\n')

        except Exception as e:
            logger.error(f"Failed to save health data: {e}")

    async def continuous_monitoring(self, interval_seconds: int = 60):
        """Run continuous health monitoring"""
        logger.info(f"Starting continuous monitoring with {interval_seconds}s interval")

        while True:
            try:
                health_data = await self.run_health_check()
                self.save_health_data(health_data)

                # Print summary
                summary = health_data['summary']
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Servers: {summary['healthy_servers']}/{summary['total_servers']} healthy, "
                      f"Containers: {summary['containers_monitored']} monitored")

                await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(interval_seconds)

async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Physics Assistant Health Monitor')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--output', choices=['json', 'summary'], default='summary', help='Output format')

    args = parser.parse_args()

    monitor = HealthMonitor()

    if args.continuous:
        await monitor.continuous_monitoring(args.interval)
    else:
        health_data = await monitor.run_health_check()

        if args.output == 'json':
            print(json.dumps(health_data, indent=2))
        else:
            # Print summary
            summary = health_data['summary']
            print(f"\n🏥 Physics Assistant Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print(f"MCP Servers: {summary['healthy_servers']}/{summary['total_servers']} healthy")

            for name, status in health_data['mcp_servers'].items():
                status_icon = "✅" if status['status'] == 'healthy' else "❌"
                response_time = f" ({status.get('response_time_ms', 0):.1f}ms)" if status.get('response_time_ms') else ""
                print(f"  {status_icon} {name}: {status['status']}{response_time}")

            if health_data['container_stats']:
                print(f"\nContainer Resources:")
                for name, stats in health_data['container_stats'].items():
                    print(f"  📦 {name}: CPU {stats['cpu_percent']:.1f}%, Memory {stats['memory_percent']:.1f}%")

            if health_data['disk_space']:
                disk = health_data['disk_space']
                print(f"\nDisk Space: {disk['use_percent']}% used ({disk['available']} available)")

if __name__ == "__main__":
    asyncio.run(main())