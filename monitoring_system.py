"""
Comprehensive Monitoring System for Continuous Learning

Tracks learning progress, storage usage, context window limits, and provides
insights into whether the continuous learning plan is working effectively.
"""

import os
import json
import time
import psutil
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import threading
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import subprocess

logger = logging.getLogger(__name__)

class LearningMetricsTracker:
    """Tracks and analyzes learning system metrics"""

    def __init__(self, log_directory: str = "monitoring_logs"):
        self.log_directory = log_directory
        os.makedirs(log_directory, exist_ok=True)

        # Metrics storage
        self.performance_history = []
        self.storage_history = []
        self.context_window_usage = []
        self.learning_efficiency = []

        # Baseline metrics for comparison
        self.baseline_performance = {
            "knowledge_items_generated": 0,
            "conversations_completed": 0,
            "architectures_evolved": 0,
            "languages_supported": 1
        }

    def track_performance_metrics(self) -> Dict[str, Any]:
        """Track current performance metrics"""
        current_time = datetime.now()

        # Read latest performance log
        performance_data = self._read_latest_performance_log()

        # Calculate growth rates
        growth_metrics = self._calculate_growth_rates(performance_data)

        # Track context window usage
        context_metrics = self._monitor_context_window_usage()

        # Storage usage
        storage_metrics = self._monitor_storage_usage()

        metrics = {
            "timestamp": current_time.isoformat(),
            "performance": performance_data,
            "growth_rates": growth_metrics,
            "context_window": context_metrics,
            "storage": storage_metrics,
            "efficiency_score": self._calculate_efficiency_score(performance_data, storage_metrics)
        }

        # Store metrics
        self.performance_history.append(metrics)

        # Save to file
        self._save_metrics_to_file(metrics)

        return metrics

    def _read_latest_performance_log(self) -> Dict[str, Any]:
        """Read the latest performance log entries"""
        performance_file = "performance_log.jsonl"

        if not os.path.exists(performance_file):
            return {
                "knowledge_items_generated": 0,
                "conversations_completed": 0,
                "architectures_evolved": 0,
                "languages_supported": 1,
                "active_agent_threads": 0,
                "runtime_hours": 0
            }

        # Read last 10 entries for recent trends
        entries = []
        try:
            with open(performance_file, 'r') as f:
                lines = f.readlines()[-10:]  # Last 10 entries
                for line in lines:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Error reading performance log: {e}")
            return self.baseline_performance.copy()

        if not entries:
            return self.baseline_performance.copy()

        # Use latest entry
        latest = entries[-1]
        return {
            "knowledge_items_generated": latest.get("knowledge_items_generated", 0),
            "conversations_completed": latest.get("conversations_completed", 0),
            "architectures_evolved": latest.get("architectures_evolved", 0),
            "languages_supported": latest.get("supported_languages", 1),
            "active_agent_threads": latest.get("active_agent_threads", 0),
            "runtime_hours": latest.get("runtime_hours", 0)
        }

    def _calculate_growth_rates(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate growth rates compared to previous measurements"""
        if len(self.performance_history) < 2:
            return {key: 0.0 for key in current_metrics.keys() if key != "timestamp"}

        previous = self.performance_history[-2]["performance"]
        time_diff_hours = 0.5  # Assuming measurements every 30 minutes

        growth_rates = {}
        for key, current_value in current_metrics.items():
            if key == "timestamp":
                continue

            previous_value = previous.get(key, 0)
            if previous_value == 0:
                growth_rate = float('inf') if current_value > 0 else 0.0
            else:
                growth_rate = (current_value - previous_value) / previous_value / time_diff_hours

            growth_rates[f"{key}_growth_rate"] = growth_rate

        return growth_rates

    def _monitor_context_window_usage(self) -> Dict[str, Any]:
        """Monitor Gemini API context window usage"""
        # Check for API rate limiting indicators
        api_error_log = "api_errors.log"

        context_metrics = {
            "estimated_tokens_used": 0,
            "api_errors_today": 0,
            "rate_limit_warnings": 0,
            "context_window_efficiency": 1.0
        }

        # Check for recent API errors
        if os.path.exists(api_error_log):
            try:
                with open(api_error_log, 'r') as f:
                    lines = f.readlines()
                    # Count errors in last 24 hours
                    recent_errors = 0
                    cutoff_time = time.time() - 86400  # 24 hours ago

                    for line in lines:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                if entry.get("timestamp", 0) > cutoff_time:
                                    recent_errors += 1
                            except:
                                continue

                    context_metrics["api_errors_today"] = recent_errors
            except Exception as e:
                logger.warning(f"Error reading API error log: {e}")

        # Estimate token usage (rough calculation based on data generation)
        recent_knowledge = self._read_latest_performance_log().get("knowledge_items_generated", 0)
        # Assume average of 1000 tokens per knowledge item
        context_metrics["estimated_tokens_used"] = recent_knowledge * 1000

        return context_metrics

    def _monitor_storage_usage(self) -> Dict[str, Any]:
        """Monitor storage usage of the learning system"""
        directories_to_monitor = [
            "data",
            "checkpoints",
            "monitoring_logs",
            "performance_log.jsonl",
            "continuous_learning.log"
        ]

        total_size = 0
        breakdown = {}

        for path in directories_to_monitor:
            if os.path.exists(path):
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                else:
                    size = self._get_directory_size(path)

                total_size += size
                breakdown[path] = size

        # Convert to MB
        total_mb = total_size / (1024 * 1024)

        return {
            "total_storage_mb": total_mb,
            "breakdown": {k: v / (1024 * 1024) for k, v in breakdown.items()},
            "growth_rate_mb_per_hour": self._calculate_storage_growth_rate()
        }

    def _get_directory_size(self, path: str) -> int:
        """Calculate total size of directory recursively"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    continue
        return total_size

    def _calculate_storage_growth_rate(self) -> float:
        """Calculate storage growth rate"""
        if len(self.storage_history) < 2:
            return 0.0

        current = self.storage_history[-1]["total_storage_mb"]
        previous = self.storage_history[-2]["total_storage_mb"]
        time_diff_hours = 0.5  # Assuming measurements every 30 minutes

        return (current - previous) / time_diff_hours

    def _calculate_efficiency_score(self, performance: Dict[str, Any], storage: Dict[str, Any]) -> float:
        """Calculate overall learning efficiency score"""
        # Efficiency = (knowledge_generated + conversations) / (storage_used + 1)
        knowledge_score = performance.get("knowledge_items_generated", 0) * 0.1
        conversation_score = performance.get("conversations_completed", 0) * 0.5
        language_bonus = performance.get("languages_supported", 1) * 10

        storage_penalty = storage.get("total_storage_mb", 0) * 0.001

        efficiency = (knowledge_score + conversation_score + language_bonus) / (storage_penalty + 1)

        return min(efficiency, 100.0)  # Cap at 100

    def _save_metrics_to_file(self, metrics: Dict[str, Any]):
        """Save metrics to monitoring log file"""
        metrics_file = os.path.join(self.log_directory, "learning_metrics.jsonl")

        with open(metrics_file, 'a') as f:
            json.dump(metrics, f)
            f.write('\n')

    def generate_monitoring_report(self) -> str:
        """Generate a comprehensive monitoring report"""
        if not self.performance_history:
            return "No monitoring data available yet."

        latest = self.performance_history[-1]

        report = f"""
# Continuous Learning System Monitoring Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Status
- Runtime: {latest['performance'].get('runtime_hours', 0):.1f} hours
- Active Agent Threads: {latest['performance'].get('active_agent_threads', 0)}
- Efficiency Score: {latest['efficiency_score']:.2f}/100

## Learning Progress
- Knowledge Items Generated: {latest['performance'].get('knowledge_items_generated', 0)}
- Conversations Completed: {latest['performance'].get('conversations_completed', 0)}
- Architectures Evolved: {latest['performance'].get('architectures_evolved', 0)}
- Languages Supported: {latest['performance'].get('languages_supported', 1)}

## Growth Rates (per hour)
"""

        growth_rates = latest.get('growth_rates', {})
        for key, rate in growth_rates.items():
            if rate == float('inf'):
                report += f"- {key}: ∞ (from 0 to positive)\n"
            else:
                report += f"- {key}: {rate:.3f}\n"

        report += f"""
## Storage Usage
- Total Storage: {latest['storage']['total_storage_mb']:.1f} MB
- Growth Rate: {latest['storage']['growth_rate_mb_per_hour']:.2f} MB/hour

## Context Window Usage
- Estimated Tokens Used: {latest['context_window']['estimated_tokens_used']:,}
- API Errors Today: {latest['context_window']['api_errors_today']}

## Health Indicators
"""

        # Health assessment
        health_indicators = self._assess_system_health(latest)
        for indicator, status in health_indicators.items():
            report += f"- {indicator}: {status}\n"

        return report

    def _assess_system_health(self, latest_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall system health"""
        indicators = {}

        # Learning progress
        knowledge_growth = latest_metrics.get('growth_rates', {}).get('knowledge_items_generated_growth_rate', 0)
        indicators["Learning Progress"] = "Good" if knowledge_growth > 0.1 else "Slow"

        # Storage efficiency
        storage_mb = latest_metrics['storage']['total_storage_mb']
        knowledge_count = latest_metrics['performance'].get('knowledge_items_generated', 0)
        storage_efficiency = knowledge_count / (storage_mb + 1)
        indicators["Storage Efficiency"] = "Good" if storage_efficiency > 10 else "Needs Optimization"

        # API health
        api_errors = latest_metrics['context_window']['api_errors_today']
        indicators["API Health"] = "Good" if api_errors < 10 else "Issues Detected"

        # System stability
        runtime_hours = latest_metrics['performance'].get('runtime_hours', 0)
        indicators["System Stability"] = "Stable" if runtime_hours > 1 else "Initializing"

        return indicators

    def create_monitoring_dashboard(self):
        """Create a visual monitoring dashboard"""
        if len(self.performance_history) < 2:
            print("Not enough data for dashboard yet.")
            return

        # Create plots
        self._create_performance_plot()
        self._create_storage_plot()
        self._create_efficiency_plot()

        print("Monitoring dashboard created. Check the 'monitoring_logs' directory.")

    def _create_performance_plot(self):
        """Create performance over time plot"""
        timestamps = []
        knowledge_counts = []
        conversation_counts = []

        for entry in self.performance_history[-50:]:  # Last 50 measurements
            timestamps.append(datetime.fromisoformat(entry['timestamp']))
            knowledge_counts.append(entry['performance'].get('knowledge_items_generated', 0))
            conversation_counts.append(entry['performance'].get('conversations_completed', 0))

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, knowledge_counts, label='Knowledge Items', marker='o')
        plt.plot(timestamps, conversation_counts, label='Conversations', marker='s')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.title('Learning Progress Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_directory, 'performance_progress.png'))
        plt.close()

    def _create_storage_plot(self):
        """Create storage usage plot"""
        timestamps = []
        storage_usage = []

        for entry in self.performance_history[-50:]:
            timestamps.append(datetime.fromisoformat(entry['timestamp']))
            storage_usage.append(entry['storage']['total_storage_mb'])

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, storage_usage, label='Storage Used (MB)', color='orange', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Storage (MB)')
        plt.title('Storage Usage Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_directory, 'storage_usage.png'))
        plt.close()

    def _create_efficiency_plot(self):
        """Create efficiency score plot"""
        timestamps = []
        efficiency_scores = []

        for entry in self.performance_history[-50:]:
            timestamps.append(datetime.fromisoformat(entry['timestamp']))
            efficiency_scores.append(entry.get('efficiency_score', 0))

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, efficiency_scores, label='Efficiency Score', color='green', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Efficiency Score')
        plt.title('Learning Efficiency Over Time')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_directory, 'efficiency_score.png'))
        plt.close()


class MonitoringDashboard:
    """Interactive monitoring dashboard"""

    def __init__(self, tracker: LearningMetricsTracker):
        self.tracker = tracker
        self.running = False

    def start_dashboard(self):
        """Start the monitoring dashboard"""
        self.running = True
        print("Starting Continuous Learning Monitoring Dashboard...")
        print("Press Ctrl+C to stop monitoring")

        try:
            while self.running:
                # Track metrics
                metrics = self.tracker.track_performance_metrics()

                # Display current status
                self._display_current_status(metrics)

                # Check for alerts
                self._check_alerts(metrics)

                # Wait before next update
                time.sleep(1800)  # 30 minutes

        except KeyboardInterrupt:
            print("\nStopping monitoring dashboard...")
            self.running = False

    def _display_current_status(self, metrics: Dict[str, Any]):
        """Display current system status"""
        os.system('clear')  # Clear terminal

        print("=" * 80)
        print("CONTINUOUS LEARNING SYSTEM MONITORING")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: {metrics['performance'].get('runtime_hours', 0):.1f} hours")
        print()

        print("LEARNING PROGRESS:")
        print(f"  Knowledge Items: {metrics['performance'].get('knowledge_items_generated', 0)}")
        print(f"  Conversations: {metrics['performance'].get('conversations_completed', 0)}")
        print(f"  Languages: {metrics['performance'].get('languages_supported', 1)}")
        print()

        print("GROWTH RATES (per hour):")
        growth_rates = metrics.get('growth_rates', {})
        for key, rate in growth_rates.items():
            if rate == float('inf'):
                print(f"  {key}: ∞")
            else:
                print(f"  {key}: {rate:.3f}")
        print()

        print("STORAGE USAGE:")
        print(f"  Total: {metrics['storage']['total_storage_mb']:.1f} MB")
        print(f"  Growth Rate: {metrics['storage']['growth_rate_mb_per_hour']:.2f} MB/hour")
        print()

        print("CONTEXT WINDOW:")
        print(f"  Estimated Tokens: {metrics['context_window']['estimated_tokens_used']:,}")
        print(f"  API Errors Today: {metrics['context_window']['api_errors_today']}")
        print()

        print("EFFICIENCY SCORE:")
        efficiency = metrics['efficiency_score']
        efficiency_bar = "█" * int(efficiency / 2)  # Scale to 50 characters
        print(f"  {efficiency:.1f}/100 [{efficiency_bar:<50}]")
        print()

        print("SYSTEM HEALTH:")
        health_indicators = self.tracker._assess_system_health(metrics)
        for indicator, status in health_indicators.items():
            status_color = "🟢" if status == "Good" else "🟡" if status == "Slow" else "🔴"
            print(f"  {status_color} {indicator}: {status}")
        print()

    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for system alerts"""
        alerts = []

        # Storage alert
        if metrics['storage']['total_storage_mb'] > 10000:  # 10GB
            alerts.append("⚠️  High storage usage detected (>10GB)")

        # API error alert
        if metrics['context_window']['api_errors_today'] > 50:
            alerts.append("⚠️  High API error rate detected")

        # Learning stagnation alert
        knowledge_growth = metrics.get('growth_rates', {}).get('knowledge_items_generated_growth_rate', 0)
        if len(self.tracker.performance_history) > 10 and knowledge_growth < 0.01:
            alerts.append("⚠️  Learning growth appears stagnant")

        # Display alerts
        if alerts:
            print("ALERTS:")
            for alert in alerts:
                print(f"  {alert}")
            print()


def main():
    """Main monitoring system entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Continuous Learning Monitoring System")
    parser.add_argument("--dashboard", action="store_true", help="Start interactive dashboard")
    parser.add_argument("--report", action="store_true", help="Generate monitoring report")
    parser.add_argument("--plots", action="store_true", help="Create monitoring plots")
    parser.add_argument("--log-dir", type=str, default="monitoring_logs", help="Monitoring log directory")

    args = parser.parse_args()

    # Initialize tracker
    tracker = LearningMetricsTracker(args.log_dir)

    if args.dashboard:
        dashboard = MonitoringDashboard(tracker)
        dashboard.start_dashboard()
    elif args.report:
        report = tracker.generate_monitoring_report()
        print(report)
    elif args.plots:
        tracker.create_monitoring_dashboard()
    else:
        # Default: track metrics once
        metrics = tracker.track_performance_metrics()
        print("Metrics tracked. Use --dashboard for continuous monitoring.")


if __name__ == "__main__":
    main()
