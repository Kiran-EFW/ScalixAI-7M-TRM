#!/usr/bin/env python3
"""
Progress Monitoring Script for Continuous Learning

Checks the status of the continuous learning system and provides
regular progress updates.
"""

import os
import time
import json
import subprocess
import psutil
from datetime import datetime, timedelta
from pathlib import Path

def check_process_status():
    """Check if the continuous learning process is still running"""
    try:
        if os.path.exists("continuous_learning.pid"):
            with open("continuous_learning.pid", "r") as f:
                pid = int(f.read().strip())

            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                if process.is_running():
                    return True, pid
    except Exception as e:
        print(f"Error checking process status: {e}")

    return False, None

def get_system_metrics():
    """Get current system metrics"""
    metrics = {}

    # Performance log
    if os.path.exists("performance_log.jsonl"):
        try:
            with open("performance_log.jsonl", "r") as f:
                lines = f.readlines()[-1:]  # Last entry
                if lines:
                    latest = json.loads(lines[0])
                    metrics["performance"] = latest
        except Exception as e:
            metrics["performance_error"] = str(e)

    # Storage usage
    try:
        storage_result = subprocess.run(
            ["python3", "storage_manager.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if storage_result.returncode == 0:
            # Extract key metrics from output
            output = storage_result.stdout
            metrics["storage_status"] = "OK"
            if "CRITICAL" in output:
                metrics["storage_status"] = "CRITICAL"
            elif "WARNING" in output:
                metrics["storage_status"] = "WARNING"
        else:
            metrics["storage_error"] = storage_result.stderr
    except Exception as e:
        metrics["storage_error"] = str(e)

    # Log file sizes
    log_files = ["continuous_learning.log", "performance_log.jsonl"]
    metrics["log_sizes"] = {}
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            metrics["log_sizes"][log_file] = size

    return metrics

def generate_progress_report():
    """Generate a comprehensive progress report"""
    report_time = datetime.now()

    # Check process status
    is_running, pid = check_process_status()

    # Get metrics
    metrics = get_system_metrics()

    # Create report
    report = f"""
Continuous Learning Progress Report
===================================
Generated: {report_time.strftime('%Y-%m-%d %H:%M:%S')}

Process Status:
- Running: {'✓ YES' if is_running else '✗ NO'}
"""

    if pid:
        report += f"- PID: {pid}\n"

    # Performance metrics
    runtime_hours = 0
    knowledge_per_hour = 0
    conversations_per_hour = 0

    if "performance" in metrics:
        perf = metrics["performance"]
        runtime_hours = perf.get("runtime_hours", 0)
        knowledge_items = perf.get("knowledge_items_generated", 0)
        conversations = perf.get("conversations_completed", 0)
        languages = perf.get("languages_supported", 1)

        report += f"""
Learning Progress:
- Runtime: {runtime_hours:.1f} hours
- Knowledge Items Generated: {knowledge_items}
- Conversations Completed: {conversations}
- Languages Supported: {languages}
"""

        # Calculate efficiency
        if runtime_hours > 0:
            knowledge_per_hour = knowledge_items / runtime_hours
            conversations_per_hour = conversations / runtime_hours
            report += f"- Knowledge Generation Rate: {knowledge_per_hour:.1f}/hour\n"
            report += f"- Conversation Rate: {conversations_per_hour:.1f}/hour\n"

    # Storage status
    if "storage_status" in metrics:
        report += f"""
Storage Status: {metrics['storage_status']}
"""

    # Log sizes
    if "log_sizes" in metrics:
        report += """
Log File Sizes:
"""
        for log_file, size in metrics["log_sizes"].items():
            report += f"- {log_file}: {size / 1024:.1f} KB\n"

    # Warnings/Errors
    warnings = []
    if not is_running:
        warnings.append("⚠️  Continuous learning process is not running!")
    if metrics.get("storage_status") == "CRITICAL":
        warnings.append("⚠️  Storage usage is critical!")
    if "performance_error" in metrics:
        warnings.append(f"⚠️  Performance monitoring error: {metrics['performance_error']}")
    if "storage_error" in metrics:
        warnings.append(f"⚠️  Storage monitoring error: {metrics['storage_error']}")

    if warnings:
        report += """
Warnings:
"""
        for warning in warnings:
            report += f"{warning}\n"

    # Recommendations
    recommendations = []
    if is_running and runtime_hours < 24:  # Less than 1 day
        recommendations.append("System is still in initialization phase. Continue monitoring.")
    elif is_running and knowledge_per_hour < 1:
        recommendations.append("Knowledge generation rate is low. Check API connectivity.")
    elif not is_running:
        recommendations.append("Restart the continuous learning system.")

    if recommendations:
        report += """
Recommendations:
"""
        for rec in recommendations:
            report += f"- {rec}\n"

    return report

def create_monitoring_loop():
    """Create a monitoring loop that runs continuously"""
    print("Starting continuous monitoring... (Press Ctrl+C to stop)")

    try:
        while True:
            # Generate and display report
            report = generate_progress_report()
            print("\n" + "="*80)
            print(report)
            print("="*80)

            # Save report to file
            with open("progress_report.txt", "w") as f:
                f.write(report)

            # Wait before next check (15 minutes)
            print("Next check in 15 minutes... (or press Ctrl+C to stop)")
            time.sleep(900)  # 15 minutes

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Continuous Learning Progress Monitor")
    parser.add_argument("--once", action="store_true", help="Generate single report and exit")
    parser.add_argument("--loop", action="store_true", help="Start continuous monitoring loop")

    args = parser.parse_args()

    if args.once or not args.loop:
        # Single report
        report = generate_progress_report()
        print(report)

        # Save to file
        with open("progress_report.txt", "w") as f:
            f.write(report)

    elif args.loop:
        create_monitoring_loop()

if __name__ == "__main__":
    main()
