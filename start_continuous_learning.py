#!/usr/bin/env python3
"""
Continuous Learning Startup Script

Sets up and starts the continuous learning system with Gemini 2.5 Flash integration,
including monitoring and storage management.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json

def setup_environment():
    """Set up the environment for continuous learning"""
    print("Setting up continuous learning environment...")

    # Create necessary directories
    directories = [
        "data/coding", "data/reasoning", "data/agentic_browsing", "data/conversation",
        "data/multimodal", "checkpoints/multi_agent_system", "checkpoints/continuous_learning",
        "checkpoints/architecture_evolution", "checkpoints/cross_modal",
        "monitoring_logs", "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Set environment variables
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set!")
        print("Please set it with: export GEMINI_API_KEY='your_api_key_here'")
        return False

    print("Environment setup complete!")
    return True

def start_monitoring_system():
    """Start the monitoring system in background"""
    print("Starting monitoring system...")

    try:
        # Start monitoring in background
        monitor_cmd = [sys.executable, "monitoring_system.py", "--dashboard"]
        monitor_process = subprocess.Popen(
            monitor_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )

        # Wait a moment for it to start
        time.sleep(2)

        if monitor_process.poll() is None:
            print("✓ Monitoring system started successfully")
            return monitor_process
        else:
            stdout, stderr = monitor_process.communicate()
            print(f"✗ Monitoring system failed to start: {stderr.decode()}")
            return None

    except Exception as e:
        print(f"✗ Error starting monitoring: {e}")
        return None

def start_continuous_learning(duration_days=3, num_agents=4):
    """Start the continuous learning system"""
    print(f"Starting continuous learning for {duration_days} days with {num_agents} agents...")

    try:
        # Prepare command
        cmd = [
            sys.executable, "continuous_learning_system.py",
            "--api-key", os.getenv("GEMINI_API_KEY"),
            "--duration-days", str(duration_days),
            "--num-agents", str(num_agents),
            "--languages", "en", "es", "fr", "de", "zh"
        ]

        print(f"Running command: {' '.join(cmd)}")

        # Start the learning system
        learning_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        return learning_process

    except Exception as e:
        print(f"✗ Error starting continuous learning: {e}")
        return None

def start_storage_manager():
    """Start storage management monitoring"""
    print("Starting storage management...")

    try:
        # Start storage monitoring in background
        storage_cmd = [sys.executable, "storage_manager.py", "--report"]
        storage_process = subprocess.Popen(
            storage_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )

        # Run once to get initial report
        stdout, stderr = storage_process.communicate(timeout=30)

        if storage_process.returncode == 0:
            print("✓ Storage management initialized")
            print("Storage Report:")
            print(stdout.decode())
        else:
            print(f"✗ Storage management error: {stderr.decode()}")

    except Exception as e:
        print(f"✗ Error with storage management: {e}")

def monitor_processes(processes):
    """Monitor running processes and handle shutdown"""
    learning_process, monitor_process = processes

    try:
        while True:
            # Check if learning process is still running
            if learning_process and learning_process.poll() is not None:
                print("Continuous learning process has finished!")
                break

            # Check if monitoring process is still running
            if monitor_process and monitor_process.poll() is not None:
                print("Warning: Monitoring process stopped, restarting...")
                monitor_process = start_monitoring_system()

            # Periodic storage check
            storage_status = subprocess.run(
                [sys.executable, "storage_manager.py"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=10
            )

            if storage_status.returncode == 0:
                # Check for critical storage warnings
                output = storage_status.stdout
                if "CRITICAL" in output:
                    print("⚠️  CRITICAL STORAGE WARNING!")
                    print(output)

                    # Run emergency cleanup
                    print("Running emergency storage cleanup...")
                    cleanup_result = subprocess.run(
                        [sys.executable, "storage_manager.py", "--emergency"],
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd()
                    )
                    if cleanup_result.returncode == 0:
                        print("✓ Emergency cleanup completed")
                    else:
                        print("✗ Emergency cleanup failed")

            time.sleep(300)  # Check every 5 minutes

    except KeyboardInterrupt:
        print("\nShutdown requested by user...")

    finally:
        # Clean shutdown
        print("Shutting down processes...")

        if learning_process and learning_process.poll() is None:
            print("Terminating continuous learning process...")
            learning_process.terminate()
            try:
                learning_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                learning_process.kill()

        if monitor_process and monitor_process.poll() is None:
            print("Terminating monitoring process...")
            monitor_process.terminate()
            try:
                monitor_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor_process.kill()

        print("✓ All processes shut down")

def create_startup_report():
    """Create a startup report"""
    report = f"""
Continuous Learning System Startup Report
=========================================
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

System Configuration:
- API Key: {'✓ Set' if os.getenv('GEMINI_API_KEY') else '✗ Not Set'}
- Python Version: {sys.version.split()[0]}
- Working Directory: {os.getcwd()}

Directories Created:
- data/ (with subdirectories for different modalities)
- checkpoints/ (for model checkpoints)
- monitoring_logs/ (for monitoring data)
- logs/ (for system logs)

Components:
- Continuous Learning Orchestrator
- Multi-Agent System
- Cross-Modal Transfer
- Architecture Evolution
- Monitoring System
- Storage Management

To monitor progress:
1. Check monitoring dashboard: python monitoring_system.py --dashboard
2. View storage usage: python storage_manager.py --report
3. Generate reports: python monitoring_system.py --report

The system will run continuously and automatically manage:
- Learning progress tracking
- Storage optimization
- Context window monitoring
- Performance metrics
- Checkpoint saving

Press Ctrl+C to stop gracefully.
"""

    with open("startup_report.txt", "w") as f:
        f.write(report)

    print(report)

def main():
    parser = argparse.ArgumentParser(description="Start Continuous Learning System")
    parser.add_argument("--duration-days", type=float, default=3.0,
                       help="Duration to run learning in days")
    parser.add_argument("--num-agents", type=int, default=4,
                       help="Number of learning agents")
    parser.add_argument("--api-key", type=str,
                       help="Gemini API key (can also use GEMINI_API_KEY env var)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be started without actually starting")

    args = parser.parse_args()

    print("=" * 80)
    print("CONTINUOUS LEARNING SYSTEM STARTUP")
    print("=" * 80)

    # Set API key if provided
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key

    # Validate environment
    if not setup_environment():
        print("✗ Environment setup failed. Please fix issues and try again.")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN - Would start:")
        print(f"- Continuous learning for {args.duration_days} days")
        print(f"- {args.num_agents} agents")
        print("- Monitoring system")
        print("- Storage management")
        return

    # Create startup report
    create_startup_report()

    # Start components
    print("Starting system components...")

    # Start storage management (one-time)
    start_storage_manager()

    # Start monitoring system
    monitor_process = start_monitoring_system()
    if not monitor_process:
        print("✗ Failed to start monitoring system. Continuing anyway...")

    # Start continuous learning
    learning_process = start_continuous_learning(args.duration_days, args.num_agents)
    if not learning_process:
        print("✗ Failed to start continuous learning system.")
        sys.exit(1)

    print("=" * 80)
    print("✓ CONTINUOUS LEARNING SYSTEM STARTED SUCCESSFULLY!")
    print("=" * 80)
    print("The system is now running. Check startup_report.txt for details.")
    print("Monitor progress with: python monitoring_system.py --dashboard")
    print("=" * 80)

    # Monitor processes
    monitor_processes((learning_process, monitor_process))

if __name__ == "__main__":
    main()
