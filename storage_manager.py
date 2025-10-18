"""
Storage Management System for Continuous Learning

Manages data growth, cleanup, compression, and storage optimization
to ensure the learning system doesn't run out of storage space.
"""

import os
import json
import shutil
import gzip
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages storage usage and optimization for continuous learning"""

    def __init__(self, base_directory: str = ".", max_storage_gb: float = 50.0):
        self.base_directory = Path(base_directory)
        self.max_storage_gb = max_storage_gb
        self.max_storage_bytes = max_storage_gb * 1024 * 1024 * 1024

        # Aggressive storage policies for efficiency
        self.retention_policies = {
            "performance_logs": {"max_age_days": 1, "compress_after_days": 0.5, "delete_immediately": True},
            "checkpoints": {"max_age_days": 1, "keep_recent": 2, "compress_after_days": 0.25},
            "monitoring_logs": {"max_age_days": 3, "compress_after_days": 1, "delete_immediately": True},
            "api_errors": {"max_age_days": 1, "compress_after_days": 0.5, "delete_immediately": True},
            "training_data": {"max_age_days": 7, "compress_after_days": 1},
            "data": {"max_age_days": 1, "compress_after_days": 0.5},  # Generated training data
            "logs": {"max_age_days": 1, "compress_after_days": 0.5, "delete_immediately": True}
        }

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get comprehensive storage usage statistics"""
        usage = {
            "total_size_bytes": 0,
            "total_size_gb": 0.0,
            "breakdown": {},
            "largest_files": [],
            "oldest_files": [],
            "compression_candidates": []
        }

        # Walk through all files
        for root, dirs, files in os.walk(self.base_directory):
            for file in files:
                filepath = Path(root) / file
                try:
                    size = filepath.stat().st_size
                    usage["total_size_bytes"] += size

                    # Categorize by directory
                    relative_path = filepath.relative_to(self.base_directory)
                    category = str(relative_path.parts[0]) if len(relative_path.parts) > 0 else "root"

                    if category not in usage["breakdown"]:
                        usage["breakdown"][category] = 0
                    usage["breakdown"][category] += size

                    # Track largest files
                    usage["largest_files"].append((filepath, size))
                    usage["largest_files"].sort(key=lambda x: x[1], reverse=True)
                    usage["largest_files"] = usage["largest_files"][:10]  # Keep top 10

                    # Track oldest files
                    mtime = filepath.stat().st_mtime
                    usage["oldest_files"].append((filepath, mtime))
                    usage["oldest_files"].sort(key=lambda x: x[1])
                    usage["oldest_files"] = usage["oldest_files"][:10]  # Keep top 10

                    # Check for compression candidates
                    age_days = (time.time() - mtime) / (24 * 3600)
                    if age_days > 1 and not str(filepath).endswith('.gz'):
                        usage["compression_candidates"].append((filepath, age_days))

                except OSError:
                    continue

        usage["total_size_gb"] = usage["total_size_bytes"] / (1024**3)

        # Convert breakdowns to GB
        usage["breakdown"] = {k: v / (1024**3) for k, v in usage["breakdown"].items()}

        return usage

    def optimize_storage(self) -> Dict[str, Any]:
        """Perform aggressive storage optimization operations"""
        optimization_results = {
            "files_compressed": 0,
            "files_deleted": 0,
            "space_saved_bytes": 0,
            "errors": []
        }

        try:
            # Immediate cleanup for aggressive policies
            immediate_cleanup = self._immediate_cleanup()
            optimization_results["files_deleted"] += immediate_cleanup["files_deleted"]
            optimization_results["space_saved_bytes"] += immediate_cleanup["space_saved_bytes"]

            # Compress old files
            compression_results = self._compress_old_files()
            optimization_results["files_compressed"] += compression_results["files_compressed"]
            optimization_results["space_saved_bytes"] += compression_results["space_saved_bytes"]

            # Clean up old files based on retention policies
            cleanup_results = self._cleanup_old_files()
            optimization_results["files_deleted"] += cleanup_results["files_deleted"]
            optimization_results["space_saved_bytes"] += cleanup_results["space_saved_bytes"]

            # Remove duplicate or temporary files
            dedup_results = self._remove_duplicates()
            optimization_results["files_deleted"] += dedup_results["files_deleted"]
            optimization_results["space_saved_bytes"] += dedup_results["space_saved_bytes"]

        except Exception as e:
            optimization_results["errors"].append(f"Optimization error: {str(e)}")

        return optimization_results

    def _immediate_cleanup(self) -> Dict[str, Any]:
        """Perform immediate cleanup for files marked with delete_immediately"""
        results = {"files_deleted": 0, "space_saved_bytes": 0}

        for category, policy in self.retention_policies.items():
            if not policy.get("delete_immediately", False):
                continue

            category_path = self.base_directory / category
            if not category_path.exists():
                continue

            max_age_days = policy.get("max_age_days", 1)

            for filepath in category_path.rglob("*"):
                if not filepath.is_file():
                    continue

                try:
                    age_days = (time.time() - filepath.stat().st_mtime) / (24 * 3600)

                    # Delete immediately if older than max_age_days
                    if age_days > max_age_days:
                        size = filepath.stat().st_size
                        filepath.unlink()
                        results["files_deleted"] += 1
                        results["space_saved_bytes"] += size
                        print(f"Immediately deleted: {filepath} (age: {age_days:.2f} days)")

                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

        return results

    def _compress_old_files(self) -> Dict[str, Any]:
        """Compress old files to save space"""
        results = {"files_compressed": 0, "space_saved_bytes": 0}

        for category, policy in self.retention_policies.items():
            compress_after = policy.get("compress_after_days", 7)
            category_path = self.base_directory / category

            if not category_path.exists():
                continue

            for filepath in category_path.rglob("*"):
                if not filepath.is_file() or str(filepath).endswith('.gz'):
                    continue

                try:
                    age_days = (time.time() - filepath.stat().st_mtime) / (24 * 3600)

                    if age_days > compress_after:
                        original_size = filepath.stat().st_size
                        compressed_path = filepath.with_suffix(filepath.suffix + '.gz')

                        # Compress file
                        with open(filepath, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        compressed_size = compressed_path.stat().st_size

                        # Remove original if compression successful
                        if compressed_size < original_size:
                            filepath.unlink()
                            space_saved = original_size - compressed_size
                            results["files_compressed"] += 1
                            results["space_saved_bytes"] += space_saved

                            logger.info(f"Compressed {filepath} ({original_size} -> {compressed_size} bytes)")

                except Exception as e:
                    logger.warning(f"Failed to compress {filepath}: {e}")

        return results

    def _cleanup_old_files(self) -> Dict[str, Any]:
        """Clean up old files based on retention policies"""
        results = {"files_deleted": 0, "space_saved_bytes": 0}

        for category, policy in self.retention_policies.items():
            max_age_days = policy.get("max_age_days", 30)
            keep_recent = policy.get("keep_recent", 0)

            category_path = self.base_directory / category
            if not category_path.exists():
                continue

            # Get all files in category
            files = []
            for filepath in category_path.rglob("*"):
                if filepath.is_file():
                    try:
                        age_days = (time.time() - filepath.stat().st_mtime) / (24 * 3600)
                        files.append((filepath, age_days, filepath.stat().st_size))
                    except OSError:
                        continue

            # Sort by age (newest first)
            files.sort(key=lambda x: x[1])

            # Keep recent files, delete old ones
            if keep_recent > 0 and len(files) > keep_recent:
                files_to_delete = files[keep_recent:]
            else:
                # Delete files older than max_age_days
                files_to_delete = [f for f in files if f[1] > max_age_days]

            for filepath, age, size in files_to_delete:
                try:
                    filepath.unlink()
                    results["files_deleted"] += 1
                    results["space_saved_bytes"] += size
                    logger.info(f"Deleted old file: {filepath} (age: {age:.1f} days)")
                except Exception as e:
                    logger.warning(f"Failed to delete {filepath}: {e}")

        return results

    def _remove_duplicates(self) -> Dict[str, Any]:
        """Remove duplicate files and temporary files"""
        results = {"files_deleted": 0, "space_saved_bytes": 0}

        # Remove temporary files
        temp_patterns = ["*.tmp", "*.temp", "*~", "*.bak", "*.swp"]
        for pattern in temp_patterns:
            for filepath in self.base_directory.rglob(pattern):
                if filepath.is_file():
                    try:
                        size = filepath.stat().st_size
                        filepath.unlink()
                        results["files_deleted"] += 1
                        results["space_saved_bytes"] += size
                        logger.info(f"Removed temporary file: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {filepath}: {e}")

        # Remove duplicate checkpoint files (keep latest)
        checkpoint_dir = self.base_directory / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = {}
            for filepath in checkpoint_dir.rglob("checkpoint_*.json"):
                try:
                    # Extract timestamp from filename
                    timestamp = int(filepath.stem.split('_')[1])
                    key = filepath.parent.name  # subdirectory name

                    if key not in checkpoints:
                        checkpoints[key] = []
                    checkpoints[key].append((filepath, timestamp))
                except (ValueError, IndexError):
                    continue

            # For each checkpoint type, keep only the 3 most recent
            for checkpoint_type, files in checkpoints.items():
                if len(files) > 3:
                    files.sort(key=lambda x: x[1], reverse=True)  # Newest first
                    to_delete = files[3:]  # Keep first 3

                    for filepath, _ in to_delete:
                        try:
                            size = filepath.stat().st_size
                            filepath.unlink()
                            results["files_deleted"] += 1
                            results["space_saved_bytes"] += size
                            logger.info(f"Removed old checkpoint: {filepath}")
                        except Exception as e:
                            logger.warning(f"Failed to remove checkpoint {filepath}: {e}")

        return results

    def check_storage_limits(self) -> Dict[str, Any]:
        """Check if storage limits are being approached"""
        usage = self.get_storage_usage()

        status = {
            "current_usage_gb": usage["total_size_gb"],
            "max_allowed_gb": self.max_storage_gb,
            "usage_percentage": (usage["total_size_gb"] / self.max_storage_gb) * 100,
            "status": "OK",
            "warnings": [],
            "critical": False
        }

        # Check warning levels
        if status["usage_percentage"] > 90:
            status["status"] = "CRITICAL"
            status["critical"] = True
            status["warnings"].append(f"Storage usage is {status['usage_percentage']:.1f}% - immediate action required!")
        elif status["usage_percentage"] > 75:
            status["status"] = "WARNING"
            status["warnings"].append(f"Storage usage is {status['usage_percentage']:.1f}% - consider cleanup")
        elif status["usage_percentage"] > 50:
            status["status"] = "MONITOR"
            status["warnings"].append(f"Storage usage is {status['usage_percentage']:.1f}%")

        # Check for rapidly growing directories
        for category, size_gb in usage["breakdown"].items():
            if size_gb > 10:  # Any directory > 10GB
                status["warnings"].append(f"Large directory '{category}': {size_gb:.1f} GB")

        return status

    def get_storage_recommendations(self) -> List[str]:
        """Get storage optimization recommendations"""
        recommendations = []
        usage = self.get_storage_usage()
        status = self.check_storage_limits()

        # Basic recommendations
        if status["usage_percentage"] > 75:
            recommendations.append("Run storage optimization immediately")
            recommendations.append("Consider increasing max_storage_gb if learning continues to grow")

        # Check for uncompressed old files
        old_uncompressed = [f for f, age in usage["compression_candidates"] if age > 7]
        if old_uncompressed:
            recommendations.append(f"Compress {len(old_uncompressed)} old files to save space")

        # Check retention policies
        for category, policy in self.retention_policies.items():
            max_age = policy.get("max_age_days", 30)
            recommendations.append(f"Configure {category} cleanup: delete files older than {max_age} days")

        # Disk space recommendations
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 80:
            recommendations.append(f"System disk usage is {disk_usage.percent}% - monitor closely")

        return recommendations

    def create_storage_report(self) -> str:
        """Create a comprehensive storage report"""
        usage = self.get_storage_usage()
        status = self.check_storage_limits()
        recommendations = self.get_storage_recommendations()

        report = f"""
# Storage Management Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Storage Usage Summary
- Current Usage: {usage['total_size_gb']:.2f} GB
- Maximum Allowed: {self.max_storage_gb:.1f} GB
- Usage Percentage: {status['usage_percentage']:.1f}%
- Status: {status['status']}

## Storage Breakdown by Category
"""

        for category, size_gb in sorted(usage['breakdown'].items(), key=lambda x: x[1], reverse=True):
            percentage = (size_gb / usage['total_size_gb']) * 100 if usage['total_size_gb'] > 0 else 0
            report += f"- {category}: {size_gb:.2f} GB ({percentage:.1f}%)\n"

        if usage['largest_files']:
            report += "\n## Largest Files\n"
            for filepath, size in usage['largest_files'][:5]:
                report += f"- {filepath}: {size / (1024**2):.1f} MB\n"

        if status['warnings']:
            report += "\n## Warnings\n"
            for warning in status['warnings']:
                report += f"- ⚠️  {warning}\n"

        if recommendations:
            report += "\n## Recommendations\n"
            for rec in recommendations:
                report += f"- 💡 {rec}\n"

        return report

    def emergency_cleanup(self) -> Dict[str, Any]:
        """Perform emergency cleanup when storage is critical"""
        logger.warning("Performing emergency storage cleanup!")

        results = {"files_deleted": 0, "space_freed_gb": 0.0, "actions_taken": []}

        # Delete oldest log files first
        log_dirs = ["monitoring_logs", "performance_log.jsonl", "continuous_learning.log"]
        for log_path in log_dirs:
            if os.path.exists(log_path):
                if os.path.isfile(log_path):
                    # Keep last 1000 lines of log files
                    try:
                        with open(log_path, 'r') as f:
                            lines = f.readlines()

                        if len(lines) > 1000:
                            # Keep only last 1000 lines
                            with open(log_path, 'w') as f:
                                f.writelines(lines[-1000:])

                            results["actions_taken"].append(f"Truncated {log_path} to last 1000 lines")
                    except Exception as e:
                        logger.error(f"Failed to truncate {log_path}: {e}")
                else:
                    # For directories, delete files older than 1 day
                    for root, dirs, files in os.walk(log_path):
                        for file in files:
                            filepath = os.path.join(root, file)
                            try:
                                age_days = (time.time() - os.path.getmtime(filepath)) / (24 * 3600)
                                if age_days > 1:
                                    size = os.path.getsize(filepath)
                                    os.remove(filepath)
                                    results["files_deleted"] += 1
                                    results["space_freed_gb"] += size / (1024**3)
                                    results["actions_taken"].append(f"Deleted old log: {filepath}")
                            except Exception as e:
                                logger.error(f"Failed to delete {filepath}: {e}")

        # Delete old checkpoints (keep only 2 most recent)
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            checkpoints = []
            for pattern in ["**/checkpoint_*.json", "**/checkpoint_*.pth"]:
                checkpoints.extend(list(checkpoint_dir.glob(pattern)))

            # Group by directory and sort by timestamp
            checkpoint_groups = {}
            for ckpt in checkpoints:
                key = str(ckpt.parent)
                try:
                    timestamp = int(ckpt.stem.split('_')[1])
                    if key not in checkpoint_groups:
                        checkpoint_groups[key] = []
                    checkpoint_groups[key].append((ckpt, timestamp))
                except (ValueError, IndexError):
                    continue

            # Keep only 2 most recent per group
            for group, files in checkpoint_groups.items():
                if len(files) > 2:
                    files.sort(key=lambda x: x[1], reverse=True)
                    to_delete = files[2:]

                    for filepath, _ in to_delete:
                        try:
                            size = filepath.stat().st_size
                            filepath.unlink()
                            results["files_deleted"] += 1
                            results["space_freed_gb"] += size / (1024**3)
                            results["actions_taken"].append(f"Deleted old checkpoint: {filepath}")
                        except Exception as e:
                            logger.error(f"Failed to delete checkpoint {filepath}: {e}")

        return results


class ContextWindowManager:
    """Manages Gemini API context window usage and limits"""

    def __init__(self, max_tokens_per_minute: int = 60000, max_tokens_per_day: int = 2000000):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_tokens_per_day = max_tokens_per_day

        # Usage tracking
        self.minute_usage = []
        self.daily_usage = []
        self.context_window_errors = []

    def track_token_usage(self, tokens_used: int):
        """Track token usage for rate limiting"""
        current_time = time.time()

        # Track minute usage (keep last 60 seconds)
        self.minute_usage.append((current_time, tokens_used))
        self.minute_usage = [(t, tokens) for t, tokens in self.minute_usage
                           if current_time - t < 60]

        # Track daily usage (keep last 24 hours)
        self.daily_usage.append((current_time, tokens_used))
        self.daily_usage = [(t, tokens) for t, tokens in self.daily_usage
                          if current_time - t < 86400]

    def check_rate_limits(self) -> Dict[str, Any]:
        """Check if we're approaching rate limits"""
        current_time = time.time()

        # Calculate current usage
        minute_tokens = sum(tokens for t, tokens in self.minute_usage
                          if current_time - t < 60)
        daily_tokens = sum(tokens for t, tokens in self.daily_usage
                         if current_time - t < 86400)

        status = {
            "minute_usage": minute_tokens,
            "minute_limit": self.max_tokens_per_minute,
            "minute_percentage": (minute_tokens / self.max_tokens_per_minute) * 100,
            "daily_usage": daily_tokens,
            "daily_limit": self.max_tokens_per_day,
            "daily_percentage": (daily_tokens / self.max_tokens_per_day) * 100,
            "can_make_request": True,
            "wait_seconds": 0
        }

        # Check limits
        if status["minute_percentage"] > 95:
            status["can_make_request"] = False
            status["wait_seconds"] = 60  # Wait 1 minute
        elif status["daily_percentage"] > 95:
            status["can_make_request"] = False
            status["wait_seconds"] = 3600  # Wait 1 hour

        return status

    def log_context_error(self, error_type: str, details: str):
        """Log context window related errors"""
        error_entry = {
            "timestamp": time.time(),
            "error_type": error_type,
            "details": details
        }

        self.context_window_errors.append(error_entry)

        # Keep only recent errors (last 24 hours)
        cutoff_time = time.time() - 86400
        self.context_window_errors = [e for e in self.context_window_errors
                                    if e["timestamp"] > cutoff_time]

        # Write to error log
        with open("api_errors.log", "a") as f:
            json.dump(error_entry, f)
            f.write("\n")

    def get_context_window_status(self) -> str:
        """Get current context window status report"""
        limits = self.check_rate_limits()

        status = f"""
Context Window Status:
- Minute Usage: {limits['minute_usage']:,} / {limits['minute_limit']:,} tokens ({limits['minute_percentage']:.1f}%)
- Daily Usage: {limits['daily_usage']:,} / {limits['daily_limit']:,} tokens ({limits['daily_percentage']:.1f}%)
- Recent Errors: {len(self.context_window_errors)}
"""

        if not limits["can_make_request"]:
            status += f"- ⚠️  Rate limited! Wait {limits['wait_seconds']} seconds\n"

        return status


def main():
    """Main storage management entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Storage Management System")
    parser.add_argument("--report", action="store_true", help="Generate storage report")
    parser.add_argument("--optimize", action="store_true", help="Run storage optimization")
    parser.add_argument("--emergency", action="store_true", help="Run emergency cleanup")
    parser.add_argument("--max-gb", type=float, default=50.0, help="Maximum storage in GB")

    args = parser.parse_args()

    manager = StorageManager(max_storage_gb=args.max_gb)

    if args.report:
        report = manager.create_storage_report()
        print(report)
    elif args.optimize:
        print("Running storage optimization...")
        results = manager.optimize_storage()
        print(f"Optimization complete:")
        print(f"- Files compressed: {results['files_compressed']}")
        print(f"- Files deleted: {results['files_deleted']}")
        print(f"- Space saved: {results['space_saved_bytes'] / (1024**3):.2f} GB")
    elif args.emergency:
        print("Running emergency cleanup...")
        results = manager.emergency_cleanup()
        print(f"Emergency cleanup complete:")
        print(f"- Files deleted: {results['files_deleted']}")
        print(f"- Space freed: {results['space_freed_gb']:.2f} GB")
    else:
        # Default: show status
        status = manager.check_storage_limits()
        print(f"Storage Status: {status['status']}")
        print(f"Usage: {status['current_usage_gb']:.2f}/{status['max_allowed_gb']:.1f} GB ({status['usage_percentage']:.1f}%)")

        if status['warnings']:
            print("\nWarnings:")
            for warning in status['warnings']:
                print(f"- {warning}")


if __name__ == "__main__":
    main()
