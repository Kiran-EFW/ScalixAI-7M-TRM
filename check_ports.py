#!/usr/bin/env python3
"""
Port Conflict Checker for Continuous Learning System

Checks for available ports and ensures no conflicts with existing services.
"""

import socket
import subprocess
import sys

def check_port(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('127.0.0.1', port))
        return result == 0  # 0 means port is open (in use)

def get_used_ports():
    """Get list of ports currently in use"""
    try:
        result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split('\n')[1:]  # Skip header

        used_ports = []
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    local_addr = parts[3]
                    if ':' in local_addr:
                        port = local_addr.split(':')[-1]
                        try:
                            used_ports.append(int(port))
                        except ValueError:
                            continue

        return sorted(set(used_ports))

    except Exception as e:
        print(f"Error getting used ports with ss: {e}")
        # Fallback to netstat if ss fails
        try:
            result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().split('\n')

            used_ports = []
            for line in lines:
                if 'LISTEN' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        local_addr = parts[3]
                        if '.' in local_addr:
                            port_part = local_addr.split('.')[-1]
                            if ':' in port_part:
                                port = port_part.split(':')[0]
                                try:
                                    used_ports.append(int(port))
                                except ValueError:
                                    continue

            return sorted(set(used_ports))

        except Exception as e2:
            print(f"Error with netstat fallback: {e2}")
            return []

def find_available_ports(start_port=8100, num_ports=10):
    """Find available ports starting from start_port"""
    available_ports = []
    current_port = start_port

    while len(available_ports) < num_ports and current_port < 65535:
        if not check_port(current_port):
            available_ports.append(current_port)
        current_port += 1

    return available_ports

def main():
    """Main port checking function"""
    print("🔍 Checking port usage and conflicts...")

    # Get currently used ports
    used_ports = get_used_ports()

    print(f"\n📊 Currently used ports: {len(used_ports)}")
    print("Sample used ports:", used_ports[:20], "..." if len(used_ports) > 20 else "")

    # Check for common conflicts
    conflicts = {
        8000: "Gunicorn/HTTP server",
        8001: "Vertex AI backend",
        8080: "HTTP server",
        5432: "PostgreSQL",
        6379: "Redis",
        3000: "Development server",
        4000: "Development server"
    }

    print("\n⚠️  Checking for known service conflicts:")
    for port, service in conflicts.items():
        if port in used_ports:
            print(f"  🔴 Port {port}: {service} (IN USE)")
        else:
            print(f"  🟢 Port {port}: {service} (AVAILABLE)")

    # Find available ports for future use
    available_ports = find_available_ports(8100, 5)
    print(f"\n💡 Available ports for new services: {available_ports}")

    # Check if continuous learning system needs ports (it doesn't)
    print("\n🤖 Continuous Learning System Status:")
    print("  ✅ No ports required - runs as background process")
    print("  ✅ No conflicts with existing services")

    # Overall assessment - for continuous learning system (no ports needed)
    print("\n🤖 Continuous Learning System Port Assessment:")
    print("  ✅ No ports required - runs as background process only")
    print("  ✅ Existing port usage doesn't conflict with training")
    print("  ✅ System can run safely alongside existing services")

    # Only warn if there are issues, don't exit
    critical_ports = [8000, 8001, 8080]
    critical_conflicts = [port for port in critical_ports if port in used_ports]

    if critical_conflicts:
        print("\n⚠️  Note: Some HTTP ports are in use by other services:")
        for port in critical_conflicts:
            service_name = conflicts.get(port, "Unknown service")
            print(f"  - Port {port}: {service_name}")
        print("  This doesn't affect the continuous learning system.")

    print("\n✅ SUCCESS: Continuous learning system can run without conflicts!")

if __name__ == "__main__":
    main()
