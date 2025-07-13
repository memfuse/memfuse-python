#!/usr/bin/env python3
"""Test runner script for MemFuse Python SDK.

This script runs tests layer by layer according to the testing strategy guide.
Each layer must pass before proceeding to the next layer.
"""

import sys
import subprocess
import os
from pathlib import Path
import argparse


# Test layers in order of execution
LAYERS = [
    ("smoke", "tests/smoke"),
    ("unit", "tests/unit"),
    ("error_handling", "tests/error_handling"),
    ("integration", "tests/integration"),
    ("dx", "tests/dx"),
    ("e2e", "tests/e2e")
]


def run_layer(name, path, verbose=False):
    """Run tests for a specific layer.
    
    Args:
        name: Name of the test layer
        path: Path to the test directory
        verbose: Whether to show verbose output
        
    Returns:
        bool: True if tests passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running {name} tests...")
    print(f"{'='*60}")

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", path, "-m", name]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Run the tests
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n❌ {name} tests FAILED!")
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return False

    print(f"✅ {name} tests PASSED!")
    if verbose:
        print(result.stdout)
    return True


def run_specific_layer(layer_name, verbose=False):
    """Run tests for a specific layer only.
    
    Args:
        layer_name: Name of the layer to run
        verbose: Whether to show verbose output
        
    Returns:
        bool: True if tests passed, False otherwise
    """
    for name, path in LAYERS:
        if name == layer_name:
            if not Path(path).exists():
                print(f"❌ Layer '{layer_name}' not found at path {path}")
                return False
            return run_layer(name, path, verbose)
    
    print(f"❌ Unknown layer '{layer_name}'")
    print(f"Available layers: {', '.join([name for name, _ in LAYERS])}")
    return False


def main():
    """Run all test layers in sequence or a specific layer."""
    parser = argparse.ArgumentParser(description='Run MemFuse Python SDK tests')
    parser.add_argument('--layer', '-l', help='Run specific layer only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--list', action='store_true', help='List available layers')
    
    args = parser.parse_args()
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    if args.list:
        print("Available test layers:")
        for name, path in LAYERS:
            exists = "✅" if Path(path).exists() else "❌"
            print(f"  {exists} {name:<15} - {path}")
        return
    
    if args.layer:
        success = run_specific_layer(args.layer, args.verbose)
        sys.exit(0 if success else 1)
    
    # Run all layers in sequence
    print("🚀 Starting MemFuse Python SDK test suite...")
    print(f"Project root: {project_root}")
    
    for name, path in LAYERS:
        if not Path(path).exists():
            print(f"⏭️  Skipping {name} tests - path {path} not found")
            continue

        if not run_layer(name, path, args.verbose):
            print(f"\n❌ Stopping test run due to {name} layer failure")
            sys.exit(1)

    print("\n🎉 All test layers passed!")
    print("✅ MemFuse Python SDK tests completed successfully!")


if __name__ == "__main__":
    main() 