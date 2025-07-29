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
import time


# Test layers in order of execution
LAYERS = [
    ("smoke", "tests/smoke"),
    ("unit", "tests/unit"),
    ("error_handling", "tests/error_handling"),
    ("integration", "tests/integration"),
    ("dx", "tests/dx"),
    ("e2e", "tests/e2e"),
    ("benchmarks", "tests/benchmarks")
]


def run_layer(name, path, verbose=False, show_output=False):
    """Run tests for a specific layer.
    
    Args:
        name: Name of the test layer
        path: Path to the test directory
        verbose: Whether to show verbose output
        show_output: Whether to show test output (stdout/print statements)
        
    Returns:
        bool: True if tests passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running {name} tests...")
    print(f"{'='*60}")

    # Start timing
    start_time = time.time()

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", path, "-m", name]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if show_output:
        cmd.append("-s")  # Show output (don't capture stdout)
    
    # Run the tests
    if show_output:
        # Don't capture output so we can see print statements in real-time
        result = subprocess.run(cmd)
    else:
        # Capture output for processing
        result = subprocess.run(cmd, capture_output=True, text=True)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    if result.returncode == 5:
        # Exit code 5 means "no tests collected" - treat as success for empty layers
        print(f"⏭️  {name} tests - no tests found (empty layer) ({elapsed_time:.2f}s)")
        return True
    elif result.returncode != 0:
        print(f"\n❌ {name} tests FAILED! ({elapsed_time:.2f}s)")
        if not show_output:  # Only print captured output if we captured it
            print("STDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
        return False

    print(f"✅ {name} tests PASSED! ({elapsed_time:.2f}s)")
    if verbose and not show_output:  # Only print captured output if we captured it
        print(result.stdout)
    return True


def run_specific_layer(layer_name, verbose=False, show_output=False):
    """Run tests for a specific layer only.
    
    Args:
        layer_name: Name of the layer to run
        verbose: Whether to show verbose output
        show_output: Whether to show test output (stdout/print statements)
        
    Returns:
        bool: True if tests passed, False otherwise
    """
    for name, path in LAYERS:
        if name == layer_name:
            if not Path(path).exists():
                print(f"❌ Layer '{layer_name}' not found at path {path}")
                return False
            return run_layer(name, path, verbose, show_output)
    
    print(f"❌ Unknown layer '{layer_name}'")
    print(f"Available layers: {', '.join([name for name, _ in LAYERS])}")
    return False


def main():
    """Run all test layers in sequence or a specific layer."""
    parser = argparse.ArgumentParser(description='Run MemFuse Python SDK tests')
    parser.add_argument('--layer', '-l', help='Run specific layer only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--show-output', '-s', action='store_true', help='Show test output (stdout/print statements)')
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
        start_time = time.time()
        success = run_specific_layer(args.layer, args.verbose, args.show_output)
        elapsed_time = time.time() - start_time
        if success:
            print(f"\n✅ Layer '{args.layer}' completed successfully! (Total time: {elapsed_time:.2f}s)")
        else:
            print(f"\n❌ Layer '{args.layer}' failed! (Total time: {elapsed_time:.2f}s)")
        sys.exit(0 if success else 1)
    
    # Run all layers in sequence
    print("🚀 Starting MemFuse Python SDK test suite...")
    print(f"Project root: {project_root}")
    
    # Start total timing
    total_start_time = time.time()
    
    for name, path in LAYERS:
        if not Path(path).exists():
            print(f"⏭️  Skipping {name} tests - path {path} not found")
            continue

        if not run_layer(name, path, args.verbose, args.show_output):
            print(f"\n❌ Stopping test run due to {name} layer failure")
            sys.exit(1)

    # Calculate total elapsed time
    total_elapsed_time = time.time() - total_start_time
    
    print("\n🎉 All test layers passed!")
    print(f"✅ MemFuse Python SDK tests completed successfully! (Total time: {total_elapsed_time:.2f}s)")


if __name__ == "__main__":
    main() 