#!/usr/bin/env python3
"""
Simple test runner for the crypto trading bot.
"""

import subprocess
import sys
import os

def run_tests():
    """Run the test suite."""
    print("🧪 Running Crypto Trading Bot Tests...")
    print("=" * 50)
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "pytest-mock"])
    
    # Run tests
    print("🚀 Starting test execution...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--tb=short"
    ])
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"🧪 Running specific test: {test_file}")
    print("=" * 50)
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        test_file, 
        "-v", 
        "--tb=short"
    ])
    
    if result.returncode == 0:
        print(f"\n✅ Test {test_file} passed!")
    else:
        print(f"\n❌ Test {test_file} failed!")
        sys.exit(1)

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        if os.path.exists(test_file):
            run_specific_test(test_file)
        else:
            print(f"❌ Test file not found: {test_file}")
            sys.exit(1)
    else:
        run_tests()

if __name__ == "__main__":
    main()
