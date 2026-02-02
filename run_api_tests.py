#!/usr/bin/env python
"""
Quick test runner to verify API tests are working.
"""
import subprocess
import sys

def run_tests():
    """Run API tests and capture output."""
    print("=" * 70)
    print("Running Backend API Tests")
    print("=" * 70)
    
    test_files = [
        "tests/unit/test_api_config.py",
        "tests/unit/test_api_reconstruction_service.py",
        "tests/unit/test_api_supabase_client.py",
        "tests/integration/test_api_endpoints.py",
        "tests/integration/test_api_simulation_service.py",
        "tests/integration/test_api_optimization_service.py",
    ]
    
    for test_file in test_files:
        print(f"\n{'=' * 70}")
        print(f"Testing: {test_file}")
        print('=' * 70)
        
        result = subprocess.run(
            ["pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Tests failed for {test_file}")
            print(f"Return code: {result.returncode}")
        else:
            print(f"✅ Tests passed for {test_file}")
    
    print("\n" + "=" * 70)
    print("Test run complete!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
