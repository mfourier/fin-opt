#!/usr/bin/env python3
"""Simple test runner to debug pytest issues."""

import sys
from datetime import date

print("=" * 60)
print("WITHDRAWAL MODULE TEST RUNNER")
print("=" * 60)

# Test 1: Import modules
print("\n[TEST 1] Importing modules...")
try:
    from src.withdrawal import (
        WithdrawalEvent,
        WithdrawalSchedule,
        StochasticWithdrawal,
        WithdrawalModel
    )
    from src.portfolio import Account
    print("✓ PASSED: All imports successful")
except ImportError as e:
    print(f"✗ FAILED: Import error - {e}")
    sys.exit(1)

# Test 2: Create WithdrawalEvent
print("\n[TEST 2] Creating WithdrawalEvent...")
try:
    event = WithdrawalEvent(
        account="Conservador",
        amount=400_000,
        date=date(2025, 6, 1),
        description="Compra bicicleta"
    )
    assert event.account == "Conservador"
    assert event.amount == 400_000
    assert event.date == date(2025, 6, 1)
    print(f"✓ PASSED: {event}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create accounts
print("\n[TEST 3] Creating accounts...")
try:
    accounts = [
        Account.from_annual("Conservador", annual_return=0.06, annual_volatility=0.08),
        Account.from_annual("Agresivo", annual_return=0.12, annual_volatility=0.15)
    ]
    assert len(accounts) == 2
    print(f"✓ PASSED: Created {len(accounts)} accounts")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: WithdrawalSchedule.to_array()
print("\n[TEST 4] Testing WithdrawalSchedule.to_array()...")
try:
    schedule = WithdrawalSchedule(events=[event])
    D = schedule.to_array(T=12, start_date=date(2025, 1, 1), accounts=accounts)
    
    assert D.shape == (12, 2), f"Expected shape (12, 2), got {D.shape}"
    assert D[5, 0] == 400_000, f"Expected D[5,0]=400000, got {D[5,0]}"
    assert D[5, 1] == 0, f"Expected D[5,1]=0, got {D[5,1]}"
    
    print(f"✓ PASSED: Shape {D.shape}, D[5,0]={D[5,0]:,.0f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: StochasticWithdrawal
print("\n[TEST 5] Testing StochasticWithdrawal...")
try:
    import numpy as np
    withdrawal = StochasticWithdrawal(
        account="Conservador",
        base_amount=300_000,
        sigma=50_000,
        month=6,
        floor=200_000,
        cap=400_000,
        seed=42
    )
    samples = withdrawal.sample(n_sims=100)
    
    assert samples.shape == (100,), f"Expected shape (100,), got {samples.shape}"
    assert (samples >= 200_000).all(), "Some samples below floor"
    assert (samples <= 400_000).all(), "Some samples above cap"
    
    print(f"✓ PASSED: Generated {len(samples)} samples, mean={samples.mean():,.0f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: WithdrawalModel
print("\n[TEST 6] Testing WithdrawalModel...")
try:
    model = WithdrawalModel(
        scheduled=WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
        ]),
        stochastic=[
            StochasticWithdrawal(
                account="Conservador",
                base_amount=300_000,
                sigma=50_000,
                date=date(2025, 9, 1),
                seed=42
            )
        ]
    )
    
    D = model.to_array(
        T=12,
        start_date=date(2025, 1, 1),
        accounts=accounts,
        n_sims=10,
        seed=42
    )
    
    assert D.shape == (10, 12, 2), f"Expected shape (10, 12, 2), got {D.shape}"
    # All scenarios should have same deterministic withdrawal
    assert np.allclose(D[:, 5, 0], 400_000), "Deterministic withdrawal mismatch"
    # Stochastic withdrawal should vary
    assert D[:, 8, 0].std() > 0, "Stochastic withdrawal has no variance"
    
    print(f"✓ PASSED: Shape {D.shape}")
    print(f"  Deterministic (June): {D[:, 5, 0].mean():,.0f} (std={D[:, 5, 0].std():.0f})")
    print(f"  Stochastic (Sep): {D[:, 8, 0].mean():,.0f} (std={D[:, 8, 0].std():,.0f})")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
