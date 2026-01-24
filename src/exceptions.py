"""
Custom exceptions for FinOpt.

Purpose
-------
Provides a unified exception hierarchy for consistent error handling
across all FinOpt modules. All exceptions inherit from FinOptError,
enabling catch-all handling when needed.

Exception Hierarchy
-------------------
FinOptError (base)
├── ConfigurationError - Invalid configuration or parameters
├── ValidationError - Data validation failures
│   ├── TimeIndexError - Month/date indexing errors
│   └── AllocationConstraintError - Allocation policy violations
├── OptimizationError - Solver failures
│   └── InfeasibleError - No feasible solution exists
└── MemoryLimitError - Memory limit exceeded

Usage
-----
>>> from src.exceptions import ValidationError, InfeasibleError
>>>
>>> # Raise specific exception
>>> raise ValidationError("T must be positive, got -1")
>>>
>>> # Catch all FinOpt exceptions
>>> try:
...     result = model.optimize(...)
>>> except FinOptError as e:
...     print(f"FinOpt error: {e}")
"""


class FinOptError(Exception):
    """
    Base exception for all FinOpt errors.

    All FinOpt-specific exceptions inherit from this class,
    enabling unified error handling when needed.

    Examples
    --------
    >>> try:
    ...     model.optimize(goals)
    ... except FinOptError as e:
    ...     logger.error(f"Optimization failed: {e}")
    """
    pass


class ConfigurationError(FinOptError):
    """
    Invalid configuration or parameters.

    Raised when model configuration is invalid, such as:
    - Invalid annual_growth values (must be > -1)
    - Incompatible parameter combinations
    - Missing required configuration fields

    Examples
    --------
    >>> raise ConfigurationError(
    ...     "annual_growth must be > -1, got -2.0. "
    ...     "Value <= -1 would cause income to become zero or negative."
    ... )
    """
    pass


class ValidationError(FinOptError):
    """
    Data validation failures.

    Raised when input data fails validation checks, such as:
    - Invalid array shapes
    - Out-of-bounds values
    - Constraint violations

    Examples
    --------
    >>> raise ValidationError(
    ...     f"T must be positive, got {T}. "
    ...     f"Use T >= 1 for valid return simulations."
    ... )
    """
    pass


class TimeIndexError(ValidationError):
    """
    Month/date indexing errors.

    Raised when month or date specifications are invalid:
    - Month out of valid range
    - Date before simulation start
    - Inconsistent date/month specifications

    Examples
    --------
    >>> raise TimeIndexError(
    ...     f"Withdrawal month {month} is before simulation start (month 1). "
    ...     f"Months are 1-indexed: month 1 = end of first simulation month."
    ... )
    """
    pass


class AllocationConstraintError(ValidationError):
    """
    Allocation policy constraint violations.

    Raised when allocation policy X violates constraints:
    - Negative allocations (x_t^m < 0)
    - Simplex violations (sum of allocations != 1)
    - Invalid allocation array shape

    Examples
    --------
    >>> raise AllocationConstraintError(
    ...     f"Allocation X has negative values at positions: {negative_locs}. "
    ...     f"Min value: {X.min():.6f}. Allocations must be non-negative."
    ... )
    """
    pass


class OptimizationError(FinOptError):
    """
    Optimization solver failures.

    Raised when the optimization solver encounters errors:
    - Solver convergence failure
    - Numerical instability
    - Unexpected solver status

    Examples
    --------
    >>> raise OptimizationError(
    ...     f"Solver returned unexpected status: {prob.status}. "
    ...     f"Try adjusting solver settings or using a different solver."
    ... )
    """
    pass


class InfeasibleError(OptimizationError):
    """
    No feasible solution exists.

    Raised when the optimization problem has no feasible solution:
    - Goals cannot be achieved within the horizon
    - Withdrawal constraints cannot be satisfied
    - Conflicting constraint requirements

    Examples
    --------
    >>> raise InfeasibleError(
    ...     f"No feasible solution found in T ∈ [{T_min}, {T_max}]. "
    ...     f"Consider: (1) increasing T_max, (2) relaxing goal thresholds, "
    ...     f"(3) reducing withdrawal amounts, (4) increasing epsilon tolerances."
    ... )
    """
    pass


class MemoryLimitError(FinOptError):
    """
    Memory limit exceeded.

    Raised when an operation would exceed memory limits:
    - Large accumulation factor arrays
    - Too many Monte Carlo scenarios
    - Excessive horizon length

    Examples
    --------
    >>> raise MemoryLimitError(
    ...     f"Accumulation factors would require {memory_gb:.1f} GB. "
    ...     f"Use LazyAccumulationFactors for large problems, or reduce: "
    ...     f"n_sims={n_sims}, T={T}, M={M}"
    ... )
    """
    pass
