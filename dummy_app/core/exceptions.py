"""Custom exception types used to control expected execution flows."""


class ValidationOptimalityConfirmed(Exception):
    """Raised when validation proves the previously found MILP solution is truly optimal."""
