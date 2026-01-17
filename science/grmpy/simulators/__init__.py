"""
Simulators module for generating synthetic Roy model data.

Design Decision: AVAILABLE_FUNCTIONS exported for dynamic validation,
enabling contracts.py to derive valid options rather than hard-coding.
"""

from typing import List

from grmpy.simulators.simulate import simulate

# Available simulation functions - used by contracts.py for validation
AVAILABLE_FUNCTIONS: List[str] = ["roy_model"]


__all__ = ["simulate", "AVAILABLE_FUNCTIONS"]
