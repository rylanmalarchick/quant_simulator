"""
Crisis/Bubble Detection Module

Implements statistical mechanics inspired indicators to detect
market stress and bubble conditions.
"""

from .detector import CrisisDetector, run_crisis_check

__all__ = ['CrisisDetector', 'run_crisis_check']
