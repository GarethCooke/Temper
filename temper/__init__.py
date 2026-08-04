"""Temper — a reinforcement-learning execution agent graded against its oracle.

Sub-packages land per milestone (see ``ROADMAP.md``): ``oracle`` (M0, here),
``env`` (M1), ``agents`` (M2), ``eval`` (M2).

Constitution invariant 8: nothing under ``temper/`` performs network I/O. The
Anvil participant lives in ``client/`` and consumes this package.
"""

__all__ = ["__version__"]

__version__ = "0.1.0.dev0"
