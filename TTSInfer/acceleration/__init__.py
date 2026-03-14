"""
Acceleration module

Default inference wrapper: rollout/pruner_warpper_test_stream.py
Default training wrapper: rollout/pruner_warpper_train.py
"""

from .rollout.pruner_warpper_test_stream import CachePrunerWrapper

__all__ = ["CachePrunerWrapper"]
