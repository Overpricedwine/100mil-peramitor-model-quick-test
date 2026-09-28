"""Persistent, reversible, CUDA-rendered scene construction primitives."""

# Keep core memory importable on machines before the CUDA dependencies are installed.
from .core import SceneFunction, SceneSession

__all__ = ["SceneFunction", "SceneSession"]
