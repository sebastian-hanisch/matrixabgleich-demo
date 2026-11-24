"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def hand():
    """Handbeispiel 1: alte Matrix [[1, 2], [3, 4]], Zeilenziele (5, 6), Spaltenziele (5, 6). Grenzwert [[2, 3], [3, 3]] (von Hand gerechnet, siehe Expander der App)."""
    return np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([5.0, 6.0]), np.array([5.0, 6.0])
