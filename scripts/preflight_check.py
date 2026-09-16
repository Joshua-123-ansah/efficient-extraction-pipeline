#!/usr/bin/env python3
"""Wrapper so `python scripts/preflight_check.py` runs the package-root module."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preflight_check import main

if __name__ == "__main__":
    main()
