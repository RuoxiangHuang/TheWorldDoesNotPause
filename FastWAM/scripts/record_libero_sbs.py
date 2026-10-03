#!/usr/bin/env python3
"""Deprecated entrypoint — prefer integrations.libero_viz.record_wallclock_sbs.

Forwards to the sim-step recorder for compatibility with older commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from integrations.libero_viz.record_simstep_sbs import main

if __name__ == "__main__":
    main()
