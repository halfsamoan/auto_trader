#!/usr/bin/env python3
"""Build KOSPI200 + KOSDAQ representative AI universe JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.universe_builder import build_ai_universe


def main() -> int:
    result = build_ai_universe(write=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
