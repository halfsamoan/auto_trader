# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/gaussian_score_engine.py, core/fetcher_daily.py
# Description: Gaussian 점수 엔진 학습/점검용 데이터셋을 생성합니다.
# ================================================================================

"""Optional dataset inspection utility for the deterministic Gaussian engine."""

from __future__ import annotations

import argparse

from core.fetcher_daily import fetch_daily
from core.gaussian_score_engine import GaussianScoreEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Gaussian score dataset preview")
    parser.add_argument("codes", nargs="*", default=["005930", "000660", "035720"])
    parser.add_argument("--period", default="1y")
    args = parser.parse_args()

    engine = GaussianScoreEngine()
    for code in args.codes:
        df = fetch_daily(code, period=args.period)
        score = engine.compute(code, df)
        print(f"{code}: {score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
