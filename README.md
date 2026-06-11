# README.md

## KIS Open API 기반 Intraday 자동 매매 봇 (MVP)

이 프로젝트는 **한국투자증권(KIS) Open API V3.0** 을 활용한 **데이 트레이딩** 자동화 봇의 첫 번째 MVP 버전입니다.

### 주요 특징
- **Python 전용** 로 구현 (다른 언어 사용 금지)
- `--dry-run` 옵션으로 실제 주문 없이 시뮬레이션만 수행
- `yfinance` 로 **005930 (삼성전자), 000660 (SK하이닉스), 035720 (NAVER)** 의 실시간(1분) 데이터를 받아 분석
- **Gaussian Score Engine** (trend, momentum, fundamental, risk, market 5축) 적용
- **Intraday Scorer** (5분봉) 와 Gaussian Score 를 혼합하여 `final_score` 계산
- `final_score` 에 따라 **매수 / 보류 / 회피** (buy/hold/avoid) 신호 출력 (한국어)
- KIS 실제 주문 모듈은 **stub** 으로 제공되어 절대 실행되지 않음
- 민감 정보는 `.env.example` 에만 명시하고 실제 `.env` 파일은 **.gitignore**에 추가 (절대 커밋 금지)

### 설치 방법
```bash
# 프로젝트 디렉터리 이동
cd ~/auto_trader

# 가상환경 생성 (선택)
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 실행 방법 (Dry‑Run)
```bash
python main.py --dry-run
```
위 명령을 실행하면 각 종목에 대한 최종 점수와 매수/보류/회피 신호가 콘솔에 한국어로 출력됩니다.

### 백테스트 (Intraday)
```bash
python backtest_intraday.py 005930 --period 60d
```
- `--period` 는 조회 기간을 **일(day)** 단위로 지정합니다.
- 지정된 기간 동안 5분봉으로 점수를 계산하고, 각 구간별 신호 요약을 제공합니다.

### 파일 구조
```
auto_trader/
├─ .env.example          # KIS API 키 샘플 (실제 값은 넣지 않음)
├─ .gitignore            # 민감 파일 및 캐시 무시
├─ README.md
├─ requirements.txt
├─ logger.py              # 한글 로그 출력 유틸
├─ main.py                # 진입점 (dry‑run)
├─ backtest_intraday.py   # 백테스트 스크립트
└─ core/
   ├─ technical.py        # 기술 지표 (SMA, EMA, RSI 등)
   ├─ gaussian_score_engine.py
   ├─ intraday_scorer.py
   ├─ fetcher_daily.py    # 일간 데이터 (예비)
   ├─ fetcher_intraday.py
   └─ strategy.py
```

### 주의 사항
- **실제 주문** 은 아직 구현되지 않았으며, `kis_order_stub.py` 로 대체되었습니다. 나중에 실제 주문 로직을 연결하기 전까지는 절대 활성화되지 않으니 안심하세요.
- `.env` 파일은 자동으로 생성되지 않으며, 반드시 **`.env.example`** 을 복사해서 사용자는 직접 채워야 합니다.
- 모든 출력 및 주석은 **한국어** 로 작성되었습니다.

---

### 향후 작업 계획
1. 실제 KIS 주문 모듈 구현 및 테스트 (안전 검증 후 활성화)
2. 추가 기술 지표 및 파라미터 튜닝
3. 리스크 관리 로직 (포지션 사이징, 손절/익절) 추가
4. Docker 이미지 제공 및 CI/CD 파이프라인 구축

---

> 프로젝트에 대한 질문이나 개선 요청이 있으면 언제든 알려 주세요!
