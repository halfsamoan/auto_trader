# auto_trader V3.2 Paper Lab KR + Overseas Futures Final

한국투자증권 KIS OpenAPI 기반 intraday auto-trading bot입니다. 이번 버전은 **국내주식 + CME 해외선물 모의투자 준비**만 다룹니다. 해외주식은 제외했습니다.

최종 목표는 KIS 해외선물 모의투자 계좌에서 `MNQ/MES` 지정가 모의 주문을 실행하는 것입니다. 다만 KIS 해외선물 모의투자 지원 여부, endpoint, TR ID, body field가 공식 문서 또는 샘플 코드로 검증되기 전까지 실제 주문 API 호출은 차단됩니다.

## 대상 자산

국내주식 실험 자본: `5,000,000 KRW`

- `005930` 삼성전자
- `000660` SK하이닉스
- `035720` 카카오는 제외 유지

해외선물 전략 배정 자본: `5,000,000 KRW`

- `MNQ` Micro Nasdaq 100 Futures
- `MES` Micro S&P 500 Futures

`MNQ/MES`는 국내선물이 아니라 CME 해외선물입니다. KIS 국내주식 paper API와 KIS 해외선물 paper API는 다른 상품군이며, KIS 해외선물 OpenAPI/모의투자 지원 여부, endpoint, TR ID, body field는 수동 검증이 필요합니다. 모의계좌 잔고가 5백만원보다 커도 봇은 `FUTURES_PAPER_CAPITAL_KRW=5000000` 안에서만 수량을 계산합니다.

## 실행

```bash
python main.py --dry-run --asset domestic-stock
python main.py --paper-check --asset domestic-stock
python main.py --dry-run --asset futures
python main.py --paper-check --asset futures
python main.py --paper-check --asset futures --kis-live-probe
python main.py --paper --asset futures
python main.py --paper-sim --asset futures
python backtest_futures.py --watchlist MNQ,MES --period 60d --capital 5000000
```

`--kis-live-probe`는 주문 API를 호출하지 않고 해외선물 조회 API만 실검증합니다. `KIS_FUTURES_ACCOUNT_NO`, `KIS_FUTURES_ACCOUNT_PRODUCT_CODE`가 필요하며, 해외선물 계좌 상품코드는 계좌별 확인이 필요합니다. 현재 확인된 예시는 `03`입니다. 국내주식 계좌 상품코드 `KIS_ACCOUNT_PRODUCT_CODE=01`과 해외선물 계좌 상품코드는 분리해서 유지합니다. 계좌번호는 마스킹해서 출력합니다. MNQ/MES 현재가와 주문가능조회는 `KIS_FUTURES_MNQ_PRODUCT_CODE`, `KIS_FUTURES_MES_PRODUCT_CODE`가 수동 검증되어 `.env`에 들어간 경우에만 호출됩니다.

국내주식 paper 지정가 주문은 아래 조건을 모두 만족할 때만 호출될 수 있습니다.

```bash
python main.py --paper --asset domestic-stock --allow-paper-order
```

해외선물 KIS paper 지정가 주문의 최종 실행 명령은 아래입니다.

```bash
python main.py --paper --asset futures --allow-paper-order
```

단, `paper-check` 성공, `KIS_MODE=paper`, `ENABLE_REAL_ORDER=False`, `ENABLE_FUTURES_KIS_PAPER_ORDER=True`, contract metadata의 `is_kis_paper_order_enabled=True`, KIS 해외선물 TR ID/endpoint/body field 검증 완료, 현재가 조회 성공, margin check 통과, daily trade limit 통과, 전략 신호 통과 조건이 모두 맞아야 합니다. 현재 기본값은 차단입니다.

## Paper-Sim 체결 모델

해외선물 paper-sim은 보조 기능입니다. 실제 KIS 주문 API를 호출하지 않고 `data/paper_positions.json`, `data/paper_trades.json`, `data/paper_equity.json`만 갱신합니다.

- 롱 진입 limit buy: 다음 봉 `Low <= limit_price`이면 체결, 체결가 `limit_price + 1 tick`
- 롱 청산 limit sell: 다음 봉 `High >= limit_price`이면 체결, 체결가 `limit_price - 1 tick`
- 손절: 다음 봉 `Low <= stop_loss`이면 체결, 체결가 `stop_loss - 1 tick`
- 목표가: 다음 봉 `High >= target`이면 체결, 체결가 `target - 1 tick`
- 같은 봉에서 손절과 목표가가 모두 닿으면 보수적으로 손절 우선

손익 계산:

```text
pnl_usd = (price_move / tick_size) * tick_value_usd * qty
pnl_krw = pnl_usd * FX_RATE_USDKRW
```

증거금 체크:

```text
required_margin_krw = margin_per_contract_usd * qty * FX_RATE_USDKRW
required_margin_krw <= FUTURES_PAPER_CAPITAL_KRW * 0.8
```

## 국내주식 진입 필터 / paper-watch 루프

`config.py`의 `MIN_TARGET1_PROFIT_PCT`(기본 0.0045)는 target1 기대수익 최소 기준입니다. 기존 0.6% 하드코딩 필터가 대형주 5분봉에서 진입을 거의 전부 차단해 config화했습니다. 백테스트 비교 후보: 0.0035(공격적) / 0.0045(중간) / 0.006(보수적). `calculate_intraday_plan(df, min_target1_profit_pct=...)`로 호출부에서 개별 override도 가능합니다.

paper-watch 루프는 5분봉 마감 + `PAPER_WATCH_BAR_OFFSET_SEC`(기본 5초) 후 봉당 1회만 신호를 계산합니다. 같은 봉에서는 재실행하지 않으며(`last_processed_bar_key` 기준), `--interval-sec`는 루프 wake 주기로 그대로 반영됩니다. 봉 간격은 `PAPER_WATCH_BAR_INTERVAL_SEC`(기본 300초)로 조정합니다.

신호 분포 분석(주문 API 호출 없음):

```bash
python scripts/analyze_signal_distribution.py
```

## PatchTST AI Shadow Signal

국내주식 paper-watch에는 PatchTST-lite 기반 로컬 AI 판단 모듈을 shadow mode로만 연결합니다. 기본값은 `AI_SHADOW_MODE=True`이며, `--ai-gate`를 주지 않으면 AI 결과는 `data/signal_log.json`에만 기록되고 기존 rule-based 주문 판단을 바꾸지 않습니다. `--ai-gate` 옵션은 구조만 준비되어 있으며, 현재 기본 설정에서는 shadow mode 사유가 로그에 남습니다. AI 단독 buy는 허용하지 않습니다.

AI 의존성은 자동 설치하지 않습니다. 필요한 경우 직접 설치하세요.

```bash
.venv/bin/python -m pip install -r requirements-ai.txt
.venv/bin/python ai/train_patchtst.py --watchlist 005930,000660 --period 120d --split-dry-run
.venv/bin/python ai/train_patchtst.py --watchlist 005930,000660 --period 120d --epochs 10
.venv/bin/python ai/evaluate_ai_signal.py --model ai/model_store/domestic_patchtst_model.pt
.venv/bin/python main.py --paper-watch --asset domestic-stock --allow-paper-order --interval-sec 300 --ai-shadow
```

국내 5분봉 데이터는 KIS 국내주식 당일분봉조회 `quotations` API를 primary로 사용하고, KIS WebSocket tick cache가 있으면 보조 source로 resample하며, yfinance는 seed/fallback으로만 사용합니다. 주문/계좌 API는 학습 데이터 수집에서 호출하지 않습니다. yfinance 5m 데이터는 최근 60일 제한이 있으므로 `data/intraday_cache`는 반복 수집으로 누적되는 저장소입니다. 새로 받은 데이터와 캐시의 timestamp가 겹치면 last-wins로 새 값을 남깁니다.

캐시 schema는 `timestamp, open, high, low, close, volume, code, source, collected_at`입니다. source 값은 `kis_intraday`, `kis_websocket_cache`, `yfinance_fallback` 중 하나로 기록됩니다. tick/호가 scaffold는 각각 `data/tick_cache/{code}_ticks.csv`, `data/orderbook_cache/{code}_quote.csv`를 사용합니다.

```bash
.venv/bin/python scripts/collect_intraday_data.py --universe ai_train --source auto --period 5d --interval 5m
.venv/bin/python scripts/run_kis_tick_collector.py --universe watchlist --safe-mode
```

AI 학습 universe와 market breadth sensor는 KOSPI200 + KOSDAQ 대표 50종목입니다. KOSPI200 구성은 `pykrx.get_index_portfolio_deposit_file("1028", date)`로 동적 조회하고, KOSDAQ 대표 50은 KOSDAQ150 또는 KOSDAQ 전체에서 최근 거래대금/거래량 상위 종목으로 산정합니다. 조회 실패나 `pykrx` 미설치 시 `config.py`의 `AI_TRAIN_UNIVERSE_FALLBACK`을 사용합니다. 생성 명령은 아래와 같습니다.

```bash
.venv/bin/python scripts/build_ai_universe.py
```

생성 결과는 `data/ai_universe/ai_train_universe.json`에 저장됩니다. 이 universe는 PatchTST pretraining/fine-tuning과 market breadth sensor 용도이며, 사용자의 최신 요청에 따라 국내주식 paper-watch의 주문 가능 감시 대상도 같은 universe를 읽습니다. 그래도 실전 주문은 금지되어 있고, paper 주문도 `KIS_MODE=paper`, `--allow-paper-order`, 기존 안전장치 없이는 호출되지 않습니다.

AI feature는 5분봉 sequence length 96을 사용합니다. 현재 진행 중인 미완성 5분봉은 제외하고, VWAP는 Asia/Seoul 기준 날짜별 anchored VWAP로 계산합니다. 12:00~12:59 점심시간 봉이 데이터에 존재하면 `AI_EXCLUDE_LUNCH_BARS=True` 기본값에 따라 제외합니다. NaN/inf는 미래 데이터를 쓰지 않고 과거 방향 forward fill 후 남은 값만 0으로 대체합니다.

Label은 다음 6개 5분봉 안에서 target `+0.0035`와 stop `-0.0025` 중 먼저 닿는 쪽으로 만듭니다. 기본 `AI_LABEL_MODE="drop_neutral"`에서는 target=1, stop=0, neutral은 학습에서 제외합니다. `three_class` 모드는 target=2, neutral=1, stop=0으로 유지하며 neutral class weight를 낮춥니다. neutral을 stop=0으로 단순 병합하지 않습니다.

Split은 랜덤 분할을 쓰지 않고 전 종목 공통 달력 시간 컷오프로 나눕니다. 종목별 샘플을 concat한 뒤 row 순서 기준으로 split하지 않습니다. 전체 timestamp의 70%, 85% 지점을 train/validation/test 경계로 삼고 각 경계에서 기본 `AI_PURGE_GAP_BARS=96`을 시간 단위로 차감합니다. Binary 학습은 기본 `AI_LOSS_TYPE="bce_pos_weight"`로 positive/negative 비율에서 `pos_weight`를 자동 계산합니다.

Full 학습 전 `--split-dry-run`은 production 설정인 sequence length 96, purge gap 96 그대로 split 크기만 계산합니다. 기본적으로 validation/test가 각각 300샘플 미만이면 full 학습을 시작하지 않고 부족분을 리포트합니다. Label 분포에서 neutral 비율이 5% 미만이면 barrier 또는 horizon 재검토 경고를 출력합니다.

확률은 raw neural output을 그대로 신뢰하지 않고 validation 예측으로 Platt scaling calibrator를 저장합니다. 모델 파일은 `ai/model_store/domestic_patchtst_model.pt`, calibration은 `ai/model_store/domestic_patchtst_calibrator.pkl`, metadata는 `ai/model_store/domestic_patchtst_metadata.json`에 저장됩니다. Calibrator가 없으면 추론 로그의 `ai_status`가 `uncalibrated`로 남습니다.

권장 재학습 주기는 4주마다 최근 120영업일 또는 가능한 최신 데이터 기준입니다. 새 모델은 이전 모델과 validation/test AUC, PR-AUC, Brier score, probability bin별 target hit rate를 비교한 뒤 더 나쁠 경우 자동 교체하지 않습니다.

Phase 2는 masked patch pretraining smoke-test 단계입니다. 라벨이 부족한 상태에서 action label을 바로 학습하지 않고, 캐시에 쌓인 5분봉 sequence의 일부 patch를 가린 뒤 PatchTST encoder가 복원하도록 학습해 표현 학습 파이프라인을 검증합니다. 캐시 데이터가 부족하면 성능 검증은 불가능하며, 이 단계는 실제 주문 판단과 무관합니다.

```bash
.venv/bin/python ai/pretrain_patchtst.py --source cache --universe ai_train --smoke-test --epochs 2
```

Policy fine-tuning은 triple-barrier 기반 진입 action label을 사용합니다. target 먼저 도달 시 BUY, stop 먼저 도달 시 NO_ACTION, neutral은 기본적으로 drop하며 손실 class로 합치지 않습니다. 보유/청산 action은 실제 포지션 로그가 부족하므로 synthetic label로 별도 metadata에 표시합니다.

```bash
.venv/bin/python ai/train_patchtst_policy.py --source cache --universe ai_train --smoke-test --epochs 2
.venv/bin/python ai/evaluate_policy_model.py --model ai/model_store/domestic_patchtst_policy.pt
```

Behavior Tree는 shadow 검증 단계에서는 주문 API를 호출하지 않고 어떤 branch를 탔을지만 `signal_log.json`에 기록합니다. `bt-execute`는 scaffold만 있으며 기본 `ENABLE_BT_EXECUTE=False`라서 `--allow-paper-order`가 있어도 차단됩니다. 실행을 열기 전에는 최소 10거래일 이상의 bt-shadow 로그, `bt_would_enter`의 target hit rate, confidence bin별 성과, BUY precision 기준 충족이 필요합니다.

```bash
.venv/bin/python ai/evaluate_bt_shadow.py
.venv/bin/python main.py --paper-watch --asset domestic-stock --ai-policy --bt-shadow
.venv/bin/python main.py --paper-watch --asset domestic-stock --ai-policy --bt-execute
```

데이터 소스 전략에서 pykrx는 universe 구성과 일봉/거래대금 필터 용도로만 사용합니다. KIS WebSocket collector는 현재 safe scaffold이며, 실제 연결을 활성화하려면 별도 인증/구독 메시지 검증 후 진행해야 합니다.

TFT, Chronos, TimesFM, FinRL, RL 기반 청산/수량 조절은 이번 작업에서 구현하지 않습니다. 한국 5분봉 단타 데이터에 맞춘 보정과 별도 실험 설계가 필요하므로 향후 연구 항목으로만 둡니다.

## TODO

- KIS 해외선물 모의투자 지원 여부 확인
- KIS 해외선물 endpoint/TR ID/body field 수동 검증
- KIS 해외선물 현재가/잔고/증거금/미결제/지정가 주문 API 샘플 코드 대조
- KIS 해외선물 모의계좌 상품코드와 MNQ/MES 월물/거래소코드 매핑 검증
- MNQ/MES tick size, tick value, margin, 거래시간 수동 검증
- yfinance `MNQ=F`, `MES=F`는 front-month 연속 데이터라 롤오버 갭 검증 필요
- CME/KIS 해외선물 정확한 maintenance break 검증
- 숏 진입은 V3.3에서 별도 구현
- 실거래 수수료/환전/슬리피지 모델 정교화

## 안전 조건

- 기본 실행은 주문 없는 dry-run입니다.
- 실전 주문은 `ENABLE_REAL_ORDER=True`와 `--live` 없이는 차단됩니다.
- 시장가 주문은 사용하지 않습니다. 지정가만 사용합니다.
- `.env`, API 키, 계좌번호, access token은 커밋하지 마세요.
- 백테스트와 paper-sim 성과는 실전 수익을 보장하지 않습니다.
