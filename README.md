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
python main.py --paper --asset futures
python main.py --paper-sim --asset futures
python backtest_futures.py --watchlist MNQ,MES --period 60d --capital 5000000
```

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
