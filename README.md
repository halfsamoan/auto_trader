# Auto Trader

국내 주식과 해외 선물의 전략을 백테스트하고 모의 실행하는 Python 도구입니다.

KIS 시세·계좌 어댑터, 장중 데이터 수집, 전략 신호와 위험 제한을 함께 실험합니다.

## Install

Python 환경에서 저장소 루트에서 실행합니다.

```sh
python -m pip install -r requirements.txt
```

## Quickstart

주문을 보내지 않는 신호 실행입니다. 시세 데이터 접근에는 네트워크가 필요합니다.

```sh
python main.py --dry-run --asset domestic-stock
```

KIS 기능을 사용하려면 `.env.example`을 바탕으로 로컬 `.env`에 인증 정보를 설정합니다. 데이터 소스와 실행 모드에 따른 설정은 [실행 안내](CONTRIBUTING.md)를 따릅니다.

## 실험 범위

- **국내 주식:** 시세 조회, 전략 신호, 장중 백테스트와 모의 흐름을 실행합니다.
- **해외 선물:** MNQ·MES 데이터와 모의 시뮬레이션을 다룹니다.
- **학습·리플레이:** TOA AI와 정책 모델 실험 도구를 포함합니다.
- **위험 제한:** 주문 모드와 자산별 게이트를 구분합니다.

## 주문 상태

`ENABLE_REAL_ORDER=False`가 기본값입니다. 해외 선물 KIS 모의 주문도 검증 플래그가 꺼져 있어 차단됩니다. `--kis-live-probe`는 시세·계좌 조회 경로입니다.

백테스트와 모의 결과는 실거래 성과를 입증하지 않습니다. 브로커 주문 경로의 검증이 남아 있으며, 이 README의 빠른 시작은 주문 권한을 켜지 않습니다.

[환경 설정·명령 목록](CONTRIBUTING.md) · [별도 TOA AI 패키지](https://github.com/halfsamoan/TOa_ai)

## License

[MIT](LICENSE).
