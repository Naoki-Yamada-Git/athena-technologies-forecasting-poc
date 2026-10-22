# ETT 油温予測 PoC

ETT-small の4系列を用いて、変圧器の油温（`OT`）の同時点推定と6時間・24時間先の予測可能性を確認する解析コードです。GitHub には、再現に必要なコード・依存関係・実行手順のみを置いています。

## リポジトリの内容

- `src/run_analysis.py`: データ品質確認、EDA、時系列分割、ベースライン比較、モデル評価を実行します。
- `requirements.txt`: 実行に必要な Python パッケージを固定しています。
- `report/Athena_ML_PoC_report_final.pdf`: 分析結果と検討内容をまとめた課題報告スライドです。
- データ本体と実行結果はこのリポジトリに含めていません。

## データの準備

ETT-small のデータは [ETDataset 公式リポジトリ](https://github.com/zhouhaoyi/ETDataset) から取得してください。次の4ファイルをローカルの `data/` フォルダに配置します。

```text
data/
  ETTh1.csv
  ETTh2.csv
  ETTm1.csv
  ETTm2.csv
```

データ本体は配布条件を確認したうえで各自取得してください。データファイルと実行結果は `.gitignore` により Git の追跡対象から除外しています。

## 実行方法

Python 3.10 以降で依存パッケージをインストールし、リポジトリのルートから実行します。

```bash
python -m pip install -r requirements.txt
python src/run_analysis.py --data-dir data --output-dir results
```

実行すると、データ品質・EDA・評価指標・予測プレビューなどの CSV/JSON がローカルの `results/` に出力されます。

## 評価設計

- 4系列を個別に評価し、予測対象時刻を基準に時系列順で学習60%、検証20%、テスト20%に分割します。
- 予測時点 `t` で利用可能な負荷系列、過去の油温、予測対象時刻に既知のカレンダー特徴を使います。
- `h=0` は同時点推定として扱い、`OT(t)` 自体は特徴量に含めません。6時間・24時間先の予測では、時点 `t` までに観測した油温を使います。
- 検証期間の MAE で Persistence、日周期・週周期の季節ナイーブ、Ridge、HistGradientBoostingRegressor を比較し、選択後に学習期間と検証期間を合わせて再学習します。
- 指標には MAE、RMSE、R² を用います。`OT` に0付近や負値があるため MAPE は使用しません。

## 解釈上の制約

ETT には故障・異常・点検・停止のラベル、警報の運用履歴、周囲温度、負荷や油温の物理単位がありません。したがって、このデータでの油温予測誤差だけから故障予測や保全費削減、ROIを主張しません。実機での判断には、設備固有の履歴を使った後ろ向き評価と現場でのシャドー運用が必要です。

## 参照元

- ETTDataset [公式リポジトリ](https://github.com/zhouhaoyi/ETDataset)
