# Momentum Screener

## Historical Signals

Historical Screening Engine 支持任意自然日闭区间 `[start_date, end_date]`，批量回放
Monthly Reversal 6.2 和顺向火车2，自动处理交易日和 warmup。
输出是可删除重建的 derived signal store（默认 `data/signals/`），包含匹配股票完整
diagnostics 及零匹配日期 coverage；它不模拟交易或计算策略绩效。

```bash
uv run python -m momentum_screener.historical_screening \
  --start-date 2026-08-01 --end-date latest

# 只运行一个策略，并使用另一个本地 store
uv run python -m momentum_screener.historical_screening \
  --start-date 2024-06-01 --end-date 2025-03-31 \
  --strategy trend_reacceleration --output-store /tmp/historical-signals
```

第一阶段每次重算并安全替换请求区间。通过 `get_signal_calendar`、
`get_signals_for_date`、`get_signal_detail` 查询结果，无需实时重跑策略。
数据语义为 `retrospective_latest_data`，并非严格 point-in-time backtest。
完整合约、存储 schema 与阶段规划见 [Historical Signals 设计](docs/historical-signals-design.md)。
