# 顺向火车2 / Trend Re-acceleration 1.0

策略模块为 `momentum_screener.trend_reacceleration`，id 为 `trend_reacceleration`。
语义：Strong Momentum + Healthy Pullback + Trend Re-acceleration。

## 数据与公式

所有 OHLC 指标复用 `technical_features.add_adjusted_ohlc()`：
`C = adj_close`，`H = high * adj_close / close`，`L = low * adj_close / close`。
原始 OHLC 不参与技术公式，不重复复权、不填充缺失值、不读取未来数据。

默认配置 `DEFAULT_CONFIG = TrendReaccelerationConfig()` 定义以下条件：

| 模块 | 条件 |
| --- | --- |
| Momentum | RPS120 + RPS250 **> 185**，两项 RPS 均可用 |
| Trend | C > MA20；最近 30 行至少 25 行各自 C > MA250、C > MA200；最近 4 行至少 3 行 C > MA10 **或** C > MA20 |
| Pullback | 最近 20 行最高 adjusted high 之后的最大回撤 **≤ 25%**；C / 最近 250 行最高 adjusted close **> 0.80** |
| Re-acceleration | MA20 连续 5 行不下降；MA10 连续 5 行 ≥ MA20；当前 MA10 和 MA20 均严格上升；当前 MA10 ≥ MA20 |

高点相同时取最近一次。低点搜索范围严格为高点的**下一行到当前行**，不包含高点当日
或高点之前的 low。若最高点就在今天，则 `recent_low = L[t]`、`days_since_low = 0`、
`drawdown = 0`，即使今天日内振幅很大也如此。
`no_deep_drawdown_since_high` 与 `drawdown_ok` 完全一致，不另设回撤定义。

窗口按股票自身的交易行计数。MA250 的 30 行 COUNT 要求 `250 + 30 - 1 = 279`
条有效历史行；默认加载 320 个 XNYS session 对应的年度分区，并保留首年的额外行，
与 Monthly Reversal 的加载方式一致。自定义窗口会自动增加 warmup。
NaN、不完整窗口不能产生信号；evaluate 返回 `insufficient_history`、
`invalid_adjusted_ohlc`、`rps_unavailable` 或 `price_unavailable` 等解释。

`setup` 为四个模块在历史可完整评价时的合取，`signal = setup`。
**没有任何信号去重或首次出现要求，连续满足三天就连续三天 signal=True。**
Monthly Reversal 原有的前 14 行去重规则保持不变。

## API

```python
from momentum_screener.trend_reacceleration import (
    TrendReaccelerationConfig,
    calculate_trend_reacceleration_features,
    calculate_trend_reacceleration_history,
    evaluate_trend_reacceleration,
    screen_trend_reacceleration,
)

# 已准备好的单 ticker 原始 OHLC + adj_close + RPS120/RPS250。
features = calculate_trend_reacceleration_features(frame)
history = calculate_trend_reacceleration_history(
    "NVDA", start_date="2026-08-03", end_date="2026-09-03",
)
diagnosis = evaluate_trend_reacceleration("NVDA", "2026-09-03")
matches = screen_trend_reacceleration("2026-09-03")
custom = screen_trend_reacceleration(
    "2026-09-03", config=TrendReaccelerationConfig(rps_sum_threshold=190),
)
```

storage API 均支持与 Monthly Reversal 一致的 `prices_root`、`universe_path`、
`rps_root`、`rps_snapshots` 参数，以及 `config`。
日期必须是 XNYS session，不自动回退。history 默认日期来自 price manifest。
screen 以 momentum 必要条件预筛选，只对候选计算完整技术特征，返回全部诊断字段；
`signal_only=False` 返回 setup 匹配项，与 signal 匹配项一致。
诊断字段的 `_30`、`_4`、`_5`、`20d`、`250d` 后缀保留默认名称，使用自定义配置时
实际计算窗口服从 config。未新增任何持久化 schema 或技术特征存储。

## 公共数据层与通知

`strategy_data` 统一负责历史分区加载、日期/session 解析、任意 RPS lookback 的
读取/缺失回退，以及按 `(date, ticker)` 合并。RPS 排名始终基于完整 Universe，
单 ticker 查询只投影结果。显式注入优先于存储；`-1` 和显式 NaN 不可用值不会被
回退覆盖。完整注入无需再次读 RPS 存储。

`daily_screening_notification` 一次准备最近 15 个 session 的 RPS50/120/250，
优先读取，缺失 session 批量计算；当天快照持久化一次，两个策略复用同一份 RPS，
最终只发一封包含两个独立 section 的 text/HTML 邮件。无匹配的 section 仍显示空结果。
顺向火车2每个满足条件的交易日都继续列出股票，不读取昨天的通知状态。

```bash
uv run python -m momentum_screener.daily_screening_notification --dry-run
```

dry-run 不持久化、不加载 SMTP 配置、不连接 SMTP。旧
`monthly_reversal_notification` CLI 转调新入口；旧 Python renderer/runner 的
Monthly Reversal-only 合约保留，供现有调用方使用。生产 workflow 只调用新入口。
