# 月线反转 6.2

实现位于 `momentum_screener.monthly_reversal`，运行时读取经过现有校验器验证的
`daily_prices_v2` 年度分区。策略不会修改价格数据，也不会持久化 MA、HHV、LLV、FYX、
YXFZ 或 signal。

## 价格与窗口口径

所有技术指标统一使用 adjusted OHLC。逐行复权因子及价格定义为：

```text
adjust_factor = adj_close / close
C = adj_close
O = open * adjust_factor
H = high * adjust_factor
L = low * adjust_factor
```

`add_adjusted_ohlc()` 对非正数、缺失值和非有限值 fail closed：无效行的全部 adjusted
OHLC 都是 unavailable，不进行 forward-fill，也不产生 infinity。

MA、HHV、LLV、COUNT 和 REF 的窗口单位都是股票自身已有的 trading rows。rolling
窗口包含当前行并要求完整窗口；REF 通过 `shift` 实现，不读取未来数据。

## 策略与信号

RPS 输入只有现有 generic RPS engine 的 RPS50 和 RPS120。公式集中在
`calculate_monthly_reversal_features()`：

```text
YXFZ = FYX1 AND FYX2 AND FYX3 AND FYX4 AND FYX5 AND FYX6 AND FYX7

signal =
    current YXFZ
    AND no YXFZ in the previous 14 trading rows
```

因此 t-14 的 YXFZ 会阻止当前信号，t-15 的 YXFZ 不会。YXFZ 与 signal 是两个独立
字段。单日 signal 至少要求 264 条股票自身历史行，默认加载器另外保留缓冲；不足或
必要指标不可用时条件为 False，并在 explain 结果中显示 readiness/status。

## API

```python
from momentum_screener.monthly_reversal import (
    calculate_monthly_reversal_history,
    evaluate_monthly_reversal,
    screen_monthly_reversal,
)

history = calculate_monthly_reversal_history(
    "NVDA",
    start_date="2026-08-01",
    end_date="2026-09-03",
)
explanation = evaluate_monthly_reversal("NVDA", "2026-09-03")
signals = screen_monthly_reversal("2026-09-03")
yxfz_candidates = screen_monthly_reversal("2026-09-03", signal_only=False)
```

日期必须是有效 XNYS session，与 RPS API 一样不会自动回退到上一交易日。

全市场 screen 优先从 `data/processed/rps/` 读取当前及前 14 个 session 的
RPS50/RPS120。调用方可注入已经计算好的当日 snapshot；持久化历史或注入数据已完整的
session 不会重复计算，只有缺失 session 才使用 generic RPS engine 基于同一批价格 rows
回退计算。随后用当日 `RPS50 > 87 OR RPS120 > 90` 做严格等价的 FYX1 预筛选。单股票
evaluate 不做该预筛选，所以即使 FYX1=False 也能返回完整诊断。

screen 返回 DataFrame，并在 `DataFrame.attrs` 提供 `universe_count`、
`fyx1_candidate_count`、`yxfz_count`、`signal_count`、
`loaded_price_session_count` 和 `rps_snapshot_count`。

数据访问由公共 `strategy_data` 层负责；FYX/YXFZ/signal 公式及上述窗口口径保持不变。
生产邮件通过 `daily_screening_notification` 与[顺向火车2](trend-reacceleration.md)
合并发送，两个策略共享 RPS 准备结果。
