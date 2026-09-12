# Momentum Screener

## Signal Research UI

安装项目依赖后启动：

```bash
uv run python -m momentum_screener.signal_ui
```

浏览器自动打开 <http://127.0.0.1:8000>；终端保持运行，`Ctrl+C` 停止。
Each line can be either a CSV file path or a directory path. A directory loads all CSV files directly inside that directory; subdirectories are not scanned.
默认 signal folder 为 `/home/gooder/momentum-screener-research/`，可自由修改。
点击 **Load Signals**，选择日期及一个或多个 strategy；多选时显示 ticker 严格交集。
结果表显示本地 RPS storage 的 RPS20/RPS50/RPS120/RPS250（缺失为 `N/A`，不自动重算），点击股票行查看价格图。
结果表还从 `data/universe/ticker_metadata.csv` 动态显示 Sector / Industry；每日 signal
邮件显示 Industry。Company metadata 仅用于展示，不参与 signal generation。
CSV 需要 `session`（或 `date`）及 `ticker`；缺少 `strategy_id` 时使用文件名。
图表显示本地 Adjusted Close，默认视窗为信号日前 3 个月至后 1 个月，
可缩放范围为前 2 年至后 1 年，并裁剪到可用价格边界。

The UI is local-only and reads local CSV/market data.
价格沿用 `data/processed/prices/` 的本地 manifest 与 parquet，缺失时提示，不自动同步。

Ticker 表还动态显示 **40D DD / 40D Gain / 120D DD / 120D Gain**：以信号当日原始
Close 为基准，读取之后最多 40/120 个可用 ticker 交易日的 Low 最小值与 High 最大值，
计算相对涨跌幅；不包含信号当天，也不是 peak-to-trough 回撤。不足窗口时使用已有数据，
无后续价格或缺少当日 Close 时显示 `N/A`。这些指标不写入 signal CSV，价格更新后缓存自动失效。
所有策略共用 `forward_performance.calculate_forward_performance_for_signals()`；
单笔查询使用 `calculate_signal_forward_performance(ticker, signal_date)`。

更新 Universe 后可手动刷新并 review、commit company metadata：

```bash
uv run python -m momentum_screener.company_metadata refresh --force
```

该命令使用当前 Universe 和 Yahoo Finance 的原始 Sector / Industry 分类；daily workflow
不会自动刷新 metadata。删除或修改 metadata CSV 不会改变任何策略选择结果。

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

### Blue Diamond / 蓝色钻石

`blue_diamond` v1.0 使用 RPS20/50、受控回撤、MA20 附近的强均线结构和长期趋势，
每个满足条件的交易日都产生信号。技术形态仅使用 adjusted OHLC；turnover 使用
`raw_close * volume / market_cap`。RPS 优先复用本地持久化数据，MarketCap 必须按
`date,ticker` exact join，不填补缺失，也不使用当前市值回填历史。

只有显式选择蓝色钻石才读取 MarketCap；未指定策略时仍运行原有两个策略。
目标区间任何 session 没有 MarketCap 观测则在替换 signal store 前失败；单个 ticker
缺失则不产生信号。可筛选范围从实际 MarketCap 记录开始日期起，不提供历史 backfill。

```bash
uv run python -m momentum_screener.historical_screening \
  --start-date 2026-09-04 --end-date 2026-09-04 \
  --strategy blue_diamond \
  --market-cap-root data/processed/market_cap \
  --export-csv /home/gooder/momentum-screener-research/2026-09-04_blue_diamond_signals.csv
```

`--export-csv` 是通用导出入口，保留所选策略在本次区间内的全部 signal diagnostics；
只替换指定 CSV。Python 也可调用 `signal_store.export_signal_csv()`。
UI 点击 **Load Signals** 后，CSV 中的 `blue_diamond` 自动进入 Strategies pills，
多选仍为 ticker 交集；零信号日期保留在 store coverage，CSV 不伪造 signal 行。

## Daily research datasets

默认持久化 RPS horizons 为 **20 / 50 / 120 / 250**（`rps_v2`），计算、wide schema 与校验
从同一配置生成。旧 `rpsData` 需要一次性迁移，保留既有指标，仅补算缺少的 horizon。
新增独立的 **point-in-time daily market-cap snapshot**：固定 Universe，按成功 price session
保存到 `data/processed/market_cap/`，并通过 `marketCapData` Release 保留历史；缺失值显式报告。
[迁移命令与 MarketCap bootstrap](docs/github-actions-data-bootstrap.md#existing-rps_v1-one-time-migration-before-enabling-daily-jobs)。
