# GitHub Actions 行情数据迁移与 Bootstrap

每日工作流把固定 GitHub Release `marketData` 作为行情数据的远端权威存储。源码
checkout 只提供代码、当前 Universe 和数据集身份；工作流不会把 Parquet 提交回源码
分支，也不会把 Actions cache 或 workflow artifact 当作长期行情存储。

## `daily_prices_v2` 字段与来源语义

当前 canonical dataset schema version 是 `daily_prices_v2`，年度 Parquet 严格按以下
顺序持久化字段：

```text
date        date32
ticker      string
open        float64
high        float64
low         float64
close       float64
adj_close   float64
volume      int64
```

数据源是 Yahoo Finance，通过 `yfinance.download` 获取。下载明确使用
`auto_adjust=False`、`actions=False`、`repair=True`，因此 `open/high/low/close` 分别
保存 Yahoo 返回的 `Open/High/Low/Close` 原始价格语义，`adj_close` 单独保存 Yahoo 的
`Adj Close`，`volume` 保存 `Volume`。`close` 与 `adj_close` 不是同一字段，也不会在
price dataset 中把 OHLC 转成 adjusted OHLC。后续技术分析可以在 feature layer 统一
计算 adjusted OHLC。

full backfill 与 daily incremental update 共用相同的 Yahoo downloader、normalizer 和
OHLC validation。on-demand RPS engine 的默认 horizons 是 20、50、120、250 个 XNYS 交易
session；lookback 不是自然日。RPS20/RPS50/RPS120/RPS250 都只读取 `adj_close`，其公式不因 v2
增加 OHLC 而改变。生产 daily email 使用月线反转 6.2 的最终 `signal=True` 结果；它不再
发送 RPS120/RPS250 threshold list，但四个 RPS horizon 仍会每天计算并持久化。

## 当前迁移场景

旧的 `daily_prices_v1` 只有 `date/ticker/close/adj_close/volume`，没有真实历史
`Open/High/Low`，因此不能无损转换为 v2，也不能与 v2 增量数据混合。不要用 close 伪造
OHLC；必须从 Yahoo 重新下载 `2016-01-01` 至最新已完成交易日的完整历史。旧 Release
或本地 v1 dataset 不能直接供每日更新继续使用，schema identity mismatch 会在访问
Yahoo 或替换本地文件前失败。

本地完整 rebuild 的安全顺序如下。第一条命令先把现有正式目录原子重命名为带时间戳的
`data/processed/prices_legacy_YYYYmmdd_HHMMSS` 备份，并重新创建空的正式目录；它不会删除
旧数据：

```bash
uv run python -c 'from momentum_screener.prices import rotate_price_output_to_legacy; print(rotate_price_output_to_legacy())'

uv run python -m momentum_screener.prices backfill \
  --start 2016-01-01

uv run python -m momentum_screener.release_storage bootstrap \
  --release-tag marketData \
  --dry-run
```

`prices backfill` 保持 `daily/year=YYYY/prices.parquet`、ZSTD、`date,ticker` 排序和
transactional publish。`bootstrap --dry-run` 会重新读取并严格验证 Universe、v2 manifest、
coverage、全部年度 Parquet、文件大小和 SHA-256，只生成本地 migration plan，不连接或
修改 GitHub Release。

每日 workflow 和普通 `pull-update-inputs` 会比较以下身份字段：

- `schema_version`
- `universe_sha256`
- `requested_start`
- `universe_ticker_count`

任一字段不匹配都会在下载年度分区或访问 Yahoo 前失败，并提示先手动执行 Bootstrap。

## 推荐迁移顺序

先在保存完整新数据集的本地工作区执行：

```bash
uv run python -m momentum_screener.universe validate

uv run python -m momentum_screener.release_storage bootstrap \
  --release-tag marketData \
  --dry-run

uv run python -m momentum_screener.release_storage bootstrap \
  --release-tag marketData \
  --confirm-replace-dataset

uv run python -m momentum_screener.release_storage check \
  --release-tag marketData
```

`bootstrap --dry-run` 和没有确认参数的 `bootstrap` 只验证本地数据并生成
`data/processed/prices/release_migration_plan.json`，不会构造 GitHub 客户端或修改
远端。实际替换必须显式传入 `--confirm-replace-dataset`。该命令只允许管理员手动
运行，daily workflow 不会调用它。

实际 Bootstrap 会在上传前重新验证 Universe、manifest、coverage、全部年度 Parquet、
文件大小和 SHA-256。发布顺序为：

1. `prices-year-2016.parquet` 至当前年份；
2. `prices-ticker-coverage.csv`；
3. `prices-download-failures.csv`；
4. 可选的 update report 和 missing-tickers；
5. 最后上传 `prices-manifest.json`。

GitHub Release API 不提供本实现可安全依赖的原子资产重命名，因此 Bootstrap 使用正式
资产名逐个覆盖，并坚持 manifest 最后上传。若 manifest 前的上传失败，旧 manifest
仍保留，但部分同名资产可能已经替换；此时不要运行 daily workflow，应重新执行
Bootstrap 直至完整验证成功。

Bootstrap 成功后：

1. 手动触发 `Update daily prices` workflow；
2. 检查 job summary 中的 identity、latest session 和 workflow ready；
3. 若发生更新，确认 target coverage 达到配置门槛；
4. 在另一台已更新代码和 Universe 的本地环境运行：

   ```bash
   uv run python -m momentum_screener.release_storage pull \
     --release-tag marketData
   ```

5. 确认本地和远端 `latest_session` 一致。

## 每日工作流

workflow 先验证 Universe，恢复 price update inputs、RPS history 和独立的 MarketCap
history；尚未初始化的 MarketCap Release 允许从空数据集开始。随后按以下顺序运行：

1. 增量更新并验证本地 `daily_prices_v2`；
2. 抓取 MarketCap 并按最新成功 price session 保存 snapshot，显式记录缺失；
3. 对最新完整 session 一次计算全 Universe 的 RPS20/RPS50/RPS120/RPS250；
4. 以 `(date, ticker)` 幂等写入本地 RPS 年度分区；
5. 把同一个当日 snapshot 注入现有双策略，并优先读取已持久化的前 14 个 session；
6. 只选择各策略的 `signal=True`，渲染并发送一次邮件；
7. 发布 price update（若有变化）并复核 `marketData`；
8. 发布 MarketCap 年度分区，再上传并复核 `market-cap-manifest.json`；
9. 发布 RPS 年度分区并最后上传 RPS manifest，再复核 `rpsData`。

任何 price update、MarketCap refresh、RPS calculation/persistence 或策略失败都会中止后续
正常邮件/发布步骤。`signal_count=0` 是成功结果，仍发送明确的空结果邮件。price 数据集
身份不匹配时不会下载旧分区、访问 Yahoo 或上传资产。

`momentum_screener.prices update` 是纯本地命令，不解析 repository、不读取 GitHub token，
也不构建 `release_publish_plan.json`。它成功提交 Parquet、coverage、update report 和
manifest 后，以 `local_update_success=true` 报告本地状态。单独执行的
`momentum_screener.release_storage publish-update` 才解析 repository 和认证信息、从已
提交的 update report 构建发布计划并上传，成功时报告
`release_publish_success=true`。

### 每日双策略筛选邮件

生产入口 `momentum_screener.daily_screening_notification` 从已验证的
`data/processed/prices/manifest.json` 读取 `latest_session`。公共 `strategy_data`
一次准备最近 15 个 session 的 RPS20/50/120/250：优先读取现有存储，仅批量计算缺失
session，当天 RPS 持久化一次。Monthly Reversal 与顺向火车2复用该批 RPS，
不会分别重新计算当天全市场快照。

主题格式为 `Momentum Screener — YYYY-MM-DD`。一封 text/HTML 邮件包含两个独立
section：Monthly Reversal 6.2 与顺向火车2。各 section 仅显示自己的 `signal=True`
股票；没有匹配时仍显示明确空结果，不影响另一个 section。顺向火车2连续满足时每天
继续显示；Monthly Reversal 原有 14 日首次出现规则不变。RPS120/RPS250 继续计算和
持久化，不恢复独立的高 RPS 股票列表推送。

旧 `monthly_reversal_notification` CLI 转调新入口；其旧 Python renderer/runner
保留 Monthly Reversal-only 行为。`rps_notification` 仍仅作为手工诊断 API，生产
workflow 不调用它。新策略公式、API 和 warmup 详见 [顺向火车2](trend-reacceleration.md)。

workflow 需要配置以下 GitHub Actions Secrets：

- `GMAIL_USER`
- `GMAIL_APP_PASSWORD`
- `EMAIL_TO`（一个地址，或逗号分隔的多个地址）

workflow 把这些 Secrets 映射为 notification 模块使用的 `RPS_*` 环境变量，并固定使用
Gmail 的 `smtp.gmail.com:587` + STARTTLS。`GMAIL_APP_PASSWORD` 应使用启用两步验证后
生成的 Google App Password，不应使用或提交普通 Google account password。

本地 CLI 会从 repository root 的 `.env` 加载配置，但不会覆盖 shell 中已经存在的环境
变量。它支持与 workflow 相同的 `RPS_*` 名称，也兼容本地已有的 Gmail 配置名：

```text
GMAIL_USER
GMAIL_APP_PASSWORD
EMAIL_TO
```

在 Gmail 配置模式下，host 默认为 `smtp.gmail.com`，port 默认为 `587`，From 默认为
`GMAIL_USER`。`.env` 必须保持未跟踪，并由 `.gitignore` 排除。

本地手动运行默认使用 price dataset latest session：

```bash
uv run python -m momentum_screener.daily_screening_notification
```

先进行不写入 RPS dataset、不连接 SMTP、不发送邮件的完整计算和渲染检查：

```bash
uv run python -m momentum_screener.daily_screening_notification --dry-run
```

也可显式指定有效 XNYS session；计算仍使用完整 Universe 的四个 RPS horizon：

```bash
uv run python -m momentum_screener.daily_screening_notification \
  --as-of-date 2026-08-31 \
  --dry-run
```

命令会记录 session、Universe/RPS rows、FYX1/YXFZ/signal、顺向火车2信号数量、脱敏收件人及发送
结果；不会记录 SMTP password。非 dry-run 会在计算 RPS 前验证完整邮件配置；RPS
计算、持久化、策略、渲染或发送失败都会返回非零退出码。

## RPS history dataset

RPS 与价格数据逻辑和物理分离：

```text
data/processed/rps/
    manifest.json
    daily/
        year=2016/rps.parquet
        ...
        year=2026/rps.parquet
```

schema version 为 `rps_v2`。每个 `(date, ticker)` 唯一行按 `date,ticker` 稳定排序，列为：

```text
date, ticker,
rps20, rps50, rps120, rps250,
return_20, return_50, return_120, return_250,
rps20_base_date, rps50_base_date, rps120_base_date, rps250_base_date
```

manifest 记录 `schema_version/latest_session/actual_min_date`、Universe SHA-256 与 ticker
count、`lookbacks=[20,50,120,250]`、`price_field=adj_close`、逐年 row count、文件 size 和
SHA-256。daily upsert 只重写当前年份分区；重复运行同一个 session 会替换该日完整
Universe，不会追加重复键。

读取 API 隐藏了年度物理布局：

```python
from momentum_screener.rps_storage import (
    read_rps_history,
    read_rps_snapshot,
    read_stock_rps_history,
)
```

首次历史 backfill 一次读取价格分区的 `date/ticker/adj_close`，构造一个 session panel，
用 vectorized shift/return 和逐日横截面 rank 生成全部 horizon，不逐日重复读取 Parquet。
最早不足 lookback 的结果沿用 `INVALID_RPS`/null return，不伪造历史。实际命令为：

```bash
UV_CACHE_DIR=/tmp/momentum-uv-cache uv run python \
  -m momentum_screener.rps_storage backfill --start 2016-01-01

UV_CACHE_DIR=/tmp/momentum-uv-cache uv run python \
  -m momentum_screener.rps_storage validate

UV_CACHE_DIR=/tmp/momentum-uv-cache uv run python \
  -m momentum_screener.rps_storage update
```

最后一条是 daily persistence 的独立手工入口；正常 workflow 由 notification
orchestrator 计算一次并写入，避免重复 cross-section calculation。

### 独立 `rpsData` Release

RPS history 使用独立、大小写敏感的 Release tag `rpsData`，manifest asset 为
`rps-manifest.json`；不会改变 `marketData` 的 price identity/check/bootstrap/publish
流程。RPS 发布同样先上传有变化的年度 Parquet，最后上传 manifest。首次上线顺序：

1. 运行完整本地 tests；
2. 执行上述 2016+ RPS backfill 和 validate；
3. commit/merge 本次代码（由维护者执行）；
4. 在 GitHub 创建独立的 `rpsData` Release（只需一次）；
5. 先预览再显式确认 bootstrap；
6. 运行远端 check，并在需要时从另一环境 pull 验证；
7. 运行 monthly reversal notification dry-run；
8. 手动触发并确认 production workflow 后，再保留 schedule 正常运行。

对应命令：

```bash
gh release create rpsData --title "RPS data" --notes "Managed RPS history"

uv run python -m momentum_screener.rps_release_storage bootstrap \
  --repository OWNER/REPOSITORY --release-tag rpsData --dry-run

uv run python -m momentum_screener.rps_release_storage bootstrap \
  --repository OWNER/REPOSITORY --release-tag rpsData --confirm-bootstrap

uv run python -m momentum_screener.rps_release_storage check \
  --repository OWNER/REPOSITORY --release-tag rpsData

uv run python -m momentum_screener.rps_release_storage pull \
  --repository OWNER/REPOSITORY --release-tag rpsData

uv run python -m momentum_screener.daily_screening_notification --dry-run
```

这些 RPS 远端命令都必须由维护者显式运行；RPS tooling 不会创建 Release，也不会自动覆盖首次
production dataset。`rpsData` 未完成 bootstrap 前不要启用新版 workflow，否则远端
RPS check 会按 fail-closed 语义失败。

增量更新的 refresh 下限来自远端 manifest 的 `requested_start`：

```text
max(requested_start, target_session - refresh_calendar_days)
```

因此 2016+ 数据集不会读取或创建 2010–2015 分区。

## 旧远端资产

`prices-year-2010.parquet` 至 `prices-year-2015.parquet` 可以保留到新 workflow 验证
成功。新读取逻辑只信任新 manifest 管理的资产，因此这些旧文件不会被下载或使用。
Bootstrap 不会自动删除任何旧远端资产；成功后会生成
`data/processed/prices/remote_obsolete_assets.json`，由管理员确认稳定运行后决定是否
手动删除其中列出的资产。

## 日常本地同步

```bash
uv run python -m momentum_screener.release_storage check \
  --release-tag marketData
uv run python -m momentum_screener.release_storage pull \
  --release-tag marketData
```

强制重新校验和下载全部受 manifest 管理的资产：

```bash
uv run python -m momentum_screener.release_storage pull \
  --release-tag marketData \
  --force
```

只同步指定年份：

```bash
uv run python -m momentum_screener.release_storage pull \
  --release-tag marketData \
  --year 2025 \
  --year 2026
```

Repository 解析优先级为 `--repository`、`GITHUB_REPOSITORY`、
`MOMENTUM_SCREENER_REPOSITORY`、只读解析 `.git/config`。认证优先使用
`GITHUB_TOKEN`，其次使用 `GH_TOKEN`；token 不会写入报告或日志。

### Existing rps_v1: one-time migration before enabling daily jobs

以下命令由维护者手动执行，需要完整本地 price history（迁移前校验分区 size/hash）。迁移只计算新增 horizon，
保留旧 50/120/250 的每一行和数值，并返回原始文件 archive 路径；重复 migrate 是 no-op。
`rpsData` 仍为 v1 时 daily check 明确报 migration required，不会在 runner 中重算历史。
在隔离目录迁移后先检查 dry-run，再由维护者决定发布（这些命令不会由开发过程自动执行）：

```bash
uv run python -m momentum_screener.release_storage pull --repository GhcFrank/momentum-screener --release-tag marketData
uv run python -m momentum_screener.rps_release_storage pull --repository GhcFrank/momentum-screener --release-tag rpsData --allow-legacy --rps-root data/processed/rps-migration
uv run python -m momentum_screener.rps_storage migrate --rps-root data/processed/rps-migration --prices-root data/processed/prices
uv run python -m momentum_screener.rps_release_storage publish --repository GhcFrank/momentum-screener --release-tag rpsData --rps-root data/processed/rps-migration --allow-migration --dry-run
# Review first. The following command writes the migrated dataset to the Release:
uv run python -m momentum_screener.rps_release_storage publish --repository GhcFrank/momentum-screener --release-tag rpsData --rps-root data/processed/rps-migration --allow-migration
uv run python -m momentum_screener.rps_release_storage pull --repository GhcFrank/momentum-screener --release-tag rpsData
```

### Daily MarketCap observations

`market_cap_v1` 是独立的 point-in-time daily market-cap snapshot，位于
`data/processed/market_cap/daily/year=YYYY/market_cap.parquet`，列为 `date,ticker,market_cap`。
日期来自有效价格 manifest 的最新已结算 session；`observed_at` 记录实际抓取时间，
不声称是精确收盘市值，也不允许用今天的市值回填陈旧历史。
仅保留现有 Universe tickers；missing 不填零、不前向填充，计数逐日保留在 manifest，
本次 missing 清单写入 `missing_tickers.csv`。

```bash
uv run python -m momentum_screener.market_cap_release_storage check --repository GhcFrank/momentum-screener
uv run python -m momentum_screener.market_cap_release_storage pull --repository GhcFrank/momentum-screener --allow-bootstrap
uv run python -m momentum_screener.market_cap_storage refresh
uv run python -m momentum_screener.market_cap_release_storage publish --repository GhcFrank/momentum-screener --allow-bootstrap --dry-run
```

独立 tag `marketCapData`，年度资产 `market-cap-year-YYYY.parquet`，最后上传
`market-cap-manifest.json`。确认仓库可访问且 tag 确实 404 时才允许首次初始化；权限、网络、
损坏数据均报错；已有年度资产但缺失 manifest 时停止，须先恢复 manifest。
`publish --allow-bootstrap` 的实际执行会创建缺失的 Release。
Daily 顺序：恢复 price/RPS/MarketCap → 更新并验收 price → MarketCap refresh →
增量 RPS 和现有通知 → 发布 price → 发布 MarketCap → 发布 RPS。
