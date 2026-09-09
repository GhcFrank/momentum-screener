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

自动尝试使用 `America/New_York` 时区：Mon–Fri **18:30、21:30**，Tue–Sat
**00:30、05:30**；保留 `workflow_dispatch`。使用 GitHub 的 `timezone` 字段自动处理
DST，不硬编码 UTC offset（[GitHub schedule 文档](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onschedule)）。

`daily_update preflight` 先读取本次 workflow run 的 `created_at`，结合触发的 cron
还原 nominal attempt time，再复用 `prices.determine_target_session` 的 XNYS calendar
和实际收盘 + 90 分钟逻辑。凌晨尝试指向前一个已完成交易日，Saturday 指向 Friday；
节假日、提前收盘、重跑和排队延迟不会简单使用 runner 的 `date.today()`。
读取 run metadata 使用 `actions: read` 权限。

preflight 在恢复数据前检查 `marketData` 中该 session 的完成凭据。已 complete 时，
验证、恢复、Yahoo/MarketCap、RPS、signals、发布和正常邮件步骤全部跳过。手动触发同样
默认 skip，没有新增 force mode。未完成时顺序为：

1. 验证 Universe/数据身份，恢复 price update inputs；
2. 增量更新并验收本地 `daily_prices_v2`；已经存在的 target prices 不重新访问 Yahoo，
   但仍校验 target coverage；
3. 仅 `ready=true` 时恢复 RPS/MarketCap，刷新并保存 MarketCap snapshot；
4. 共享一次 RPS 准备、持久化和现有双策略计算，以 `--prepare-email` 保存邮件内容；
5. 发布 price（有变化时）、MarketCap、RPS，并复核远端结果；
6. 确认三个 dataset 都对应 target session 后，发送刚才准备的正常邮件，再标记 complete。

所有下游 step 都显式要求 preflight `action=run`、price `ready=true` 和前置步骤成功。
`signal_count=0` 仍是成功结果，发送明确的空结果邮件。strategy 公式及现有邮件内容不变。

Yahoo 请求完成且 `unresolved_failure_count=0`，但 target coverage 未达到 **0.97** 时，
price 层抛出带结构化报告的 `ProviderNotSettledError`，orchestration 转为
`provider_not_settled`、`ready=false`。缺少 Close 的 bar 仍被 normalizer 排除，不填补
任何价格，也不提交或发布不完整 canonical data。前三次 attempt 正常结束，只保存
`.update_diagnostics/**`，不跑下游、不发正常或失败邮件；手动 run 同样不自动发最终失败邮件。
真正的下载、校验、意外错误保留失败退出码和日志 traceback，不伪装成 provider 待完成。

只有触发 cron 为 **05:30 Tue–Sat**、且 session 尚未 complete 时，最终失败步骤才复用
现有 SMTP infrastructure 发一封 `Momentum Screener — Daily Update Failed — YYYY-MM-DD`。
邮件包含 session、attempt time/final 标志、失败原因、实际/要求 coverage、expected active、
missing 和 unresolved 数量。provider 未完成时明确说明本次未推进 canonical 数据、未生成
RPS/signals；若失败发生在下游，则如实说明前面的发布或计算可能已成功。发信后该最终失败
attempt 返回非零，便于 Actions 监控。每次 run/attempt 的 diagnostics artifact 独立命名。

完成凭据是小型 Release asset `daily-screening-YYYY-MM-DD.json`，内容记录发布验收和
screening 结果。发信前创建占位，SMTP 成功后通过原位 PATCH 把 asset metadata `label`
设为 `complete`，避免删除/重传凭据产生空窗。失败邮件使用独立的
`daily-failure-YYYY-MM-DD.json`，发送成功后的 label 为 `sent`。
这些凭据不进入 price manifest/schema，不会被正常 publish 覆盖；旧的 Release check
可能把它们列在 `obsolete_remote_assets` 中，但它们是需要保留的通知凭据。

SMTP 与 Release 无法构成一个原子事务。若 SMTP 结果或发送后的 marker 更新无法确认，
占位会保留，后续自动尝试明确失败并要求人工核对原 run/SMTP 结果，既不算 complete，也不
自动重发。只有确认发送成功且所有发布验收已完成后，才可人工将对应 label 设为 complete；
无法确认时保留占位。这个取舍保证自动正常邮件最多发送一次，不能同时承诺网络故障下必达。
本地原有 notification 命令保持直接发送语义；session 去重由 daily workflow adapter 负责。
旧 workflow 没有这类发送凭据，因此不能从价格 manifest 追认历史邮件已经发送。首次启用
应从尚未处理的新 session 开始；若与旧流程在同一 session 交接，应先核实原 run 的完整
成功结果并建立对应完成凭据，避免把缺少凭据的旧 session 再次作为未完成任务处理。

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
结果；不会记录 SMTP password。默认非 dry-run 会在计算 RPS 前验证完整邮件配置；
workflow 的 `--prepare-email PATH` 只持久化/准备，不加载 SMTP 配置，由发布后的发送阶段验证。RPS
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
Daily 顺序：completion preflight → 恢复并验收 price → 恢复 RPS/MarketCap → MarketCap refresh →
共享 RPS/signals 并准备邮件 → 发布/复核 price、MarketCap、RPS → 发送邮件 → 标记 complete。
