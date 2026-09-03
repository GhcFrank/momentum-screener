# GitHub Actions 行情数据迁移与 Bootstrap

每日工作流把固定 GitHub Release `marketData` 作为行情数据的远端权威存储。源码
checkout 只提供代码、当前 Universe 和数据集身份；工作流不会把 Parquet 提交回源码
分支，也不会把 Actions cache 或 workflow artifact 当作长期行情存储。

## 当前迁移场景

旧 Release 对应错误 Universe 及 2010 年开始的数据集。当前本地权威数据集使用重新验证
的 2,000 个普通股 Universe，`requested_start` 为 `2016-01-01`，正式年份从 2016
开始。旧 Release 不能直接供每日更新继续使用，否则会混合两个不同的数据集。

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

workflow 依次验证本地 Universe、运行只读 `release_storage check`、只拉取 refresh
window 涉及的年份和 coverage、执行 update dry planning、执行正式增量更新、验证本地
数据集、由 `release_storage publish-update` 构建发布计划并以 manifest 最后顺序发布，
再次运行远端 check，最后计算并发送 RPS 邮件。RPS 邮件步骤是同一 job 中的普通后续
步骤；任何 update、发布或复核步骤失败时都不会运行，也不会发送正常筛选结果。身份不
匹配时不会下载
2010–2015 分区、不会访问 Yahoo、不会上传任何资产。

`momentum_screener.prices update` 是纯本地命令，不解析 repository、不读取 GitHub token，
也不构建 `release_publish_plan.json`。它成功提交 Parquet、coverage、update report 和
manifest 后，以 `local_update_success=true` 报告本地状态。单独执行的
`momentum_screener.release_storage publish-update` 才解析 repository 和认证信息、从已
提交的 update report 构建发布计划并上传，成功时报告
`release_publish_success=true`。

### 每日 RPS 邮件

`momentum_screener.rps_notification` 从已验证的
`data/processed/prices/manifest.json` 读取 `latest_session`，不使用系统日期，也不通过
单只股票推断日期。随后调用完整 Universe 的 `calculate_rps_snapshot(latest_session)`，
独立筛选 `rps120 > threshold` 与 `rps250 > threshold`，生成 plain text + HTML 邮件，
并通过认证 STARTTLS SMTP 发送一次。默认 threshold 为 `87.0`；workflow 可通过
Repository Variable `RPS_EMAIL_THRESHOLD` 覆盖。

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

本地手动补发默认使用 dataset latest session：

```bash
uv run python -m momentum_screener.rps_notification
```

先进行不连接 SMTP、不发送邮件的完整计算和渲染检查：

```bash
uv run python -m momentum_screener.rps_notification --dry-run
```

也可显式指定 session 和 threshold；计算仍然使用完整 Universe 的 RPS snapshot：

```bash
uv run python -m momentum_screener.rps_notification \
  --as-of-date 2026-08-31 \
  --threshold 90 \
  --dry-run
```

命令会记录 latest session、snapshot ticker count、两个筛选数量、收件人及发送结果；
不会记录 SMTP password。非 dry-run 会在计算 RPS 前验证完整邮件配置；RPS 计算、渲染
或发送失败都会返回非零退出码，已经成功落地或发布的行情数据不会被回滚。

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
