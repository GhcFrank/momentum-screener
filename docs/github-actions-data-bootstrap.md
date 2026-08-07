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
再次运行远端 check。身份不匹配时不会下载
2010–2015 分区、不会访问 Yahoo、不会上传任何资产。

`momentum_screener.prices update` 是纯本地命令，不解析 repository、不读取 GitHub token，
也不构建 `release_publish_plan.json`。它成功提交 Parquet、coverage、update report 和
manifest 后，以 `local_update_success=true` 报告本地状态。单独执行的
`momentum_screener.release_storage publish-update` 才解析 repository 和认证信息、从已
提交的 update report 构建发布计划并上传，成功时报告
`release_publish_success=true`。

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
