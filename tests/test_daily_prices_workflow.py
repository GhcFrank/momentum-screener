from pathlib import Path


def test_daily_workflow_schedule_concurrency_and_permissions() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )
    assert 'cron: "17 18 * * 1-5"' in content
    assert 'timezone: "America/New_York"' in content
    assert "workflow_dispatch:" in content
    assert "group: daily-price-update" in content
    assert "cancel-in-progress: false" in content
    assert "contents: write" in content
    assert "timeout-minutes: 45" in content
    assert "retention-days: 14" in content
    assert content.count("RELEASE_TAG: marketData") == 1
    assert content.count("RPS_RELEASE_TAG: rpsData") == 1


def test_daily_workflow_uses_release_commands_without_git_writes() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )
    assert "momentum_screener.release_storage pull-update-inputs" in content
    assert "momentum_screener.prices update" in content
    assert "momentum_screener.release_storage publish-update" in content
    assert content.count("momentum_screener.release_storage check") == 2
    assert "momentum_screener.rps_release_storage pull" in content
    assert "momentum_screener.rps_release_storage publish" in content
    assert content.count("momentum_screener.rps_release_storage check") == 2
    assert "momentum_screener.universe validate" in content
    assert "momentum_screener.prices update --dry-run" in content
    assert "--allow-partial-session" not in content
    assert "bootstrap" not in content
    assert "backfill" not in content
    assert "year=2010" not in content
    assert "git add" not in content
    assert "git commit" not in content
    assert "git push" not in content
    assert "market" + "-data" not in content
    assert content.count('--repository "${{ github.repository }}"') == 8
    assert content.count('--release-tag "${RELEASE_TAG}"') == 4
    assert content.count('--release-tag "${RPS_RELEASE_TAG}"') == 4
    assert 'echo "- Release tag: ${RELEASE_TAG}"' in content


def test_daily_workflow_checks_identity_before_pull_and_manifest_publish() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )

    validate = content.index("momentum_screener.universe validate")
    first_check = content.index("momentum_screener.release_storage check")
    pull = content.index("momentum_screener.release_storage pull-update-inputs")
    rps_check = content.index("momentum_screener.rps_release_storage check")
    rps_pull = content.index("momentum_screener.rps_release_storage pull")
    dry_plan = content.index("momentum_screener.prices update --dry-run")
    update = content.index('--result-json "$RUNNER_TEMP/price-update-result.json"')
    incremental_acceptance = content.index(
        "validate_local_incremental_update_acceptance"
    )
    notification = content.index("momentum_screener.daily_screening_notification")
    publish = content.index("momentum_screener.release_storage publish-update")
    final_check = content.rindex("momentum_screener.release_storage check")
    rps_publish = content.index("momentum_screener.rps_release_storage publish")
    final_rps_check = content.rindex("momentum_screener.rps_release_storage check")

    assert (
        validate
        < first_check
        < pull
        < rps_check
        < rps_pull
        < dry_plan
        < update
        < incremental_acceptance
        < notification
        < publish
        < final_check
        < rps_publish
        < final_rps_check
    )
    assert "Validate incremental update acceptance" in content
    assert "validate_local_dataset_acceptance" not in content
    assert "build_publish_plan" not in content
    assert "Local Universe ticker count" in content
    assert "Expected requested start" in content
    assert "local_update_success" in content
    assert "release_publish_success" in content


def test_daily_screening_notification_is_success_only_before_publication() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )

    refresh = content.index("- name: Refresh daily prices")
    acceptance = content.index("- name: Validate incremental update acceptance")
    notification = content.index(
        "- name: Persist RPS and email daily screening signals"
    )
    price_publish = content.index("- name: Publish update to Release")
    notification_step = content[notification:price_publish]

    assert refresh < acceptance < notification < price_publish
    assert "if:" not in notification_step
    assert content.count("momentum_screener.daily_screening_notification") == 1
    assert "momentum_screener.rps_notification" not in content
    assert "RPS_EMAIL_THRESHOLD" not in content
    assert "RPS120 >" not in content
    assert "RPS250 >" not in content
    assert "calculate_rps_snapshot" not in content


def test_daily_screening_notification_reuses_email_secrets() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )

    assert "RPS_SMTP_HOST: smtp.gmail.com" in content
    assert 'RPS_SMTP_PORT: "587"' in content
    assert "RPS_SMTP_USERNAME: ${{ secrets.GMAIL_USER }}" in content
    assert "RPS_SMTP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}" in content
    assert "RPS_EMAIL_FROM: ${{ secrets.GMAIL_USER }}" in content
    assert "RPS_EMAIL_TO: ${{ secrets.EMAIL_TO }}" in content


def test_rps_publish_and_summary_include_persisted_dataset_results() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )

    assert "data/processed/rps/manifest.json" in content
    assert "daily-screening-notification.json" in content
    assert "rps-publish-result.json" in content
    assert "rps-check-after.json" in content
    assert "RPS rows persisted" in content
    assert "Monthly Reversal signal count" in content
    assert "Trend Re-acceleration signal count" in content


def test_production_documentation_uses_market_data_tag() -> None:
    content = Path("docs/github-actions-data-bootstrap.md").read_text(encoding="utf-8")
    assert "marketData" in content
    assert "market" + "-data" not in content
