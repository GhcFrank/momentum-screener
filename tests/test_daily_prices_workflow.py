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


def test_daily_workflow_uses_release_commands_without_git_writes() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )
    assert "momentum_screener.release_storage pull-update-inputs" in content
    assert "momentum_screener.prices update" in content
    assert "momentum_screener.release_storage publish-update" in content
    assert content.count("momentum_screener.release_storage check") == 2
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
    assert content.count('--repository "${{ github.repository }}"') == 4
    assert content.count('--release-tag "${RELEASE_TAG}"') == 4
    assert 'echo "- Release tag: ${RELEASE_TAG}"' in content


def test_daily_workflow_checks_identity_before_pull_and_manifest_publish() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )

    validate = content.index("momentum_screener.universe validate")
    first_check = content.index("momentum_screener.release_storage check")
    pull = content.index("momentum_screener.release_storage pull-update-inputs")
    dry_plan = content.index("momentum_screener.prices update --dry-run")
    update = content.index('--result-json "$RUNNER_TEMP/price-update-result.json"')
    incremental_acceptance = content.index(
        "validate_local_incremental_update_acceptance"
    )
    publish = content.index("momentum_screener.release_storage publish-update")
    final_check = content.rindex("momentum_screener.release_storage check")
    notification = content.index("momentum_screener.rps_notification")

    assert (
        validate
        < first_check
        < pull
        < dry_plan
        < update
        < incremental_acceptance
        < publish
        < final_check
        < notification
    )
    assert "Validate incremental update acceptance" in content
    assert "validate_local_dataset_acceptance" not in content
    assert "build_publish_plan" not in content
    assert "Local Universe ticker count" in content
    assert "Expected requested start" in content
    assert "local_update_success" in content
    assert "release_publish_success" in content


def test_daily_rps_notification_is_a_success_only_downstream_step() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )

    refresh = content.index("- name: Refresh daily prices")
    final_check = content.index("- name: Verify remote manifest after update")
    notification = content.index("- name: Calculate and email daily RPS screen")
    summary = content.index("- name: Write job summary")
    notification_step = content[notification:summary]

    assert refresh < final_check < notification < summary
    assert "if:" not in notification_step
    assert content.count("momentum_screener.rps_notification") == 1
    assert '--threshold "${RPS_EMAIL_THRESHOLD}"' in notification_step
    assert "calculate_rps_snapshot" not in content


def test_daily_rps_notification_uses_secrets_without_hardcoded_credentials() -> None:
    content = Path(".github/workflows/update-daily-prices.yml").read_text(
        encoding="utf-8"
    )

    assert "RPS_SMTP_HOST: smtp.gmail.com" in content
    assert 'RPS_SMTP_PORT: "587"' in content
    assert "RPS_SMTP_USERNAME: ${{ secrets.GMAIL_USER }}" in content
    assert "RPS_SMTP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}" in content
    assert "RPS_EMAIL_FROM: ${{ secrets.GMAIL_USER }}" in content
    assert "RPS_EMAIL_TO: ${{ secrets.EMAIL_TO }}" in content
    assert "RPS_EMAIL_THRESHOLD: ${{ vars.RPS_EMAIL_THRESHOLD || '87' }}" in content


def test_production_documentation_uses_market_data_tag() -> None:
    content = Path("docs/github-actions-data-bootstrap.md").read_text(encoding="utf-8")
    assert "marketData" in content
    assert "market" + "-data" not in content
