from __future__ import annotations

from datetime import date

import pandas as pd

from momentum_screener.company_metadata import (
    enrich_company_metadata,
    get_company_metadata,
    load_company_metadata,
    refresh_company_metadata,
)


def test_loader_lookup_and_left_enrichment_keep_missing_tickers(tmp_path) -> None:
    path = tmp_path / "ticker_metadata.csv"
    path.write_text(
        "ticker,sector,industry\nNVDA,Technology,Semiconductors\n",
        encoding="utf-8",
    )

    metadata = load_company_metadata(path)
    assert load_company_metadata(tmp_path / "missing.csv").empty
    lookup = get_company_metadata(["NVDA", "XYZ"], metadata)
    assert lookup["ticker"].tolist() == ["NVDA", "XYZ"]
    assert lookup.loc[0, "sector"] == "Technology"
    assert lookup.loc[0, "industry"] == "Semiconductors"
    assert lookup.loc[1, ["sector", "industry"]].isna().all()

    signals = pd.DataFrame(
        {"ticker": ["XYZ", "NVDA"], "signal": [True, True], "score": [1, 2]}
    )
    signals.attrs["selection"] = "already-complete"
    enriched = enrich_company_metadata(signals, metadata)
    assert enriched["ticker"].tolist() == ["XYZ", "NVDA"]
    assert enriched["signal"].tolist() == [True, True]
    assert enriched["score"].tolist() == [1, 2]
    assert pd.isna(enriched.loc[0, "industry"])
    assert enriched.loc[1, "industry"] == "Semiconductors"
    assert enriched.attrs["selection"] == "already-complete"


def test_refresh_fetches_missing_and_force_preserves_good_values_on_failure(
    tmp_path,
) -> None:
    universe = tmp_path / "universe.csv"
    universe.write_text("ticker\nAAPL\nNVDA\n", encoding="utf-8")
    output = tmp_path / "ticker_metadata.csv"
    output.write_text(
        "ticker,sector,industry,updated_at\n"
        "AAPL,Technology,Consumer Electronics,2026-08-01\n",
        encoding="utf-8",
    )

    requested: list[str] = []

    def missing_provider(ticker: str) -> dict[str, object]:
        requested.append(ticker)
        return {"sector": "Technology", "industry": "Semiconductors"}

    initial = refresh_company_metadata(
        universe_path=universe,
        output_path=output,
        provider=missing_provider,
        max_workers=1,
        request_interval_seconds=0,
        refreshed_on=date(2026, 9, 11),
    )
    assert initial.requested_count == 1
    assert requested == ["NVDA"]

    def force_provider(ticker: str) -> dict[str, object]:
        if ticker == "AAPL":
            raise RuntimeError("temporary Yahoo failure")
        return {"sector": "Technology", "industry": "Semiconductors"}

    result = refresh_company_metadata(
        force=True,
        universe_path=universe,
        output_path=output,
        provider=force_provider,
        max_workers=1,
        request_interval_seconds=0,
        refreshed_on=date(2026, 9, 12),
    )
    refreshed = load_company_metadata(output).set_index("ticker")

    assert result.universe_count == 2
    assert result.requested_count == 2
    assert result.request_failure_count == 1
    assert refreshed.loc["AAPL", "sector"] == "Technology"
    assert refreshed.loc["AAPL", "industry"] == "Consumer Electronics"
    assert refreshed.loc["AAPL", "updated_at"] == "2026-08-01"
    assert refreshed.loc["NVDA", "industry"] == "Semiconductors"
    assert refreshed.loc["NVDA", "updated_at"] == "2026-09-12"
