from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BACKTEST_DIR = SCRIPT_PATH.parents[1] / "backtest"
if str(BACKTEST_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKTEST_DIR),
    )

from quant_metrics import glossary_frame


def _pct(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.2f}%"


def _num(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _money(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Print a concise summary of the saved "
            "ML portfolio-diagnostics outputs without "
            "retraining any models."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    args = ap.parse_args()

    root = Path(
        args.root
    ).resolve()
    report = (
        root
        / "reports/ml/"
        "portfolio_diagnostics"
    )

    required = {
        "ensemble": (
            report
            / "validation_ensemble_search.csv"
        ),
        "validation_tail": (
            report
            / "validation_top_tail_summary.csv"
        ),
        "walkforward_folds": (
            report
            / "walkforward_fold_diagnostics.csv"
        ),
        "walkforward_tail": (
            report
            / "walkforward_top_tail_summary.csv"
        ),
        "walkforward_persistence": (
            report
            / "walkforward_score_persistence_summary.csv"
        ),
        "walkforward_portfolios": (
            report
            / "walkforward_portfolio_comparison.csv"
        ),
        "walkforward_gross_net": (
            report
            / "walkforward_gross_vs_net.csv"
        ),
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.is_file()
    ]

    if missing:
        raise SystemExit(
            "Missing diagnostics outputs:\n"
            + "\n".join(
                missing
            )
        )

    ensemble = pd.read_csv(
        required["ensemble"]
    )
    validation_tail = pd.read_csv(
        required[
            "validation_tail"
        ]
    )
    folds = pd.read_csv(
        required[
            "walkforward_folds"
        ]
    )
    walk_tail = pd.read_csv(
        required[
            "walkforward_tail"
        ]
    )
    persistence = pd.read_csv(
        required[
            "walkforward_persistence"
        ]
    )
    portfolios = pd.read_csv(
        required[
            "walkforward_portfolios"
        ]
    )
    gross_net = pd.read_csv(
        required[
            "walkforward_gross_net"
        ]
    )

    print(
        "\n=== VALIDATION ENSEMBLE SEARCH ==="
    )
    cols = [
        c
        for c in (
            "alpha",
            "top_k",
            "cagr",
            "sharpe",
            "max_drawdown",
            "ending_equity",
            "selected_alpha",
        )
        if c in ensemble.columns
    ]
    e = ensemble[
        cols
    ].copy()

    if "cagr" in e.columns:
        e["cagr"] = e[
            "cagr"
        ].map(_pct)
    if (
        "max_drawdown"
        in e.columns
    ):
        e[
            "max_drawdown"
        ] = e[
            "max_drawdown"
        ].map(_pct)
    if "sharpe" in e.columns:
        e["sharpe"] = e[
            "sharpe"
        ].map(
            lambda x: _num(
                x,
                3,
            )
        )
    if (
        "ending_equity"
        in e.columns
    ):
        e[
            "ending_equity"
        ] = e[
            "ending_equity"
        ].map(_money)

    print(
        e.to_string(
            index=False
        )
    )

    print(
        "\n=== VALIDATION TOP TAIL ==="
    )
    t = validation_tail[
        [
            c
            for c in (
                "selection",
                "mean_realized_return",
                "mean_excess_return",
                "positive_excess_fraction",
                "mean_selected_rows",
            )
            if c
            in validation_tail.columns
        ]
    ].copy()

    for col in (
        "mean_realized_return",
        "mean_excess_return",
        "positive_excess_fraction",
    ):
        if col in t.columns:
            t[col] = t[
                col
            ].map(_pct)

    print(
        t.to_string(
            index=False
        )
    )

    print(
        "\n=== WALK-FORWARD FOLDS ==="
    )
    f = folds[
        [
            c
            for c in (
                "fold_year",
                "train_rows",
                "mean_daily_ic",
                "positive_ic_fraction",
                "mean_top_decile_excess",
            )
            if c in folds.columns
        ]
    ].copy()

    if (
        "mean_daily_ic"
        in f.columns
    ):
        f[
            "mean_daily_ic"
        ] = f[
            "mean_daily_ic"
        ].map(
            lambda x: _num(
                x,
                4,
            )
        )

    for col in (
        "positive_ic_fraction",
        "mean_top_decile_excess",
    ):
        if col in f.columns:
            f[col] = f[
                col
            ].map(_pct)

    print(
        f.to_string(
            index=False
        )
    )

    print(
        "\n=== WALK-FORWARD TOP TAIL ==="
    )
    wt = walk_tail[
        [
            c
            for c in (
                "selection",
                "mean_realized_return",
                "mean_excess_return",
                "positive_excess_fraction",
                "mean_selected_rows",
            )
            if c in walk_tail.columns
        ]
    ].copy()

    for col in (
        "mean_realized_return",
        "mean_excess_return",
        "positive_excess_fraction",
    ):
        if col in wt.columns:
            wt[col] = wt[
                col
            ].map(_pct)

    print(
        wt.to_string(
            index=False
        )
    )

    print(
        "\n=== WALK-FORWARD SCORE PERSISTENCE ==="
    )
    p = persistence[
        [
            c
            for c in (
                "lag_sessions",
                "mean_rank_spearman",
                "mean_top_decile_overlap",
                "mean_top5_turnover_proxy",
                "mean_top10_turnover_proxy",
                "mean_top20_turnover_proxy",
                "mean_top50_turnover_proxy",
            )
            if c in persistence.columns
        ]
    ].copy()

    for col in (
        "mean_rank_spearman",
        "mean_top_decile_overlap",
        "mean_top5_turnover_proxy",
        "mean_top10_turnover_proxy",
        "mean_top20_turnover_proxy",
        "mean_top50_turnover_proxy",
    ):
        if col in p.columns:
            p[col] = p[
                col
            ].map(
                lambda x: _num(
                    x,
                    3,
                )
            )

    print(
        p.to_string(
            index=False
        )
    )

    print(
        "\n=== WALK-FORWARD GROSS VS NET ==="
    )
    g = gross_net[
        [
            c
            for c in (
                "name",
                "top_k",
                "gross_cagr",
                "net_cagr",
                "cagr_cost_drag_pp",
                "gross_sharpe",
                "net_sharpe",
                "net_total_fees",
                "gross_trades",
                "net_trades",
            )
            if c in gross_net.columns
        ]
    ].copy()

    for col in (
        "gross_cagr",
        "net_cagr",
    ):
        if col in g.columns:
            g[col] = g[
                col
            ].map(_pct)

    for col in (
        "gross_sharpe",
        "net_sharpe",
        "cagr_cost_drag_pp",
    ):
        if col in g.columns:
            g[col] = g[
                col
            ].map(
                lambda x: _num(
                    x,
                    3,
                )
            )

    if (
        "net_total_fees"
        in g.columns
    ):
        g[
            "net_total_fees"
        ] = g[
            "net_total_fees"
        ].map(_money)

    print(
        g.to_string(
            index=False
        )
    )

    print(
        "\n=== WALK-FORWARD PORTFOLIOS ==="
    )
    q = portfolios[
        [
            c
            for c in (
                "name",
                "kind",
                "top_k",
                "ending_equity",
                "cagr",
                "max_drawdown",
                "sharpe",
                "sortino",
                "calmar",
                "max_drawdown_duration_sessions",
                "profit_factor",
                "turnover_multiple",
                "trades",
                "total_fees",
            )
            if c in portfolios.columns
        ]
    ].copy()

    q = q.sort_values(
        [
            "cagr",
            "sharpe",
        ],
        ascending=[
            False,
            False,
        ],
    )

    for col in (
        "cagr",
        "max_drawdown",
    ):
        if col in q.columns:
            q[col] = q[
                col
            ].map(_pct)

    for ratio_col in (
        "sharpe",
        "sortino",
        "calmar",
        "profit_factor",
        "turnover_multiple",
    ):
        if ratio_col in q.columns:
            q[
                ratio_col
            ] = q[
                ratio_col
            ].map(
                lambda x: _num(
                    x,
                    3,
                )
            )

    for col in (
        "ending_equity",
        "total_fees",
    ):
        if col in q.columns:
            q[col] = q[
                col
            ].map(_money)

    print(
        q.to_string(
            index=False
        )
    )

    print(
        "\n=== JARGON — ONE-LINE CAPTIONS ==="
    )
    glossary = glossary_frame()
    for metric in (
        "cagr",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "max_drawdown_duration_sessions",
        "profit_factor",
        "turnover_multiple",
        "mean_daily_ic",
        "icir",
        "mean_top_decile_excess",
        "walk_forward_testing",
        "transaction_cost_adjusted_return",
    ):
        row = glossary.loc[
            glossary[
                "metric"
            ].eq(metric)
        ]
        if row.empty:
            continue
        item = row.iloc[0]
        print(
            f"{item['label']} — "
            f"{item['caption']}"
        )


if __name__ == "__main__":
    main()
