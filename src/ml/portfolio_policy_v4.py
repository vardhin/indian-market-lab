from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ML_DIR = SCRIPT_PATH.parent
BACKTEST_DIR = SCRIPT_PATH.parents[1] / "backtest"

for path in (ML_DIR, BACKTEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines import (  # noqa: E402
    CostProfile,
    build_day_index,
    build_events_by_market_index,
    load_mechanical_events,
)
from controller_v1 import (  # noqa: E402
    attach_scores,
    load_panel,
)
from portfolio_oracle_v4 import (  # noqa: E402
    action_features,
    apply_mechanical_actions,
    candidate_frame,
    clone_state,
    day_lookup,
    decode_target,
    equity_at_price,
    execute_target,
    market_value_weights,
    target_signature,
)


RANDOM_STATE = 4242


def student_action_search(
    *,
    state: dict,
    day: pd.DataFrame,
    model_bundle: dict,
    slots: int,
    particles: int,
    iterations: int,
    restarts: int,
    random_state: int,
) -> dict:
    candidates = candidate_frame(
        day
    )
    if candidates.empty:
        return {
            "target": {},
            "predicted_q": 0.0,
        }

    model = model_bundle[
        "model"
    ]
    feature_order = list(
        model_bundle[
            "features"
        ]
    )

    dim = (
        slots
        + slots
        + 1
    )
    global_best = None

    for restart in range(
        restarts
    ):
        rng = np.random.default_rng(
            random_state
            + restart
            * 100_003
        )
        positions = np.empty(
            (
                particles,
                dim,
            ),
            dtype=float,
        )
        positions[
            :,
            :slots,
        ] = rng.uniform(
            0.0,
            1.0,
            size=(
                particles,
                slots,
            ),
        )
        positions[
            :,
            slots:,
        ] = rng.normal(
            0.0,
            1.0,
            size=(
                particles,
                slots + 1,
            ),
        )
        velocities = rng.normal(
            0.0,
            0.12,
            size=(
                particles,
                dim,
            ),
        )

        pbest = positions.copy()
        pbest_values = np.full(
            particles,
            -1e18,
            dtype=float,
        )
        gbest = positions[
            0
        ].copy()
        gbest_value = -1e18

        for _iteration in range(
            iterations
        ):
            feature_rows = []
            targets = []

            for i in range(
                particles
            ):
                target = decode_target(
                    positions[
                        i,
                        :slots,
                    ],
                    positions[
                        i,
                        slots:,
                    ],
                    candidates,
                    slots=slots,
                )
                targets.append(
                    target
                )
                feature_rows.append(
                    action_features(
                        state=state,
                        target=target,
                        day=day,
                        slots=slots,
                    )
                )

            feature_frame = pd.DataFrame(
                feature_rows
            ).reindex(
                columns=feature_order
            )
            predictions = model.predict(
                feature_frame
            )

            for i, value in enumerate(
                predictions
            ):
                value = float(
                    value
                )
                if value > pbest_values[i]:
                    pbest_values[i] = value
                    pbest[i] = (
                        positions[i].copy()
                    )
                if value > gbest_value:
                    gbest_value = value
                    gbest = (
                        positions[i].copy()
                    )

            r1 = rng.random(
                size=(
                    particles,
                    dim,
                )
            )
            r2 = rng.random(
                size=(
                    particles,
                    dim,
                )
            )
            velocities = (
                0.72
                * velocities
                + 1.49
                * r1
                * (
                    pbest
                    - positions
                )
                + 1.49
                * r2
                * (
                    gbest[
                        None,
                        :
                    ]
                    - positions
                )
            )
            positions += velocities
            positions[
                :,
                :slots,
            ] = np.clip(
                positions[
                    :,
                    :slots,
                ],
                0.0,
                np.nextafter(
                    1.0,
                    0.0,
                ),
            )
            positions[
                :,
                slots:,
            ] = np.clip(
                positions[
                    :,
                    slots:,
                ],
                -6.0,
                6.0,
            )

        target = decode_target(
            gbest[
                :slots
            ],
            gbest[
                slots:
            ],
            candidates,
            slots=slots,
        )
        features = pd.DataFrame([
            action_features(
                state=state,
                target=target,
                day=day,
                slots=slots,
            )
        ]).reindex(
            columns=feature_order
        )
        predicted_q = float(
            model.predict(
                features
            )[
                0
            ]
        )

        if (
            global_best is None
            or predicted_q
            > global_best[
                "predicted_q"
            ]
        ):
            global_best = {
                "target": (
                    target
                ),
                "predicted_q": (
                    predicted_q
                ),
            }

    if global_best is None:
        raise RuntimeError(
            "Student PSO did not find "
            "a valid action."
        )

    return global_best


def performance_metrics(
    equity: pd.DataFrame,
    *,
    initial_capital: float,
    annual_risk_free_rate: float,
) -> dict:
    values = equity[
        "equity"
    ].to_numpy(
        dtype=float
    )
    dates = pd.to_datetime(
        equity[
            "date"
        ]
    )

    if len(values) < 2:
        return {}

    daily = (
        pd.Series(
            values
        )
        .pct_change()
        .dropna()
    )
    years = max(
        (
            dates.iloc[
                -1
            ]
            - dates.iloc[
                0
            ]
        ).days
        / 365.25,
        1.0
        / 365.25,
    )
    cagr = (
        values[
            -1
        ]
        / float(
            initial_capital
        )
    ) ** (
        1.0
        / years
    ) - 1.0

    annual_return = float(
        daily.mean()
        * 252.0
    )
    annual_vol = float(
        daily.std(
            ddof=1
        )
        * math.sqrt(
            252.0
        )
    )
    excess_return = (
        annual_return
        - float(
            annual_risk_free_rate
        )
    )
    sharpe = (
        excess_return
        / annual_vol
        if annual_vol > 0
        else np.nan
    )

    downside = daily.loc[
        daily.lt(
            0.0
        )
    ]
    downside_vol = (
        float(
            downside.std(
                ddof=1
            )
            * math.sqrt(
                252.0
            )
        )
        if len(
            downside
        )
        > 1
        else np.nan
    )
    sortino = (
        excess_return
        / downside_vol
        if (
            math.isfinite(
                downside_vol
            )
            and downside_vol > 0
        )
        else np.nan
    )

    peaks = np.maximum.accumulate(
        values
    )
    drawdowns = (
        values
        / peaks
        - 1.0
    )

    max_drawdown = float(
        np.min(
            drawdowns
        )
    )
    calmar = (
        float(
            cagr
            / abs(
                max_drawdown
            )
        )
        if max_drawdown < 0
        else np.nan
    )

    return {
        "starting_capital": float(
            initial_capital
        ),
        "ending_equity": float(
            values[
                -1
            ]
        ),
        "total_return": float(
            values[
                -1
            ]
            / float(
                initial_capital
            )
            - 1.0
        ),
        "cagr": float(
            cagr
        ),
        "annualized_arithmetic_return": (
            annual_return
        ),
        "annualized_volatility": (
            annual_vol
        ),
        "risk_free_rate_annual": float(
            annual_risk_free_rate
        ),
        "sharpe": float(
            sharpe
        ),
        "sortino": float(
            sortino
        ),
        "max_drawdown": (
            max_drawdown
        ),
        "calmar": float(
            calmar
        ),
    }


def run_policy(
    root: Path,
    *,
    year: int,
    model_path: Path,
    slots: int,
    particles: int,
    iterations: int,
    restarts: int,
    initial_capital: float,
    costs: CostProfile,
    annual_risk_free_rate: float,
) -> dict:
    panel = load_panel(
        root,
        years=[
            int(
                year
            )
        ],
    )
    attach_scores(
        panel,
        root=root,
        stage="development",
        years=[
            int(
                year
            )
        ],
    )

    model_bundle = joblib.load(
        model_path
    )

    day_groups, dates = (
        build_day_index(
            panel
        )
    )
    market_indices = sorted(
        day_groups
    )
    if len(market_indices) < 2:
        raise RuntimeError(
            "Policy panel has fewer "
            "than two market days."
        )

    events = load_mechanical_events(
        root
    )
    events_by_index = (
        build_events_by_market_index(
            events,
            dates,
        )
    )

    state = {
        "cash": float(
            initial_capital
        ),
        "fees": 0.0,
        "holdings": {},
    }

    equity_rows = []
    decision_rows = []
    total_turnover = 0.0

    first_index = int(
        market_indices[
            0
        ]
    )
    first_day = panel.loc[
        day_groups[
            first_index
        ]
    ]
    initial_lookup = day_lookup(
        first_day
    )
    initial_equity = equity_at_price(
        state,
        initial_lookup,
        "close",
    )
    equity_rows.append({
        "date": pd.Timestamp(
            dates[
                first_index
            ]
        ).normalize(),
        "market_index": (
            first_index
        ),
        "equity": float(
            initial_equity
        ),
        "cash": float(
            state[
                "cash"
            ]
        ),
        "exposure": 0.0,
        "holdings": 0,
    })

    for position in range(
        len(
            market_indices
        )
        - 1
    ):
        signal_index = int(
            market_indices[
                position
            ]
        )
        execution_index = int(
            market_indices[
                position
                + 1
            ]
        )
        signal_day = panel.loc[
            day_groups[
                signal_index
            ]
        ]
        signal_date = pd.Timestamp(
            dates[
                signal_index
            ]
        ).normalize()

        result = student_action_search(
            state=state,
            day=signal_day,
            model_bundle=(
                model_bundle
            ),
            slots=slots,
            particles=particles,
            iterations=iterations,
            restarts=restarts,
            random_state=(
                RANDOM_STATE
                + signal_index
            ),
        )
        target = result[
            "target"
        ]

        decision_rows.append({
            "date": (
                signal_date
            ),
            "market_index": (
                signal_index
            ),
            "predicted_q": float(
                result[
                    "predicted_q"
                ]
            ),
            "current_assets": json.dumps(
                sorted(
                    state[
                        "holdings"
                    ]
                )
            ),
            "target_assets": json.dumps(
                sorted(
                    target
                )
            ),
            "target_signature": (
                target_signature(
                    target
                )
            ),
        })

        next_state = clone_state(
            state
        )
        apply_mechanical_actions(
            next_state,
            events_by_index,
            execution_index,
        )
        execution_day = panel.loc[
            day_groups[
                execution_index
            ]
        ]
        executed = execute_target(
            next_state,
            target_weights=target,
            execution_day=execution_day,
            costs=costs,
        )
        if executed is None:
            raise RuntimeError(
                "Student action could not "
                "execute at next open."
            )
        state = executed[
            0
        ]
        total_turnover += float(
            executed[
                1
            ]
        )

        close_lookup = day_lookup(
            execution_day
        )
        equity = equity_at_price(
            state,
            close_lookup,
            "close",
        )
        if (
            equity is None
            or equity <= 0
        ):
            raise RuntimeError(
                "Invalid policy equity."
            )

        _weights, cash_weight = (
            market_value_weights(
                state,
                execution_day,
            )
        )

        equity_rows.append({
            "date": pd.Timestamp(
                dates[
                    execution_index
                ]
            ).normalize(),
            "market_index": (
                execution_index
            ),
            "equity": float(
                equity
            ),
            "cash": float(
                state[
                    "cash"
                ]
            ),
            "exposure": float(
                max(
                    0.0,
                    1.0
                    - cash_weight,
                )
            ),
            "holdings": int(
                len(
                    state[
                        "holdings"
                    ]
                )
            ),
        })

        if (
            (
                position
                + 1
            )
            % 20
            == 0
        ):
            print(
                "Policy decisions: "
                f"{position + 1:,}; "
                "equity="
                f"{float(equity):,.2f}",
                flush=True,
            )

    equity = pd.DataFrame(
        equity_rows
    )
    decisions = pd.DataFrame(
        decision_rows
    )
    metrics = performance_metrics(
        equity,
        initial_capital=(
            initial_capital
        ),
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
    )
    metrics[
        "average_exposure"
    ] = float(
        equity[
            "exposure"
        ].mean()
    )
    metrics[
        "average_holdings"
    ] = float(
        equity[
            "holdings"
        ].mean()
    )
    metrics[
        "total_fees"
    ] = float(
        state.get(
            "fees",
            0.0,
        )
    )
    metrics[
        "turnover_value"
    ] = float(
        total_turnover
    )
    metrics[
        "turnover_multiple"
    ] = float(
        total_turnover
        / initial_capital
        if initial_capital > 0
        else np.nan
    )

    output_root = (
        root
        / "reports/ml/"
        "portfolio_policy_v4"
        / f"development_{year}"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    equity.to_parquet(
        output_root
        / "equity_curve.parquet",
        index=False,
        compression="zstd",
    )
    decisions.to_parquet(
        output_root
        / "decisions.parquet",
        index=False,
        compression="zstd",
    )
    (
        output_root
        / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
            default=float,
        )
        + "\n"
    )

    print(
        "\n=== PORTFOLIO POLICY V4 COMPLETE ==="
    )
    print(
        "Year:    "
        f"{year}"
    )
    print(
        "CAGR:    "
        f"{float(metrics['cagr']):.2%}"
    )
    print(
        "Sharpe:  "
        f"{float(metrics['sharpe']):.3f}"
    )
    print(
        "MDD:     "
        f"{float(metrics['max_drawdown']):.2%}"
    )
    print(
        "Exposure:"
        f" {float(metrics['average_exposure']):.1%}"
    )
    print(
        "Output:  "
        f"{output_root}"
    )

    return metrics


def self_test() -> None:
    assert callable(
        student_action_search
    )
    assert callable(
        performance_metrics
    )
    print(
        "Portfolio-policy-v4 self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Use the frozen tree action-value model to search and execute "
            "whole-portfolio actions without future information."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--year",
        type=int,
        default=2023,
    )
    ap.add_argument(
        "--model-path",
        default=None,
    )
    ap.add_argument(
        "--slots",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--particles",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=60,
    )
    ap.add_argument(
        "--restarts",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--capital",
        type=float,
        default=50_000.0,
    )
    ap.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.065,
    )
    ap.add_argument(
        "--brokerage-per-order",
        type=float,
        default=15.0,
    )
    ap.add_argument(
        "--dp-charge-per-sell",
        type=float,
        default=0.0,
    )
    ap.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    root = Path(
        args.root
    ).resolve()

    model_path = (
        Path(
            args.model_path
        ).resolve()
        if args.model_path
        is not None
        else (
            root
            / "data/processed/"
            "portfolio_student_v4/"
            "development_best.joblib"
        )
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen student model: {model_path}"
        )

    costs = CostProfile(
        brokerage_per_order=float(
            args.brokerage_per_order
        ),
        dp_charge_per_sell=float(
            args.dp_charge_per_sell
        ),
        slippage_bps=float(
            args.slippage_bps
        ),
    )

    run_policy(
        root,
        year=int(
            args.year
        ),
        model_path=model_path,
        slots=int(
            args.slots
        ),
        particles=int(
            args.particles
        ),
        iterations=int(
            args.iterations
        ),
        restarts=int(
            args.restarts
        ),
        initial_capital=float(
            args.capital
        ),
        costs=costs,
        annual_risk_free_rate=float(
            args.risk_free_rate
        ),
    )


if __name__ == "__main__":
    main()
