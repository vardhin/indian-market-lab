from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

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
    buy_execution,
    load_mechanical_events,
    max_affordable_quantity,
    sell_execution,
)
from controller_v1 import (  # noqa: E402
    B4_FEATURES,
    DEVELOPMENT_YEARS,
    attach_scores,
    load_panel,
)
from experiment_progress import ProgressReporter  # noqa: E402


DEFAULT_SLOTS = 5
DEFAULT_LOOKAHEAD = 20
DEFAULT_PARTICLES = 96
DEFAULT_ITERATIONS = 80
DEFAULT_RESTARTS = 3
DEFAULT_SAMPLES_PER_STATE = 32
RANDOM_STATE = 242


def clone_state(state: dict) -> dict:
    return {
        "cash": float(state["cash"]),
        "fees": float(
            state.get(
                "fees",
                0.0,
            )
        ),
        "holdings": {
            str(cid): float(qty)
            for cid, qty in state["holdings"].items()
            if float(qty) > 0
        },
    }


def softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x - float(np.max(x))
    ex = np.exp(x)
    total = float(ex.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full(len(x), 1.0 / len(x))
    return ex / total


def candidate_frame(day: pd.DataFrame) -> pd.DataFrame:
    work = day.loc[
        day["eligible_universe"].fillna(False)
        & day["persistent_score"].notna()
        & day["persistent_rank"].notna()
        & day["persistent_score_pct"].notna()
        & day["open"].notna()
        & day["close"].notna()
    ].copy()

    return (
        work.sort_values(
            [
                "persistent_rank",
                "turnover_median_20d",
                "canonical_security_id",
            ],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def decode_target(
    selectors: np.ndarray,
    logits: np.ndarray,
    candidates: pd.DataFrame,
    *,
    slots: int,
) -> dict[str, float]:
    if candidates.empty:
        return {}

    n = len(candidates)
    chosen: list[str] = []
    used: set[str] = set()

    for raw in np.asarray(selectors, dtype=float)[:slots]:
        start = min(
            n - 1,
            max(0, int(math.floor(float(raw) * n))),
        )
        selected = None

        for offset in range(n):
            idx = (start + offset) % n
            cid = str(
                candidates.iloc[idx]["canonical_security_id"]
            )
            if cid not in used:
                selected = cid
                break

        if selected is None:
            break

        used.add(selected)
        chosen.append(selected)

    weights = softmax(
        np.asarray(logits, dtype=float)[: len(chosen) + 1]
    )

    target: dict[str, float] = {}
    for cid, weight in zip(chosen, weights[:-1]):
        if float(weight) > 1e-6:
            target[cid] = float(weight)

    # The final softmax coordinate is cash. Asset weights therefore sum <= 1.
    return target


def day_lookup(day: pd.DataFrame) -> pd.DataFrame:
    return day.set_index(
        "canonical_security_id",
        drop=False,
    )


def apply_mechanical_actions(
    state: dict,
    events_by_index: dict,
    market_index: int,
) -> None:
    for cid, multiplier in events_by_index.get(
        int(market_index),
        [],
    ):
        cid = str(cid)
        if cid not in state["holdings"]:
            continue
        post = (
            float(state["holdings"][cid])
            * float(multiplier)
        )
        state["holdings"][cid] = float(
            math.floor(post + 1e-12)
        )
        if state["holdings"][cid] <= 0:
            del state["holdings"][cid]


def equity_at_price(
    state: dict,
    lookup: pd.DataFrame,
    column: str,
) -> float | None:
    value = float(state["cash"])

    for cid, quantity in state["holdings"].items():
        if cid not in lookup.index:
            return None

        price = float(
            lookup.loc[cid, column]
        )
        if (
            not math.isfinite(price)
            or price <= 0
        ):
            return None

        value += float(quantity) * price

    return value


def execute_target(
    state: dict,
    *,
    target_weights: dict[str, float],
    execution_day: pd.DataFrame,
    costs: CostProfile,
) -> tuple[dict, float] | None:
    state = clone_state(state)
    lookup = day_lookup(execution_day)

    pretrade_equity = equity_at_price(
        state,
        lookup,
        "open",
    )
    if (
        pretrade_equity is None
        or pretrade_equity <= 0
    ):
        return None

    desired_qty: dict[str, int] = {}

    for cid, weight in target_weights.items():
        if cid not in lookup.index:
            return None

        quoted_open = float(
            lookup.loc[cid, "open"]
        )
        if (
            not math.isfinite(quoted_open)
            or quoted_open <= 0
        ):
            return None

        per_share = quoted_open * (
            1.0
            + costs.slippage_bps
            / 10_000.0
        )
        desired_qty[cid] = max(
            0,
            int(
                math.floor(
                    pretrade_equity
                    * float(weight)
                    / max(per_share, 1e-12)
                )
            ),
        )

    turnover = 0.0

    # Sells first so switches can fund their buys.
    for cid in list(state["holdings"]):
        current = int(
            math.floor(
                float(state["holdings"][cid])
                + 1e-12
            )
        )
        desired = int(
            desired_qty.get(cid, 0)
        )
        quantity = current - desired

        if quantity <= 0:
            continue

        if cid not in lookup.index:
            return None

        quoted_open = float(
            lookup.loc[cid, "open"]
        )
        execution = sell_execution(
            quoted_open,
            quantity,
            costs,
        )
        state["cash"] += float(
            execution["cash_in"]
        )
        state["fees"] += float(
            execution["fees"]
        )
        turnover += float(
            execution["trade_value"]
        )
        remaining = current - quantity

        if remaining > 0:
            state["holdings"][cid] = float(
                remaining
            )
        else:
            del state["holdings"][cid]

    buy_order = sorted(
        target_weights,
        key=lambda cid: (
            -float(target_weights[cid]),
            str(cid),
        ),
    )

    for cid in buy_order:
        desired = int(
            desired_qty.get(cid, 0)
        )
        current = int(
            math.floor(
                float(
                    state["holdings"].get(
                        cid,
                        0.0,
                    )
                )
                + 1e-12
            )
        )
        need = desired - current
        if need <= 0:
            continue

        if cid not in lookup.index:
            return None

        quoted_open = float(
            lookup.loc[cid, "open"]
        )

        affordable = max_affordable_quantity(
            float(state["cash"]),
            quoted_open,
            costs,
        )
        quantity = min(
            int(need),
            int(affordable),
        )
        if quantity <= 0:
            continue

        execution = buy_execution(
            quoted_open,
            quantity,
            costs,
        )
        state["cash"] -= float(
            execution["cash_out"]
        )
        state["fees"] += float(
            execution["fees"]
        )
        turnover += float(
            execution["trade_value"]
        )
        state["holdings"][cid] = float(
            current + quantity
        )

    # Tiny negative floating residue should not invalidate a path.
    if state["cash"] < -1e-6:
        return None
    state["cash"] = max(
        0.0,
        float(state["cash"]),
    )

    return state, float(turnover)


def market_value_weights(
    state: dict,
    day: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    lookup = day_lookup(day)
    total = equity_at_price(
        state,
        lookup,
        "close",
    )
    if total is None or total <= 0:
        return {}, 1.0

    weights: dict[str, float] = {}
    for cid, quantity in state["holdings"].items():
        if cid not in lookup.index:
            continue
        close = float(
            lookup.loc[cid, "close"]
        )
        if math.isfinite(close) and close > 0:
            weights[cid] = (
                float(quantity)
                * close
                / total
            )

    cash_weight = max(
        0.0,
        min(
            1.0,
            float(state["cash"])
            / total,
        ),
    )
    return weights, cash_weight


def target_signature(
    target: dict[str, float],
) -> str:
    rows = sorted(
        (
            str(cid),
            round(float(weight), 4),
        )
        for cid, weight in target.items()
        if float(weight) > 1e-5
    )
    return json.dumps(
        rows,
        separators=(",", ":"),
    )


def plan_dimensions(
    *,
    horizon: int,
    slots: int,
) -> int:
    # Per decision: K selectors + K asset logits + 1 cash logit.
    return int(
        horizon
        * (
            slots
            + slots
            + 1
        )
    )


def decode_plan_step(
    vector: np.ndarray,
    *,
    step: int,
    candidates: pd.DataFrame,
    slots: int,
) -> dict[str, float]:
    width = (
        slots
        + slots
        + 1
    )
    start = step * width
    selectors = vector[
        start : start + slots
    ]
    logits = vector[
        start + slots : start + width
    ]
    return decode_target(
        selectors,
        logits,
        candidates,
        slots=slots,
    )


def simulate_plan(
    vector: np.ndarray,
    *,
    initial_state: dict,
    market_indices: list[int],
    day_groups: dict,
    panel: pd.DataFrame,
    events_by_index: dict,
    candidates_by_index: dict[int, pd.DataFrame],
    slots: int,
    costs: CostProfile,
    drawdown_penalty: float,
) -> dict:
    state = clone_state(
        initial_state
    )

    first_index = int(
        market_indices[0]
    )
    first_day = panel.loc[
        day_groups[first_index]
    ]
    first_lookup = day_lookup(
        first_day
    )
    start_equity = equity_at_price(
        state,
        first_lookup,
        "close",
    )
    if (
        start_equity is None
        or start_equity <= 0
    ):
        return {
            "fitness": -1e9,
            "terminal_return": -1.0,
            "max_drawdown": -1.0,
            "turnover": np.inf,
            "first_target": {},
        }

    equity_path = [
        float(start_equity)
    ]
    total_turnover = 0.0
    first_target: dict[
        str,
        float,
    ] = {}

    horizon = (
        len(market_indices)
        - 1
    )

    for step in range(horizon):
        signal_index = int(
            market_indices[step]
        )
        execution_index = int(
            market_indices[
                step + 1
            ]
        )

        target = decode_plan_step(
            vector,
            step=step,
            candidates=(
                candidates_by_index[
                    signal_index
                ]
            ),
            slots=slots,
        )
        if step == 0:
            first_target = dict(
                target
            )

        apply_mechanical_actions(
            state,
            events_by_index,
            execution_index,
        )

        execution_day = panel.loc[
            day_groups[
                execution_index
            ]
        ]

        result = execute_target(
            state,
            target_weights=target,
            execution_day=execution_day,
            costs=costs,
        )
        if result is None:
            return {
                "fitness": -1e9,
                "terminal_return": -1.0,
                "max_drawdown": -1.0,
                "turnover": np.inf,
                "first_target": first_target,
            }

        state, turnover = result
        total_turnover += float(
            turnover
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
            return {
                "fitness": -1e9,
                "terminal_return": -1.0,
                "max_drawdown": -1.0,
                "turnover": np.inf,
                "first_target": first_target,
            }

        equity_path.append(
            float(equity)
        )

    values = np.asarray(
        equity_path,
        dtype=float,
    )
    peaks = np.maximum.accumulate(
        values
    )
    drawdowns = (
        values / peaks - 1.0
    )
    max_drawdown = float(
        drawdowns.min()
    )
    terminal_return = float(
        values[-1]
        / values[0]
        - 1.0
    )

    log_growth = math.log(
        max(
            values[-1]
            / values[0],
            1e-12,
        )
    )
    fitness = (
        log_growth
        - float(drawdown_penalty)
        * abs(max_drawdown)
    )

    return {
        "fitness": float(
            fitness
        ),
        "terminal_return": (
            terminal_return
        ),
        "max_drawdown": (
            max_drawdown
        ),
        "turnover": float(
            total_turnover
        ),
        "first_target": (
            first_target
        ),
    }


def pso_search(
    *,
    initial_state: dict,
    market_indices: list[int],
    day_groups: dict,
    panel: pd.DataFrame,
    events_by_index: dict,
    candidates_by_index: dict[int, pd.DataFrame],
    slots: int,
    particles: int,
    iterations: int,
    restarts: int,
    samples_per_state: int,
    costs: CostProfile,
    drawdown_penalty: float,
    random_state: int,
    progress: ProgressReporter | None = None,
    progress_offset: int = 0,
    progress_message: str = "",
) -> tuple[dict, list[dict]]:
    horizon = (
        len(market_indices)
        - 1
    )
    dim = plan_dimensions(
        horizon=horizon,
        slots=slots,
    )
    width = (
        slots
        + slots
        + 1
    )

    all_candidates: dict[
        str,
        dict,
    ] = {}
    global_best = None

    for restart in range(restarts):
        rng = np.random.default_rng(
            random_state
            + restart
            * 100_003
        )

        positions = np.empty(
            (particles, dim),
            dtype=float,
        )
        velocities = rng.normal(
            0.0,
            0.12,
            size=(
                particles,
                dim,
            ),
        )

        for step in range(horizon):
            start = step * width
            positions[
                :,
                start : start + slots,
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
                start + slots : start + width,
            ] = rng.normal(
                0.0,
                1.0,
                size=(
                    particles,
                    slots + 1,
                ),
            )

        pbest = positions.copy()
        pbest_values = np.full(
            particles,
            -1e18,
            dtype=float,
        )
        pbest_details: list[
            dict | None
        ] = [
            None
            for _ in range(
                particles
            )
        ]

        gbest = positions[0].copy()
        gbest_value = -1e18
        gbest_detail = None

        for _iteration in range(
            iterations
        ):
            for i in range(
                particles
            ):
                detail = simulate_plan(
                    positions[i],
                    initial_state=initial_state,
                    market_indices=(
                        market_indices
                    ),
                    day_groups=(
                        day_groups
                    ),
                    panel=panel,
                    events_by_index=(
                        events_by_index
                    ),
                    candidates_by_index=(
                        candidates_by_index
                    ),
                    slots=slots,
                    costs=costs,
                    drawdown_penalty=(
                        drawdown_penalty
                    ),
                )
                value = float(
                    detail["fitness"]
                )

                if value > pbest_values[i]:
                    pbest_values[i] = value
                    pbest[i] = (
                        positions[i].copy()
                    )
                    pbest_details[i] = (
                        detail
                    )

                if value > gbest_value:
                    gbest_value = value
                    gbest = (
                        positions[i].copy()
                    )
                    gbest_detail = detail

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

            for step in range(
                horizon
            ):
                start = step * width
                positions[
                    :,
                    start : start + slots,
                ] = np.clip(
                    positions[
                        :,
                        start : start + slots,
                    ],
                    0.0,
                    np.nextafter(
                        1.0,
                        0.0,
                    ),
                )
                positions[
                    :,
                    start + slots : start + width,
                ] = np.clip(
                    positions[
                        :,
                        start + slots : start + width,
                    ],
                    -6.0,
                    6.0,
                )

            if progress is not None:
                local_iterations = (
                    restart
                    * iterations
                    + _iteration
                    + 1
                )
                progress.update(
                    progress_offset
                    + local_iterations
                    * particles,
                    message=(
                        progress_message
                    ),
                    detail=(
                        f"restart "
                        f"{restart + 1}/{restarts} · "
                        f"iteration "
                        f"{_iteration + 1}/{iterations} · "
                        f"best "
                        f"{gbest_value:+.6f}"
                    ),
                    force=(
                        restart
                        == restarts - 1
                        and _iteration
                        == iterations - 1
                    ),
                )

        for i in range(
            particles
        ):
            detail = (
                pbest_details[i]
            )
            if detail is None:
                continue

            signature = target_signature(
                detail[
                    "first_target"
                ]
            )
            previous = (
                all_candidates.get(
                    signature
                )
            )
            if (
                previous is None
                or float(
                    detail[
                        "fitness"
                    ]
                )
                > float(
                    previous[
                        "fitness"
                    ]
                )
            ):
                all_candidates[
                    signature
                ] = {
                    **detail,
                    "plan": (
                        pbest[i].copy()
                    ),
                }

        if (
            gbest_detail
            is not None
            and (
                global_best
                is None
                or float(
                    gbest_detail[
                        "fitness"
                    ]
                )
                > float(
                    global_best[
                        "fitness"
                    ]
                )
            )
        ):
            global_best = {
                **gbest_detail,
                "plan": (
                    gbest.copy()
                ),
            }

    if global_best is None:
        raise RuntimeError(
            "PSO failed to produce a valid "
            "portfolio trajectory."
        )

    ranked = sorted(
        all_candidates.values(),
        key=lambda row: float(
            row["fitness"]
        ),
        reverse=True,
    )

    if not ranked:
        ranked = [
            global_best
        ]

    # Ensure the actual global best is represented even if its rounded
    # first-target signature collided with another continuation.
    best_signature = target_signature(
        global_best[
            "first_target"
        ]
    )
    if (
        not ranked
        or target_signature(
            ranked[0][
                "first_target"
            ]
        )
        != best_signature
    ):
        ranked.insert(
            0,
            global_best,
        )

    return (
        global_best,
        ranked[
            : max(
                1,
                int(
                    samples_per_state
                ),
            )
        ],
    )


def portfolio_summary(
    state: dict,
    day: pd.DataFrame,
) -> dict:
    weights, cash_weight = (
        market_value_weights(
            state,
            day,
        )
    )

    lookup = day_lookup(
        day
    )
    score_pcts = []
    ranks = []

    for cid, weight in weights.items():
        if (
            cid in lookup.index
            and pd.notna(
                lookup.loc[
                    cid,
                    "persistent_score_pct",
                ]
            )
        ):
            score_pcts.append(
                (
                    float(weight),
                    float(
                        lookup.loc[
                            cid,
                            "persistent_score_pct",
                        ]
                    ),
                )
            )
        if (
            cid in lookup.index
            and pd.notna(
                lookup.loc[
                    cid,
                    "persistent_rank",
                ]
            )
        ):
            ranks.append(
                (
                    float(weight),
                    float(
                        lookup.loc[
                            cid,
                            "persistent_rank",
                        ]
                    ),
                )
            )

    def weighted_mean(
        values: list[
            tuple[
                float,
                float,
            ]
        ],
    ) -> float:
        if not values:
            return np.nan
        total_weight = sum(
            weight
            for weight, _ in values
        )
        if total_weight <= 0:
            return np.nan
        return float(
            sum(
                weight * value
                for weight, value
                in values
            )
            / total_weight
        )

    return {
        "holdings_count": float(
            len(weights)
        ),
        "cash_fraction": float(
            cash_weight
        ),
        "exposure": float(
            max(
                0.0,
                1.0
                - cash_weight,
            )
        ),
        "weighted_score_pct": (
            weighted_mean(
                score_pcts
            )
        ),
        "weighted_rank": (
            weighted_mean(
                ranks
            )
        ),
    }


def target_summary(
    target: dict[str, float],
    day: pd.DataFrame,
) -> dict:
    lookup = day_lookup(
        day
    )
    rows = []

    for cid, weight in target.items():
        if cid not in lookup.index:
            continue

        row = lookup.loc[
            cid
        ]
        rows.append(
            (
                float(weight),
                float(
                    row[
                        "persistent_score_pct"
                    ]
                )
                if pd.notna(
                    row[
                        "persistent_score_pct"
                    ]
                )
                else np.nan,
                float(
                    row[
                        "persistent_rank"
                    ]
                )
                if pd.notna(
                    row[
                        "persistent_rank"
                    ]
                )
                else np.nan,
            )
        )

    asset_weight = float(
        sum(
            target.values()
        )
    )
    valid_scores = [
        (
            weight,
            score,
        )
        for weight, score, _rank
        in rows
        if math.isfinite(score)
    ]
    valid_ranks = [
        (
            weight,
            rank,
        )
        for weight, _score, rank
        in rows
        if math.isfinite(rank)
    ]

    def weighted(
        values,
    ):
        denom = sum(
            weight
            for weight, _ in values
        )
        if denom <= 0:
            return np.nan
        return float(
            sum(
                weight * value
                for weight, value
                in values
            )
            / denom
        )

    return {
        "holdings_count": float(
            len(target)
        ),
        "cash_fraction": float(
            max(
                0.0,
                1.0
                - asset_weight,
            )
        ),
        "exposure": float(
            min(
                1.0,
                max(
                    0.0,
                    asset_weight,
                ),
            )
        ),
        "weighted_score_pct": (
            weighted(
                valid_scores
            )
        ),
        "weighted_rank": (
            weighted(
                valid_ranks
            )
        ),
    }


def action_features(
    *,
    state: dict,
    target: dict[str, float],
    day: pd.DataFrame,
    slots: int,
) -> dict:
    current_weights, _cash = (
        market_value_weights(
            state,
            day,
        )
    )
    current_summary = (
        portfolio_summary(
            state,
            day,
        )
    )
    target_stats = (
        target_summary(
            target,
            day,
        )
    )

    all_cids = (
        set(
            current_weights
        )
        | set(
            target
        )
    )
    turnover_fraction = 0.5 * sum(
        abs(
            float(
                target.get(
                    cid,
                    0.0,
                )
            )
            - float(
                current_weights.get(
                    cid,
                    0.0,
                )
            )
        )
        for cid in all_cids
    )
    turnover_fraction += 0.5 * abs(
        float(
            target_stats[
                "cash_fraction"
            ]
        )
        - float(
            current_summary[
                "cash_fraction"
            ]
        )
    )

    features = {
        "f__state_holdings_count": (
            current_summary[
                "holdings_count"
            ]
        ),
        "f__state_cash_fraction": (
            current_summary[
                "cash_fraction"
            ]
        ),
        "f__state_exposure": (
            current_summary[
                "exposure"
            ]
        ),
        "f__state_weighted_score_pct": (
            current_summary[
                "weighted_score_pct"
            ]
        ),
        "f__state_weighted_rank": (
            current_summary[
                "weighted_rank"
            ]
        ),
        "f__target_holdings_count": (
            target_stats[
                "holdings_count"
            ]
        ),
        "f__target_cash_fraction": (
            target_stats[
                "cash_fraction"
            ]
        ),
        "f__target_exposure": (
            target_stats[
                "exposure"
            ]
        ),
        "f__target_weighted_score_pct": (
            target_stats[
                "weighted_score_pct"
            ]
        ),
        "f__target_weighted_rank": (
            target_stats[
                "weighted_rank"
            ]
        ),
        "f__delta_score_pct": (
            target_stats[
                "weighted_score_pct"
            ]
            - current_summary[
                "weighted_score_pct"
            ]
            if (
                math.isfinite(
                    target_stats[
                        "weighted_score_pct"
                    ]
                )
                and math.isfinite(
                    current_summary[
                        "weighted_score_pct"
                    ]
                )
            )
            else np.nan
        ),
        "f__delta_rank": (
            target_stats[
                "weighted_rank"
            ]
            - current_summary[
                "weighted_rank"
            ]
            if (
                math.isfinite(
                    target_stats[
                        "weighted_rank"
                    ]
                )
                and math.isfinite(
                    current_summary[
                        "weighted_rank"
                    ]
                )
            )
            else np.nan
        ),
        "f__turnover_fraction": float(
            turnover_fraction
        ),
    }

    lookup = day_lookup(
        day
    )

    current_slots = sorted(
        current_weights.items(),
        key=lambda item: (
            -float(
                item[1]
            ),
            str(
                item[0]
            ),
        ),
    )[
        :slots
    ]
    target_slots = sorted(
        target.items(),
        key=lambda item: (
            -float(
                item[1]
            ),
            str(
                item[0]
            ),
        ),
    )[
        :slots
    ]

    for side, rows in (
        (
            "source",
            current_slots,
        ),
        (
            "target",
            target_slots,
        ),
    ):
        for slot in range(
            slots
        ):
            prefix = (
                f"f__{side}{slot}_"
            )
            if slot >= len(rows):
                features[
                    prefix
                    + "weight"
                ] = 0.0
                features[
                    prefix
                    + "score_pct"
                ] = np.nan
                features[
                    prefix
                    + "rank"
                ] = np.nan
                for feature in (
                    B4_FEATURES
                ):
                    features[
                        prefix
                        + feature
                    ] = np.nan
                continue

            cid, weight = rows[
                slot
            ]
            features[
                prefix
                + "weight"
            ] = float(
                weight
            )

            if cid not in lookup.index:
                features[
                    prefix
                    + "score_pct"
                ] = np.nan
                features[
                    prefix
                    + "rank"
                ] = np.nan
                for feature in (
                    B4_FEATURES
                ):
                    features[
                        prefix
                        + feature
                    ] = np.nan
                continue

            row = lookup.loc[
                cid
            ]
            features[
                prefix
                + "score_pct"
            ] = (
                float(
                    row[
                        "persistent_score_pct"
                    ]
                )
                if pd.notna(
                    row[
                        "persistent_score_pct"
                    ]
                )
                else np.nan
            )
            features[
                prefix
                + "rank"
            ] = (
                float(
                    row[
                        "persistent_rank"
                    ]
                )
                if pd.notna(
                    row[
                        "persistent_rank"
                    ]
                )
                else np.nan
            )

            for feature in (
                B4_FEATURES
            ):
                value = row[
                    feature
                ]
                features[
                    prefix
                    + feature
                ] = (
                    float(
                        value
                    )
                    if pd.notna(
                        value
                    )
                    else np.nan
                )

    return features


def execute_first_action(
    state: dict,
    *,
    best: dict,
    signal_index: int,
    execution_index: int,
    panel: pd.DataFrame,
    day_groups: dict,
    events_by_index: dict,
    costs: CostProfile,
) -> dict:
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
    result = execute_target(
        next_state,
        target_weights=(
            best[
                "first_target"
            ]
        ),
        execution_day=execution_day,
        costs=costs,
    )
    if result is None:
        raise RuntimeError(
            "Best oracle action could not "
            "be executed."
        )
    return result[0]


def build_oracle(
    root: Path,
    *,
    years: list[int],
    slots: int,
    lookahead: int,
    particles: int,
    iterations: int,
    restarts: int,
    samples_per_state: int,
    initial_capital: float,
    costs: CostProfile,
    drawdown_penalty: float,
    max_decision_days: int | None,
    engine: str = "cpu",
    device: str = "cuda",
    gpu_finalists: int = 128,
) -> dict:
    panel = load_panel(
        root,
        years=years,
    )
    attach_scores(
        panel,
        root=root,
        stage="development",
        years=years,
    )

    day_groups, dates = (
        build_day_index(
            panel
        )
    )
    market_indices = sorted(
        day_groups
    )
    events = (
        load_mechanical_events(
            root
        )
    )
    events_by_index = (
        build_events_by_market_index(
            events,
            dates,
        )
    )

    candidates_by_index = {
        int(index): candidate_frame(
            panel.loc[
                day_groups[
                    index
                ]
            ]
        )
        for index in market_indices
    }

    torch_cache = None
    cuda_search = None

    if engine == "torch-cuda":
        from portfolio_oracle_cuda import (
            build_torch_market_cache,
            pso_search_cuda,
        )

        torch_cache = build_torch_market_cache(
            panel=panel,
            day_groups=day_groups,
            market_indices=market_indices,
            candidates_by_index=(
                candidates_by_index
            ),
            events_by_index=(
                events_by_index
            ),
            device=device,
        )
        cuda_search = pso_search_cuda

        try:
            import torch

            print(
                "CUDA oracle engine: "
                f"{torch.cuda.get_device_name(0)}",
                flush=True,
            )
        except Exception:
            pass
    elif engine != "cpu":
        raise ValueError(
            "Unknown oracle engine: "
            f"{engine}. Use cpu or torch-cuda."
        )

    output_root = (
        root
        / "data/processed/"
        "portfolio_oracle_v4"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_root = (
        root
        / "reports/ml/"
        "portfolio_oracle_v4"
        / "development"
    )
    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    teacher_rows: list[
        dict
    ] = []
    trajectory_rows: list[
        dict
    ] = []

    processed_days = 0

    total_decisions = 0
    for planned_year in years:
        planned_indices = [
            int(index)
            for index in market_indices
            if int(
                pd.Timestamp(
                    dates[index]
                ).year
            )
            == int(
                planned_year
            )
        ]
        if len(
            planned_indices
        ) <= lookahead:
            continue

        for planned_position in range(
            0,
            len(
                planned_indices
            )
            - lookahead,
        ):
            planned_signal_index = int(
                planned_indices[
                    planned_position
                ]
            )
            if (
                candidates_by_index[
                    planned_signal_index
                ].empty
            ):
                continue

            total_decisions += 1
            if (
                max_decision_days
                is not None
                and total_decisions
                >= max_decision_days
            ):
                break

        if (
            max_decision_days
            is not None
            and total_decisions
            >= max_decision_days
        ):
            break

    work_per_decision = int(
        particles
        * iterations
        * restarts
    )
    total_work = max(
        1,
        int(
            total_decisions
            * work_per_decision
        ),
    )
    progress = ProgressReporter(
        phase="portfolio_oracle_v4",
        total=total_work,
    )
    progress.update(
        0,
        message=(
            f"0/{total_decisions} "
            "decision states"
        ),
        detail=(
            f"{particles} particles · "
            f"{restarts} restarts · "
            f"{iterations} iterations"
        ),
        force=True,
    )

    for year in years:
        year_indices = [
            int(index)
            for index
            in market_indices
            if int(
                pd.Timestamp(
                    dates[index]
                ).year
            )
            == int(
                year
            )
        ]
        if (
            len(
                year_indices
            )
            <= lookahead
        ):
            continue

        state = {
            "cash": float(
                initial_capital
            ),
            "fees": 0.0,
            "holdings": {},
        }

        for position in range(
            0,
            len(
                year_indices
            )
            - lookahead,
        ):
            if (
                max_decision_days
                is not None
                and processed_days
                >= max_decision_days
            ):
                break

            window = year_indices[
                position : position
                + lookahead
                + 1
            ]
            signal_index = int(
                window[0]
            )
            execution_index = int(
                window[1]
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

            if (
                candidates_by_index[
                    signal_index
                ].empty
            ):
                continue

            progress_message = (
                f"state "
                f"{processed_days + 1}/"
                f"{total_decisions} · "
                f"{signal_date.date()}"
            )

            if engine == "torch-cuda":
                assert (
                    torch_cache
                    is not None
                    and cuda_search
                    is not None
                )

                exact_kwargs = {
                    "initial_state": state,
                    "market_indices": window,
                    "day_groups": day_groups,
                    "panel": panel,
                    "events_by_index": (
                        events_by_index
                    ),
                    "candidates_by_index": (
                        candidates_by_index
                    ),
                    "slots": slots,
                    "costs": costs,
                    "drawdown_penalty": (
                        drawdown_penalty
                    ),
                }

                best, samples = cuda_search(
                    initial_state=state,
                    market_indices=window,
                    cache=torch_cache,
                    slots=slots,
                    particles=particles,
                    iterations=iterations,
                    restarts=restarts,
                    samples_per_state=(
                        samples_per_state
                    ),
                    costs=costs,
                    drawdown_penalty=(
                        drawdown_penalty
                    ),
                    random_state=(
                        RANDOM_STATE
                        + signal_index
                    ),
                    progress=progress,
                    progress_offset=(
                        processed_days
                        * work_per_decision
                    ),
                    progress_message=(
                        progress_message
                    ),
                    gpu_finalists=(
                        gpu_finalists
                    ),
                    exact_simulate=(
                        simulate_plan
                    ),
                    exact_kwargs=(
                        exact_kwargs
                    ),
                    target_signature=(
                        target_signature
                    ),
                )
            else:
                best, samples = (
                    pso_search(
                        initial_state=state,
                        market_indices=window,
                        day_groups=day_groups,
                        panel=panel,
                        events_by_index=(
                            events_by_index
                        ),
                        candidates_by_index=(
                            candidates_by_index
                        ),
                        slots=slots,
                        particles=particles,
                        iterations=iterations,
                        restarts=restarts,
                        samples_per_state=(
                            samples_per_state
                        ),
                        costs=costs,
                        drawdown_penalty=(
                            drawdown_penalty
                        ),
                        random_state=(
                            RANDOM_STATE
                            + signal_index
                        ),
                        progress=progress,
                        progress_offset=(
                            processed_days
                            * work_per_decision
                        ),
                        progress_message=(
                            progress_message
                        ),
                    )
                )

            ranked_samples = sorted(
                samples,
                key=lambda row: float(
                    row[
                        "fitness"
                    ]
                ),
                reverse=True,
            )

            state_key = (
                f"{signal_date.date()}"
            )

            for rank, sample in enumerate(
                ranked_samples,
                start=1,
            ):
                target = sample[
                    "first_target"
                ]
                row = {
                    "state_key": (
                        state_key
                    ),
                    "state_date": (
                        signal_date
                    ),
                    "state_market_index": (
                        signal_index
                    ),
                    "year": int(
                        year
                    ),
                    "action_rank": int(
                        rank
                    ),
                    "is_oracle_action": bool(
                        rank
                        == 1
                    ),
                    "oracle_q": float(
                        sample[
                            "fitness"
                        ]
                    ),
                    "oracle_terminal_return": float(
                        sample[
                            "terminal_return"
                        ]
                    ),
                    "oracle_max_drawdown": float(
                        sample[
                            "max_drawdown"
                        ]
                    ),
                    "oracle_turnover": float(
                        sample[
                            "turnover"
                        ]
                    ),
                    # IDs are metadata only. Student code explicitly uses only
                    # f__* columns, so company identity cannot become a predictor.
                    "meta_current_assets": json.dumps(
                        sorted(
                            state[
                                "holdings"
                            ]
                        )
                    ),
                    "meta_target_assets": json.dumps(
                        sorted(
                            target
                        )
                    ),
                    "meta_target_signature": (
                        target_signature(
                            target
                        )
                    ),
                }
                row.update(
                    action_features(
                        state=state,
                        target=target,
                        day=signal_day,
                        slots=slots,
                    )
                )
                teacher_rows.append(
                    row
                )

            current_lookup = day_lookup(
                signal_day
            )
            current_equity = equity_at_price(
                state,
                current_lookup,
                "close",
            )

            trajectory_rows.append({
                "date": (
                    signal_date
                ),
                "market_index": (
                    signal_index
                ),
                "year": int(
                    year
                ),
                "equity_before_action": (
                    current_equity
                ),
                "oracle_q": float(
                    best[
                        "fitness"
                    ]
                ),
                "oracle_terminal_return": float(
                    best[
                        "terminal_return"
                    ]
                ),
                "oracle_max_drawdown": float(
                    best[
                        "max_drawdown"
                    ]
                ),
                "oracle_turnover": float(
                    best[
                        "turnover"
                    ]
                ),
                "target_signature": (
                    target_signature(
                        best[
                            "first_target"
                        ]
                    )
                ),
            })

            state = execute_first_action(
                state,
                best=best,
                signal_index=(
                    signal_index
                ),
                execution_index=(
                    execution_index
                ),
                panel=panel,
                day_groups=(
                    day_groups
                ),
                events_by_index=(
                    events_by_index
                ),
                costs=costs,
            )

            processed_days += 1

            progress.update(
                processed_days
                * work_per_decision,
                message=(
                    f"{processed_days}/"
                    f"{total_decisions} "
                    "decision states"
                ),
                detail=(
                    f"{len(teacher_rows):,} "
                    "teacher rows"
                ),
                force=True,
            )

        if (
            max_decision_days
            is not None
            and processed_days
            >= max_decision_days
        ):
            break

    progress.finish(
        message=(
            f"{processed_days}/"
            f"{total_decisions} "
            "decision states"
        ),
        detail=(
            f"{len(teacher_rows):,} "
            "teacher rows"
        ),
    )

    teacher = pd.DataFrame(
        teacher_rows
    )
    trajectory = pd.DataFrame(
        trajectory_rows
    )

    if teacher.empty:
        raise RuntimeError(
            "Oracle teacher is empty."
        )

    teacher_path = (
        output_root
        / "development_teacher.parquet"
    )
    trajectory_path = (
        report_root
        / "oracle_trajectory.parquet"
    )
    teacher.to_parquet(
        teacher_path,
        index=False,
        compression="zstd",
    )
    trajectory.to_parquet(
        trajectory_path,
        index=False,
        compression="zstd",
    )

    per_state = (
        teacher.groupby(
            "state_key",
            sort=False,
        )
        .agg(
            sampled_actions=(
                "oracle_q",
                "size",
            ),
            best_q=(
                "oracle_q",
                "max",
            ),
            worst_q=(
                "oracle_q",
                "min",
            ),
        )
        .reset_index()
    )
    per_state[
        "q_spread"
    ] = (
        per_state[
            "best_q"
        ]
        - per_state[
            "worst_q"
        ]
    )
    per_state.to_csv(
        report_root
        / "state_action_headroom.csv",
        index=False,
    )

    summary = {
        "algorithm": (
            "receding-horizon sparse whole-portfolio particle-swarm "
            "hindsight search"
        ),
        "status": (
            "approximate hindsight oracle; not claimed globally exact"
        ),
        "years": [
            int(year)
            for year in years
        ],
        "decision_interval": (
            "every market close; target portfolio executes next open"
        ),
        "portfolio_slots": int(
            slots
        ),
        "lookahead_sessions": int(
            lookahead
        ),
        "particles": int(
            particles
        ),
        "pso_iterations": int(
            iterations
        ),
        "restarts": int(
            restarts
        ),
        "samples_per_state": int(
            samples_per_state
        ),
        "engine": str(
            engine
        ),
        "device": str(
            device
        ),
        "gpu_finalists": int(
            gpu_finalists
        )
        if engine == "torch-cuda"
        else None,
        "cuda_fidelity": (
            "GPU float32 PSO proposal search; all retained finalist plans "
            "are rescored by the original CPU simulator before teacher Q "
            "values and oracle actions are recorded."
            if engine == "torch-cuda"
            else "reference CPU simulator"
        ),
        "drawdown_penalty": float(
            drawdown_penalty
        ),
        "initial_capital": float(
            initial_capital
        ),
        "cost_profile": asdict(
            costs
        ),
        "decision_states": int(
            teacher[
                "state_key"
            ].nunique()
        ),
        "teacher_rows": int(
            len(
                teacher
            )
        ),
        "feature_columns": int(
            sum(
                str(column).startswith(
                    "f__"
                )
                for column
                in teacher.columns
            )
        ),
        "identity_rule": (
            "security IDs/symbols are metadata only; student predictors "
            "must use columns prefixed f__"
        ),
        "teacher_path": str(
            teacher_path
        ),
        "trajectory_path": str(
            trajectory_path
        ),
    }
    (
        report_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=float,
        )
        + "\n"
    )

    print(
        "\n=== PORTFOLIO ORACLE V4 COMPLETE ==="
    )
    print(
        "Decision states: "
        f"{summary['decision_states']:,}"
    )
    print(
        "Teacher rows:    "
        f"{summary['teacher_rows']:,}"
    )
    print(
        "Features:        "
        f"{summary['feature_columns']:,}"
    )
    print(
        "Teacher:         "
        f"{teacher_path}"
    )

    return summary


def self_test() -> None:
    candidates = pd.DataFrame({
        "canonical_security_id": [
            "A",
            "B",
            "C",
        ],
        "persistent_rank": [
            1.0,
            2.0,
            3.0,
        ],
        "persistent_score_pct": [
            1.0,
            0.8,
            0.6,
        ],
        "turnover_median_20d": [
            3.0,
            2.0,
            1.0,
        ],
    })

    target = decode_target(
        np.asarray([
            0.0,
            0.34,
        ]),
        np.asarray([
            2.0,
            1.0,
            -2.0,
        ]),
        candidates,
        slots=2,
    )
    assert len(target) == 2
    assert set(target).issubset(
        {"A", "B", "C"}
    )
    assert sum(target.values()) < 1.0

    assert (
        plan_dimensions(
            horizon=20,
            slots=5,
        )
        == 220
    )

    # Identity must never be a predictive feature.
    bad = [
        column
        for column in (
            "meta_current_assets",
            "meta_target_assets",
            "canonical_security_id",
            "symbol",
        )
        if str(column).startswith(
            "f__"
        )
    ]
    assert not bad

    print(
        "Portfolio-oracle-v4 self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a daily whole-portfolio hindsight action-value teacher "
            "with sparse particle-swarm trajectory search."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=(
            DEVELOPMENT_YEARS
        ),
    )
    ap.add_argument(
        "--slots",
        type=int,
        default=DEFAULT_SLOTS,
    )
    ap.add_argument(
        "--lookahead",
        type=int,
        default=DEFAULT_LOOKAHEAD,
    )
    ap.add_argument(
        "--particles",
        type=int,
        default=DEFAULT_PARTICLES,
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
    )
    ap.add_argument(
        "--restarts",
        type=int,
        default=DEFAULT_RESTARTS,
    )
    ap.add_argument(
        "--samples-per-state",
        type=int,
        default=(
            DEFAULT_SAMPLES_PER_STATE
        ),
    )
    ap.add_argument(
        "--capital",
        type=float,
        default=50_000.0,
    )
    ap.add_argument(
        "--drawdown-penalty",
        type=float,
        default=0.0,
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
        "--engine",
        choices=[
            "cpu",
            "torch-cuda",
        ],
        default="cpu",
        help=(
            "PSO proposal engine. torch-cuda batches particles on a CUDA "
            "GPU and exactly rescoring finalists with the CPU reference "
            "simulator."
        ),
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help=(
            "Torch device used by --engine torch-cuda, for example cuda "
            "or cuda:0."
        ),
    )
    ap.add_argument(
        "--gpu-finalists",
        type=int,
        default=128,
        help=(
            "Top GPU particle-best plans per restart to rescore with the "
            "exact CPU simulator."
        ),
    )
    ap.add_argument(
        "--max-decision-days",
        type=int,
        default=None,
        help=(
            "Optional smoke-test cap. Omit for the full daily teacher."
        ),
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.slots <= 0:
        raise SystemExit(
            "--slots must be positive"
        )
    if args.lookahead <= 0:
        raise SystemExit(
            "--lookahead must be positive"
        )
    if args.particles < 4:
        raise SystemExit(
            "--particles must be >= 4"
        )
    if args.iterations <= 0:
        raise SystemExit(
            "--iterations must be positive"
        )
    if args.restarts <= 0:
        raise SystemExit(
            "--restarts must be positive"
        )
    if args.gpu_finalists <= 0:
        raise SystemExit(
            "--gpu-finalists must be positive"
        )

    root = Path(
        args.root
    ).resolve()

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

    build_oracle(
        root,
        years=list(
            args.years
        ),
        slots=int(
            args.slots
        ),
        lookahead=int(
            args.lookahead
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
        samples_per_state=int(
            args.samples_per_state
        ),
        initial_capital=float(
            args.capital
        ),
        costs=costs,
        drawdown_penalty=float(
            args.drawdown_penalty
        ),
        max_decision_days=(
            int(
                args.max_decision_days
            )
            if args.max_decision_days
            is not None
            else None
        ),
        engine=str(
            args.engine
        ),
        device=str(
            args.device
        ),
        gpu_finalists=int(
            args.gpu_finalists
        ),
    )


if __name__ == "__main__":
    main()
