from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class TorchMarketCache:
    device: object
    dtype: object
    cid_to_col: dict[str, int]
    col_to_cid: list[str]
    market_row: dict[int, int]
    open_prices: object
    close_prices: object
    multipliers: object
    candidate_cols: dict[int, object]


def build_torch_market_cache(
    *,
    panel: pd.DataFrame,
    day_groups: dict,
    market_indices: list[int],
    candidates_by_index: dict[int, pd.DataFrame],
    events_by_index: dict,
    device: str,
) -> TorchMarketCache:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Torch CUDA engine requested but torch is not installed. "
            "Install the project with --extra deep."
        ) from exc

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError(
            "The torch oracle backend is intended for CUDA. "
            "Use --engine cpu for the reference implementation."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false."
        )

    universe = sorted(
        str(value)
        for value in panel[
            "canonical_security_id"
        ].dropna().unique()
    )
    cid_to_col = {
        cid: index
        for index, cid in enumerate(universe)
    }
    market_row = {
        int(market_index): row
        for row, market_index in enumerate(market_indices)
    }

    n_days = len(market_indices)
    n_assets = len(universe)

    opens = np.full(
        (n_days, n_assets),
        np.nan,
        dtype=np.float32,
    )
    closes = np.full(
        (n_days, n_assets),
        np.nan,
        dtype=np.float32,
    )
    multipliers = np.ones(
        (n_days, n_assets),
        dtype=np.float32,
    )

    candidate_cols: dict[int, object] = {}

    for market_index in market_indices:
        market_index = int(market_index)
        row = market_row[market_index]
        day = panel.loc[
            day_groups[market_index],
            [
                "canonical_security_id",
                "open",
                "close",
            ],
        ]

        cols = np.fromiter(
            (
                cid_to_col[str(cid)]
                for cid in day[
                    "canonical_security_id"
                ]
            ),
            dtype=np.int64,
            count=len(day),
        )
        opens[
            row,
            cols,
        ] = pd.to_numeric(
            day["open"],
            errors="coerce",
        ).to_numpy(
            dtype=np.float32,
        )
        closes[
            row,
            cols,
        ] = pd.to_numeric(
            day["close"],
            errors="coerce",
        ).to_numpy(
            dtype=np.float32,
        )

        for cid, multiplier in events_by_index.get(
            market_index,
            [],
        ):
            col = cid_to_col.get(
                str(cid)
            )
            if col is None:
                continue
            multipliers[
                row,
                col,
            ] *= float(multiplier)

        candidates = candidates_by_index[
            market_index
        ]
        candidate_np = np.asarray(
            [
                cid_to_col[
                    str(cid)
                ]
                for cid in candidates[
                    "canonical_security_id"
                ].tolist()
            ],
            dtype=np.int64,
        )
        candidate_cols[
            market_index
        ] = torch.as_tensor(
            candidate_np,
            dtype=torch.long,
            device=target_device,
        )

    print(
        "Building CUDA market cache: "
        f"{n_days:,} sessions × "
        f"{n_assets:,} securities",
        flush=True,
    )

    return TorchMarketCache(
        device=target_device,
        dtype=torch.float32,
        cid_to_col=cid_to_col,
        col_to_cid=universe,
        market_row=market_row,
        open_prices=torch.as_tensor(
            opens,
            dtype=torch.float32,
            device=target_device,
        ),
        close_prices=torch.as_tensor(
            closes,
            dtype=torch.float32,
            device=target_device,
        ),
        multipliers=torch.as_tensor(
            multipliers,
            dtype=torch.float32,
            device=target_device,
        ),
        candidate_cols=candidate_cols,
    )


def _decode_targets(
    step_vector,
    *,
    candidates,
    slots: int,
):
    import torch

    particles = int(
        step_vector.shape[0]
    )
    n_candidates = int(
        candidates.numel()
    )
    active_slots = min(
        int(slots),
        n_candidates,
    )

    target_ids = torch.full(
        (
            particles,
            slots,
        ),
        -1,
        dtype=torch.long,
        device=step_vector.device,
    )
    target_weights = torch.zeros(
        (
            particles,
            slots,
        ),
        dtype=step_vector.dtype,
        device=step_vector.device,
    )

    if active_slots <= 0:
        return (
            target_ids,
            target_weights,
        )

    selectors = step_vector[
        :,
        :active_slots,
    ]
    starts = torch.floor(
        selectors
        * float(n_candidates)
    ).to(
        torch.long
    )
    starts.clamp_(
        0,
        n_candidates - 1,
    )

    chosen: list[object] = []

    for slot in range(
        active_slots
    ):
        positions = starts[
            :,
            slot,
        ].clone()
        candidate = candidates[
            positions
        ]

        if chosen:
            # Resolve duplicate selectors without any host-device
            # synchronization. With <slot> previously selected assets,
            # at most <slot> cyclic increments are required to find an
            # unused candidate because candidates are unique.
            for _ in range(
                slot
            ):
                duplicate = torch.zeros(
                    particles,
                    dtype=torch.bool,
                    device=step_vector.device,
                )
                for previous in chosen:
                    duplicate |= (
                        candidate
                        == previous
                    )

                positions = torch.where(
                    duplicate,
                    (
                        positions
                        + 1
                    )
                    % n_candidates,
                    positions,
                )
                candidate = candidates[
                    positions
                ]

        chosen.append(
            candidate
        )
        target_ids[
            :,
            slot,
        ] = candidate

    logits = step_vector[
        :,
        slots:
        slots
        + active_slots
        + 1,
    ]
    weights = torch.softmax(
        logits,
        dim=1,
    )
    target_weights[
        :,
        :active_slots,
    ] = weights[
        :,
        :active_slots,
    ]

    active = (
        target_weights
        > 1e-6
    )
    target_ids = torch.where(
        active,
        target_ids,
        torch.full_like(
            target_ids,
            -1,
        ),
    )
    target_weights = torch.where(
        active,
        target_weights,
        torch.zeros_like(
            target_weights
        ),
    )

    return (
        target_ids,
        target_weights,
    )


def _gather_row(
    row,
    ids,
    *,
    missing_value: float = 0.0,
):
    import torch

    valid_ids = ids.ge(0)
    safe = ids.clamp_min(
        0
    )
    values = row[
        safe
    ]
    return torch.where(
        valid_ids,
        values,
        torch.full_like(
            values,
            float(missing_value),
        ),
    )


def _initial_holdings(
    *,
    state: dict,
    cache: TorchMarketCache,
    particles: int,
    slots: int,
):
    import torch

    items = [
        (
            str(cid),
            float(quantity),
        )
        for cid, quantity
        in state[
            "holdings"
        ].items()
        if float(quantity) > 0
    ]

    if len(items) > slots:
        raise RuntimeError(
            "Initial portfolio has more holdings than --slots."
        )

    ids = torch.full(
        (
            particles,
            slots,
        ),
        -1,
        dtype=torch.long,
        device=cache.device,
    )
    quantities = torch.zeros(
        (
            particles,
            slots,
        ),
        dtype=cache.dtype,
        device=cache.device,
    )

    for slot, (
        cid,
        quantity,
    ) in enumerate(
        items
    ):
        if cid not in cache.cid_to_col:
            raise RuntimeError(
                f"Portfolio security {cid} is absent from CUDA market cache."
            )
        ids[
            :,
            slot,
        ] = int(
            cache.cid_to_col[
                cid
            ]
        )
        quantities[
            :,
            slot,
        ] = float(
            quantity
        )

    cash = torch.full(
        (
            particles,
        ),
        float(
            state[
                "cash"
            ]
        ),
        dtype=cache.dtype,
        device=cache.device,
    )

    return (
        ids,
        quantities,
        cash,
    )


def evaluate_plans_cuda(
    plans,
    *,
    initial_state: dict,
    market_indices: list[int],
    cache: TorchMarketCache,
    slots: int,
    costs,
    drawdown_penalty: float,
):
    import torch

    particles = int(
        plans.shape[0]
    )
    horizon = len(
        market_indices
    ) - 1

    holding_ids, holding_qty, cash = (
        _initial_holdings(
            state=initial_state,
            cache=cache,
            particles=particles,
            slots=slots,
        )
    )

    signal_row = cache.market_row[
        int(
            market_indices[0]
        )
    ]
    start_close = _gather_row(
        cache.close_prices[
            signal_row
        ],
        holding_ids,
    )
    held = holding_qty.gt(
        0
    )
    valid = (
        (~held)
        | (
            torch.isfinite(
                start_close
            )
            & start_close.gt(
                0
            )
        )
    ).all(
        dim=1
    )
    start_equity = (
        cash
        + (
            holding_qty
            * torch.nan_to_num(
                start_close,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        ).sum(
            dim=1
        )
    )
    valid &= (
        torch.isfinite(
            start_equity
        )
        & start_equity.gt(
            0
        )
    )

    running_peak = torch.clamp(
        start_equity,
        min=1e-12,
    )
    max_drawdown = torch.zeros_like(
        start_equity
    )
    total_turnover = torch.zeros_like(
        start_equity
    )

    first_ids = None
    first_weights = None

    buy_slip = (
        1.0
        + float(
            costs.slippage_bps
        )
        / 10_000.0
    )
    sell_slip = (
        1.0
        - float(
            costs.slippage_bps
        )
        / 10_000.0
    )

    buy_variable_rate = (
        float(
            costs.stt_buy_rate
        )
        + float(
            costs.stamp_buy_rate
        )
        + float(
            costs.exchange_rate
        )
        + float(
            costs.sebi_rate
        )
        + float(
            costs.gst_rate
        )
        * (
            float(
                costs.exchange_rate
            )
            + float(
                costs.sebi_rate
            )
        )
    )
    sell_variable_rate = (
        float(
            costs.stt_sell_rate
        )
        + float(
            costs.exchange_rate
        )
        + float(
            costs.sebi_rate
        )
        + float(
            costs.gst_rate
        )
        * (
            float(
                costs.exchange_rate
            )
            + float(
                costs.sebi_rate
            )
        )
    )
    buy_variable_multiplier = (
        1.0
        + buy_variable_rate
    )
    fixed_buy = (
        float(
            costs.brokerage_per_order
        )
        * (
            1.0
            + float(
                costs.gst_rate
            )
        )
    )
    fixed_sell = (
        float(
            costs.brokerage_per_order
        )
        * (
            1.0
            + float(
                costs.gst_rate
            )
        )
        + float(
            costs.dp_charge_per_sell
        )
    )

    for step in range(
        horizon
    ):
        signal_index = int(
            market_indices[
                step
            ]
        )
        execution_index = int(
            market_indices[
                step + 1
            ]
        )
        execution_row = (
            cache.market_row[
                execution_index
            ]
        )

        target_ids, target_weights = (
            _decode_targets(
                plans[
                    :,
                    step,
                    :,
                ],
                candidates=(
                    cache.candidate_cols[
                        signal_index
                    ]
                ),
                slots=slots,
            )
        )

        if step == 0:
            first_ids = (
                target_ids.clone()
            )
            first_weights = (
                target_weights.clone()
            )

        multiplier = _gather_row(
            cache.multipliers[
                execution_row
            ],
            holding_ids,
            missing_value=1.0,
        )
        multiplier = torch.where(
            holding_ids.ge(
                0
            ),
            multiplier,
            torch.ones_like(
                multiplier
            ),
        )
        holding_qty = torch.floor(
            holding_qty
            * multiplier
            + 1e-6
        )

        execution_open = _gather_row(
            cache.open_prices[
                execution_row
            ],
            holding_ids,
        )
        held = holding_qty.gt(
            0
        )
        valid &= (
            (~held)
            | (
                torch.isfinite(
                    execution_open
                )
                & execution_open.gt(
                    0
                )
            )
        ).all(
            dim=1
        )

        safe_execution_open = (
            torch.nan_to_num(
                execution_open,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        pretrade_equity = (
            cash
            + (
                holding_qty
                * safe_execution_open
            ).sum(
                dim=1
            )
        )
        valid &= (
            torch.isfinite(
                pretrade_equity
            )
            & pretrade_equity.gt(
                0
            )
        )

        target_open = _gather_row(
            cache.open_prices[
                execution_row
            ],
            target_ids,
        )
        target_active = (
            target_ids.ge(
                0
            )
            & target_weights.gt(
                0
            )
        )
        valid &= (
            (~target_active)
            | (
                torch.isfinite(
                    target_open
                )
                & target_open.gt(
                    0
                )
            )
        ).all(
            dim=1
        )
        safe_target_open = (
            torch.nan_to_num(
                target_open,
                nan=1.0,
                posinf=1.0,
                neginf=1.0,
            )
        )

        desired_target_qty = torch.floor(
            pretrade_equity[
                :,
                None,
            ]
            * target_weights
            / torch.clamp(
                safe_target_open
                * buy_slip,
                min=1e-12,
            )
        )
        desired_target_qty = torch.where(
            target_active,
            desired_target_qty,
            torch.zeros_like(
                desired_target_qty
            ),
        )

        matches = (
            holding_ids[
                :,
                :,
                None,
            ]
            == target_ids[
                :,
                None,
                :,
            ]
        )
        matches &= (
            holding_ids[
                :,
                :,
                None,
            ].ge(
                0
            )
            & target_ids[
                :,
                None,
                :,
            ].ge(
                0
            )
        )

        desired_for_current = (
            matches.to(
                holding_qty.dtype
            )
            * desired_target_qty[
                :,
                None,
                :,
            ]
        ).sum(
            dim=2
        )
        sell_qty = torch.clamp(
            holding_qty
            - desired_for_current,
            min=0.0,
        )
        sell_mask = sell_qty.gt(
            0
        )

        sell_execution_price = (
            safe_execution_open
            * sell_slip
        )
        sell_trade_value = (
            sell_execution_price
            * sell_qty
        )
        sell_fees = torch.where(
            sell_mask,
            (
                sell_trade_value
                * sell_variable_rate
                + fixed_sell
            ),
            torch.zeros_like(
                sell_trade_value
            ),
        )
        cash = (
            cash
            + (
                sell_trade_value
                - sell_fees
            ).sum(
                dim=1
            )
        )
        total_turnover += (
            sell_trade_value.sum(
                dim=1
            )
        )
        holding_qty = (
            holding_qty
            - sell_qty
        )

        current_for_target = (
            matches.to(
                holding_qty.dtype
            ).transpose(
                1,
                2,
            )
            @ holding_qty[
                :,
                :,
                None,
            ]
        ).squeeze(
            2
        )

        # CPU reference buys by descending target weight, then security ID.
        # Stable sort by ID first and by weight second preserves the exact
        # tie-break order without perturbing the weights.
        id_order = torch.argsort(
            target_ids,
            dim=1,
            stable=True,
        )
        sorted_ids = torch.gather(
            target_ids,
            1,
            id_order,
        )
        sorted_weights = torch.gather(
            target_weights,
            1,
            id_order,
        )
        sorted_open = torch.gather(
            safe_target_open,
            1,
            id_order,
        )
        sorted_desired = torch.gather(
            desired_target_qty,
            1,
            id_order,
        )
        sorted_current = torch.gather(
            current_for_target,
            1,
            id_order,
        )
        sorted_active = torch.gather(
            target_active,
            1,
            id_order,
        )

        weight_order = torch.argsort(
            sorted_weights,
            dim=1,
            descending=True,
            stable=True,
        )
        sorted_ids = torch.gather(
            sorted_ids,
            1,
            weight_order,
        )
        sorted_weights = torch.gather(
            sorted_weights,
            1,
            weight_order,
        )
        sorted_open = torch.gather(
            sorted_open,
            1,
            weight_order,
        )
        sorted_desired = torch.gather(
            sorted_desired,
            1,
            weight_order,
        )
        sorted_current = torch.gather(
            sorted_current,
            1,
            weight_order,
        )
        sorted_active = torch.gather(
            sorted_active,
            1,
            weight_order,
        )

        final_qty = sorted_current.clone()

        for order_slot in range(
            slots
        ):
            active_order = (
                sorted_active[
                    :,
                    order_slot,
                ]
            )
            need = torch.clamp(
                sorted_desired[
                    :,
                    order_slot,
                ]
                - final_qty[
                    :,
                    order_slot,
                ],
                min=0.0,
            )
            active_buy = (
                active_order
                & need.gt(
                    0
                )
            )

            quoted_open = sorted_open[
                :,
                order_slot,
            ]
            execution_price = (
                quoted_open
                * buy_slip
            )

            available_after_fixed = (
                cash
                - fixed_buy
            )
            affordable = torch.floor(
                torch.clamp(
                    available_after_fixed,
                    min=0.0,
                )
                / torch.clamp(
                    execution_price
                    * buy_variable_multiplier,
                    min=1e-12,
                )
            )
            affordable = torch.where(
                cash.gt(
                    fixed_buy
                ),
                affordable,
                torch.zeros_like(
                    affordable
                ),
            )
            quantity = torch.minimum(
                need,
                affordable,
            )
            quantity = torch.where(
                active_buy,
                quantity,
                torch.zeros_like(
                    quantity
                ),
            )

            for _ in range(
                2
            ):
                trade_value = (
                    execution_price
                    * quantity
                )
                fees = torch.where(
                    quantity.gt(
                        0
                    ),
                    (
                        trade_value
                        * buy_variable_rate
                        + fixed_buy
                    ),
                    torch.zeros_like(
                        trade_value
                    ),
                )
                cash_out = (
                    trade_value
                    + fees
                )
                too_large = (
                    quantity.gt(
                        0
                    )
                    & cash_out.gt(
                        cash
                        + 1e-4
                    )
                )
                quantity = torch.where(
                    too_large,
                    torch.clamp(
                        quantity
                        - 1.0,
                        min=0.0,
                    ),
                    quantity,
                )

            trade_value = (
                execution_price
                * quantity
            )
            fees = torch.where(
                quantity.gt(
                    0
                ),
                (
                    trade_value
                    * buy_variable_rate
                    + fixed_buy
                ),
                torch.zeros_like(
                    trade_value
                ),
            )
            cash -= (
                trade_value
                + fees
            )
            total_turnover += (
                trade_value
            )
            final_qty[
                :,
                order_slot,
            ] += quantity

        tiny_negative = (
            cash.ge(
                -1e-3
            )
        )
        valid &= tiny_negative
        cash = torch.clamp(
            cash,
            min=0.0,
        )

        holding_ids = sorted_ids
        holding_qty = final_qty

        execution_close = _gather_row(
            cache.close_prices[
                execution_row
            ],
            holding_ids,
        )
        held = holding_qty.gt(
            0
        )
        valid &= (
            (~held)
            | (
                torch.isfinite(
                    execution_close
                )
                & execution_close.gt(
                    0
                )
            )
        ).all(
            dim=1
        )
        safe_close = torch.nan_to_num(
            execution_close,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        equity = (
            cash
            + (
                holding_qty
                * safe_close
            ).sum(
                dim=1
            )
        )
        valid &= (
            torch.isfinite(
                equity
            )
            & equity.gt(
                0
            )
        )

        safe_equity = torch.where(
            valid,
            equity,
            torch.clamp(
                running_peak,
                min=1e-12,
            ),
        )
        running_peak = torch.maximum(
            running_peak,
            safe_equity,
        )
        drawdown = (
            safe_equity
            / torch.clamp(
                running_peak,
                min=1e-12,
            )
            - 1.0
        )
        max_drawdown = torch.minimum(
            max_drawdown,
            drawdown,
        )

    terminal_equity = (
        cash
        + (
            holding_qty
            * torch.nan_to_num(
                _gather_row(
                    cache.close_prices[
                        cache.market_row[
                            int(
                                market_indices[-1]
                            )
                        ]
                    ],
                    holding_ids,
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        ).sum(
            dim=1
        )
    )
    terminal_ratio = (
        terminal_equity
        / torch.clamp(
            start_equity,
            min=1e-12,
        )
    )
    terminal_return = (
        terminal_ratio
        - 1.0
    )
    fitness = (
        torch.log(
            torch.clamp(
                terminal_ratio,
                min=1e-12,
            )
        )
        - float(
            drawdown_penalty
        )
        * torch.abs(
            max_drawdown
        )
    )
    fitness = torch.where(
        valid,
        fitness,
        torch.full_like(
            fitness,
            -1e9,
        ),
    )

    return {
        "fitness": fitness,
        "terminal_return": terminal_return,
        "max_drawdown": max_drawdown,
        "turnover": total_turnover,
        "first_ids": first_ids,
        "first_weights": first_weights,
    }


def pso_search_cuda(
    *,
    initial_state: dict,
    market_indices: list[int],
    cache: TorchMarketCache,
    slots: int,
    particles: int,
    iterations: int,
    restarts: int,
    samples_per_state: int,
    costs,
    drawdown_penalty: float,
    random_state: int,
    progress=None,
    progress_offset: int = 0,
    progress_message: str = "",
    gpu_finalists: int = 128,
    finalist_strategy: str = "top",
    exact_simulate: Callable,
    exact_kwargs: dict,
    target_signature: Callable,
) -> tuple[dict, list[dict]]:
    import torch

    if particles < 4:
        raise ValueError(
            "particles must be >= 4"
        )

    horizon = (
        len(
            market_indices
        )
        - 1
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

    with torch.inference_mode():
        for restart in range(
            restarts
        ):
            generator = torch.Generator(
                device=cache.device
            )
            generator.manual_seed(
                int(
                    random_state
                    + restart
                    * 100_003
                )
            )

            positions = torch.empty(
                (
                    particles,
                    horizon,
                    width,
                ),
                dtype=cache.dtype,
                device=cache.device,
            )
            positions[
                :,
                :,
                :slots,
            ] = torch.rand(
                (
                    particles,
                    horizon,
                    slots,
                ),
                dtype=cache.dtype,
                device=cache.device,
                generator=generator,
            )
            positions[
                :,
                :,
                slots:,
            ] = torch.randn(
                (
                    particles,
                    horizon,
                    slots + 1,
                ),
                dtype=cache.dtype,
                device=cache.device,
                generator=generator,
            )
            velocities = (
                torch.randn(
                    positions.shape,
                    dtype=cache.dtype,
                    device=cache.device,
                    generator=generator,
                )
                * 0.12
            )

            pbest = positions.clone()
            pbest_values = torch.full(
                (
                    particles,
                ),
                -1e18,
                dtype=cache.dtype,
                device=cache.device,
            )
            gbest = positions[
                0
            ].clone()
            gbest_value = torch.full(
                (),
                -1e18,
                dtype=cache.dtype,
                device=cache.device,
            )

            for iteration in range(
                iterations
            ):
                detail = (
                    evaluate_plans_cuda(
                        positions,
                        initial_state=(
                            initial_state
                        ),
                        market_indices=(
                            market_indices
                        ),
                        cache=cache,
                        slots=slots,
                        costs=costs,
                        drawdown_penalty=(
                            drawdown_penalty
                        ),
                    )
                )
                values = detail[
                    "fitness"
                ]

                improved = (
                    values
                    > pbest_values
                )
                pbest_values = torch.where(
                    improved,
                    values,
                    pbest_values,
                )
                pbest[
                    improved
                ] = positions[
                    improved
                ]

                iteration_best_value, (
                    iteration_best_index
                ) = torch.max(
                    values,
                    dim=0,
                )

                # Keep global-best selection entirely on-device. The old
                # implementation called .item() here every iteration,
                # introducing an unnecessary CUDA synchronization.
                better = (
                    iteration_best_value
                    > gbest_value
                )
                candidate_gbest = positions[
                    iteration_best_index
                ]
                gbest = torch.where(
                    better,
                    candidate_gbest,
                    gbest,
                )
                gbest_value = torch.maximum(
                    gbest_value,
                    iteration_best_value,
                )

                r1 = torch.rand(
                    positions.shape,
                    dtype=cache.dtype,
                    device=cache.device,
                    generator=generator,
                )
                r2 = torch.rand(
                    positions.shape,
                    dtype=cache.dtype,
                    device=cache.device,
                    generator=generator,
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
                            :,
                            :,
                        ]
                        - positions
                    )
                )
                positions += velocities
                positions[
                    :,
                    :,
                    :slots,
                ].clamp_(
                    0.0,
                    1.0 - 1e-7,
                )
                positions[
                    :,
                    :,
                    slots:,
                ].clamp_(
                    -6.0,
                    6.0,
                )

                if progress is not None:
                    is_checkpoint = (
                        iteration
                        == iterations - 1
                        or (
                            iteration
                            + 1
                        )
                        % 10
                        == 0
                    )

                    # Reading a CUDA scalar for display synchronizes the
                    # device. Only do that at sparse progress checkpoints.
                    if is_checkpoint:
                        best_text = (
                            f"{float(gbest_value.item()):+.6f}"
                        )
                    else:
                        best_text = "running"

                    progress.update(
                        progress_offset
                        + (
                            restart
                            * iterations
                            + iteration
                            + 1
                        )
                        * particles,
                        message=(
                            progress_message
                        ),
                        detail=(
                            f"CUDA restart "
                            f"{restart + 1}/{restarts} · "
                            f"iteration "
                            f"{iteration + 1}/{iterations} · "
                            f"best "
                            f"{best_text}"
                        ),
                        force=(
                            restart
                            == restarts - 1
                            and iteration
                            == iterations - 1
                        ),
                    )

            # GPU float32 is the proposal engine. Recorded teacher Q values
            # are recomputed with the original CPU reference simulator so
            # costs, integer quantities and execution semantics stay exact.
            finalist_count = min(
                particles,
                max(
                    samples_per_state,
                    int(
                        gpu_finalists
                    ),
                ),
            )

            if finalist_strategy == "top":
                finalist_indices = torch.topk(
                    pbest_values,
                    k=finalist_count,
                    largest=True,
                    sorted=True,
                ).indices
            elif finalist_strategy == "stratified":
                # V5 teacher mode: exact-rescore particle bests across the
                # swarm's quality spectrum instead of only the elite tail.
                # This preserves the PSO search/trajectory optimum while
                # creating a much broader action-value supervision pool.
                sorted_indices = torch.argsort(
                    pbest_values,
                    descending=True,
                    stable=True,
                )
                quartile_edges = torch.linspace(
                    0,
                    particles,
                    steps=5,
                    device=cache.device,
                ).round().to(
                    dtype=torch.long
                )

                base = finalist_count // 4
                remainder = finalist_count % 4
                selected_parts = []

                for quartile in range(4):
                    count = base + (
                        1
                        if quartile < remainder
                        else 0
                    )
                    if count <= 0:
                        continue

                    start = int(
                        quartile_edges[
                            quartile
                        ].item()
                    )
                    stop = int(
                        quartile_edges[
                            quartile + 1
                        ].item()
                    )
                    width_q = max(
                        1,
                        stop - start,
                    )
                    count = min(
                        count,
                        width_q,
                    )

                    offsets = torch.linspace(
                        0,
                        width_q - 1,
                        steps=count,
                        device=cache.device,
                    ).round().to(
                        dtype=torch.long
                    )
                    selected_parts.append(
                        sorted_indices[
                            start + offsets
                        ]
                    )

                finalist_indices = torch.cat(
                    selected_parts,
                    dim=0,
                )
            else:
                raise ValueError(
                    "Unknown finalist_strategy: "
                    f"{finalist_strategy}. "
                    "Use top or stratified."
                )

            finalist_plans = (
                pbest[
                    finalist_indices
                ]
                .detach()
                .cpu()
                .numpy()
            )

            for plan_tensor in finalist_plans:
                plan = np.asarray(
                    plan_tensor,
                    dtype=float,
                ).reshape(
                    -1
                )
                exact = exact_simulate(
                    plan,
                    **exact_kwargs,
                )
                if float(
                    exact[
                        "fitness"
                    ]
                ) <= -1e8:
                    continue

                signature = target_signature(
                    exact[
                        "first_target"
                    ]
                )
                previous = (
                    all_candidates.get(
                        signature
                    )
                )
                candidate = {
                    **exact,
                    "plan": plan.copy(),
                }
                if (
                    previous is None
                    or float(
                        exact[
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
                    ] = candidate

                if (
                    global_best is None
                    or float(
                        exact[
                            "fitness"
                        ]
                    )
                    > float(
                        global_best[
                            "fitness"
                        ]
                    )
                ):
                    global_best = candidate

    if global_best is None:
        raise RuntimeError(
            "CUDA PSO failed to produce a valid exactly-rescored trajectory."
        )

    ranked = sorted(
        all_candidates.values(),
        key=lambda row: float(
            row[
                "fitness"
            ]
        ),
        reverse=True,
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
