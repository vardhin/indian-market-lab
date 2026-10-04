# Controller V1: finite-horizon optimal stopping

The frozen B4 XGBoost selector remains unchanged. Controller V1 addresses the
separate question: after a selected stock has been bought, should the portfolio
continue holding it or exit at the next market open?

## Decision clock

- Selector signal: close(t)
- Entry: open(t+1)
- Portfolio policy used to create episodes: top 5, 20-session rebalance
- Controller observation: each subsequent close while the position is active
- Controller action: HOLD or EXIT
- EXIT execution: next market open
- Maximum episode length: 20 sessions
- Terminal action: forced EXIT at the next open after the final controller state

The first controller is deliberately an optimal-stopping problem, not yet a
full allocation RL problem.

## Hindsight teacher

For a historical episode we know all future opens after the fact. For each
controller state, the teacher compares:

1. net return from selling at the immediately following open; and
2. the best net sale return available at a later open before the episode ends.

Transaction costs and slippage are included in each hypothetical sale. The
teacher action is HOLD only when a later permitted sale is better.

This is an exact finite-horizon oracle for the restricted HOLD/EXIT problem.
Dynamic programming is preferable to simulated annealing or PSO here because
the action space is tiny and the exact optimum is available.

The future path is used **only for labels**. Controller predictors contain only
information available by that state's close.

Episodes containing a split/bonus are excluded from Controller V1 rather than
comparing unadjusted prices through a quantity-changing event.

Episodes are also kept inside one calendar year, so a December training episode
cannot leak later states or continuation labels into the next validation year.

## Controller state

The controller receives the 86 frozen B4 features plus position/opportunity
state such as:

- age and remaining horizon
- entry score/rank and current score/rank
- rank and score deterioration since entry
- unrealized return
- running peak return and drawdown from the post-entry peak
- best currently available alternative B4 score
- score gap versus that alternative
- current top-5 cutoff and distance from it
- number of eligible large-cap candidates

All large-cap B4 scores are recomputed/attached in parallel for each close, so
the controller observes opportunity cost rather than judging a holding in
isolation.

## Development protocol

The selector's annual walk-forward scores remain frozen.

- Controller teacher: 2021-2023
- Controller model train years: 2021-2022
- Controller internal validation: 2023
- 2024-2026 stays out of controller model selection

The initial tournament compares HistGradientBoosting, Random Forest and
ExtraTrees regressors. They predict the oracle continuation advantage. The
implied action is:

- predicted advantage > 0: HOLD
- otherwise: EXIT

Primary development ranking is lowest one-step hindsight-action regret, then
lower advantage MAE, then higher Spearman correlation.

These diagnostics are **not** yet a sequential portfolio backtest: after a
predicted HOLD, the one-step diagnostic assumes the hindsight oracle acts
optimally later. The next stage must recursively run the learned controller
through the actual simulator.

## Why no PSO / simulated annealing yet?

They become useful once the action is expanded to:

- HOLD current stock
- EXIT to cash
- SWITCH to one of several alternative stocks
- resize positions
- allocate across several candidates

That state/action space becomes combinatorial and path-dependent. Controller V2
can use SA/PSO/CMA-ES to search parameterized policies or generate approximate
teacher trajectories, followed by imitation learning or RL.

Controller V1 first answers the more basic question: is optimal exit timing
valuable, and is the hindsight stopping boundary learnable from causal market
state?
