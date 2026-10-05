# Portfolio Oracle V4: whole-portfolio hindsight action values

V4 replaces the old single-position HOLD/EXIT teacher with a portfolio-level
decision problem.

## Decision problem

At every market close, the system may choose a new sparse target portfolio of up
to K securities plus cash. The target executes at the next market open.

Therefore a single target transition can contain all of the familiar actions:

- HOLD A: A remains in the target at roughly the same weight.
- SELL A -> CASH: A disappears and cash weight rises.
- BUY B: B appears in the target.
- SWITCH A -> B: A disappears while B appears.
- MULTI-SWITCH: several holdings can be replaced in the same decision.

There is no 20-session rebalance clock. A decision is available after every
market close.

## Hindsight search oracle

The historical oracle knows the realized future path during teacher generation.
It uses particle-swarm optimization (PSO) to search a sequence of whole target
portfolios over a configurable lookahead.

A particle contains, for each future decision:

1. K continuous security selectors. Each selector can address any currently
   eligible security; there is no top-N candidate truncation.
2. K+1 allocation logits. Softmax converts these into K asset weights plus a
   cash weight.

The fitness function simulates next-open execution, current Indian delivery
costs, slippage, integer shares, subsequent daily switches, corporate-action
quantity changes, and close-to-close portfolio wealth. By default the objective
is terminal log wealth; an optional drawdown penalty can be added.

Only the first target portfolio is executed. At the next close the oracle
re-optimizes with hindsight. This is receding-horizon control at a daily
decision interval.

PSO is a stochastic optimizer, so V4 deliberately calls this an
approximate hindsight oracle rather than pretending global optimality has been
proven. Multiple deterministic-seed restarts are used. Search-strength and
restart-sensitivity audits should be reported before publication.

## Q-like teacher matrix

Every PSO particle represents an action and a continuation trajectory.
Particles that share the same first target action are collapsed by keeping the
best continuation value found for that action.

The resulting training row is conceptually:

    (causal state at t, proposed whole-portfolio action at t)
        -> hindsight continuation value Q_oracle

The dataset keeps many good and bad action samples for each state, not only the
winning action. This gives the student both a ranking target and a continuous
value target.

## Identity anonymization

Security IDs are required internally to execute historical trades, but they are
metadata only.

Student predictors are only columns prefixed with f__.

The student sees source/target asset characteristics, B4 features, ranks,
scores, weights, cash, exposure, and portfolio transition statistics. It never
receives ticker, symbol, company name, canonical security ID, or an identity
embedding.

This lets the same learned rule apply to arbitrary A -> B transitions.

## Student

portfolio_student_v4.py compares:

- Random Forest
- ExtraTrees
- HistGradientBoosting

The target is oracle_q.

Model selection is not based only on regression error. For every 2023 state,
the student ranks the candidate actions produced by the oracle search. The
primary development metric is mean oracle regret of the student's selected
action.

2021-2022 are training years. 2023 is development validation.

2024-2026 must remain untouched until the oracle configuration, student model,
student action-search procedure, costs, features, and selection rule are frozen.

## Research interpretation

The old architecture was:

    B4 chooses WHAT to buy
    controller chooses WHEN to sell

V4 changes this to:

    causal market + portfolio state
        -> generate feasible whole-portfolio actions
        -> estimate Q(state, action)
        -> choose the highest-value action

Thus what-to-buy, when-to-buy, what-to-sell, when-to-sell, switching, and cash
become one portfolio decision problem.

The B4 model still contributes causal asset-quality features, but it is no
longer the final portfolio decision maker.
