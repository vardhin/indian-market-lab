from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "reports" / "ui_runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
PANEL_ROOT = ROOT / "data" / "processed" / "largecap_model_panel_v2"
PRICE_ROOT = ROOT / "data" / "processed" / "equities_adjusted"

BACKTEST_DIR = ROOT / "src" / "backtest"
ML_DIR = ROOT / "src" / "ml"
for path in (BACKTEST_DIR, ML_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines import CostProfile, buy_execution, sell_execution  # noqa: E402

app = FastAPI(title="Indian Market Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "portfolio_oracle_v4": {
        "label": "Portfolio Oracle V4",
        "description": "Daily whole-portfolio PSO hindsight teacher.",
        "script": "src/ml/portfolio_oracle_v4.py",
        "group": "V4",
        "params": [
            {"name": "years", "flag": "--years", "type": "int_list", "default": [2021, 2022, 2023], "label": "Years"},
            {"name": "slots", "flag": "--slots", "type": "int", "default": 5, "min": 1, "max": 20, "label": "Max holdings"},
            {"name": "lookahead", "flag": "--lookahead", "type": "int", "default": 20, "min": 1, "max": 120, "label": "Oracle lookahead"},
            {"name": "particles", "flag": "--particles", "type": "int", "default": 96, "min": 4, "max": 2048, "label": "PSO particles"},
            {"name": "iterations", "flag": "--iterations", "type": "int", "default": 80, "min": 1, "max": 2000, "label": "PSO iterations"},
            {"name": "restarts", "flag": "--restarts", "type": "int", "default": 3, "min": 1, "max": 50, "label": "PSO restarts"},
            {"name": "samples_per_state", "flag": "--samples-per-state", "type": "int", "default": 32, "min": 1, "max": 1024, "label": "Actions kept / state"},
            {"name": "capital", "flag": "--capital", "type": "float", "default": 50000.0, "min": 1000.0, "label": "Initial capital"},
            {"name": "drawdown_penalty", "flag": "--drawdown-penalty", "type": "float", "default": 0.0, "min": 0.0, "label": "Drawdown penalty"},
            {"name": "brokerage_per_order", "flag": "--brokerage-per-order", "type": "float", "default": 15.0, "min": 0.0, "label": "Brokerage / order"},
            {"name": "dp_charge_per_sell", "flag": "--dp-charge-per-sell", "type": "float", "default": 0.0, "min": 0.0, "label": "DP charge / sell"},
            {"name": "slippage_bps", "flag": "--slippage-bps", "type": "float", "default": 5.0, "min": 0.0, "label": "Slippage (bps)"},
            {"name": "max_decision_days", "flag": "--max-decision-days", "type": "optional_int", "default": None, "min": 1, "label": "Decision cap (smoke test)"},
        ],
    },
    "portfolio_student_v4": {
        "label": "Portfolio Student V4",
        "description": "Train tree ensembles on anonymized oracle action values.",
        "script": "src/ml/portfolio_student_v4.py",
        "group": "V4",
        "params": [
            {"name": "models", "flag": "--models", "type": "str_list", "default": ["random_forest", "extra_trees", "histgb"], "choices": ["random_forest", "extra_trees", "histgb"], "label": "Models"},
        ],
    },
    "portfolio_policy_v4": {
        "label": "Portfolio Policy V4",
        "description": "Causal student policy using PSO over predicted Q.",
        "script": "src/ml/portfolio_policy_v4.py",
        "group": "V4",
        "params": [
            {"name": "year", "flag": "--year", "type": "int", "default": 2023, "min": 2010, "max": 2030, "label": "Year"},
            {"name": "slots", "flag": "--slots", "type": "int", "default": 5, "min": 1, "max": 20, "label": "Max holdings"},
            {"name": "particles", "flag": "--particles", "type": "int", "default": 128, "min": 4, "max": 2048, "label": "Search particles"},
            {"name": "iterations", "flag": "--iterations", "type": "int", "default": 60, "min": 1, "max": 2000, "label": "Search iterations"},
            {"name": "restarts", "flag": "--restarts", "type": "int", "default": 3, "min": 1, "max": 50, "label": "Search restarts"},
            {"name": "capital", "flag": "--capital", "type": "float", "default": 50000.0, "min": 1000.0, "label": "Initial capital"},
            {"name": "risk_free_rate", "flag": "--risk-free-rate", "type": "float", "default": 0.065, "min": 0.0, "max": 1.0, "label": "Risk-free rate"},
            {"name": "brokerage_per_order", "flag": "--brokerage-per-order", "type": "float", "default": 15.0, "min": 0.0, "label": "Brokerage / order"},
            {"name": "dp_charge_per_sell", "flag": "--dp-charge-per-sell", "type": "float", "default": 0.0, "min": 0.0, "label": "DP charge / sell"},
            {"name": "slippage_bps", "flag": "--slippage-bps", "type": "float", "default": 5.0, "min": 0.0, "label": "Slippage (bps)"},
        ],
    },
    "controller_fqi_v2": {
        "label": "Controller FQI V2",
        "description": "Legacy finite-horizon HOLD/EXIT fitted-Q experiment.",
        "script": "src/ml/controller_fqi_v2.py",
        "group": "Legacy",
        "params": [
            {"name": "models", "flag": "--models", "type": "str_list", "default": ["random_forest"], "choices": ["histgb", "random_forest", "extra_trees"], "label": "Models"},
            {"name": "iterations", "flag": "--iterations", "type": "int", "default": 12, "min": 1, "max": 200, "label": "FQI iterations"},
            {"name": "random_state", "flag": "--random-state", "type": "int", "default": 42, "label": "Random state"},
        ],
    },
    "controller_cycle_portfolio_v3": {
        "label": "Controller Portfolio V3",
        "description": "Matched 2023 portfolio evaluation for the legacy controller.",
        "script": "src/ml/controller_cycle_portfolio_v3.py",
        "group": "Legacy",
        "params": [
            {"name": "fqi_iterations", "flag": "--fqi-iterations", "type": "int", "default": 12, "min": 1, "max": 200, "label": "FQI iterations"},
            {"name": "capital", "flag": "--capital", "type": "float", "default": 50000.0, "min": 1000.0, "label": "Initial capital"},
            {"name": "risk_free_rate", "flag": "--risk-free-rate", "type": "float", "default": 0.065, "min": 0.0, "max": 1.0, "label": "Risk-free rate"},
            {"name": "brokerage_per_order", "flag": "--brokerage-per-order", "type": "float", "default": 15.0, "min": 0.0, "label": "Brokerage / order"},
            {"name": "dp_charge_per_sell", "flag": "--dp-charge-per-sell", "type": "float", "default": 0.0, "min": 0.0, "label": "DP charge / sell"},
            {"name": "slippage_bps", "flag": "--slippage-bps", "type": "float", "default": 5.0, "min": 0.0, "label": "Slippage (bps)"},
        ],
    },
}


class RunRequest(BaseModel):
    experiment_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None


class CustomRunRequest(BaseModel):
    script: str
    args: list[str] = Field(default_factory=list)
    name: str | None = None


class GameCreate(BaseModel):
    year: int = 2023
    initial_capital: float = 50_000.0
    max_holdings: int = 5
    brokerage_per_order: float = 15.0
    dp_charge_per_sell: float = 0.0
    slippage_bps: float = 5.0


class GameAction(BaseModel):
    type: Literal["HOLD", "BUY", "SELL", "SWITCH", "LIQUIDATE"]
    symbol: str | None = None
    from_symbol: str | None = None
    to_symbol: str | None = None
    weight: float | None = None


RUNS: dict[str, dict[str, Any]] = {}
PROCESSES: dict[str, asyncio.subprocess.Process] = {}
TASKS: dict[str, asyncio.Task] = {}
GAMES: dict[str, dict[str, Any]] = {}


def _run_dir(run_id: str) -> Path:
    return RUN_ROOT / run_id


def _save_run(run: dict[str, Any]) -> None:
    folder = _run_dir(run["id"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(
        json.dumps(run, indent=2, default=str) + "\n"
    )


def _load_runs() -> None:
    for metadata in RUN_ROOT.glob("*/metadata.json"):
        try:
            run = json.loads(metadata.read_text())
            if run.get("status") == "running":
                run["status"] = "unknown"
            RUNS[str(run["id"])] = run
        except Exception:
            continue


_load_runs()


def _coerce_param(spec: dict[str, Any], value: Any) -> Any:
    kind = spec["type"]
    if value is None:
        return None
    if kind in {"int", "optional_int"}:
        value = int(value)
    elif kind == "float":
        value = float(value)
    elif kind == "int_list":
        value = (
            [int(v.strip()) for v in value.split(",") if v.strip()]
            if isinstance(value, str)
            else [int(v) for v in value]
        )
    elif kind == "str_list":
        value = (
            [v.strip() for v in value.split(",") if v.strip()]
            if isinstance(value, str)
            else [str(v) for v in value]
        )

    if isinstance(value, (int, float)):
        if "min" in spec and value < spec["min"]:
            raise ValueError(f"{spec['name']} must be >= {spec['min']}")
        if "max" in spec and value > spec["max"]:
            raise ValueError(f"{spec['name']} must be <= {spec['max']}")
    if "choices" in spec and isinstance(value, list):
        invalid = [v for v in value if v not in spec["choices"]]
        if invalid:
            raise ValueError(f"Invalid {spec['name']}: {invalid}")
    return value


def build_command(
    experiment_id: str,
    config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    experiment = EXPERIMENTS.get(experiment_id)
    if experiment is None:
        raise ValueError(f"Unknown experiment: {experiment_id}")

    clean: dict[str, Any] = {}
    argv = [sys.executable, "-u", experiment["script"]]
    known = {spec["name"]: spec for spec in experiment["params"]}
    unknown = sorted(set(config) - set(known))
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    for name, spec in known.items():
        value = _coerce_param(spec, config.get(name, spec.get("default")))
        clean[name] = value
        if value is None:
            continue

        argv.append(spec["flag"])
        if spec["type"] in {"int_list", "str_list"}:
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))

    return argv, clean


async def _run_experiment(run_id: str) -> None:
    run = RUNS[run_id]
    folder = _run_dir(run_id)
    log_path = folder / "run.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    run["status"] = "running"
    run["started_at"] = utc_now()
    _save_run(run)

    try:
        proc = await asyncio.create_subprocess_exec(
            *run["command"],
            cwd=str(ROOT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        PROCESSES[run_id] = proc
        run["pid"] = proc.pid
        _save_run(run)

        with log_path.open("w", encoding="utf-8") as handle:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                handle.write(text)
                handle.flush()

            code = await proc.wait()

        run["return_code"] = int(code)
        run["status"] = "completed" if code == 0 else "failed"
    except asyncio.CancelledError:
        run["status"] = "cancelled"
        raise
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = repr(exc)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[ui-runner] {exc!r}\n")
    finally:
        run["finished_at"] = utc_now()
        PROCESSES.pop(run_id, None)
        TASKS.pop(run_id, None)
        _save_run(run)


def _safe_under(base: Path, requested: str) -> Path:
    candidate = (base / requested).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Path escapes allowed root",
        ) from exc
    return candidate


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "root": str(ROOT),
        "python": sys.version.split()[0],
        "time": utc_now(),
    }


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]:
    return [
        {"id": key, **value}
        for key, value in EXPERIMENTS.items()
    ]


@app.get("/api/scripts")
def scripts() -> list[str]:
    return sorted(
        path.name
        for path in (ROOT / "src" / "ml").glob("*.py")
        if not path.name.startswith("_")
    )


@app.post("/api/runs")
async def start_run(payload: RunRequest) -> dict[str, Any]:
    try:
        command, clean = build_command(
            payload.experiment_id,
            payload.config,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    run_id = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    run = {
        "id": run_id,
        "name": payload.name
        or f"{payload.experiment_id}-{run_id[-6:]}",
        "experiment_id": payload.experiment_id,
        "config": clean,
        "command": command,
        "status": "queued",
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "return_code": None,
        "pid": None,
    }
    RUNS[run_id] = run
    _save_run(run)
    TASKS[run_id] = asyncio.create_task(
        _run_experiment(run_id)
    )
    return run


@app.post("/api/runs/custom")
async def start_custom_run(
    payload: CustomRunRequest,
) -> dict[str, Any]:
    script_path = _safe_under(ROOT / "src" / "ml", payload.script)
    if (
        script_path.suffix != ".py"
        or not script_path.is_file()
    ):
        raise HTTPException(
            status_code=400,
            detail="Custom scripts must be existing .py files under src/ml",
        )

    run_id = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    command = [
        sys.executable,
        "-u",
        str(script_path.relative_to(ROOT)),
        *[str(arg) for arg in payload.args],
    ]
    run = {
        "id": run_id,
        "name": payload.name or f"custom-{script_path.stem}-{run_id[-6:]}",
        "experiment_id": "custom",
        "config": {"script": payload.script, "args": payload.args},
        "command": command,
        "status": "queued",
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "return_code": None,
        "pid": None,
    }
    RUNS[run_id] = run
    _save_run(run)
    TASKS[run_id] = asyncio.create_task(
        _run_experiment(run_id)
    )
    return run


@app.get("/api/runs")
def list_runs(
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    runs = sorted(
        RUNS.values(),
        key=lambda row: row.get("created_at", ""),
        reverse=True,
    )
    return runs[:limit]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Run not found",
        )
    return run


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Run not found",
        )

    proc = PROCESSES.get(run_id)
    if (
        proc is not None
        and proc.returncode is None
    ):
        proc.send_signal(signal.SIGTERM)
        run["status"] = "stopping"
        _save_run(run)
    return run


@app.get("/api/runs/{run_id}/logs")
def run_logs(
    run_id: str,
    after: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
) -> dict[str, Any]:
    if run_id not in RUNS:
        raise HTTPException(
            status_code=404,
            detail="Run not found",
        )

    log_path = _run_dir(run_id) / "run.log"
    lines = (
        log_path.read_text(errors="replace").splitlines()
        if log_path.is_file()
        else []
    )
    chunk = lines[after : after + limit]
    return {
        "lines": chunk,
        "next": after + len(chunk),
        "total": len(lines),
        "status": RUNS[run_id]["status"],
    }


@app.get("/api/runs/{run_id}/events")
async def run_events(
    run_id: str,
    after: int = Query(0, ge=0),
) -> StreamingResponse:
    if run_id not in RUNS:
        raise HTTPException(
            status_code=404,
            detail="Run not found",
        )

    async def stream():
        cursor = after
        idle = 0

        while True:
            log_path = _run_dir(run_id) / "run.log"
            lines = (
                log_path.read_text(
                    errors="replace"
                ).splitlines()
                if log_path.is_file()
                else []
            )

            while cursor < len(lines):
                payload = json.dumps(
                    {
                        "index": cursor,
                        "line": lines[cursor],
                    }
                )
                yield f"data: {payload}\n\n"
                cursor += 1
                idle = 0

            status = RUNS[run_id]["status"]
            if (
                status
                in {
                    "completed",
                    "failed",
                    "cancelled",
                    "unknown",
                }
                and cursor >= len(lines)
            ):
                yield (
                    "event: end\n"
                    f"data: {json.dumps({'status': status})}\n\n"
                )
                break

            idle += 1
            if idle % 15 == 0:
                yield ": keepalive\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
    )


@lru_cache(maxsize=8)
def _year_frame(year: int) -> pd.DataFrame:
    if not PANEL_ROOT.exists():
        raise FileNotFoundError(
            f"Missing panel root: {PANEL_ROOT}"
        )

    paths: list[Path] = []
    for path in sorted(
        PANEL_ROOT.glob("date=*/data.parquet")
    ):
        date_text = path.parent.name.removeprefix(
            "date="
        )
        try:
            date = pd.Timestamp(date_text)
        except Exception:
            continue

        if date.year == year:
            paths.append(path)

    if not paths:
        raise FileNotFoundError(
            f"No panel partitions for {year}"
        )

    schema = pq.read_schema(paths[0])
    available = set(schema.names)
    requested = [
        column
        for column
        in [
            "date",
            "market_day_index",
            "canonical_security_id",
            "symbol",
            "eligible_universe",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "turnover_median_20d",
            "return_5d",
            "return_20d",
            "return_60d",
            "volatility_20d",
            "volatility_60d",
        ]
        if column in available
    ]

    frames = [
        pd.read_parquet(
            path,
            columns=requested,
        )
        for path in paths
    ]
    frame = pd.concat(
        frames,
        ignore_index=True,
    )
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    ).dt.normalize()
    frame["symbol"] = (
        frame["symbol"]
        .astype("string")
        .str.strip()
    )
    frame["canonical_security_id"] = (
        frame["canonical_security_id"]
        .astype("string")
        .str.strip()
    )

    numeric_columns = [
        column
        for column in requested
        if column
        not in {
            "date",
            "symbol",
            "canonical_security_id",
            "eligible_universe",
        }
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if "high" not in frame.columns:
        frame["high"] = frame[
            ["open", "close"]
        ].max(axis=1)
    if "low" not in frame.columns:
        frame["low"] = frame[
            ["open", "close"]
        ].min(axis=1)
    if "volume" not in frame.columns:
        if "turnover" in frame.columns:
            frame["volume"] = (
                frame["turnover"]
                / frame["close"].replace(
                    0,
                    np.nan,
                )
            )
        else:
            frame["volume"] = np.nan

    return (
        frame.sort_values(
            ["date", "symbol"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


@lru_cache(maxsize=8)
def _price_year_frame(
    year: int,
) -> tuple[pd.DataFrame, str]:
    """
    Prefer the mechanically adjusted OHLCV layer because it preserves the
    official raw NSE high/low columns as well as adjusted columns. Fall back to
    the model panel only for line-style price viewing.
    """
    if PRICE_ROOT.exists():
        paths: list[Path] = []
        for path in sorted(
            PRICE_ROOT.glob("date=*/data.parquet")
        ):
            try:
                date = pd.Timestamp(
                    path.parent.name.removeprefix(
                        "date="
                    )
                )
            except Exception:
                continue
            if date.year == year:
                paths.append(path)

        if paths:
            schema = pq.read_schema(
                paths[0]
            )
            available = set(
                schema.names
            )
            columns = [
                column
                for column in [
                    "date",
                    "canonical_security_id",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
                if column in available
            ]
            required = {
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
            }
            if required.issubset(
                set(columns)
            ):
                frame = pd.concat(
                    [
                        pd.read_parquet(
                            path,
                            columns=columns,
                        )
                        for path in paths
                    ],
                    ignore_index=True,
                )
                frame["date"] = pd.to_datetime(
                    frame["date"],
                    errors="coerce",
                ).dt.normalize()
                frame["symbol"] = (
                    frame["symbol"]
                    .astype("string")
                    .str.strip()
                )
                for column in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]:
                    if column in frame.columns:
                        frame[column] = (
                            pd.to_numeric(
                                frame[column],
                                errors="coerce",
                            )
                        )
                if "volume" not in frame.columns:
                    frame["volume"] = np.nan
                return (
                    frame.sort_values(
                        ["date", "symbol"],
                        kind="stable",
                    ).reset_index(
                        drop=True
                    ),
                    "true_ohlc",
                )

    # Fallback deliberately does not invent candle wicks.
    model = _year_frame(
        year
    )[
        [
            "date",
            "symbol",
            "open",
            "close",
            "volume",
        ]
    ].copy()
    model["high"] = np.nan
    model["low"] = np.nan
    return model, "open_close_only"


def _resample_candles(
    frame: pd.DataFrame,
    interval: str,
) -> pd.DataFrame:
    if interval == "1D":
        return frame

    rule = {
        "1W": "W-FRI",
        "1M": "ME",
        "3M": "QE",
    }.get(interval)

    if rule is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported interval. Available from the "
                "daily research panel: 1D, 1W, 1M, 3M"
            ),
        )

    work = frame.set_index("date")
    agg = (
        work.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(
            subset=["open", "close"]
        )
    )
    return agg.reset_index()


@app.get("/api/market/years")
def market_years() -> list[int]:
    years: set[int] = set()
    if PANEL_ROOT.exists():
        for path in PANEL_ROOT.glob(
            "date=*/data.parquet"
        ):
            try:
                years.add(
                    pd.Timestamp(
                        path.parent.name.removeprefix(
                            "date="
                        )
                    ).year
                )
            except Exception:
                pass

    return sorted(years)


@app.get("/api/market/symbols")
def market_symbols(
    year: int = 2023,
    query: str = "",
    limit: int = Query(
        200,
        ge=1,
        le=2000,
    ),
) -> list[dict[str, Any]]:
    try:
        frame = _year_frame(year)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    columns = [
        "symbol",
        "canonical_security_id",
    ]
    symbols = (
        frame[columns]
        .drop_duplicates()
        .dropna()
    )

    if query:
        q = query.upper()
        symbols = symbols[
            symbols["symbol"]
            .str.upper()
            .str.contains(
                q,
                regex=False,
            )
        ]

    return symbols.head(limit).to_dict(
        orient="records"
    )


@app.get("/api/market/candles")
def market_candles(
    symbol: str,
    year: int = 2023,
    interval: str = "1D",
    end_date: str | None = None,
) -> dict[str, Any]:
    try:
        frame, ohlc_quality = _price_year_frame(
            year
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    subset = frame[
        frame["symbol"]
        .str.upper()
        .eq(symbol.upper())
    ][
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].dropna(
        subset=["open", "close"]
    )

    if end_date:
        cutoff = pd.Timestamp(
            end_date
        ).normalize()
        subset = subset.loc[
            subset["date"].le(
                cutoff
            )
        ]

    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No candles for {symbol} in {year}"
            ),
        )

    if (
        ohlc_quality
        != "true_ohlc"
        and interval != "1D"
    ):
        # Resampling open/close-only data is still legitimate for a line/area
        # chart, but it is not a real candlestick series.
        pass

    subset = _resample_candles(
        subset,
        interval,
    )
    candles = [
        {
            "time": pd.Timestamp(
                row.date
            ).strftime("%Y-%m-%d"),
            "open": float(row.open),
            "high": (
                None
                if pd.isna(row.high)
                else float(row.high)
            ),
            "low": (
                None
                if pd.isna(row.low)
                else float(row.low)
            ),
            "close": float(row.close),
            "volume": (
                None
                if pd.isna(row.volume)
                else float(row.volume)
            ),
        }
        for row in subset.itertuples(
            index=False
        )
    ]

    return {
        "symbol": symbol.upper(),
        "year": year,
        "interval": interval,
        "available_intervals": [
            "1D",
            "1W",
            "1M",
            "3M",
        ],
        "ohlc_quality": (
            ohlc_quality
        ),
        "candles": candles,
    }


def _report_files() -> list[dict[str, Any]]:
    report_root = ROOT / "reports" / "ml"
    if not report_root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for pattern, kind in [
        ("**/summary.json", "summary"),
        ("**/metrics.json", "metrics"),
        ("**/leaderboard.csv", "leaderboard"),
    ]:
        for path in report_root.glob(pattern):
            rows.append(
                {
                    "path": str(
                        path.relative_to(ROOT)
                    ),
                    "kind": kind,
                    "mtime": path.stat().st_mtime,
                    "size": path.stat().st_size,
                }
            )

    return sorted(
        rows,
        key=lambda row: row["mtime"],
        reverse=True,
    )


@app.get("/api/reports")
def reports(
    limit: int = Query(
        200,
        ge=1,
        le=2000,
    ),
) -> list[dict[str, Any]]:
    return _report_files()[:limit]


@app.get("/api/artifact")
def artifact(
    path: str,
    limit: int = Query(
        1000,
        ge=1,
        le=10000,
    ),
) -> dict[str, Any]:
    full = _safe_under(
        ROOT,
        path,
    )
    allowed = [
        ROOT / "reports",
        ROOT / "data" / "processed",
    ]

    if not any(
        full.is_relative_to(
            base.resolve()
        )
        for base in allowed
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "Artifact path is not under reports/ "
                "or data/processed/"
            ),
        )

    if not full.is_file():
        raise HTTPException(
            status_code=404,
            detail="Artifact not found",
        )

    suffix = full.suffix.lower()
    if suffix == ".json":
        return {
            "path": path,
            "kind": "json",
            "data": json.loads(
                full.read_text()
            ),
        }

    if suffix == ".csv":
        frame = pd.read_csv(
            full
        ).head(limit)
        return {
            "path": path,
            "kind": "table",
            "columns": list(frame.columns),
            "rows": (
                frame.where(
                    pd.notna(frame),
                    None,
                ).to_dict(
                    orient="records"
                )
            ),
        }

    if suffix == ".parquet":
        frame = pd.read_parquet(
            full
        ).head(limit)
        frame = frame.copy()
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(
                frame[column]
            ):
                frame[column] = frame[
                    column
                ].astype("string")

        return {
            "path": path,
            "kind": "table",
            "columns": list(frame.columns),
            "rows": (
                frame.where(
                    pd.notna(frame),
                    None,
                ).to_dict(
                    orient="records"
                )
            ),
        }

    return {
        "path": path,
        "kind": "text",
        "data": full.read_text(
            errors="replace"
        )[:2_000_000],
    }


def _latest_policy_dir(
    year: int = 2023,
) -> Path:
    return (
        ROOT
        / "reports"
        / "ml"
        / "portfolio_policy_v4"
        / f"development_{year}"
    )


@app.get("/api/dashboard")
def dashboard(
    year: int = 2023,
) -> dict[str, Any]:
    base = _latest_policy_dir(
        year
    )
    metrics: dict[str, Any] = {}
    decisions: list[
        dict[str, Any]
    ] = []
    equity: list[
        dict[str, Any]
    ] = []

    metrics_path = (
        base / "metrics.json"
    )
    if metrics_path.is_file():
        metrics = json.loads(
            metrics_path.read_text()
        )

    decisions_path = (
        base / "decisions.parquet"
    )
    if decisions_path.is_file():
        frame = (
            pd.read_parquet(
                decisions_path
            )
            .tail(500)
            .copy()
        )
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(
                frame[column]
            ):
                frame[column] = frame[
                    column
                ].astype("string")
        decisions = (
            frame.where(
                pd.notna(frame),
                None,
            )
            .to_dict(
                orient="records"
            )
        )

    equity_path = (
        base / "equity_curve.parquet"
    )
    if equity_path.is_file():
        frame = pd.read_parquet(
            equity_path
        ).copy()
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(
                frame[column]
            ):
                frame[column] = frame[
                    column
                ].astype("string")
        equity = (
            frame.where(
                pd.notna(frame),
                None,
            )
            .to_dict(
                orient="records"
            )
        )

    return {
        "year": year,
        "metrics": metrics,
        "decisions": decisions,
        "equity": equity,
        "reports": _report_files()[:30],
    }


def _game_snapshot(
    game: dict[str, Any],
) -> dict[str, Any]:
    frame = _year_frame(
        game["year"]
    )
    dates = game["dates"]
    index = game["position"]
    date = pd.Timestamp(
        dates[index]
    )
    day = frame[
        frame["date"].eq(
            date
        )
    ]
    close_map = (
        day.set_index("symbol")[
            "close"
        ].to_dict()
    )

    holdings_value = 0.0
    holdings = []
    for symbol, quantity in sorted(
        game["holdings"].items()
    ):
        price = float(
            close_map.get(
                symbol,
                game["last_prices"].get(
                    symbol,
                    0.0,
                ),
            )
        )
        game["last_prices"][
            symbol
        ] = price
        value = quantity * price
        holdings_value += value
        holdings.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "value": value,
            }
        )

    equity = float(
        game["cash"]
        + holdings_value
    )
    date_text = date.strftime(
        "%Y-%m-%d"
    )
    if (
        not game["equity"]
        or game["equity"][-1][
            "date"
        ]
        != date_text
    ):
        game["equity"].append(
            {
                "date": date_text,
                "equity": equity,
            }
        )

    returns = (
        pd.Series(
            [
                row["equity"]
                for row in game[
                    "equity"
                ]
            ],
            dtype=float,
        )
        .pct_change()
        .dropna()
    )
    values = np.asarray(
        [
            row["equity"]
            for row in game[
                "equity"
            ]
        ],
        dtype=float,
    )

    peaks = (
        np.maximum.accumulate(
            values
        )
        if len(values)
        else np.asarray([])
    )
    max_drawdown = (
        float(
            np.min(
                values
                / peaks
                - 1.0
            )
        )
        if len(values)
        else 0.0
    )
    years = max(
        (
            date
            - pd.Timestamp(
                dates[0]
            )
        ).days
        / 365.25,
        1.0 / 365.25,
    )
    cagr = (
        (
            equity
            / game[
                "initial_capital"
            ]
        )
        ** (1 / years)
        - 1
        if equity > 0
        else -1.0
    )
    volatility = (
        float(
            returns.std(
                ddof=1
            )
            * math.sqrt(252.0)
        )
        if len(returns) > 1
        else 0.0
    )
    annual_return = (
        float(
            returns.mean()
            * 252.0
        )
        if len(returns)
        else 0.0
    )
    sharpe = (
        (
            annual_return
            - 0.065
        )
        / volatility
        if volatility > 0
        else None
    )
    downside = returns[
        returns < 0
    ]
    downside_vol = (
        float(
            downside.std(
                ddof=1
            )
            * math.sqrt(252.0)
        )
        if len(downside) > 1
        else 0.0
    )
    sortino = (
        (
            annual_return
            - 0.065
        )
        / downside_vol
        if downside_vol > 0
        else None
    )

    return {
        "id": game["id"],
        "year": game["year"],
        "date": date_text,
        "position": index,
        "total_steps": len(dates),
        "progress": (
            index
            / max(
                1,
                len(dates)
                - 1,
            )
        ),
        "cash": game["cash"],
        "equity": equity,
        "fees": game["fees"],
        "holdings": holdings,
        "history": game["history"],
        "equity_curve": game["equity"],
        "finished": game["finished"],
        "metrics": {
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": (
                max_drawdown
            ),
            "total_return": (
                equity
                / game[
                    "initial_capital"
                ]
                - 1
            ),
        },
    }


@app.post("/api/game")
def create_game(
    payload: GameCreate,
) -> dict[str, Any]:
    try:
        frame = _year_frame(
            payload.year
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    dates = [
        pd.Timestamp(date)
        for date in sorted(
            frame["date"]
            .dropna()
            .unique()
        )
    ]
    if len(dates) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough market days",
        )

    game_id = uuid.uuid4().hex[:10]
    GAMES[game_id] = {
        "id": game_id,
        "year": payload.year,
        "initial_capital": float(
            payload.initial_capital
        ),
        "max_holdings": int(
            payload.max_holdings
        ),
        "costs": CostProfile(
            brokerage_per_order=float(
                payload.brokerage_per_order
            ),
            dp_charge_per_sell=float(
                payload.dp_charge_per_sell
            ),
            slippage_bps=float(
                payload.slippage_bps
            ),
        ),
        "dates": dates,
        "position": 0,
        "cash": float(
            payload.initial_capital
        ),
        "holdings": {},
        "fees": 0.0,
        "history": [],
        "equity": [],
        "last_prices": {},
        "finished": False,
    }
    return _game_snapshot(
        GAMES[game_id]
    )


@app.get("/api/game/{game_id}")
def get_game(
    game_id: str,
) -> dict[str, Any]:
    game = GAMES.get(
        game_id
    )
    if game is None:
        raise HTTPException(
            status_code=404,
            detail="Game not found",
        )
    return _game_snapshot(
        game
    )


def _game_day(
    frame: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    return (
        frame[
            frame["date"].eq(
                date
            )
        ]
        .set_index(
            "symbol",
            drop=False,
        )
    )


@app.post("/api/game/{game_id}/action")
def game_action(
    game_id: str,
    action: GameAction,
) -> dict[str, Any]:
    game = GAMES.get(
        game_id
    )
    if game is None:
        raise HTTPException(
            status_code=404,
            detail="Game not found",
        )
    if game["finished"]:
        raise HTTPException(
            status_code=400,
            detail="Game already finished",
        )
    if (
        game["position"]
        >= len(game["dates"])
        - 1
    ):
        game["finished"] = True
        return _game_snapshot(
            game
        )

    frame = _year_frame(
        game["year"]
    )
    signal_date = pd.Timestamp(
        game["dates"][
            game["position"]
        ]
    )
    next_date = pd.Timestamp(
        game["dates"][
            game["position"]
            + 1
        ]
    )
    next_day = _game_day(
        frame,
        next_date,
    )
    costs: CostProfile = game[
        "costs"
    ]
    kind = action.type
    fees_before = game["fees"]

    def sell_symbol(
        symbol: str,
    ) -> None:
        symbol = symbol.upper()
        quantity = float(
            game["holdings"].get(
                symbol,
                0.0,
            )
        )
        if quantity <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No holding in {symbol}"
                ),
            )
        if symbol not in next_day.index:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No next-open quote for {symbol}"
                ),
            )

        execution = sell_execution(
            float(
                next_day.loc[
                    symbol,
                    "open",
                ]
            ),
            quantity,
            costs,
        )
        game["cash"] += float(
            execution["cash_in"]
        )
        game["fees"] += float(
            execution["fees"]
        )
        game["holdings"].pop(
            symbol,
            None,
        )

    def buy_symbol(
        symbol: str,
        weight: float | None,
    ) -> None:
        symbol = symbol.upper()
        if symbol in game["holdings"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Already holding {symbol}"
                ),
            )
        if (
            len(game["holdings"])
            >= game["max_holdings"]
        ):
            raise HTTPException(
                status_code=400,
                detail="Max holdings reached",
            )
        if symbol not in next_day.index:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No next-open quote for {symbol}"
                ),
            )

        quoted = float(
            next_day.loc[
                symbol,
                "open",
            ]
        )
        target_weight = float(
            weight
            if weight is not None
            else (
                1.0
                / game[
                    "max_holdings"
                ]
            )
        )
        target_weight = min(
            max(
                target_weight,
                0.01,
            ),
            1.0,
        )
        signal_frame = _year_frame(
            game["year"]
        )
        signal_date = pd.Timestamp(
            game["dates"][
                game["position"]
            ]
        )
        signal_day = signal_frame.loc[
            signal_frame["date"].eq(
                signal_date
            )
        ]
        close_map = (
            signal_day.set_index(
                "symbol"
            )["close"].to_dict()
        )
        equity_before = float(
            game["cash"]
            + sum(
                float(quantity)
                * float(
                    close_map.get(
                        held_symbol,
                        game["last_prices"].get(
                            held_symbol,
                            0.0,
                        ),
                    )
                )
                for held_symbol, quantity
                in game["holdings"].items()
            )
        )
        budget = min(
            game["cash"],
            equity_before
            * target_weight,
        )

        quantity = int(
            max(
                0,
                math.floor(
                    budget
                    / max(
                        quoted,
                        1e-12,
                    )
                ),
            )
        )

        while quantity > 0:
            execution = buy_execution(
                quoted,
                quantity,
                costs,
            )
            if (
                execution["cash_out"]
                <= game["cash"]
                + 1e-9
            ):
                break
            quantity -= 1

        if quantity <= 0:
            raise HTTPException(
                status_code=400,
                detail="Insufficient cash",
            )

        game["cash"] -= float(
            execution["cash_out"]
        )
        game["fees"] += float(
            execution["fees"]
        )
        game["holdings"][
            symbol
        ] = float(
            quantity
        )

    cash_before = float(
        game["cash"]
    )
    holdings_before = dict(
        game["holdings"]
    )
    fees_state_before = float(
        game["fees"]
    )

    try:
        if kind == "BUY":
            if not action.symbol:
                raise HTTPException(
                    status_code=400,
                    detail="BUY requires symbol",
                )
            buy_symbol(
                action.symbol,
                action.weight,
            )
        elif kind == "SELL":
            if not action.symbol:
                raise HTTPException(
                    status_code=400,
                    detail="SELL requires symbol",
                )
            sell_symbol(
                action.symbol
            )
        elif kind == "SWITCH":
            if (
                not action.from_symbol
                or not action.to_symbol
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SWITCH requires from_symbol "
                        "and to_symbol"
                    ),
                )
            sell_symbol(
                action.from_symbol
            )
            buy_symbol(
                action.to_symbol,
                action.weight,
            )
        elif kind == "LIQUIDATE":
            for symbol in list(
                game["holdings"]
            ):
                sell_symbol(
                    symbol
                )
            game["finished"] = True
        elif kind != "HOLD":
            raise HTTPException(
                status_code=400,
                detail="Unknown game action",
            )
    except HTTPException:
        game["cash"] = (
            cash_before
        )
        game["holdings"] = (
            holdings_before
        )
        game["fees"] = (
            fees_state_before
        )
        raise

    game["history"].append(
        {
            "signal_date": (
                signal_date.strftime(
                    "%Y-%m-%d"
                )
            ),
            "execution_date": (
                next_date.strftime(
                    "%Y-%m-%d"
                )
            ),
            "action": kind,
            "symbol": action.symbol,
            "from_symbol": (
                action.from_symbol
            ),
            "to_symbol": (
                action.to_symbol
            ),
            "fees": (
                game["fees"]
                - fees_before
            ),
        }
    )
    game["position"] += 1

    if (
        game["position"]
        >= len(game["dates"])
        - 1
        and kind != "LIQUIDATE"
    ):
        game["finished"] = True

    return _game_snapshot(
        game
    )


@app.get("/api/game/{game_id}/market")
def game_market(
    game_id: str,
    query: str = "",
    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
) -> list[dict[str, Any]]:
    game = GAMES.get(
        game_id
    )
    if game is None:
        raise HTTPException(
            status_code=404,
            detail="Game not found",
        )

    frame = _year_frame(
        game["year"]
    )
    date = pd.Timestamp(
        game["dates"][
            game["position"]
        ]
    )
    day = frame[
        frame["date"].eq(
            date
        )
    ].copy()

    if (
        "eligible_universe"
        in day.columns
    ):
        day = day[
            day[
                "eligible_universe"
            ].fillna(False)
        ]

    if query:
        day = day[
            day["symbol"]
            .str.upper()
            .str.contains(
                query.upper(),
                regex=False,
            )
        ]

    sort_column = (
        "turnover_median_20d"
        if "turnover_median_20d"
        in day.columns
        else "symbol"
    )
    day = (
        day.sort_values(
            sort_column,
            ascending=(
                False
                if sort_column
                != "symbol"
                else True
            ),
        )
        .head(limit)
    )

    columns = [
        column
        for column in [
            "symbol",
            "close",
            "return_5d",
            "return_20d",
            "return_60d",
            "volatility_20d",
        ]
        if column in day.columns
    ]

    return (
        day[columns]
        .where(
            pd.notna(
                day[columns]
            ),
            None,
        )
        .to_dict(
            orient="records"
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
