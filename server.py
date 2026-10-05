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
    start_date: str | None = None
    history_bars: int = Field(
        default=260,
        ge=20,
        le=2000,
    )
    initial_capital: float = 50_000.0
    max_holdings: int = 5
    brokerage_per_order: float = 15.0
    dp_charge_per_sell: float = 0.0
    slippage_bps: float = 5.0


class GameAction(BaseModel):
    type: Literal[
        "HOLD",
        "BUY",
        "SELL",
        "SWITCH",
        "SET_TARGET",
        "LIQUIDATE",
    ]
    symbol: str | None = None
    from_symbol: str | None = None
    to_symbol: str | None = None
    sizing_mode: Literal[
        "shares",
        "rupees",
        "equity_fraction",
        "target_weight",
    ] = "shares"
    quantity: int | None = Field(
        default=None,
        ge=1,
    )
    amount: float | None = Field(
        default=None,
        gt=0.0,
    )
    fraction: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )
    target_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    # Backward-compatible alias from the first UI version.
    weight: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )


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
            "IML_PROGRESS_JSON": "1",
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

                if text.startswith(
                    "IML_PROGRESS "
                ):
                    try:
                        progress_payload = (
                            json.loads(
                                text[
                                    len(
                                        "IML_PROGRESS "
                                    ):
                                ].strip()
                            )
                        )
                        run[
                            "progress"
                        ] = progress_payload
                        _save_run(
                            run
                        )
                    except Exception:
                        handle.write(
                            text
                        )
                        handle.flush()
                    continue

                handle.write(text)
                handle.flush()

            code = await proc.wait()

        run["return_code"] = int(code)
        run["status"] = "completed" if code == 0 else "failed"
        if (
            code == 0
            and isinstance(
                run.get("progress"),
                dict,
            )
        ):
            run[
                "progress"
            ][
                "completed"
            ] = run[
                "progress"
            ].get(
                "total",
                1,
            )
            run[
                "progress"
            ][
                "fraction"
            ] = 1.0
            run[
                "progress"
            ][
                "percent"
            ] = 100.0
            run[
                "progress"
            ][
                "eta_seconds"
            ] = 0.0
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
        "progress": {
            "phase": payload.experiment_id,
            "completed": 0,
            "total": 1,
            "fraction": 0.0,
            "percent": 0.0,
            "eta_seconds": None,
            "message": "queued",
            "detail": "",
        },
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
    lookback_bars: int | None = Query(
        default=None,
        ge=20,
        le=5000,
    ),
) -> dict[str, Any]:
    frames: list[
        pd.DataFrame
    ] = []
    qualities: list[str] = []

    cutoff = (
        pd.Timestamp(
            end_date
        ).normalize()
        if end_date
        else None
    )
    final_year = int(
        cutoff.year
        if cutoff
        is not None
        else year
    )

    years_to_load = [
        final_year
    ]
    if (
        cutoff is not None
        and lookback_bars
        is not None
    ):
        # Bring in prior years so a game can start anywhere without the
        # chart looking as if market history began on the start date.
        years_to_load.extend(
            range(
                final_year - 1,
                max(
                    final_year - 8,
                    2009,
                ),
                -1,
            )
        )

    for source_year in years_to_load:
        try:
            source, quality = (
                _price_year_frame(
                    int(
                        source_year
                    )
                )
            )
        except FileNotFoundError:
            continue

        part = source[
            source["symbol"]
            .str.upper()
            .eq(
                symbol.upper()
            )
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
            subset=[
                "open",
                "close",
            ]
        )

        if cutoff is not None:
            part = part.loc[
                part[
                    "date"
                ].le(
                    cutoff
                )
            ]

        if not part.empty:
            frames.append(
                part
            )
            qualities.append(
                quality
            )

        if (
            lookback_bars
            is not None
            and sum(
                len(
                    frame
                )
                for frame in frames
            )
            >= lookback_bars
        ):
            break

    if not frames:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No candles for {symbol}"
            ),
        )

    subset = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            "date",
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "date",
            ],
            keep="last",
        )
    )

    if lookback_bars is not None:
        subset = subset.tail(
            int(
                lookback_bars
            )
        )

    ohlc_quality = (
        "true_ohlc"
        if qualities
        and all(
            quality
            == "true_ohlc"
            for quality in qualities
        )
        else "open_close_only"
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
        "end_date": (
            None
            if cutoff is None
            else cutoff.strftime(
                "%Y-%m-%d"
            )
        ),
        "lookback_bars": (
            lookback_bars
        ),
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


@app.get("/api/action-matrix")
def action_matrix(
    date: str | None = None,
    model: str | None = None,
    limit: int = Query(
        250,
        ge=1,
        le=5000,
    ),
) -> dict[str, Any]:
    path = (
        ROOT
        / "reports"
        / "ml"
        / "portfolio_student_v4"
        / "development"
        / "action_value_predictions.parquet"
    )
    if not path.is_file():
        return {
            "state_date": None,
            "rows": [],
        }

    frame = pd.read_parquet(
        path
    )
    frame["state_date"] = (
        pd.to_datetime(
            frame["state_date"],
            errors="coerce",
        ).dt.normalize()
    )

    if model:
        frame = frame.loc[
            frame["model"].eq(
                model
            )
        ]

    if date:
        state_date = pd.Timestamp(
            date
        ).normalize()
    else:
        state_date = frame[
            "state_date"
        ].max()

    frame = frame.loc[
        frame["state_date"].eq(
            state_date
        )
    ].copy()

    if frame.empty:
        return {
            "state_date": (
                None
                if pd.isna(
                    state_date
                )
                else pd.Timestamp(
                    state_date
                ).strftime(
                    "%Y-%m-%d"
                )
            ),
            "rows": [],
        }

    frame["regret"] = (
        frame.groupby(
            [
                "state_key",
                "model",
            ],
            sort=False,
        )["oracle_q"]
        .transform("max")
        - frame["oracle_q"]
    )
    frame = (
        frame.sort_values(
            [
                "predicted_q",
                "oracle_q",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )
        .head(limit)
    )

    symbol_map: dict[
        str,
        str,
    ] = {}
    try:
        market = _year_frame(
            int(
                pd.Timestamp(
                    state_date
                ).year
            )
        )
        day = market.loc[
            market["date"].eq(
                state_date
            ),
            [
                "canonical_security_id",
                "symbol",
            ],
        ].drop_duplicates()
        symbol_map = dict(
            zip(
                day[
                    "canonical_security_id"
                ].astype(str),
                day[
                    "symbol"
                ].astype(str),
            )
        )
    except Exception:
        symbol_map = {}

    rows = []
    for row in frame.to_dict(
        orient="records"
    ):
        target = []
        try:
            encoded = json.loads(
                str(
                    row.get(
                        "meta_target_signature",
                        "[]",
                    )
                )
            )
            for cid, weight in encoded:
                target.append(
                    {
                        "asset": (
                            symbol_map.get(
                                str(cid),
                                str(cid),
                            )
                        ),
                        "weight": float(
                            weight
                        ),
                    }
                )
        except Exception:
            target = []

        rows.append({
            "state_key": row.get(
                "state_key"
            ),
            "state_date": pd.Timestamp(
                row[
                    "state_date"
                ]
            ).strftime(
                "%Y-%m-%d"
            ),
            "model": row.get(
                "model"
            ),
            "oracle_rank": int(
                row.get(
                    "action_rank",
                    0,
                )
            ),
            "is_oracle_action": bool(
                row.get(
                    "is_oracle_action",
                    False,
                )
            ),
            "predicted_q": float(
                row[
                    "predicted_q"
                ]
            ),
            "oracle_q": float(
                row[
                    "oracle_q"
                ]
            ),
            "regret": float(
                row[
                    "regret"
                ]
            ),
            "oracle_terminal_return": float(
                row[
                    "oracle_terminal_return"
                ]
            ),
            "oracle_max_drawdown": float(
                row[
                    "oracle_max_drawdown"
                ]
            ),
            "oracle_turnover": float(
                row[
                    "oracle_turnover"
                ]
            ),
            "target": target,
        })

    return {
        "state_date": pd.Timestamp(
            state_date
        ).strftime(
            "%Y-%m-%d"
        ),
        "rows": rows,
    }


def _report_files() -> list[dict[str, Any]]:
    report_root = ROOT / "reports" / "ml"
    if not report_root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for suffix in (
        ".json",
        ".csv",
        ".parquet",
    ):
        for path in report_root.glob(
            f"**/*{suffix}"
        ):
            stem = path.stem.lower()
            if (
                "leaderboard"
                in stem
            ):
                kind = "leaderboard"
            elif (
                stem
                in {
                    "summary",
                    "metrics",
                }
                or stem.endswith(
                    "_summary"
                )
            ):
                kind = (
                    "summary"
                    if "summary"
                    in stem
                    else "metrics"
                )
            elif (
                "diagnostic"
                in stem
            ):
                kind = "diagnostics"
            elif (
                "prediction"
                in stem
                or "state_value"
                in stem
            ):
                kind = "predictions"
            elif (
                "trade"
                in stem
            ):
                kind = "trades"
            elif (
                "equity"
                in stem
            ):
                kind = "equity"
            else:
                kind = (
                    suffix.lstrip(
                        "."
                    )
                )

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
        quantity = int(
            quantity
        )
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

        value = (
            quantity
            * price
        )
        holdings_value += value

        average_cost = float(
            game.get(
                "average_cost",
                {},
            ).get(
                symbol,
                price,
            )
        )
        cost_basis = (
            quantity
            * average_cost
        )
        unrealized_pnl = (
            value
            - cost_basis
        )
        unrealized_return = (
            unrealized_pnl
            / cost_basis
            if cost_basis > 0
            else 0.0
        )

        holdings.append({
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "value": value,
            "average_cost": (
                average_cost
            ),
            "cost_basis": (
                cost_basis
            ),
            "unrealized_pnl": (
                unrealized_pnl
            ),
            "unrealized_return": (
                unrealized_return
            ),
        })

    equity = float(
        game["cash"]
        + holdings_value
    )

    for holding in holdings:
        holding["weight"] = (
            float(
                holding[
                    "value"
                ]
            )
            / equity
            if equity > 0
            else 0.0
        )
        reference_price = max(
            float(
                holding[
                    "price"
                ]
            ),
            1e-12,
        )
        holding[
            "estimated_max_add_shares"
        ] = int(
            max(
                0,
                math.floor(
                    game["cash"]
                    / reference_price
                ),
            )
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
                dates[
                    game[
                        "start_position"
                    ]
                ]
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
        ** (
            1.0
            / years
        )
        - 1.0
        if equity > 0
        else -1.0
    )
    volatility = (
        float(
            returns.std(
                ddof=1
            )
            * math.sqrt(
                252.0
            )
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
            * math.sqrt(
                252.0
            )
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
    calmar = (
        cagr
        / abs(
            max_drawdown
        )
        if max_drawdown < 0
        else None
    )

    elapsed_calendar_days = int(
        max(
            0,
            (
                date
                - pd.Timestamp(
                    dates[
                        game[
                            "start_position"
                        ]
                    ]
                )
            ).days,
        )
    )
    elapsed_sessions = int(
        max(
            0,
            index
            - game[
                "start_position"
            ],
        )
    )
    average_daily_return = (
        float(
            returns.mean()
        )
        if len(
            returns
        )
        else 0.0
    )

    execution_legs = [
        execution
        for event
        in game[
            "history"
        ]
        for execution
        in event.get(
            "executions",
            [],
        )
    ]
    sell_legs = [
        execution
        for execution
        in execution_legs
        if execution.get(
            "side"
        )
        == "SELL"
    ]
    realized_results = [
        float(
            execution.get(
                "realized_pnl",
                0.0,
            )
        )
        for execution
        in sell_legs
    ]
    profitable_exits = sum(
        result > 0
        for result
        in realized_results
    )

    order_count = int(
        len(
            execution_legs
        )
    )
    exit_count = int(
        len(
            realized_results
        )
    )
    average_fee_per_order = (
        float(
            game[
                "fees"
            ]
        )
        / order_count
        if order_count > 0
        else 0.0
    )
    average_order_value = (
        float(
            game.get(
                "turnover",
                0.0,
            )
        )
        / order_count
        if order_count > 0
        else 0.0
    )
    average_realized_pnl = (
        float(
            np.mean(
                realized_results
            )
        )
        if realized_results
        else 0.0
    )
    realized_win_rate = (
        float(
            profitable_exits
            / exit_count
        )
        if exit_count > 0
        else None
    )
    invested_value = float(
        holdings_value
    )
    cash_fraction = (
        float(
            game[
                "cash"
            ]
        )
        / equity
        if equity > 0
        else 0.0
    )

    return {
        "id": game["id"],
        "year": game["year"],
        "start_date": pd.Timestamp(
            dates[
                game[
                    "start_position"
                ]
            ]
        ).strftime(
            "%Y-%m-%d"
        ),
        "history_bars": int(
            game[
                "history_bars"
            ]
        ),
        "date": date_text,
        "position": index,
        "start_position": int(
            game[
                "start_position"
            ]
        ),
        "total_steps": (
            len(
                dates
            )
            - game[
                "start_position"
            ]
        ),
        "progress": (
            (
                index
                - game[
                    "start_position"
                ]
            )
            / max(
                1,
                len(dates)
                - 1
                - game[
                    "start_position"
                ],
            )
        ),
        "cash": float(
            game["cash"]
        ),
        "buying_power": float(
            game["cash"]
        ),
        "invested_value": (
            invested_value
        ),
        "cash_fraction": (
            cash_fraction
        ),
        "equity": equity,
        "elapsed_calendar_days": (
            elapsed_calendar_days
        ),
        "elapsed_sessions": (
            elapsed_sessions
        ),
        "order_count": (
            order_count
        ),
        "exit_count": (
            exit_count
        ),
        "fees": float(
            game["fees"]
        ),
        "realized_pnl": float(
            game.get(
                "realized_pnl",
                0.0,
            )
        ),
        "order_rules": {
            "fractional_shares": False,
            "minimum_quantity": 1,
            "sizing_modes": [
                "shares",
                "rupees",
                "equity_fraction",
                "target_weight",
            ],
        },
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
            "calmar": calmar,
            "annualized_volatility": (
                volatility
            ),
            "average_daily_return": (
                average_daily_return
            ),
            "average_fee_per_order": (
                average_fee_per_order
            ),
            "average_order_value": (
                average_order_value
            ),
            "average_realized_pnl_per_exit": (
                average_realized_pnl
            ),
            "realized_win_rate": (
                realized_win_rate
            ),
            "turnover_multiple": (
                float(
                    game.get(
                        "turnover",
                        0.0,
                    )
                )
                / float(
                    game[
                        "initial_capital"
                    ]
                )
                if game[
                    "initial_capital"
                ] > 0
                else None
            ),
            "total_return": (
                equity
                / game[
                    "initial_capital"
                ]
                - 1.0
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
            detail=(
                "Not enough market days"
            ),
        )

    if payload.start_date:
        requested = pd.Timestamp(
            payload.start_date
        ).normalize()
        valid_positions = [
            index
            for index, date
            in enumerate(
                dates
            )
            if pd.Timestamp(
                date
            )
            >= requested
        ]
        if not valid_positions:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Start date is after the "
                    "available year."
                ),
            )
        start_position = int(
            valid_positions[
                0
            ]
        )
    else:
        start_position = 0

    if (
        start_position
        >= len(
            dates
        )
        - 1
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Start date must leave at least "
                "one future market session."
            ),
        )

    game_id = (
        uuid.uuid4().hex[
            :10
        ]
    )
    GAMES[
        game_id
    ] = {
        "id": game_id,
        "year": payload.year,
        "history_bars": int(
            payload.history_bars
        ),
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
        "start_position": (
            start_position
        ),
        "position": (
            start_position
        ),
        "cash": float(
            payload.initial_capital
        ),
        "holdings": {},
        "average_cost": {},
        "realized_pnl": 0.0,
        "fees": 0.0,
        "turnover": 0.0,
        "history": [],
        "equity": [],
        "last_prices": {},
        "finished": False,
    }
    return _game_snapshot(
        GAMES[
            game_id
        ]
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
            detail=(
                "Game already finished"
            ),
        )
    if (
        game["position"]
        >= len(
            game["dates"]
        )
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
    signal_day = _game_day(
        frame,
        signal_date,
    )
    next_day = _game_day(
        frame,
        next_date,
    )
    costs: CostProfile = game[
        "costs"
    ]
    kind = action.type

    signal_close = {
        str(symbol): float(
            row["close"]
        )
        for symbol, row
        in signal_day.iterrows()
        if pd.notna(
            row.get(
                "close"
            )
        )
    }

    def equity_at_signal() -> float:
        return float(
            game["cash"]
            + sum(
                float(quantity)
                * float(
                    signal_close.get(
                        symbol,
                        game[
                            "last_prices"
                        ].get(
                            symbol,
                            0.0,
                        ),
                    )
                )
                for symbol, quantity
                in game[
                    "holdings"
                ].items()
            )
        )

    def next_open(
        symbol: str,
    ) -> float:
        symbol = symbol.upper()
        if symbol not in next_day.index:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No next-open quote "
                    f"for {symbol}"
                ),
            )
        price = float(
            next_day.loc[
                symbol,
                "open",
            ]
        )
        if (
            not math.isfinite(
                price
            )
            or price <= 0
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid next-open quote "
                    f"for {symbol}"
                ),
            )
        return price

    def delta_to_target(
        symbol: str,
        target_weight: float,
    ) -> int:
        symbol = symbol.upper()
        equity = equity_at_signal()
        quoted = next_open(
            symbol
        )
        desired = int(
            max(
                0,
                math.floor(
                    equity
                    * float(
                        target_weight
                    )
                    / quoted
                ),
            )
        )
        current = int(
            game[
                "holdings"
            ].get(
                symbol,
                0,
            )
        )
        return (
            desired
            - current
        )

    def requested_quantity(
        *,
        symbol: str,
        side: Literal[
            "buy",
            "sell",
        ],
    ) -> int:
        symbol = symbol.upper()
        quoted = next_open(
            symbol
        )
        current = int(
            game[
                "holdings"
            ].get(
                symbol,
                0,
            )
        )
        mode = action.sizing_mode

        if mode == "shares":
            if action.quantity is None:
                if side == "sell":
                    return current
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Share-sized BUY requires "
                        "a quantity."
                    ),
                )
            return int(
                action.quantity
            )

        if mode == "rupees":
            if action.amount is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Rupee-sized order "
                        "requires amount."
                    ),
                )
            return max(
                1,
                int(
                    math.floor(
                        float(
                            action.amount
                        )
                        / quoted
                    )
                ),
            )

        if mode == "equity_fraction":
            if action.fraction is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Equity-fraction order "
                        "requires fraction."
                    ),
                )
            return max(
                1,
                int(
                    math.floor(
                        equity_at_signal()
                        * float(
                            action.fraction
                        )
                        / quoted
                    )
                ),
            )

        if mode == "target_weight":
            target = (
                action.target_weight
                if action.target_weight
                is not None
                else action.weight
            )
            if target is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Target-weight order "
                        "requires target_weight."
                    ),
                )
            delta = delta_to_target(
                symbol,
                float(
                    target
                ),
            )
            return (
                max(
                    0,
                    delta,
                )
                if side == "buy"
                else max(
                    0,
                    -delta,
                )
            )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown sizing mode: {mode}"
            ),
        )

    def execute_buy(
        symbol: str,
        quantity: int,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        quantity = int(
            quantity
        )
        if quantity <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Buy quantity resolves to zero."
                ),
            )

        is_new = (
            symbol
            not in game[
                "holdings"
            ]
        )
        if (
            is_new
            and len(
                game["holdings"]
            )
            >= game[
                "max_holdings"
            ]
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Max distinct holdings reached. "
                    "You can still add to an "
                    "existing position."
                ),
            )

        quoted = next_open(
            symbol
        )
        requested = quantity
        actual = quantity
        execution = None

        while actual > 0:
            candidate = buy_execution(
                quoted,
                actual,
                costs,
            )
            if (
                candidate[
                    "cash_out"
                ]
                <= game["cash"]
                + 1e-9
            ):
                execution = candidate
                break
            actual -= 1

        if (
            actual <= 0
            or execution is None
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Insufficient cash for even "
                    "one share after fees."
                ),
            )

        old_quantity = int(
            game[
                "holdings"
            ].get(
                symbol,
                0,
            )
        )
        old_average = float(
            game[
                "average_cost"
            ].get(
                symbol,
                0.0,
            )
        )
        new_quantity = (
            old_quantity
            + actual
        )
        total_basis = (
            old_quantity
            * old_average
            + float(
                execution[
                    "cash_out"
                ]
            )
        )

        game["cash"] -= float(
            execution[
                "cash_out"
            ]
        )
        game["fees"] += float(
            execution[
                "fees"
            ]
        )
        game["turnover"] += float(
            execution[
                "trade_value"
            ]
        )
        game["holdings"][
            symbol
        ] = new_quantity
        game["average_cost"][
            symbol
        ] = (
            total_basis
            / new_quantity
        )

        return {
            "side": "BUY",
            "symbol": symbol,
            "requested_quantity": (
                requested
            ),
            "executed_quantity": int(
                actual
            ),
            "execution_price": float(
                execution[
                    "execution_price"
                ]
            ),
            "trade_value": float(
                execution[
                    "trade_value"
                ]
            ),
            "fees": float(
                execution[
                    "fees"
                ]
            ),
            "position_quantity_after": (
                new_quantity
            ),
            "average_cost_after": float(
                game[
                    "average_cost"
                ][
                    symbol
                ]
            ),
        }

    def execute_sell(
        symbol: str,
        quantity: int,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        held = int(
            game[
                "holdings"
            ].get(
                symbol,
                0,
            )
        )
        if held <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No holding in {symbol}"
                ),
            )

        quantity = int(
            min(
                quantity,
                held,
            )
        )
        if quantity <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Sell quantity resolves to zero."
                ),
            )

        quoted = next_open(
            symbol
        )
        execution = sell_execution(
            quoted,
            quantity,
            costs,
        )
        average_cost = float(
            game[
                "average_cost"
            ].get(
                symbol,
                0.0,
            )
        )
        realized = (
            float(
                execution[
                    "cash_in"
                ]
            )
            - average_cost
            * quantity
        )

        game["cash"] += float(
            execution[
                "cash_in"
            ]
        )
        game["fees"] += float(
            execution[
                "fees"
            ]
        )
        game["turnover"] += float(
            execution[
                "trade_value"
            ]
        )
        game["realized_pnl"] += float(
            realized
        )

        remaining = (
            held
            - quantity
        )
        if remaining > 0:
            game["holdings"][
                symbol
            ] = remaining
        else:
            game["holdings"].pop(
                symbol,
                None,
            )
            game[
                "average_cost"
            ].pop(
                symbol,
                None,
            )

        return {
            "side": "SELL",
            "symbol": symbol,
            "executed_quantity": (
                quantity
            ),
            "execution_price": float(
                execution[
                    "execution_price"
                ]
            ),
            "trade_value": float(
                execution[
                    "trade_value"
                ]
            ),
            "fees": float(
                execution[
                    "fees"
                ]
            ),
            "realized_pnl": float(
                realized
            ),
            "position_quantity_after": (
                remaining
            ),
            "average_cost_after": (
                average_cost
                if remaining > 0
                else None
            ),
        }

    cash_before = float(
        game["cash"]
    )
    holdings_before = dict(
        game["holdings"]
    )
    average_cost_before = dict(
        game[
            "average_cost"
        ]
    )
    fees_before = float(
        game["fees"]
    )
    turnover_before = float(
        game.get(
            "turnover",
            0.0,
        )
    )
    realized_before = float(
        game.get(
            "realized_pnl",
            0.0,
        )
    )

    executions: list[
        dict[str, Any]
    ] = []

    try:
        if kind == "BUY":
            if not action.symbol:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "BUY requires symbol"
                    ),
                )
            executions.append(
                execute_buy(
                    action.symbol,
                    requested_quantity(
                        symbol=action.symbol,
                        side="buy",
                    ),
                )
            )

        elif kind == "SELL":
            if not action.symbol:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SELL requires symbol"
                    ),
                )
            executions.append(
                execute_sell(
                    action.symbol,
                    requested_quantity(
                        symbol=action.symbol,
                        side="sell",
                    ),
                )
            )

        elif kind == "SET_TARGET":
            if not action.symbol:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SET_TARGET requires symbol"
                    ),
                )
            target = (
                action.target_weight
                if action.target_weight
                is not None
                else action.weight
            )
            if target is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SET_TARGET requires "
                        "target_weight."
                    ),
                )
            delta = delta_to_target(
                action.symbol,
                float(
                    target
                ),
            )
            if delta > 0:
                executions.append(
                    execute_buy(
                        action.symbol,
                        delta,
                    )
                )
            elif delta < 0:
                executions.append(
                    execute_sell(
                        action.symbol,
                        -delta,
                    )
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
            source = (
                action.from_symbol.upper()
            )
            held = int(
                game[
                    "holdings"
                ].get(
                    source,
                    0,
                )
            )
            if held <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No source position in "
                        f"{source}"
                    ),
                )

            sold = execute_sell(
                source,
                held,
            )
            executions.append(
                sold
            )

            destination = (
                action.to_symbol.upper()
            )
            quoted = next_open(
                destination
            )
            redeploy = float(
                sold[
                    "trade_value"
                ]
            )
            buy_quantity = int(
                max(
                    1,
                    math.floor(
                        redeploy
                        / quoted
                    ),
                )
            )
            executions.append(
                execute_buy(
                    destination,
                    buy_quantity,
                )
            )

        elif kind == "LIQUIDATE":
            for symbol in list(
                game["holdings"]
            ):
                executions.append(
                    execute_sell(
                        symbol,
                        int(
                            game[
                                "holdings"
                            ][
                                symbol
                            ]
                        ),
                    )
                )
            game["finished"] = True

        elif kind != "HOLD":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unknown game action"
                ),
            )

    except HTTPException:
        game["cash"] = (
            cash_before
        )
        game["holdings"] = (
            holdings_before
        )
        game["average_cost"] = (
            average_cost_before
        )
        game["fees"] = (
            fees_before
        )
        game["turnover"] = (
            turnover_before
        )
        game["realized_pnl"] = (
            realized_before
        )
        raise

    game["history"].append({
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
        "sizing_mode": (
            action.sizing_mode
        ),
        "quantity": (
            action.quantity
        ),
        "amount": action.amount,
        "fraction": (
            action.fraction
        ),
        "target_weight": (
            action.target_weight
        ),
        "executions": (
            executions
        ),
        "fees": float(
            game["fees"]
            - fees_before
        ),
        "turnover": float(
            game["turnover"]
            - turnover_before
        ),
        "realized_pnl": float(
            game["realized_pnl"]
            - realized_before
        ),
    })

    game["position"] += 1

    if (
        game["position"]
        >= len(
            game["dates"]
        )
        - 1
        and kind
        != "LIQUIDATE"
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
        5000,
        ge=1,
        le=5000,
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
