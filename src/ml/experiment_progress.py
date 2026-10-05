from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass


@dataclass
class ProgressSnapshot:
    phase: str
    completed: float
    total: float
    message: str = ""
    detail: str = ""


class ProgressReporter:
    """
    Lightweight progress protocol for both terminal runs and the FastAPI UI.

    Standalone terminal:
        draws an updating progress bar on stderr.

    FastAPI-launched run:
        server sets IML_PROGRESS_JSON=1 and receives newline-delimited
        structured records prefixed with IML_PROGRESS.
    """

    def __init__(
        self,
        *,
        phase: str,
        total: float,
        width: int = 34,
        min_interval: float = 0.25,
    ) -> None:
        self.phase = str(phase)
        self.total = max(float(total), 1.0)
        self.width = int(width)
        self.min_interval = float(min_interval)
        self.started = time.monotonic()
        self.last_emit = 0.0
        self.completed = 0.0
        self.structured = (
            os.environ.get(
                "IML_PROGRESS_JSON",
                "",
            ).strip()
            == "1"
        )

    def _payload(
        self,
        *,
        completed: float,
        message: str,
        detail: str,
    ) -> dict:
        completed = min(
            max(float(completed), 0.0),
            self.total,
        )
        elapsed = max(
            time.monotonic()
            - self.started,
            1e-9,
        )
        fraction = completed / self.total
        rate = completed / elapsed
        remaining = max(
            self.total - completed,
            0.0,
        )
        eta_seconds = (
            remaining / rate
            if rate > 0
            else None
        )

        return {
            "phase": self.phase,
            "completed": completed,
            "total": self.total,
            "fraction": fraction,
            "percent": fraction * 100.0,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "rate_per_second": rate,
            "message": str(message),
            "detail": str(detail),
        }

    @staticmethod
    def _duration(
        seconds: float | None,
    ) -> str:
        if (
            seconds is None
            or not math.isfinite(
                float(seconds)
            )
        ):
            return "--:--"

        seconds = max(
            0,
            int(seconds),
        )
        hours, remainder = divmod(
            seconds,
            3600,
        )
        minutes, secs = divmod(
            remainder,
            60,
        )
        if hours:
            return (
                f"{hours:d}:"
                f"{minutes:02d}:"
                f"{secs:02d}"
            )
        return (
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    def update(
        self,
        completed: float,
        *,
        message: str = "",
        detail: str = "",
        force: bool = False,
    ) -> None:
        self.completed = min(
            max(float(completed), 0.0),
            self.total,
        )

        now = time.monotonic()
        if (
            not force
            and self.completed
            < self.total
            and now - self.last_emit
            < self.min_interval
        ):
            return

        self.last_emit = now
        payload = self._payload(
            completed=self.completed,
            message=message,
            detail=detail,
        )

        if self.structured:
            print(
                "IML_PROGRESS "
                + json.dumps(
                    payload,
                    separators=(
                        ",",
                        ":",
                    ),
                ),
                flush=True,
            )
            return

        filled = int(
            round(
                self.width
                * payload[
                    "fraction"
                ]
            )
        )
        bar = (
            "█"
            * filled
            + "░"
            * (
                self.width
                - filled
            )
        )
        suffix = (
            f" {payload['percent']:6.2f}%"
            f"  ETA {self._duration(payload['eta_seconds'])}"
        )
        if message:
            suffix += f"  {message}"
        if detail:
            suffix += f"  {detail}"

        end = (
            "\n"
            if self.completed
            >= self.total
            else "\r"
        )

        print(
            f"[{bar}]{suffix}",
            end=end,
            file=sys.stderr,
            flush=True,
        )

    def advance(
        self,
        amount: float = 1.0,
        *,
        message: str = "",
        detail: str = "",
        force: bool = False,
    ) -> None:
        self.update(
            self.completed
            + float(amount),
            message=message,
            detail=detail,
            force=force,
        )

    def finish(
        self,
        *,
        message: str = "complete",
        detail: str = "",
    ) -> None:
        self.update(
            self.total,
            message=message,
            detail=detail,
            force=True,
        )
