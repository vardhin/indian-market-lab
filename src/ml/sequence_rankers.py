from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from classical import (
    FEATURE_COLUMNS,
    HORIZON,
    TRAINING_FLAG,
    TRAIN_START,
    cross_sectional_diagnostics,
)


@dataclass
class SequenceConfig:
    lookback: int = 60
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.10
    epochs: int = 6
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_train_sequences: int = 300_000


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is optional. Install "
            "the deep tournament extras with: "
            "uv sync --extra deep"
        ) from exc
    return torch


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def build_histories(
    df: pd.DataFrame,
) -> list[np.ndarray]:
    histories: list[
        np.ndarray
    ] = []

    grouped = df.groupby(
        "canonical_security_id",
        sort=False,
    ).indices

    for rows in grouped.values():
        arr = np.asarray(
            rows,
            dtype=np.int64,
        )
        arr = arr[
            np.argsort(
                df.loc[
                    arr,
                    "market_day_index",
                ].to_numpy(
                    dtype=np.int64
                ),
                kind="stable",
            )
        ]
        histories.append(
            arr
        )

    return histories


def sequence_samples(
    histories: list[
        np.ndarray
    ],
    market_index: np.ndarray,
    mask: np.ndarray,
    *,
    lookback: int,
    max_samples: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    groups: list[
        np.ndarray
    ] = []
    positions: list[
        np.ndarray
    ] = []
    rows_out: list[
        np.ndarray
    ] = []

    for group_id, rows in enumerate(
        histories
    ):
        selected = np.flatnonzero(
            mask[
                rows
            ]
        )
        selected = selected[
            selected
            >= (
                lookback - 1
            )
        ]
        if not len(selected):
            continue

        end_rows = rows[
            selected
        ]
        start_rows = rows[
            selected
            - lookback
            + 1
        ]

        continuous = (
            market_index[
                end_rows
            ]
            - market_index[
                start_rows
            ]
            == (
                lookback - 1
            )
        )
        selected = selected[
            continuous
        ]
        if not len(selected):
            continue

        groups.append(
            np.full(
                len(selected),
                group_id,
                dtype=np.int32,
            )
        )
        positions.append(
            selected.astype(
                np.int32
            )
        )
        rows_out.append(
            rows[
                selected
            ].astype(
                np.int64
            )
        )

    if not groups:
        return (
            np.array(
                [],
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.int64,
            ),
        )

    group_ids = np.concatenate(
        groups
    )
    group_pos = np.concatenate(
        positions
    )
    current_rows = np.concatenate(
        rows_out
    )

    if (
        max_samples > 0
        and len(group_ids)
        > max_samples
    ):
        rng = np.random.default_rng(
            seed
        )
        take = rng.choice(
            len(group_ids),
            size=max_samples,
            replace=False,
        )
        take.sort()
        group_ids = group_ids[
            take
        ]
        group_pos = group_pos[
            take
        ]
        current_rows = current_rows[
            take
        ]

    return (
        group_ids,
        group_pos,
        current_rows,
    )


def normalization_stats(
    features: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    values = features[
        train_mask
    ].astype(
        np.float64,
        copy=False,
    )
    mean = np.nanmean(
        values,
        axis=0,
    )
    std = np.nanstd(
        values,
        axis=0,
    )

    mean = np.where(
        np.isfinite(mean),
        mean,
        0.0,
    )
    std = np.where(
        np.isfinite(std)
        & (
            std > 1e-8
        ),
        std,
        1.0,
    )

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


class SequenceDataset:
    def __init__(
        self,
        *,
        normalized_features: np.ndarray,
        targets: np.ndarray,
        histories: list[
            np.ndarray
        ],
        group_ids: np.ndarray,
        positions: np.ndarray,
        current_rows: np.ndarray,
        lookback: int,
        with_target: bool,
    ) -> None:
        self.features = (
            normalized_features
        )
        self.targets = targets
        self.histories = histories
        self.group_ids = group_ids
        self.positions = positions
        self.current_rows = (
            current_rows
        )
        self.lookback = int(
            lookback
        )
        self.with_target = bool(
            with_target
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.group_ids
        )

    def __getitem__(
        self,
        index: int,
    ):
        torch = require_torch()

        group_id = int(
            self.group_ids[
                index
            ]
        )
        position = int(
            self.positions[
                index
            ]
        )
        rows = self.histories[
            group_id
        ][
            position
            - self.lookback
            + 1:
            position
            + 1
        ]

        x = torch.from_numpy(
            self.features[
                rows
            ]
        )

        if not self.with_target:
            return x

        row = int(
            self.current_rows[
                index
            ]
        )
        y = torch.tensor(
            self.targets[
                row
            ],
            dtype=torch.float32,
        )
        return (
            x,
            y,
        )


def make_model(
    name: str,
    *,
    n_features: int,
    config: SequenceConfig,
):
    torch = require_torch()
    nn = torch.nn

    class SeqMLP(
        nn.Module
    ):
        def __init__(
            self,
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    config.lookback
                    * n_features,
                    512,
                ),
                nn.GELU(),
                nn.Dropout(
                    config.dropout
                ),
                nn.Linear(
                    512,
                    256,
                ),
                nn.GELU(),
                nn.Dropout(
                    config.dropout
                ),
                nn.Linear(
                    256,
                    1,
                ),
            )

        def forward(
            self,
            x,
        ):
            return (
                self.net(x)
                .squeeze(-1)
            )

    class RecurrentRanker(
        nn.Module
    ):
        def __init__(
            self,
            kind: str,
        ):
            super().__init__()
            cls = (
                nn.LSTM
                if kind == "lstm"
                else nn.GRU
            )
            self.rnn = cls(
                input_size=(
                    n_features
                ),
                hidden_size=(
                    config.hidden_size
                ),
                num_layers=(
                    config.layers
                ),
                batch_first=True,
                dropout=(
                    config.dropout
                    if config.layers > 1
                    else 0.0
                ),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(
                    config.hidden_size
                ),
                nn.Linear(
                    config.hidden_size,
                    1,
                ),
            )

        def forward(
            self,
            x,
        ):
            out, _ = self.rnn(
                x
            )
            return (
                self.head(
                    out[
                        :,
                        -1,
                    ]
                )
                .squeeze(-1)
            )

    class TCNRanker(
        nn.Module
    ):
        def __init__(
            self,
        ):
            super().__init__()
            channels = [
                config.hidden_size,
                config.hidden_size,
                config.hidden_size,
            ]
            blocks = []
            input_channels = (
                n_features
            )
            for i, output_channels in enumerate(
                channels
            ):
                dilation = 2 ** i
                padding = dilation * 2
                blocks.extend([
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        config.dropout
                    ),
                ])
                input_channels = (
                    output_channels
                )
            self.net = nn.Sequential(
                *blocks
            )
            self.head = nn.Linear(
                config.hidden_size,
                1,
            )

        def forward(
            self,
            x,
        ):
            z = x.transpose(
                1,
                2,
            )
            z = self.net(
                z
            )
            # Symmetric padding grows the sequence.
            z = z[
                :,
                :,
                :x.shape[1],
            ]
            z = z[
                :,
                :,
                -1,
            ]
            return (
                self.head(z)
                .squeeze(-1)
            )

    class TransformerRanker(
        nn.Module
    ):
        def __init__(
            self,
        ):
            super().__init__()
            d_model = (
                config.hidden_size
            )
            n_heads = 4
            self.proj = nn.Linear(
                n_features,
                d_model,
            )
            self.pos = nn.Parameter(
                torch.zeros(
                    1,
                    config.lookback,
                    d_model,
                )
            )
            encoder_layer = (
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=(
                        d_model
                        * 4
                    ),
                    dropout=(
                        config.dropout
                    ),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            self.encoder = (
                nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=(
                        config.layers
                    ),
                )
            )
            self.head = nn.Sequential(
                nn.LayerNorm(
                    d_model
                ),
                nn.Linear(
                    d_model,
                    1,
                ),
            )

        def forward(
            self,
            x,
        ):
            z = (
                self.proj(x)
                + self.pos[
                    :,
                    :x.shape[1],
                ]
            )
            z = self.encoder(
                z
            )
            return (
                self.head(
                    z[
                        :,
                        -1,
                    ]
                )
                .squeeze(-1)
            )

    class PatchTransformerRanker(
        nn.Module
    ):
        def __init__(
            self,
        ):
            super().__init__()
            self.patch_len = 10
            self.stride = 10
            self.patch_count = (
                config.lookback
                // self.stride
            )
            d_model = (
                config.hidden_size
            )
            self.patch_proj = (
                nn.Linear(
                    self.patch_len
                    * n_features,
                    d_model,
                )
            )
            self.pos = nn.Parameter(
                torch.zeros(
                    1,
                    self.patch_count,
                    d_model,
                )
            )
            layer = (
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=(
                        d_model
                        * 4
                    ),
                    dropout=(
                        config.dropout
                    ),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            self.encoder = (
                nn.TransformerEncoder(
                    layer,
                    num_layers=(
                        config.layers
                    ),
                )
            )
            self.head = nn.Sequential(
                nn.LayerNorm(
                    d_model
                ),
                nn.Linear(
                    d_model,
                    1,
                ),
            )

        def forward(
            self,
            x,
        ):
            usable = (
                self.patch_count
                * self.patch_len
            )
            z = x[
                :,
                -usable:,
            ]
            z = z.reshape(
                z.shape[0],
                self.patch_count,
                self.patch_len
                * n_features,
            )
            z = (
                self.patch_proj(z)
                + self.pos
            )
            z = self.encoder(
                z
            )
            return (
                self.head(
                    z.mean(
                        dim=1
                    )
                )
                .squeeze(-1)
            )

    class ITransformerRanker(
        nn.Module
    ):
        def __init__(
            self,
        ):
            super().__init__()
            d_model = (
                config.hidden_size
            )
            self.time_proj = (
                nn.Linear(
                    config.lookback,
                    d_model,
                )
            )
            layer = (
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=(
                        d_model
                        * 4
                    ),
                    dropout=(
                        config.dropout
                    ),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            self.encoder = (
                nn.TransformerEncoder(
                    layer,
                    num_layers=(
                        config.layers
                    ),
                )
            )
            self.head = nn.Sequential(
                nn.LayerNorm(
                    d_model
                ),
                nn.Linear(
                    d_model,
                    1,
                ),
            )

        def forward(
            self,
            x,
        ):
            # Variables become tokens; each token embeds
            # the full temporal trace for that feature.
            z = x.transpose(
                1,
                2,
            )
            z = self.time_proj(
                z
            )
            z = self.encoder(
                z
            )
            return (
                self.head(
                    z.mean(
                        dim=1
                    )
                )
                .squeeze(-1)
            )

    class NHITSStyleRanker(
        nn.Module
    ):
        def __init__(
            self,
        ):
            super().__init__()
            self.pool_sizes = [
                1,
                2,
                4,
                8,
            ]
            branch_hidden = max(
                32,
                config.hidden_size
                // 2,
            )
            self.branches = (
                nn.ModuleList()
            )
            for pool_size in (
                self.pool_sizes
            ):
                length = int(
                    math.ceil(
                        config.lookback
                        / pool_size
                    )
                )
                self.branches.append(
                    nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(
                            length
                            * n_features,
                            branch_hidden,
                        ),
                        nn.GELU(),
                        nn.Dropout(
                            config.dropout
                        ),
                        nn.Linear(
                            branch_hidden,
                            branch_hidden,
                        ),
                        nn.GELU(),
                    )
                )
            self.head = nn.Sequential(
                nn.Linear(
                    branch_hidden
                    * len(
                        self.pool_sizes
                    ),
                    config.hidden_size,
                ),
                nn.GELU(),
                nn.Dropout(
                    config.dropout
                ),
                nn.Linear(
                    config.hidden_size,
                    1,
                ),
            )

        def forward(
            self,
            x,
        ):
            outs = []
            for pool_size, branch in zip(
                self.pool_sizes,
                self.branches,
            ):
                if pool_size == 1:
                    pooled = x
                else:
                    # Pool across temporal resolution.
                    z = x.transpose(
                        1,
                        2,
                    )
                    z = (
                        nn.functional.avg_pool1d(
                            z,
                            kernel_size=(
                                pool_size
                            ),
                            stride=(
                                pool_size
                            ),
                            ceil_mode=True,
                        )
                    )
                    pooled = z.transpose(
                        1,
                        2,
                    )
                outs.append(
                    branch(
                        pooled
                    )
                )
            return (
                self.head(
                    torch.cat(
                        outs,
                        dim=1,
                    )
                )
                .squeeze(-1)
            )

    class MambaRanker(
        nn.Module
    ):
        def __init__(
            self,
        ):
            super().__init__()
            try:
                from mamba_ssm import (
                    Mamba,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "mamba_ssm is optional and "
                    "requires a compatible CUDA/"
                    "compiler environment. Install "
                    "with: uv sync --extra deep "
                    "--extra mamba"
                ) from exc

            d_model = (
                config.hidden_size
            )
            self.proj = nn.Linear(
                n_features,
                d_model,
            )
            self.blocks = (
                nn.ModuleList([
                    Mamba(
                        d_model=d_model,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    )
                    for _ in range(
                        config.layers
                    )
                ])
            )
            self.norms = (
                nn.ModuleList([
                    nn.LayerNorm(
                        d_model
                    )
                    for _ in range(
                        config.layers
                    )
                ])
            )
            self.head = nn.Linear(
                d_model,
                1,
            )

        def forward(
            self,
            x,
        ):
            z = self.proj(
                x
            )
            for block, norm in zip(
                self.blocks,
                self.norms,
            ):
                z = (
                    z
                    + block(
                        norm(z)
                    )
                )
            return (
                self.head(
                    z[
                        :,
                        -1,
                    ]
                )
                .squeeze(-1)
            )

    if name == "seq_mlp":
        return SeqMLP()
    if name == "lstm":
        return RecurrentRanker(
            "lstm"
        )
    if name == "gru":
        return RecurrentRanker(
            "gru"
        )
    if name == "tcn":
        return TCNRanker()
    if name == "transformer":
        return TransformerRanker()
    if name == "patch_transformer":
        return (
            PatchTransformerRanker()
        )
    if name == "itransformer":
        return ITransformerRanker()
    if name == "nhits_style":
        return NHITSStyleRanker()
    if name == "mamba":
        return MambaRanker()

    raise ValueError(
        f"Unknown sequence model: {name}"
    )


def train_model(
    model,
    train_dataset: SequenceDataset,
    *,
    config: SequenceConfig,
    device,
) -> None:
    torch = require_torch()

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=(
            config.batch_size
        ),
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    model.to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            config.learning_rate
        ),
        weight_decay=(
            config.weight_decay
        ),
    )
    loss_fn = (
        torch.nn.MSELoss()
    )

    for epoch in range(
        config.epochs
    ):
        model.train()
        losses = []

        for x, y in loader:
            x = x.to(
                device,
                non_blocking=True,
            )
            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )
            pred = model(
                x
            )
            loss = loss_fn(
                pred,
                y,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            losses.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

        mean_loss = (
            float(
                np.mean(
                    losses
                )
            )
            if losses
            else float("nan")
        )
        print(
            f"    epoch "
            f"{epoch + 1}/"
            f"{config.epochs} "
            f"loss={mean_loss:.6f}"
        )


def predict_model(
    model,
    dataset: SequenceDataset,
    *,
    config: SequenceConfig,
    device,
) -> np.ndarray:
    torch = require_torch()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=(
            config.batch_size
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    model.eval()
    output: list[
        np.ndarray
    ] = []

    with torch.no_grad():
        for x in loader:
            x = x.to(
                device,
                non_blocking=True,
            )
            pred = (
                model(x)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            output.append(
                pred
            )

    if not output:
        return np.array(
            [],
            dtype=float,
        )

    return np.concatenate(
        output
    ).astype(
        float,
        copy=False,
    )


def prediction_path(
    root: Path,
    *,
    model_name: str,
    seed: int,
    config: SequenceConfig,
    year: int,
) -> Path:
    n = int(
        config.max_train_sequences
    )
    key = (
        f"{model_name}_rs{seed}_"
        f"l{config.lookback}_"
        f"e{config.epochs}_"
        f"n{n}"
    )
    return (
        root
        / "reports/ml/model_tournament/"
        "predictions_temporal"
        / key
        / f"{year}.parquet"
    )


def annual_sequence_predictions(
    df: pd.DataFrame,
    *,
    root: Path,
    model_name: str,
    years: list[int],
    seed: int,
    config: SequenceConfig,
    rebuild: bool,
) -> pd.DataFrame:
    torch = require_torch()
    set_seed(
        seed
    )

    features = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )
    targets = (
        pd.to_numeric(
            df[
                "target_rank_20d"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )
    market_index = (
        df[
            "market_day_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )
    histories = build_histories(
        df
    )

    df[
        "persistent_score"
    ] = np.nan

    diagnostics_rows: list[
        dict
    ] = []

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(
        "Temporal device: "
        f"{device}"
    )

    for year in years:
        path = prediction_path(
            root,
            model_name=(
                model_name
            ),
            seed=seed,
            config=config,
            year=year,
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        score_mask = (
            df[
                "eligible_universe"
            ]
            & df[
                "date"
            ].dt.year.eq(
                year
            )
        )

        if (
            path.is_file()
            and not rebuild
        ):
            saved = pd.read_parquet(
                path
            )
            keys = [
                "date",
                "canonical_security_id",
            ]
            saved[
                "date"
            ] = pd.to_datetime(
                saved[
                    "date"
                ],
                errors="coerce",
            ).dt.normalize()
            mapping = (
                saved.set_index(
                    keys
                )[
                    "persistent_score"
                ]
            )
            idx = (
                pd.MultiIndex
                .from_frame(
                    df.loc[
                        score_mask,
                        keys,
                    ]
                )
            )
            values = (
                mapping.reindex(
                    idx
                )
                .to_numpy(
                    dtype=float
                )
            )
            df.loc[
                score_mask,
                "persistent_score",
            ] = values
            coverage = float(
                np.isfinite(
                    values
                ).mean()
            )
            print(
                f"OOS {year}: reused "
                f"temporal scores; "
                f"coverage={coverage:.2%}"
            )
        else:
            fold_start = pd.Timestamp(
                f"{year}-01-01"
            )
            prior = df.loc[
                df["date"].lt(
                    fold_start
                ),
                "market_day_index",
            ].dropna()
            if prior.empty:
                raise RuntimeError(
                    f"No prior sessions "
                    f"for year {year}."
                )
            cutoff = int(
                prior.max()
            )

            train_mask = (
                df[
                    TRAINING_FLAG
                ]
                & df[
                    "target_rank_20d"
                ].notna()
                & df[
                    "date"
                ].ge(
                    TRAIN_START
                )
                & df[
                    "market_day_index"
                ].le(
                    cutoff
                    - HORIZON
                )
            )

            train_bool = (
                train_mask
                .to_numpy(
                    dtype=bool
                )
            )
            score_bool = (
                score_mask
                .to_numpy(
                    dtype=bool
                )
            )

            (
                train_groups,
                train_pos,
                train_rows,
            ) = sequence_samples(
                histories,
                market_index,
                train_bool,
                lookback=(
                    config.lookback
                ),
                max_samples=(
                    config.max_train_sequences
                ),
                seed=(
                    seed + year
                ),
            )
            (
                score_groups,
                score_pos,
                score_rows,
            ) = sequence_samples(
                histories,
                market_index,
                score_bool,
                lookback=(
                    config.lookback
                ),
                max_samples=0,
                seed=seed,
            )

            print(
                f"OOS {year}: "
                f"train_seq="
                f"{len(train_rows):,}; "
                f"score_seq="
                f"{len(score_rows):,}"
            )

            mean, std = (
                normalization_stats(
                    features,
                    train_bool,
                )
            )
            normalized = (
                features
                - mean
            ) / std
            normalized = (
                np.nan_to_num(
                    normalized,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            train_dataset = (
                SequenceDataset(
                    normalized_features=(
                        normalized
                    ),
                    targets=targets,
                    histories=(
                        histories
                    ),
                    group_ids=(
                        train_groups
                    ),
                    positions=(
                        train_pos
                    ),
                    current_rows=(
                        train_rows
                    ),
                    lookback=(
                        config.lookback
                    ),
                    with_target=True,
                )
            )
            score_dataset = (
                SequenceDataset(
                    normalized_features=(
                        normalized
                    ),
                    targets=targets,
                    histories=(
                        histories
                    ),
                    group_ids=(
                        score_groups
                    ),
                    positions=(
                        score_pos
                    ),
                    current_rows=(
                        score_rows
                    ),
                    lookback=(
                        config.lookback
                    ),
                    with_target=False,
                )
            )

            model = make_model(
                model_name,
                n_features=len(
                    FEATURE_COLUMNS
                ),
                config=config,
            )
            train_model(
                model,
                train_dataset,
                config=config,
                device=device,
            )
            scores = predict_model(
                model,
                score_dataset,
                config=config,
                device=device,
            )

            df.loc[
                score_rows,
                "persistent_score",
            ] = scores

            saved = df.loc[
                score_rows,
                [
                    "date",
                    "canonical_security_id",
                ],
            ].copy()
            saved[
                "persistent_score"
            ] = scores
            saved.to_parquet(
                path,
                index=False,
                compression="zstd",
            )

            coverage = (
                len(
                    score_rows
                )
                / max(
                    int(
                        score_mask.sum()
                    ),
                    1,
                )
            )
            print(
                f"OOS {year}: "
                f"coverage={coverage:.2%}"
            )

            del model
            del normalized
            if (
                device.type
                == "cuda"
            ):
                torch.cuda.empty_cache()

        label_mask = (
            df[
                TRAINING_FLAG
            ]
            & df[
                "target_rank_20d"
            ].notna()
            & df[
                "date"
            ].dt.year.eq(
                year
            )
            & df[
                "persistent_score"
            ].notna()
        )
        if label_mask.any():
            label_scores = (
                df.loc[
                    label_mask,
                    "persistent_score",
                ]
                .to_numpy(
                    dtype=float
                )
            )
            diagnostics = (
                cross_sectional_diagnostics(
                    df,
                    label_mask,
                    label_scores,
                )
            )
        else:
            diagnostics = {
                "rows": 0,
                "dates": 0,
                "mean_daily_ic": None,
                "median_daily_ic": None,
                "std_daily_ic": None,
                "icir": None,
                "positive_ic_fraction": None,
                "mean_top_decile_return": None,
                "mean_universe_return": None,
                "mean_top_decile_excess": None,
            }

        diagnostics_rows.append({
            "year": int(
                year
            ),
            "model_name": (
                model_name
            ),
            **diagnostics,
        })

    add_daily_ranks(
        df
    )

    return pd.DataFrame(
        diagnostics_rows
    )
