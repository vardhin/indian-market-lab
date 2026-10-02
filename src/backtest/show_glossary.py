from __future__ import annotations

import argparse

from quant_metrics import glossary_frame


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Print the Indian Market Lab quant "
            "jargon glossary with one-line captions "
            "and implementation status."
        )
    )
    ap.add_argument(
        "--category",
        default=None,
        help=(
            "Optional category filter, e.g. "
            "risk_adjusted, benchmark, ml, "
            "research_design, execution."
        ),
    )
    args = ap.parse_args()

    glossary = glossary_frame()

    if args.category:
        glossary = glossary.loc[
            glossary[
                "category"
            ].eq(
                args.category
            )
        ]

    if glossary.empty:
        raise SystemExit(
            "No glossary entries matched."
        )

    current_category = None

    for row in glossary.itertuples(
        index=False
    ):
        if (
            row.category
            != current_category
        ):
            current_category = (
                row.category
            )
            print(
                "\n=== "
                + str(
                    current_category
                ).upper()
                + " ==="
            )

        print(
            f"{row.label} "
            f"[{row.status}] — "
            f"{row.caption}"
        )


if __name__ == "__main__":
    main()
