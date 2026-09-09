"""Extract per-brand conversation sets from the raw TWCS table.

Runs the (expensive) union-find pass once and writes one JSONL file per
requested brand, so downstream stages never touch the 516MB raw CSV again.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .threads import assign_conversations, load_raw


def extract(csv_path: str, brands: list[str], out_dir: str, nrows: int | None = None) -> dict[str, int]:
    df = load_raw(csv_path, nrows=nrows)
    df["conversation_id"] = assign_conversations(df)
    df["created_at_ts"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )

    outbound = df[~df["inbound"]].copy()
    outbound = outbound[~outbound["author_id"].str.isdigit().fillna(False)]
    n_brands = outbound.groupby("conversation_id")["author_id"].nunique()
    single = set(n_brands[n_brands == 1].index)
    conv_brand = outbound[outbound["conversation_id"].isin(single)].groupby("conversation_id")["author_id"].first()

    wanted = conv_brand[conv_brand.isin(brands)]
    df = df[df["conversation_id"].isin(set(wanted.index))].copy()
    df["brand"] = df["conversation_id"].map(wanted)
    df = df.sort_values(["conversation_id", "created_at_ts"], kind="stable")

    os.makedirs(out_dir, exist_ok=True)
    counts: dict[str, int] = {}
    for brand, bdf in df.groupby("brand", sort=False):
        path = os.path.join(out_dir, f"{brand}.jsonl")
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for conv_id, g in bdf.groupby("conversation_id", sort=False):
                turns = [
                    {
                        "tweet_id": int(r.tweet_id),
                        "author_id": str(r.author_id),
                        "inbound": bool(r.inbound),
                        "created_at": r.created_at_ts.isoformat() if pd.notna(r.created_at_ts) else None,
                        "text": r.text if isinstance(r.text, str) else "",
                    }
                    for r in g.itertuples()
                ]
                if not any(t["inbound"] for t in turns) or not any(not t["inbound"] for t in turns):
                    continue
                fh.write(json.dumps({"conversation_id": int(conv_id), "brand": brand, "turns": turns}) + "\n")
                n += 1
        counts[brand] = n
        print(f"  {brand:<20} {n:>7} conversations -> {path}")
    return counts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/twcs.csv")
    ap.add_argument("--brands", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/interim")
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()
    extract(args.csv, args.brands, args.out_dir, nrows=args.nrows)
