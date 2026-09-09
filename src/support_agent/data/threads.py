"""Reconstruct customer-support conversations from the flat TWCS tweet table.

The raw Kaggle export (`twcs.csv`) is a flat list of tweets linked by
`in_response_to_tweet_id`. A "conversation" is a connected component of that
reply graph. We rebuild components with union-find (3M rows, single pass),
then order each component by timestamp to recover the turn sequence.

Why union-find rather than following `response_tweet_id`: that column is a
comma-separated list of *children* and is inconsistent for fan-out replies,
while `in_response_to_tweet_id` is a clean single-parent pointer. Union-find
over the parent pointer is both cheaper and more faithful.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterator

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# Twitter handles are @-mentions; customer handles in this dataset are numeric
# pseudonyms (e.g. @115712) while brand handles are real names (e.g. @AmazonHelp).
_MENTION = re.compile(r"@(\w+)")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


class UnionFind:
    """Iterative union-find with path halving. Keyed by arbitrary hashables."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.parent
        if x not in p:
            p[x] = x
            return x
        root = x
        while p[root] != root:
            p[root] = p[p[root]]  # path halving
            root = p[root]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Attach the larger id under the smaller for determinism.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


@dataclass
class Turn:
    tweet_id: int
    author_id: str
    inbound: bool
    created_at: pd.Timestamp
    text: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class Conversation:
    conversation_id: int
    brand: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def customer_opening(self) -> str | None:
        for t in self.turns:
            if t.inbound:
                return t.text
        return None

    @property
    def brand_turns(self) -> list[Turn]:
        return [t for t in self.turns if not t.inbound]

    @property
    def customer_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.inbound]

    @property
    def first_brand_reply(self) -> str | None:
        bt = self.brand_turns
        return bt[0].text if bt else None

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "n_turns": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
        }


def load_raw(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load twcs.csv with memory-conscious dtypes."""
    df = pd.read_csv(
        path,
        usecols=RAW_COLUMNS,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "bool",
            "created_at": "string",
            "text": "string",
            "response_tweet_id": "string",
            "in_response_to_tweet_id": "float64",
        },
        nrows=nrows,
    )
    return df


def assign_conversations(df: pd.DataFrame) -> pd.Series:
    """Return a Series of conversation ids aligned to `df` rows."""
    uf = UnionFind()
    tweet_ids = df["tweet_id"].to_numpy()
    parents = df["in_response_to_tweet_id"].to_numpy()

    for tid in tweet_ids:
        uf.find(int(tid))
    for tid, par in zip(tweet_ids, parents):
        if not np.isnan(par):
            uf.union(int(tid), int(par))

    roots = np.fromiter((uf.find(int(t)) for t in tweet_ids), dtype=np.int64, count=len(tweet_ids))
    return pd.Series(roots, index=df.index, name="conversation_id")


def clean_text(text: str, drop_urls: bool = True) -> str:
    """Light normalisation. We deliberately keep emoji and casing.

    Rationale: emoji and shouting carry sentiment signal that matters for the
    escalation decision ("THIS IS THE THIRD TIME 😡"), so stripping them would
    destroy exactly the feature we need. We only strip URLs (which are
    t.co-shortened and carry no recoverable content) and collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    if drop_urls:
        text = _URL.sub("", text)
    return _WS.sub(" ", text).strip()


def strip_leading_mentions(text: str) -> str:
    """Remove the leading @handle tokens Twitter prepends to replies."""
    out = text
    while True:
        m = re.match(r"^\s*@\w+\s*", out)
        if not m:
            return out.strip()
        out = out[m.end():]


def is_brand_handle(author_id: str) -> bool:
    """Customer ids in TWCS are pure digits; brand ids are alphabetic handles."""
    return not str(author_id).isdigit()


def build_conversations(df: pd.DataFrame, brand: str | None = None) -> Iterator[Conversation]:
    """Yield Conversation objects, optionally filtered to a single brand."""
    if "conversation_id" not in df.columns:
        df = df.assign(conversation_id=assign_conversations(df))

    df = df.copy()
    df["created_at_ts"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )

    for conv_id, group in df.groupby("conversation_id", sort=False):
        brands = {a for a in group.loc[~group["inbound"], "author_id"].dropna().unique()}
        brands = {b for b in brands if is_brand_handle(b)}
        if len(brands) != 1:
            # Skip conversations with zero or multiple brand participants; they
            # are either customer-only threads or cross-brand mentions, and both
            # break the "one brand owns this ticket" assumption.
            continue
        conv_brand = next(iter(brands))
        if brand is not None and conv_brand != brand:
            continue

        group = group.sort_values("created_at_ts", kind="stable")
        turns = [
            Turn(
                tweet_id=int(r.tweet_id),
                author_id=str(r.author_id),
                inbound=bool(r.inbound),
                created_at=r.created_at_ts,
                text=clean_text(r.text),
            )
            for r in group.itertuples()
        ]
        yield Conversation(conversation_id=int(conv_id), brand=conv_brand, turns=turns)
