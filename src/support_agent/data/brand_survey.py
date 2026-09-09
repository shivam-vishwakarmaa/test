"""Rank brands by how *learnable* their support behaviour is.

Brand choice is the single highest-leverage decision in this assignment: it
determines whether reply-drafting has anything to ground in. Many large
handles (AppleSupport, sprintcare) resolve almost nothing in-channel -- their
modal reply is "please DM us". A retrieval-grounded drafter trained on that
brand learns to say "please DM us", which is a useless product and an
unfalsifiable evaluation.

So we score brands on a *deflection rate* and an *in-channel resolution proxy*
in addition to raw volume, and pick on that evidence rather than on size.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .threads import assign_conversations, load_raw

# Phrases whose presence in a brand's FIRST reply means the public thread is
# being closed and moved to a private channel -- i.e. no resolution is visible
# in the data, so there is nothing for a drafter to learn from.
DEFLECTION_PATTERNS = [
    r"\bDM\b", r"\bD\.M\b", r"direct message", r"private message", r"\bPM us\b",
    r"send us a (?:private |direct )?message", r"shoot us a", r"follow (?:us )?(?:and|&|then) ",
    r"click .{0,20}message", r"message us", r"reach out .{0,20}(?:privately|via DM)",
    r"provide .{0,30}(?:email|phone number|account number)",
    r"call us", r"give us a call", r"contact .{0,20}(?:support|us) at",
]
DEFLECTION_RE = re.compile("|".join(DEFLECTION_PATTERNS), re.IGNORECASE)

# Customer gratitude in a later turn is a weak but real signal that the brand's
# public reply actually helped.
THANKS_RE = re.compile(
    r"\b(?:thank you|thanks|thank u|thx|ty!|appreciate it|sorted|fixed now|worked|"
    r"that worked|got it|perfect|awesome, thanks|cheers)\b",
    re.IGNORECASE,
)

# A substantive reply contains instruction/answer verbs rather than pure apology.
SUBSTANTIVE_RE = re.compile(
    r"\b(?:you can|you'll|try|go to|tap|click|select|open|settings|make sure|check|"
    r"ensure|here's|here is|we've|we have|refund|credit|delivered|shipped|arrive|"
    r"reset|update|enable|disable|toggle|restart|because|due to|means that)\b",
    re.IGNORECASE,
)


def survey(csv_path: str, min_conversations: int = 2000, nrows: int | None = None) -> pd.DataFrame:
    """Return a per-brand table of volume and learnability metrics."""
    df = load_raw(csv_path, nrows=nrows)
    df["conversation_id"] = assign_conversations(df)
    df["created_at_ts"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )
    df = df.sort_values("created_at_ts", kind="stable")

    outbound = df[~df["inbound"]].copy()
    # Brand handles are non-numeric; drop numeric authors misflagged as outbound.
    outbound = outbound[~outbound["author_id"].str.isdigit().fillna(False)]

    # A conversation belongs to a brand only if exactly one brand speaks in it.
    brands_per_conv = outbound.groupby("conversation_id")["author_id"].nunique()
    single_brand = brands_per_conv[brands_per_conv == 1].index
    outbound = outbound[outbound["conversation_id"].isin(single_brand)]

    conv_brand = outbound.groupby("conversation_id")["author_id"].first()

    # ---- first brand reply per conversation -------------------------------
    first_reply = outbound.groupby("conversation_id")["text"].first()
    n_brand_turns = outbound.groupby("conversation_id").size()

    # ---- customer side ----------------------------------------------------
    inbound = df[df["inbound"] & df["conversation_id"].isin(single_brand)]
    opening = inbound.groupby("conversation_id")["text"].first()
    last_customer = inbound.groupby("conversation_id")["text"].last()
    n_cust_turns = inbound.groupby("conversation_id").size()

    frame = pd.DataFrame({
        "brand": conv_brand,
        "first_reply": first_reply,
        "n_brand_turns": n_brand_turns,
        "opening": opening,
        "last_customer": last_customer,
        "n_cust_turns": n_cust_turns,
    }).dropna(subset=["brand", "opening", "first_reply"])

    frame["is_deflection"] = frame["first_reply"].str.contains(DEFLECTION_RE, na=False)
    frame["is_substantive"] = frame["first_reply"].str.contains(SUBSTANTIVE_RE, na=False)
    frame["customer_thanked"] = frame["last_customer"].str.contains(THANKS_RE, na=False)
    # Resolution proxy: brand gave a substantive, non-deflecting first reply AND
    # the customer's final turn reads as gratitude.
    frame["resolved_proxy"] = (
        frame["is_substantive"] & ~frame["is_deflection"] & frame["customer_thanked"]
    )
    frame["opening_words"] = frame["opening"].str.split().str.len()

    agg = frame.groupby("brand").agg(
        n_conversations=("brand", "size"),
        deflection_rate=("is_deflection", "mean"),
        substantive_rate=("is_substantive", "mean"),
        thanks_rate=("customer_thanked", "mean"),
        resolution_proxy=("resolved_proxy", "mean"),
        median_brand_turns=("n_brand_turns", "median"),
        median_cust_turns=("n_cust_turns", "median"),
        median_opening_words=("opening_words", "median"),
    )
    agg = agg[agg["n_conversations"] >= min_conversations]

    # Learnability score: we want volume, in-channel resolution, and substance;
    # we penalise deflection. Volume is log-scaled because past ~20k threads the
    # marginal value of more data is small relative to resolution quality.
    agg["learnability"] = (
        np.log10(agg["n_conversations"]).clip(upper=5.0) / 5.0 * 0.30
        + agg["resolution_proxy"] * 0.35
        + agg["substantive_rate"] * 0.20
        + (1 - agg["deflection_rate"]) * 0.15
    )
    return agg.sort_values("learnability", ascending=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/twcs.csv")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--out", default="artifacts/brand_survey.csv")
    args = ap.parse_args()

    table = survey(args.csv, nrows=args.nrows)
    table.to_csv(args.out)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(table.head(25).round(4).to_string())
    print(f"\nwrote {args.out}  ({len(table)} brands)")
