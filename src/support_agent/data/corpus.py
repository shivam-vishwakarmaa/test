"""Turn raw brand conversations into 'cases' -- the unit the agent operates on.

A case is: one inbound customer message that opens a support thread, plus the
brand's handling of it. This is deliberately the *first* customer turn rather
than every customer turn, because that is the decision point a real helpdesk
faces: a ticket lands, and the system must classify / draft / route it before
any human has touched it.

The messy part is that many "inbound" tweets in a brand thread are not support
requests at all -- they are retweets of the brand's own marketing, quote-tweets,
or single-word noise. We filter those out explicitly and record how many we
dropped, because that number is itself a finding.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass

from .threads import strip_leading_mentions

URL_RE = re.compile(r"https?://\S+")
SIG_RE = re.compile(r"\s*\^\s*[A-Za-z]{1,3}\s*$")
WS_RE = re.compile(r"\s+")
HTML_ENT = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"}

# Openings that are the brand's own marketing copy echoed back by a retweet.
PROMO_RE = re.compile(
    r"^(watch your favorite|stream .{0,30}on hulu|get .{0,20}hulu .{0,20}free|"
    r"try hulu|sign up (?:for|to) hulu|conoce las ofertas)",
    re.IGNORECASE,
)


def unescape(t: str) -> str:
    for k, v in HTML_ENT.items():
        t = t.replace(k, v)
    return t


_DANGLING_PUNCT_RE = re.compile(r"[:\-,]\s*$")


def body_of(text: str) -> str:
    """Message content with mentions, URLs, agent signatures and entities removed.

    Removing a URL often leaves a dangling lead-in ("please call in: ") since
    the brand's original text was "...call in: <link>". We trim that trailing
    punctuation so retrieved precedents read as complete sentences instead of
    stopping mid-thought.
    """
    t = SIG_RE.sub("", URL_RE.sub("", strip_leading_mentions(unescape(text))))
    t = WS_RE.sub(" ", t).strip()
    return _DANGLING_PUNCT_RE.sub("", t).strip()


def ascii_ratio(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 1.0
    return sum(c.isascii() for c in letters) / len(letters)


@dataclass
class Case:
    case_id: int
    brand: str
    created_at: str | None
    customer_text: str          # cleaned opening, what the model sees
    customer_text_raw: str
    brand_reply: str            # first brand reply, cleaned body
    brand_reply_raw: str
    brand_reply_has_link: bool
    n_turns: int
    n_customer_turns: int
    n_brand_turns: int
    customer_followups: list[str]
    thanked: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


THANKS_RE = re.compile(
    r"\b(?:thank you|thanks|thank u|thx|appreciate it|sorted|fixed|that worked|"
    r"it works|perfect|great, thanks|cheers)\b", re.IGNORECASE)


def build_cases(
    jsonl_path: str,
    min_words: int = 4,
    min_ascii: float = 0.85,
) -> tuple[list[Case], dict[str, int]]:
    cases: list[Case] = []
    stats = {
        "total": 0, "no_customer_turn": 0, "no_brand_reply": 0, "promo": 0,
        "too_short": 0, "non_english": 0, "duplicate": 0, "kept": 0,
    }
    seen: set[str] = set()

    for line in open(jsonl_path, encoding="utf-8"):
        conv = json.loads(line)
        stats["total"] += 1
        turns = conv["turns"]
        cust = [t for t in turns if t["inbound"]]
        brand_turns = [t for t in turns if not t["inbound"]]
        if not cust:
            stats["no_customer_turn"] += 1
            continue
        if not brand_turns:
            stats["no_brand_reply"] += 1
            continue

        opening_raw = cust[0]["text"]
        opening = body_of(opening_raw)
        if PROMO_RE.match(opening):
            stats["promo"] += 1
            continue
        if len(opening.split()) < min_words:
            stats["too_short"] += 1
            continue
        if ascii_ratio(opening) < min_ascii:
            stats["non_english"] += 1
            continue

        key = WS_RE.sub(" ", opening.lower())[:160]
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)

        reply_raw = brand_turns[0]["text"]
        followups = [body_of(t["text"]) for t in cust[1:]]
        cases.append(Case(
            case_id=conv["conversation_id"],
            brand=conv["brand"],
            created_at=cust[0].get("created_at"),
            customer_text=opening,
            customer_text_raw=opening_raw,
            brand_reply=body_of(reply_raw),
            brand_reply_raw=reply_raw,
            brand_reply_has_link=bool(URL_RE.search(reply_raw)),
            n_turns=len(turns),
            n_customer_turns=len(cust),
            n_brand_turns=len(brand_turns),
            customer_followups=followups,
            thanked=any(THANKS_RE.search(f) for f in followups),
        ))
        stats["kept"] += 1

    return cases, stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="hulu_support")
    ap.add_argument("--in-dir", default="data/interim")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = os.path.join(args.in_dir, f"{args.brand}.jsonl")
    out = args.out or os.path.join(args.in_dir, f"{args.brand}_cases.jsonl")
    cases, stats = build_cases(src)
    with open(out, "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(c.to_json() + "\n")
    print(f"filter funnel for {args.brand}:")
    for k, v in stats.items():
        pct = f"{v / max(stats['total'], 1):6.1%}"
        print(f"  {k:<18} {v:>7}  {pct}")
    print(f"\nwrote {out}")
