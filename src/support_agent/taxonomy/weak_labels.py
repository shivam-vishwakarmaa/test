"""Rule-based weak labeller over the frozen taxonomy.

Two jobs, both deliberate:

1. **Stratification.** We need to sample a golden set that covers rare intents
   (`cancel_refund` is ~2% of traffic). Sampling uniformly at random would give
   ~4 cancellation cases in 200 -- too few to say anything. The weak labeller
   gives a cheap prior to stratify on, and the human label is free to disagree
   with it. Crucially the golden labels are assigned by a human reading the
   text, *not* by accepting these rules, so rule errors cost us sample balance,
   never label correctness.

2. **Distant supervision for the simple baseline.** These rules label all
   14,703 cases, and the TF-IDF + logistic-regression baseline trains on that.
   That baseline can therefore be built with no hand-labelled data at all,
   which is what makes it a fair "what would a competent engineer do in an
   afternoon" comparison rather than a straw man.

Precedence is explicit and ordered: the first matching rule wins. Order encodes
the operational cost of being wrong -- money and account issues are checked
before technical ones, because misrouting a billing problem to an auto-reply is
far worse than misrouting a buffering complaint.
"""
from __future__ import annotations

import re

# (intent, compiled pattern). Order == precedence.
_RULES: list[tuple[str, str]] = [
    # --- money / account first: these must never be swallowed by a technical rule
    ("cancel_refund", r"\b(?:refund|money back|reimburse|cancel(?:led|ling|lation)?\s+(?:my|the|our)\s+(?:account|subscription|sub)|already cancell?ed|still (?:being )?charg\w+ after)\b"),
    ("billing_payment", r"\b(?:charg\w+|billed|billing|double.?(?:charg|bill)\w*|payment|credit card|debit card|invoice|price|pricing|\$\d+|overcharg\w+|free trial)\b"),
    ("account_access", r"\b(?:can'?t (?:log|sign) ?in|log ?in|sign ?in|sign ?up|password|reset my|locked out|restart (?:my )?(?:subscription|account)|create an account|my account (?:won'?t|isn'?t|is not)|manage my account|home location|add(?:ing)? .{0,20}to (?:my|the) (?:hulu )?account|no option to)\b"),

    # --- live / rights-withheld before generic technical failure
    ("live_tv_blackout", r"\b(?:black(?:ed)? ?out|blackout|not available in (?:my|your) (?:area|region|country)|unavailable in my (?:area|region)|regional|local (?:channel|feed|affiliate)|in the uk|outside the us|channel lineup|(?:sec|big ten|espn|fs1|nbc|cbs|abc|fox) net)\b"),

    ("playback_error", r"\b(?:buffer\w*|freez\w+|frozen|error(?:s|ed)? ?(?:code|\d+)?|playback (?:error|failure|issue)|won'?t (?:load|play|stream|work)|not (?:load\w*|play\w*|work\w*)|isn'?t (?:load\w*|play\w*|work\w*)|doesn'?t (?:load\w*|play\w*|work\w*)|crash\w*|keeps? (?:crash|stopp|kick)\w*|black screen|spinning|lag\w*|stutter\w*|glitch\w*|no (?:sound|audio|video)|out of sync|repeat\w+ (?:over and over|the last)|clocking|not letting (?:us|me)|won'?t let (?:us|me)|can'?t (?:get|play|watch|access|use)|not able to (?:watch|play|use))\b"),

    ("content_availability", r"\b(?:when (?:will|is|does).{0,40}(?:available|come out|added|air)|new episodes?|season \d|why (?:don'?t|do not|doesn'?t) you have|add(?:ing)? .{0,25}(?:show|series|movie)|remove[d]?|took off|took down|take down|where (?:did|is|are) .{0,30}(?:go|gone|at)|request .{0,15}show|not (?:listed|streaming)|isn'?t (?:listed|streaming|on)|out of order|screwed up|why isn'?t .{0,25}(?:streaming|on hulu|listed))\b"),

    ("how_to_feature", r"\b(?:how (?:do|can|would|about) i|is there (?:a|any) way|do you (?:offer|support|have)|can i (?:download|change|turn|set|adjust)|closed caption\w*|subtitle\w*|offline|parental control|video quality|stream quality|restart option|watchlist|easy way to)\b"),

    ("ads_experience", r"\b(?:commercials?|\bads?\b|advertis\w+|ad breaks?|too many ads)\b"),

    ("app_ui_feedback", r"\b(?:new (?:interface|layout|design|app|experience|ui)|redesign|navigat\w+|hard to (?:use|browse|find)|old (?:interface|layout|app)|user experience|\bui\b|menu)\b"),
]

COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _RULES]
FALLBACK = "other_non_support"

# A message with none of these is very unlikely to be an actionable request.
_SUPPORT_HINT = re.compile(
    r"\b(?:help|issue|problem|error|why|how|can'?t|cannot|won'?t|not work\w*|broken|fix|"
    r"trouble|when|where|please|support|wtf|fail\w*|stuck|missing)\b|\?", re.IGNORECASE)


def weak_label(text: str) -> str:
    """First matching rule wins; unmatched text falls back to other_non_support."""
    for name, pat in COMPILED:
        if pat.search(text):
            return name
    return FALLBACK


def looks_like_support(text: str) -> bool:
    return bool(_SUPPORT_HINT.search(text))


def label_all(texts: list[str]) -> list[str]:
    return [weak_label(t) for t in texts]


if __name__ == "__main__":
    import argparse
    import collections
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/interim/hulu_support_cases.jsonl")
    ap.add_argument("--out", default="data/interim/hulu_weak_labels.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    counts: collections.Counter[str] = collections.Counter()
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            lab = weak_label(r["customer_text"])
            counts[lab] += 1
            fh.write(json.dumps({"case_id": r["case_id"], "weak_label": lab}) + "\n")

    total = sum(counts.values())
    print(f"weak-labelled {total} cases\n")
    for lab, n in counts.most_common():
        print(f"  {lab:<22} {n:>6}  {n / total:6.1%}")
    print(f"\nwrote {args.out}")
