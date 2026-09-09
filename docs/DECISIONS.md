# Decision log

Non-obvious calls made while building this, and why. Numbered for cross-reference
from code comments and `docs/REPORT.md`.

### 1. Brand: `hulu_support`, chosen by a computed "learnability" score, not by volume.

`support_agent/data/brand_survey.py` scores all 65 brands with
≥2,000 threads on volume, a *deflection rate* (does the first brand reply
hand off to DM/phone/link with no visible resolution?), a *substantive-reply
rate*, and a weak *resolution proxy* (substantive + non-deflecting reply +
customer said thanks in a later turn). AmazonHelp and AppleSupport have far
higher volume but AppleSupport deflects 52% of first replies and Amazon's
public replies are thin (many just route to a link); Hulu deflects only 5.4%
(after correcting the naive regex — see #2) and gives long, specific,
diagnostic replies in public. A retrieval-grounded drafter needs the brand's
*history* to contain real answers; volume without resolvable content is a
worse foundation than a smaller brand that actually answers in-channel. Why
this matters: an evaluator handed a brand pre-selected for volume alone would
mostly be grounding a bot on "please DM us," which is a real and instructive
failure mode but not a system worth shipping.

### 2. The first deflection-rate regex was wrong, and I kept the evidence of that in the repo instead of quietly fixing it.

The v1 regex looked for
DM/phone/email hand-offs and reported Hulu at 1.3% deflection. Reading actual
threads showed Hulu's dominant deflection pattern is a **help-article link**
("try these steps: <t.co link>"), present in 48–57% of first replies across
candidate brands — a completely different failure mode the v1 regex was blind
to. `brand_survey.py`'s final regex includes both families. This is the
single clearest illustration in the whole project of why "read 50 real
examples before trusting a metric" isn't optional.

### 3. Thread reconstruction is union-find over `in_response_to_tweet_id`, not `response_tweet_id`.

The latter is a comma-separated list of children and is
inconsistent under fan-out replies (a customer tweeting `@brand` three times
before getting one reply creates ambiguous chains); the former is a clean
single-parent pointer, so union-find over it is both cheaper (`O(n α(n))`
over 3M rows) and more faithful to what actually happened in the thread.

### 4. A "case" is the customer's OPENING message, not every customer turn.

This matches the actual decision point a helpdesk system faces (a ticket
lands; something must classify/draft/route it before a human has touched it),
and avoids the leakage of training or evaluating on later turns that already
contain the brand's diagnostic follow-up questions as context the customer
didn't have when they first wrote in.

### 5. URLs are stripped from customer and brand text, but emoji and shouting are kept.

Emoji and all-caps carry exactly the sentiment signal the `angry`
flag and the escalation policy depend on ("THIS IS THE THIRD TIME 😡" is
different from the same sentence typed calmly); URLs are t.co-shortened and
carry no recoverable content, so stripping them is pure noise reduction, not
information loss.

### 6. Classification, retrieval, drafting, and triage are four separate functions/LLM calls, not one mega-prompt.

A single "do everything" prompt
would make it impossible to separately evaluate where an error came from —
was a case escalated because the classifier got the intent wrong, or because
the drafter couldn't ground an answer? The triage decision in particular is
implemented as a **plain deterministic rule** over the classifier's own
outputs (`support_agent/agent/pipeline.py::triage`), not a further LLM call:
the thing that decides whether a human ever sees a case has to be auditable
as a rule with named reasons, not another model's opinion we'd then have to
trust recursively.

### 7. Retrieval excludes "deflection" replies from the retrievable pool (`min_reply_words=8`).

A precedent whose entire content is a link hand-off
grounds nothing (the linked page's content isn't in our data) and, if kept,
teaches the drafter to always recommend a DM — which is measurably close to
this brand's raw historical behavior and exactly the failure mode the
assignment's framing warns about. Filtering is what makes retrieval-grounded
drafting worth doing at all for a brand like this one.

### 8. The taxonomy is organized by *handling policy*, not by topic granularity.

Each of the 10 intents maps to exactly one of six dispositions
(`auto_answer`, `policy_answer`, `diagnose`, `acknowledge`, `escalate`,
`no_action`) in `config/taxonomy.yaml`. A taxonomy that split "buffering" from
"freezing" would be descriptively finer but operationally identical — both
get the same reply and the same routing. Labels earn their existence by
changing what the agent does, which is also why `cancel_refund` is kept
separate from `billing_payment` (different urgency and different verification
needs) while several UI complaints are merged into one `app_ui_feedback`.

### 9. The embedding cache is keyed per-text (content hash), not per-batch.

An
earlier version hashed the whole requested text list, so asking for any
different *subset* of previously-embedded texts (e.g. re-querying without
deflection-filtered cases) was a 100% cache miss and re-embedded everything
through Ollama from scratch — this actually happened once during development
(a ~16-minute re-embed for a query that needed zero new vectors) and is why
`support_agent/retrieval/embed.py` stores one `.npz` keyed by
`sha1(backend|model|dim|text)` instead.

### 10. The judge model is deliberately a different, stronger model than the drafter.

(`claude-opus-5` judging `claude-haiku-4-5`'s drafts by default.)
Same-family judge/drafter pairs show measurable self-enhancement bias in the
literature (arXiv 2410.21819, 2506.02592) — an LLM judge tends to rate its own
family's outputs more favorably. Every reply is also scored twice with the
rubric's criteria listed in reversed order (`judge.py::judge_reply`,
`double_score=True`); a large per-criterion delta between orderings is
reported as `order_sensitivity` rather than silently averaged away, since
order/position bias is the second most-documented LLM-judge failure mode
after self-enhancement.

### 11. Escalation is force-overridden by two flags (`angry`, `churn_risk`) in CODE, not left to the model's own triage judgment.

`build_golden.py` and
`agent/pipeline.py::triage` both apply this override identically: regardless
of intent or the classifier's own escalate recommendation, an angry or
churn-flagged message always routes to a human. This is a business-risk
argument, not an accuracy argument — an agent that's usually right about
tone but occasionally wrong on a screaming, about-to-churn customer is not a
risk worth taking for the automation it buys, and putting the override in
code rather than in a prompt instruction makes it a property you can prove
holds on every row, not one you're hoping the model followed.

### 12. The nearest-neighbour baseline reply had a self-match leakage bug, caught and fixed before any number was reported.

The first run showed
`mean top-1 similarity = 1.000` and `self-match = 220/220` — every golden
example was retrieving *its own* historical reply from the index, because
the golden cases are themselves part of the corpus the index was built from.
Fixed by threading `exclude_case_id` through
`NearestNeighborReplier.draft_batch`; the corrected, honest number is 0.817
mean top-1 similarity with 0/220 leakage. Left in the decision log instead of
silently fixed because a similarity metric of exactly 1.000 across an entire
eval set is the kind of number that should make you stop and check, not
report.

### 13. Weak (rule-based) labels are used for two different, deliberately separated purposes — stratifying the golden sample, and training the "simple" baseline classifier — never for computing a reported accuracy number.

Every
accuracy, F1, and triage number in this repo is measured against
hand-labeled `gold_intent`/`gold_escalate`, not against the weak labels the
simple baseline itself was trained on. The one place weak labels appear as a
"result" is the weak-label-vs-gold agreement table (69.1% overall) — reported
explicitly as a measurement of the rule-based labeller's own quality, which
is also why the simple TF-IDF+LR baseline (68.6% accuracy on gold) can't beat
69.1%: it is trained on, and therefore bounded by, the noisy labels it learned
from. That bound *is* the finding, not a bug in the baseline.

### 14. `LLM_BACKEND=cached` is a first-class mode, not a debugging shortcut.

`support_agent/llm/client.py` raises a loud, specific `CacheMiss` (naming the
exact request that's missing and how to fix it) rather than falling back to a
default or a mocked response. This is what makes "reproduce our headline
results in under 15 minutes" true for the LLM-dependent numbers too: once the
cache in `artifacts/llm_cache/` is committed after a real run, anyone can
replay the exact recorded model outputs with no API key, no network call, and
no risk of a different run producing different numbers than the ones in the
report.

### 15. Reproduction is split into a fast, free, always-available path and a slower, costed, optional one.

(Committed cases/weak-labels/embedding-cache → `make baselines`, <1 minute,
no key, vs. raw 493MB CSV download + Ollama embedding from scratch, or the
real LLM agent/judge run, ~$2–5.)
The trivial and simple baselines and every non-LLM metric in the report are
produced by the fast path and are true today, independent of anyone's API
key. This decision is what makes it honest to say "here are real, verified
numbers" in the README rather than "here is a pipeline that should work."
