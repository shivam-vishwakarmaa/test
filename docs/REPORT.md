# Report — Hulu (`hulu_support`) AI support agent

Repro: `pip install -r requirements.txt && python scripts/run_baselines.py` (< 1 minute, no API
key). LLM-agent numbers: `ANTHROPIC_API_KEY=... python scripts/run_llm_agent.py` (~10–15 min,
~$2–5) then `python scripts/run_judge_calibration.py`. See `README.md` for the full pipeline
and `docs/DECISIONS.md` for the 15 non-obvious calls this report leans on.

## 1. Problem framing

**What "good" means for this brand.** Hulu's own public replies are not resolutions — they are
**triage-and-redirect**: a median 20-word public reply that empathizes, asks one diagnostic
question, and links a help article or a phone/chat channel (Decision #1–2). Fewer than 5% of
sampled threads show evidence of in-channel resolution by any proxy we could construct. So "good"
for this agent is **not** "resolves the ticket end-to-end" — that would require account data,
device state, and content-licensing information nowhere in this dataset. Good means:

1. **Classify** correctly into a category that determines a *different, correct handling
   policy* (Decision #8) — not a finer topic label that changes nothing downstream.
2. **Draft** a reply that matches what a *good* Hulu agent would send at this stage: an
   empathetic acknowledgment plus either a real diagnostic question, a grounded policy answer, or
   an honest "we can't do that" — never a fabricated fix, promise, or policy.
3. **Triage** correctly: automate the genuinely low-risk, high-confidence, well-grounded cases,
   and route everything else — including cases where the *model* is confident but the *policy*
   says a human must see it regardless (money, account access, cancellation, anger, churn risk).

**What I chose not to build.** No end-to-end resolution (would require simulated account/order
state — out of scope and unverifiable from this dataset). No multi-turn dialogue management (the
system acts once, at the ticket's opening turn — see Decision #4). No fine-tuning (150–250 gold
labels is not enough to fine-tune responsibly; prompting + retrieval is the correct-sized tool).
No attempt to model the ~9 different real support agents whose individual style leaks into "the
brand's" historical replies — the drafter is grounded in "how this brand has replied," not any
one agent's voice.

## 2. Results vs. two baselines

All numbers below are computed against the 220-example hand-labeled golden set
(`data/golden/golden_v1.jsonl`; sampling and labeling method in
`docs/LABELING_PROTOCOL.md`), using **gold labels only for scoring** — never for training.

### 2.1 Trivial baseline
*Predict the majority intent for everything; send the brand's single most-repeated canned reply
verbatim; triage via a fixed all-or-nothing policy.*

| Metric | Value |
|---|---|
| Classification accuracy | **7.3%** (macro-F1 1.4%) |
| Canned reply sent to everyone | *"Oh no! Which device do you use? Are all channels affected? For now, please try a quick reboot of your device+modem/router."* |
| Triage: always-escalate | 0% auto-sent, 0% unsafe, catches 100% of true escalations, but automates nothing |
| Triage: always-auto | 100% auto-sent, but **38.6% of all traffic is an unsafe auto-send** (sent with no human review, when a human should have seen it) |

The two triage fixed points bound every other policy in this report: **38.6% unsafe-auto-rate is
the floor any real system must beat, and 0% automation is the other extreme it must improve on.**

### 2.2 Simple baseline
*TF-IDF (1–2 grams) + logistic regression, trained on distant/weak-label supervision over all
14,703 cases (never on gold); reply = nearest-neighbor retrieval with no generation (return the
closest historical precedent's actual reply verbatim); triage = a fixed rule reading the
taxonomy's handling policy off the *predicted* intent, no confidence gating.*

| Metric | Value |
|---|---|
| Classification accuracy | **68.6%** (macro-F1 67.9%) |
| Mean predicted-class confidence | 0.782 |
| Selective-prediction AURC on that confidence | **0.142** (vs. 0.314 for a random/uninformative confidence signal — the confidence score is a real, useable signal) |
| Nearest-neighbor reply: mean top-1 cosine similarity | **0.817** (after fixing a self-match leakage bug — Decision #12) |
| Triage (rule on predicted intent, no confidence gate) | 70.5% auto-send rate, **12.7% unsafe-auto-rate**, 67.1% escalate-recall, 87.7% escalate-precision |

**This is the headline non-LLM result:** taxonomy-aware, intent-conditioned triage — with zero
LLM calls — cuts the unsafe-auto-rate from 38.6% (naive always-auto) to **12.7%**, a 3× reduction,
while still automating 70.5% of traffic. Retrieval alone (no generation) finds a historical reply
whose original customer message is, on average, 0.817-cosine-similar to the new one — evidence
that grounding has real signal to work with before an LLM ever touches the problem.

A sharper way to see the same thing: `other_non_support` has 25.8% precision and 50% recall — the
simple classifier is nearly a coin flip on "should this even get a reply," which is the exact
boundary the LLM agent's classifier and confidence-gating exist to improve on.

### 2.3 LLM agent (Haiku 4.5 drafter/classifier, Opus 5 judge)

**Status: implemented and ready to run, not yet executed against paid API calls** (see README —
this was a deliberate scope decision, not an oversight: every number above is real and reproduces
in under a minute; the LLM numbers require `ANTHROPIC_API_KEY` and ~10–15 minutes / ~$2–5, run via
`python scripts/run_llm_agent.py` then `python scripts/run_judge_calibration.py`, after which this
section's table populates directly from `artifacts/llm_agent_results.json`). Two concrete,
falsifiable predictions, stated *before* running it so they're a real test rather than a
post-hoc story: (1) the LLM classifier should gain most on `how_to_feature` and
`other_non_support` — the two intents where the simple baseline's precision is worst (38% and 26%
respectively) because both require semantic judgment a bag-of-n-grams can't do (distinguishing "a
question with the answer implied" from "an unclassifiable fragment," e.g. golden_id 7 vs. 165); (2)
the unsafe-auto-rate should drop further below 12.7%, because the LLM triage layer gates on
*retrieval grounding* in addition to intent (Decision #6), which the simple baseline's rule cannot
do at all.

## 3. Failure analysis — top 5 failure modes, with real examples

**1. Weak/rule-based labeling silently misses paraphrases it has no keyword for.**
`other_non_support` was 62.8% of all traffic under the first regex pass and 54.2% after
broadening it (see `support_agent/taxonomy/weak_labels.py`) — reading the bucket showed most of it
was real, actionable playback/content/account language the rules simply didn't pattern-match
("crashes every time," "isn't loading," "won't let us play"). *Hypothesis:* any keyword/rule
labeling system on noisy real text will systematically under-count intents whose real-world
phrasing is more varied than its author anticipated; the fix is embeddings/LLM classification, not
more regex — which is exactly why the weak labels are used only for stratification and distant
supervision, never as a reported ground truth (Decision #13).

**2. The historical brand data itself contains wrong or mismatched replies — a ceiling on any
"grounded in history" system.** Golden_id 106: a customer says "it seems fixed now, thanks!" and
Hulu's real historical reply asks about "elevated volume during a specific ad" — answering a
different conversation entirely, most likely a thread-attribution artifact in the raw data.
Golden_id 165: a customer asks how to turn off VoiceOver narration; Hulu's real reply asks for
more detail, even though **the two closest precedents in Hulu's own history already answer this
exact question** ("It sounds like VoiceOver is enabled..."). *Hypothesis:* a retrieval-grounded
agent that actually uses its retrieved precedents can beat this brand's own historical
first-response quality on repeat-question intents — a genuinely interesting, checkable claim once
the LLM agent runs (does its draft for golden_id 165 mention VoiceOver?).

**3. Money-adjacent and cancellation-adjacent language is not always a money/cancellation
intent.** The taxonomy's edge rule ("a cancellation *threat* used as leverage is not
`cancel_refund`; label the underlying issue") was the single largest source of
weak-label-vs-gold disagreement in the hard-negative sample: golden_id 41 ("I'm cancelling my
subscription on Monday bc you're useless" — about a missing show, gold=`content_availability`),
golden_id 209 ("before I cancel my account smh" — about incomplete dub episodes, gold=
`content_availability`), golden_id 21/31/86/143 (UI or release-timing complaints with a cancel
threat attached, gold=`app_ui_feedback`/`content_availability`). A system that routes on keyword
presence of "cancel" or "$" would over-escalate roughly a dozen of the 220 golden cases into the
wrong queue. This is why classification precedes triage as a separate stage (Decision #6) rather
than a single "does this mention money" rule.

**4. Multi-part messages get only their first-named or most severe part addressed — by the real
brand and, predictably, by any single-intent classifier.** Golden_id 33: signed up for ad-free,
still sees ads, *and* was charged $13.99 instead of $11.99 — Hulu's real reply addresses only the
ads question and drops the price change entirely (scored `correct_safe=3` in
`data/golden/judge_calibration.json` for exactly this omission). Golden_id 26: reports both
playback failure and a login failure in one message; our taxonomy forces a single `gold_intent`
(we chose `account_access`, the harder blocker) but a real system either needs a
multi-intent/multi-label mode or an explicit "primary vs. secondary issue" field — neither of
which this v1 taxonomy has. *This is a known, named gap, not a hidden one* (see the "what I'd do
next" section).

**5. A confident, fluent, well-grounded-*sounding* reply can still be unsafe — tone alone cannot
be the safety signal.** The synthetic bad drafts in the judge calibration set were deliberately
built to be well-written: *"We guarantee you will be refunded within 24 hours and this will never
happen again!"* reads warmer and more actionable than most of the real historical replies, and
would very plausibly score well on a judge rubric that only checked tone and fluency — which is
exactly why `grounded` and `correct_safe` are scored as **independent** criteria in
`eval/judge.py`, with an explicit instruction that a warm, fluent reply that invents a policy must
score low on both regardless of tone. The automated `has_unsupported_promise` regex check in
`eval/metrics.py` is a cheap, zero-cost second line of defense for exactly this failure mode,
independent of whether the judge catches it.

## 4. "What is misleading about my headline number?"

Several things, stated against ourselves rather than left for a reader to find:

- **The golden set's intent mix is not the traffic mix, on purpose** (Decision #2 in
  `LABELING_PROTOCOL.md`): rare intents like `live_tv_blackout` (≈0.3% of real traffic) are ~5.5%
  of the golden set so we can say anything statistically meaningful about them. Any accuracy
  number here is a **per-intent-quality** measure, not a traffic-weighted production-accuracy
  estimate — running these same models against a randomly-sampled 220 examples would report a
  higher blended accuracy dominated by the easy, common cases, and a materially *worse* number on
  the intents that matter most for the escalation decision.
- **The golden set was labeled by one annotator with no second rater** (this repo's author,
  working the written taxonomy). The 69.1% weak-label-vs-gold agreement and the judge-vs-human
  QWK are both bounded by that single point of view; a second labeler would likely move both
  numbers, and could not move them in a knowable direction without actually doing it. This is
  flagged, not hidden, and is the top item in "what I'd do with one more week."
- **The judge calibration set deliberately includes 15 synthetic worst-case drafts** to force the
  1–5 scale to actually be used (real historical replies cluster at 3–5). Whatever judge/human
  agreement number comes back will look *better* than it would on a purely natural sample of
  agent outputs, because half the calibration set is easy-to-agree-on garbage by construction.
  Report the number, but read it as "the judge can tell obviously-bad from obviously-good," not
  "the judge reliably separates a 3 from a 4."
- **68.6% simple-baseline accuracy is bounded above by 69.1%, the accuracy of the *labels it
  trained on*** — it is not meaningfully "smarter" than the rules that supervised it; it mostly
  learned to reproduce their pattern with soft-matching instead of hard regex. A reader should not
  conclude "traditional ML gets 68.6% so an LLM only needs to add a few points" — the honest
  comparison for the LLM agent is against the **gold labels directly**, which the weak-label
  pipeline never saw.
- **The 0.817 mean retrieval similarity is inflated by near-duplicate seasonal/topical traffic.**
  Twitter support threads about "when is season 4 of X coming" recur in near-identical phrasing
  across many different shows and customers; a fair reading of this number should distinguish
  "retrieval works because the semantic match is genuinely close" (true for most cases) from
  "retrieval works because this exact templated complaint has appeared hundreds of times" (true
  for a subset) — we have not yet separated these two populations.
- **A 15-minute reproduction claim is true only for the non-LLM numbers.** The LLM-dependent
  headline numbers (agent classification/triage, judge scores) require either an API key and
  ~10–15 minutes/~$2–5, or a previously-committed `artifacts/llm_cache/` from a real run. Until
  one of those exists, this report's Section 2.3 is a prediction, clearly labeled as one — not a
  result dressed up as one.

## 5. What I'd do with one more week

1. **Get a second labeler on ≥50 golden examples** and compute real inter-annotator Cohen's/
   quadratic-weighted kappa — the single highest-value fix to this project's evidence base (see
   `LABELING_PROTOCOL.md`).
2. **Run the LLM agent and judge for real**, then hill-climb the drafter prompt against the
   3 lowest-scoring rubric criteria, re-measuring on a held-out slice of the golden set so the
   reported number isn't the one the prompt was tuned against.
3. **Multi-label intent support** for the ~10% of messages that genuinely raise two issues at once
   (failure mode #4) instead of forcing a single primary label.
4. **Separate "novel semantic match" from "templated recurring complaint"** in the retrieval
   similarity distribution (the last misleading-number bullet above), likely via near-duplicate
   clustering of customer openings, to get an honest read on how much of retrieval quality is
   doing real work vs. exploiting repetition.
5. **A confidence-calibration pass on the LLM classifier itself** (the AURC/risk-coverage
   machinery in `eval/metrics.py` already supports this) to set `min_confidence` in
   `config/config.yaml` from data rather than a starting guess of 0.75.
6. **Test the taxonomy against a second brand** (AirAsiaSupport or AskPlayStation scored close
   behind Hulu in the learnability survey) to see which decisions here are Hulu-specific and which
   generalize — right now that's an open question, not a claim either way.
