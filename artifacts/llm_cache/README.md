# LLM response cache

Starts empty. Populated automatically, one file per request, when you run
`scripts/run_llm_agent.py` or `scripts/run_judge_calibration.py` with
`ANTHROPIC_API_KEY` and `LLM_BACKEND=anthropic` set.

Once populated, **commit this directory** — that's what lets anyone else
reproduce the exact LLM-dependent headline numbers in the report with
`LLM_BACKEND=cached`, no API key, no network call, and no risk of a re-run
producing different numbers than the ones written up. See
Decision #14 in [`docs/DECISIONS.md`](../../docs/DECISIONS.md) and the root README's
"Reproduce the LLM-agent numbers" section.

Each file is content-addressed (`<sha256-of-the-exact-request>.json`) and
holds the full request (model, system/user prompt, schema, temperature) plus
the parsed response and token usage — so a cache hit is provably a replay of
that exact request, not a coincidence.
