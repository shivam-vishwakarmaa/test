import sys

# The raw dataset is real Twitter text -- emoji included -- and every script in
# this repo prints it (progress lines, retrieval demo output, judge reasoning).
# Windows' default console codepage (cp1252) cannot encode most emoji and
# raises UnicodeEncodeError on the first one instead of printing '?', which
# would otherwise crash `python -m support_agent.retrieval.index` and friends
# non-deterministically depending on which tweet happens to print first.
# stdout/stderr are widely available as reconfigurable TextIOWrapper objects
# (Python >=3.7); the guard skips environments where they're not (e.g. output
# captured by some test runners), since correctness there doesn't depend on this.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
del _stream
