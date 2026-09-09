"""The simple baseline -- a competent afternoon's worth of classic ML, no LLM.

- Classifier: TF-IDF (word 1-2 grams) + multinomial logistic regression,
  trained on the WEAK labels for all 14,703 cases (distant supervision -- see
  support_agent.taxonomy.weak_labels). This is deliberately not trained on any
  gold label; the golden set stays a clean held-out test set for every model
  in this repo, baseline and LLM agent alike.

- Replier: nearest-neighbour retrieval with NO generation. Embed the query,
  find the single closest historical customer message (excluding deflection
  replies, same pool the LLM agent grounds on), and return that historical
  brand reply verbatim. This isolates how much of the LLM agent's reply
  quality is coming from retrieval alone versus from generation -- a
  precise, falsifiable ablation the report leans on.

- Triage: deterministic rule straight from config/taxonomy.yaml's handling
  policy (escalate iff the *predicted* intent's default handling is
  'escalate'), with no confidence gating at all. This is what "policy without
  a model" looks like, and the gap between this and the LLM agent's
  confidence-gated triage is the entire selective-prediction story in the
  report.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from support_agent.retrieval.index import PrecedentIndex


class SimpleClassifier:
    def __init__(self, vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
        self.vectorizer = vectorizer
        self.model = model

    @classmethod
    def fit(cls, texts: list[str], labels: list[str], seed: int = 0) -> "SimpleClassifier":
        vec = TfidfVectorizer(
            sublinear_tf=True, min_df=2, max_df=0.5, ngram_range=(1, 2),
            strip_accents="unicode", lowercase=True,
        )
        X = vec.fit_transform(texts)
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=2.0, random_state=seed,
        )
        clf.fit(X, labels)
        return cls(vec, clf)

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        return list(self.model.predict(X))

    def predict_proba_top(self, texts: list[str]) -> tuple[list[str], list[float]]:
        X = self.vectorizer.transform(texts)
        proba = self.model.predict_proba(X)
        idx = proba.argmax(axis=1)
        classes = self.model.classes_
        preds = [classes[i] for i in idx]
        confs = [float(proba[r, i]) for r, i in enumerate(idx)]
        return preds, confs


class NearestNeighborReplier:
    """Retrieval with no generation: return the closest precedent's reply as-is."""

    def __init__(self, index: PrecedentIndex, embed_backend: str, embed_model: str, cache_dir: str) -> None:
        self.index = index
        self.embed_backend = embed_backend
        self.embed_model = embed_model
        self.cache_dir = cache_dir

    def draft_batch(self, texts: list[str], exclude_case_ids: list[int | None] | None = None) -> list[tuple[str, float, int]]:
        """Returns (reply_text, similarity, source_case_id) per query.

        `exclude_case_ids`, when given, excludes the query's own case from its
        own retrieval -- required whenever the queries are themselves part of
        the indexed corpus (e.g. golden-set eval), or the "nearest neighbour"
        trivially retrieves the customer's own historical reply and reports a
        meaningless 1.0 similarity. See docs/DECISIONS.md #12.
        """
        from support_agent.retrieval.embed import embed_texts

        vecs = embed_texts(texts, backend=self.embed_backend, model=self.embed_model, cache_dir=self.cache_dir)
        excl = exclude_case_ids or [None] * len(texts)
        out = []
        for v, ex in zip(vecs, excl):
            top = self.index.query(v, k=1, exclude_case_id=ex)[0]
            out.append((top.brand_reply, top.similarity, top.case_id))
        return out


def taxonomy_handling(taxonomy: dict) -> dict[str, str]:
    return {i["name"]: i["handling"] for i in taxonomy["intents"]}


def simple_triage(intents: list[str], handling_by_intent: dict[str, str]) -> list[bool]:
    return [handling_by_intent.get(intent, "escalate") == "escalate" for intent in intents]


if __name__ == "__main__":
    import argparse


    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/interim/hulu_support_cases.jsonl")
    ap.add_argument("--weak-labels", default="data/interim/hulu_weak_labels.jsonl")
    ap.add_argument("--golden", default="data/golden/golden_v1.jsonl")
    args = ap.parse_args()

    cases = [json.loads(line) for line in open(args.cases, encoding="utf-8")]
    weak = {}
    for line in open(args.weak_labels, encoding="utf-8"):
        r = json.loads(line)
        weak[r["case_id"]] = r["weak_label"]

    texts = [c["customer_text"] for c in cases if c["case_id"] in weak]
    labels = [weak[c["case_id"]] for c in cases if c["case_id"] in weak]
    clf = SimpleClassifier.fit(texts, labels)

    golden = [json.loads(line) for line in open(args.golden, encoding="utf-8")]
    g_texts = [g["customer_text"] for g in golden]
    g_gold = [g["gold_intent"] for g in golden]
    preds, confs = clf.predict_proba_top(g_texts)
    acc = sum(p == g for p, g in zip(preds, g_gold)) / len(g_gold)
    print(f"TF-IDF+LR trained on {len(texts)} weak-labelled cases")
    print(f"accuracy on {len(golden)} golden examples: {acc:.1%}")
    print(f"mean confidence on golden: {np.mean(confs):.3f}")
