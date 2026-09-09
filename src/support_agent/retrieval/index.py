"""Nearest-neighbour precedent retrieval: the agent's grounding source.

Given a new customer message, we retrieve the k most similar HISTORICAL
customer messages this brand has already handled, and hand the drafter their
brand replies as precedent. This is the entire "grounded in how the brand has
historically resolved similar issues" requirement from the brief -- done with
plain cosine similarity over embeddings, no vector DB needed at this scale
(15k vectors fits in memory as a dense matrix).

Two design decisions worth stating explicitly (both argued at length in
docs/DECISIONS.md):

1. We exclude link-only / deflection replies from the retrievable pool
   (`exclude_deflections`). A precedent whose entire content is "please DM
   us: <link>" grounds nothing -- the linked page's content isn't in our data
   -- and if kept, the drafter converges to always recommending a DM, which is
   measurably true of this brand's raw behaviour (see brand_survey.py) and
   exactly the failure mode picked out in the assignment's framing of
   AppleSupport/sprintcare-style brands. Filtering is what makes retrieval
   worth doing at all for this brand.

2. We retrieve on the CUSTOMER MESSAGE embedding (query-to-query similarity),
   not on a customer/brand-reply pair. At inference time we only have the
   query; indexing on the reply side would require a second retrieval hop
   (or leak the reply into the similarity computation) for no accuracy gain
   we could verify.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .embed import embed_texts


@dataclass
class Precedent:
    case_id: int
    customer_text: str
    brand_reply: str
    similarity: float
    thanked: bool


class PrecedentIndex:
    def __init__(
        self,
        case_ids: list[int],
        customer_texts: list[str],
        brand_replies: list[str],
        thanked: list[bool],
        embeddings: np.ndarray,
    ) -> None:
        assert len(case_ids) == len(customer_texts) == len(brand_replies) == embeddings.shape[0]
        self.case_ids = case_ids
        self.customer_texts = customer_texts
        self.brand_replies = brand_replies
        self.thanked = thanked
        self.embeddings = embeddings  # assumed L2-normalised

    def __len__(self) -> int:
        return len(self.case_ids)

    def query(self, query_vec: np.ndarray, k: int = 6, exclude_case_id: int | None = None) -> list[Precedent]:
        sims = self.embeddings @ query_vec
        order = np.argsort(-sims)
        out: list[Precedent] = []
        for idx in order:
            if exclude_case_id is not None and self.case_ids[idx] == exclude_case_id:
                continue
            out.append(Precedent(
                case_id=self.case_ids[idx],
                customer_text=self.customer_texts[idx],
                brand_reply=self.brand_replies[idx],
                similarity=float(sims[idx]),
                thanked=self.thanked[idx],
            ))
            if len(out) >= k:
                break
        return out

    def query_batch(self, query_vecs: np.ndarray, k: int = 6) -> list[list[Precedent]]:
        return [self.query(v, k=k) for v in query_vecs]

    def save(self, path_prefix: str) -> None:
        np.save(path_prefix + ".emb.npy", self.embeddings)
        with open(path_prefix + ".meta.jsonl", "w", encoding="utf-8") as fh:
            for i in range(len(self)):
                fh.write(json.dumps({
                    "case_id": self.case_ids[i],
                    "customer_text": self.customer_texts[i],
                    "brand_reply": self.brand_replies[i],
                    "thanked": self.thanked[i],
                }, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path_prefix: str) -> "PrecedentIndex":
        emb = np.load(path_prefix + ".emb.npy")
        case_ids, texts, replies, thanked = [], [], [], []
        for line in open(path_prefix + ".meta.jsonl", encoding="utf-8"):
            r = json.loads(line)
            case_ids.append(r["case_id"])
            texts.append(r["customer_text"])
            replies.append(r["brand_reply"])
            thanked.append(r["thanked"])
        return cls(case_ids, texts, replies, thanked, emb)


# Reply is excluded from the retrievable pool if it's dominated by a link
# hand-off with little substantive content of its own.
def _is_deflection_reply(reply_body: str, min_words: int) -> bool:
    return len(reply_body.split()) < min_words


def build_index(
    cases_path: str,
    embed_backend: str,
    embed_model: str,
    min_reply_words: int,
    exclude_deflections: bool,
    cache_dir: str,
) -> PrecedentIndex:
    cases = [json.loads(l) for l in open(cases_path, encoding="utf-8")]
    if exclude_deflections:
        cases = [c for c in cases if not _is_deflection_reply(c["brand_reply"], min_reply_words)]

    texts = [c["customer_text"] for c in cases]
    vecs = embed_texts(texts, backend=embed_backend, model=embed_model, cache_dir=cache_dir)
    return PrecedentIndex(
        case_ids=[c["case_id"] for c in cases],
        customer_texts=texts,
        brand_replies=[c["brand_reply"] for c in cases],
        thanked=[c["thanked"] for c in cases],
        embeddings=vecs,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/interim/hulu_support_cases.jsonl")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--min-reply-words", type=int, default=8)
    ap.add_argument("--out-prefix", default="artifacts/precedent_index")
    args = ap.parse_args()

    idx = build_index(
        args.cases, args.backend, args.model,
        min_reply_words=args.min_reply_words, exclude_deflections=True,
        cache_dir="data/interim/emb_cache",
    )
    idx.save(args.out_prefix)
    print(f"built precedent index: {len(idx)} retrievable precedents -> {args.out_prefix}.*")

    # sanity check: a couple of live queries
    demo_queries = [
        "hulu keeps buffering on my roku every few minutes",
        "why was I charged twice this month",
        "can you add season 4 of the show please",
    ]
    qvecs = embed_texts(demo_queries, backend=args.backend, model=args.model, cache_dir="data/interim/emb_cache")
    for q, v in zip(demo_queries, qvecs):
        print(f"\nquery: {q}")
        for p in idx.query(v, k=3):
            print(f"  sim={p.similarity:.3f}  cust={p.customer_text[:80]!r}")
            print(f"           reply={p.brand_reply[:100]!r}")
