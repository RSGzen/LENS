# qrels rubric v1 — Tier 2 relevance judging

Frozen 2026-11-01, before the first judgment. Do not change it mid-pass: a rubric that
moves while labels are being written makes the metric unreproducible, and the whole
point of the file is that someone else could repeat the labelling from it.

## What is being judged

**One question at a time:** *would this paper help answer the query?*

It is a judgement about the paper's **usefulness to the question**, not about whether
you believe the paper, whether it is a good paper, whether it is recent, or whether it
agrees with LENS. You are labelling a **retrieval** result, not an answer.

Read the **title and abstract only**. Do not open the PDF, do not search for the paper,
do not use knowledge of what the paper is famous for. If the abstract does not support
the judgement, the judgement is `0` — that is exactly the condition the retriever has to
work under.

## The two labels

| `rel` | Meaning | Test |
|---|---|---|
| **1** | relevant | The abstract addresses the need the query states — it reports a method, a comparison, a measurement or a resource that the query asks about, on a topic the query names. |
| **0** | not relevant | Anything else: a different topic, the right topic but the wrong need, a passing mention, an abstract you cannot judge. |

There is no middle value, no "partially", and no `2`. If you find yourself wanting a
`0.5`, the answer is `0`: a label that means "maybe" cannot be scored.

## The four decisions that actually come up

1. **Right area, wrong need.** The query asks *how* something is measured; the paper
   proposes a method and never evaluates it. → **0**. The query asks for a comparison
   and the paper reports one system. → **0**.
2. **A survey or position paper.** It covers the query's topic and would help someone
   answer the query. → **1**. Relevance is about usefulness, not novelty.
3. **A passing mention.** The abstract names the query's term once inside a list of
   related work and does nothing with it. → **0**. The test: if the abstract were the
   only thing you had, could you write a sentence answering the query from it?
4. **You cannot tell.** The abstract is truncated, or genuinely ambiguous. → **0**, and
   say so if you keep notes. Ambiguity that survives reading is not relevance.

## Worked examples (from this corpus)

Query: *Which methods compress a long input into a shorter representation that a
language model can still use?*

| Abstract, in short | `rel` | Why |
|---|---|---|
| "We propose a context compression model based on an autoencoder, which we call Recurrent Context Compression…" | **1** | It is the method the query asks about. |
| "We survey efficient attention mechanisms, including sparsity and state-space models…" | **1** | Right kind of method (making a long input tractable), and the abstract says it compares them. |
| "We introduce a benchmark for long-context reasoning with 128k-token inputs…" | **0** | Right area, wrong need: measures long-context ability, does not compress. |
| "Retrieval-Augmented Generation retrieves passages from a long context…" | **0** | Retrieval is not compression; the query names compression. |
| "KV cache compression reduces memory during decoding…" | **1** | It compresses what the model holds, which the query's "shorter representation" covers. |

## How to fill the sheet

The worksheet is `qrels-worksheet.csv`, one row per (query, paper) pair, **shuffled
with a fixed seed**, with the configuration that retrieved the paper hidden. Keep it
that way: a judge who can see which retriever found a paper is grading the retriever.

1. Rows for the same `query_id` may not be adjacent. Filter by `query_id`, read the
   `query` column once, then judge that group.
2. Fill `rel` with `1` or `0`. Leave nothing blank — a blank row is read as `0` by the
   scorer, which would silently credit the retriever for your skipped work.
3. Do not edit `query_id`, `doc_id`, `title`, `abstract` or `query`. They are the join
   keys and the evidence.
4. Stop at 45 minutes and continue later rather than rushing a block; there is no
   second annotator to catch a tired pass.
5. When the file is complete, run `scripts/score_retrieval.py --qrels <file>
   --rankings <rankings.csv>`; the extra columns are ignored.

## Optional cross-check (never a substitute)

An LLM relevance judge may be run over the **same** pool and its agreement with your
labels reported as a percentage and a κ. It never replaces a human label, and if the
two disagree the human label stands. Recording the agreement is worth the ten minutes:
it is the only evidence about how hard the task was.
