# rag-eval-harness

A small, reproducible harness for evaluating a RAG pipeline on **HotpotQA**
(distractor setting). It retrieves paragraphs with a dense model, optionally
reranks them with a cross-encoder, generates short answers with an LLM, and
scores every stage with metrics that need no LLM and with an LLM judge
(RAGAS).

```
question ──► dense search ──► top-30 ──► cross-encoder rerank ──► top-5 ──► LLM ──► answer
             (bge-m3)                    (bge-reranker-v2-m3)               (gpt-4o-mini)
                │                                │                                │
          recall@k / all_gold@k            recall@k / all_gold@k          EM / F1, RAGAS
```

## Results

200 HotpotQA validation questions (seed 42), a corpus of 1990 paragraphs,
`gpt-4o-mini` at temperature 0.

| Configuration | EM | F1 | Both gold paragraphs in context |
|---|---|---|---|
| dense, 3 paragraphs in context | 0.435 | 0.587 | 62.0% |
| dense, 5 paragraphs | 0.455 | 0.609 | 74.5% |
| dense, 10 paragraphs | 0.485 | 0.646 | 87.5% |
| **dense + rerank, 5 paragraphs** | **0.500** | **0.668** | **88.5%** |
| dense + rerank, reasoning before the answer | 0.525 | 0.699 | 88.5% |
| dense + rerank, `gpt-4o` as the generator | 0.550 | 0.722 | 88.5% |

### Retrieval

`all_gold@k`: both supporting paragraphs found in the top k, the key metric for
multi-hop questions.

| Retriever | @2 | @5 | @10 | @20 |
|---|---|---|---|---|
| dense (bge-m3) | 0.460 | 0.745 | 0.875 | 0.940 |
| BM25 | 0.230 | 0.565 | 0.795 | 0.890 |
| hybrid (RRF, k = 60) | 0.385 | 0.740 | 0.905 | 0.940 |
| **dense + rerank** (30 candidates) | **0.665** | **0.885** | 0.945 | 0.955 |
| hybrid + rerank (30 candidates) | 0.660 | 0.875 | 0.945 | 0.955 |
| hybrid + rerank (50 candidates) | 0.660 | 0.875 | 0.950 | 0.975 |

By question type at `@5`: bridge 0.706 → 0.865 with reranking, comparison
0.967 → 1.000.

Paired comparisons at `@5` (mean difference per question, 95% bootstrap
interval, exact sign test on the questions where the two retrievers differ;
the statistics are `significance.py` of `embedding-benchmark-harness`):

| Comparison | Difference | 95% interval | Better / worse | p |
|---|---|---|---|---|
| dense + rerank − dense | +0.140 | [+0.085, +0.200] | 33 / 5 | < 0.001 |
| BM25 − dense | −0.180 | [−0.250, −0.110] | 12 / 48 | < 0.001 |
| hybrid − dense | −0.005 | [−0.060, +0.050] | 14 / 15 | 1.0 |
| hybrid + rerank − dense + rerank | −0.010 | [−0.040, +0.015] | 3 / 5 | 0.73 |
| hybrid + rerank (50) − dense + rerank | −0.010 | [−0.040, +0.015] | 3 / 5 | 0.73 |

Time per question, one question at a time, on an 8 GB Apple-silicon laptop:

| Retriever | Median |
|---|---|
| dense | 40 ms |
| BM25 | 1 ms |
| hybrid | 31 ms |
| cross-encoder over 30 candidates (cold score cache) | 3–23 s |

The reranker figure is a range because it depends on memory pressure: about
3 s in an isolated micro-benchmark, about 6 s over the full 200-question run,
23 s in a run with another model loaded beside it. Either way it is the
dominant cost. Scores are cached, so a repeated run does not pay it.

### What the experiments showed

- **Twenty questions mislead.** The first 20 questions were easy
  (EM 0.65 and `all_gold@5` 0.95 there, against 0.455 and 0.745 on 200). One
  question is 5 points at n=20 and 0.5 at n=200.
- **Reranking helps most on bridge questions**, where the second supporting
  paragraph does not look like the question. A reranked top-5 covers more
  gold paragraphs than a dense top-10 with half the context.
- **The generator caps EM at about 0.54** even when both gold paragraphs are in
  the context. Of the misses in that case, 64% are partial matches of the
  reference span (`Royal Air Force (RAF)` against `Royal Air Force`), not
  wrong answers. EM measures annotation style here, so F1 or an LLM judge is
  the fairer yardstick.
- **Prompt changes were mostly noise or harmful.** Dropping the `noanswer`
  instruction fixed a refusal on a question whose answer was in the context.
  Asking for verbatim, full-name answers made answers longer (2.2 to 3.5
  words) and lowered EM from 0.500 to 0.450.
- **Reasoning before the answer helps a little.** Letting the model think in a
  few sentences and then write a `Final answer:` line (`--cot`) moves EM from
  0.500 to 0.525 and F1 from 0.668 to 0.699. That is within noise (p ≈ 0.06
  for F1, higher for EM and the judge). The gain comes from `comparison`
  questions (judge 0.811 → 0.911 on 30 questions); `bridge` questions, 85% of
  the sample, do not move.
- **A stronger generator helps more than reasoning.** Writing the answers with
  `gpt-4o` instead of `gpt-4o-mini` (`--chat-model gpt-4o`) gives EM 0.550,
  F1 0.722 and `factual_correctness` 0.778 (p ≈ 0.06 for EM, 0.2 for the
  judge). It fixes answers that need a second hop, and it adds words to some
  correct ones. Even with both gold paragraphs in the context about a fifth of
  the answers are still judged wrong, so part of the remaining error sits in
  the references and the questions, not in the model.
- **The worst answers fail in retrieval, and the judge is right about them.**
  Of the five lowest RAGAS scores in the dense baseline, four lost the second
  supporting paragraph (in three cases it was outside the top 10) and one was a
  counting mistake in generation. Reranking recovered three of the eight
  questions scored zero and `gpt-4o` one more. Three stay lost: their second
  paragraph is linked to the question only through the first one and sits
  around position 100 of 1990 in the dense ranking, out of reach of a reranker
  over the top 30.
- **Hybrid search did not pay off here.** BM25 alone is much weaker than dense
  (`all_gold@5` 0.565 against 0.745), and fusing the two with RRF leaves the
  top 5 unchanged (0.740). BM25 does find some second paragraphs that dense
  misses (one that dense ranks beyond 100 comes 6th in BM25), but fusion buries
  them, and reranking does not lift them into the top 5 either: hybrid +
  rerank (0.875) is level with dense + rerank (0.885). This is a small
  distractor corpus of 1990 paragraphs where dense retrieval is already
  strong; a large open-domain corpus with rare names may behave differently.

Caveats: one sample of 200 questions, one generator, one seed. Differences of a
few points are not significant. For the reranking gain in the final answers,
a paired comparison gives EM +0.045 (95% interval [−0.005, +0.095], 19 better
and 10 worse, p ≈ 0.14) and F1 +0.059 ([+0.007, +0.114], 28 better and 14
worse, p ≈ 0.04).

### LLM judge (RAGAS)

EM punishes correct answers that differ from the reference span, so the same
runs are also scored by an LLM judge (`gpt-4o-mini`) with four RAGAS metrics:

| Metric | What it checks | dense | + rerank |
|---|---|---|---|
| `factual_correctness` | the answer against the reference | 0.702 | 0.738 |
| `faithfulness` | the answer is supported by the contexts | 0.647 | 0.701 |
| `context_recall` | the contexts contain what the reference says | 0.775 | 0.865 |
| `context_precision` | useful contexts are ranked first | 0.589 | 0.646 |

- The judge rates answers about 24 points above EM (0.702 against 0.455 for
  dense, 0.738 against 0.500 with reranking). Both measures rank the two
  configurations the same way; only the absolute level differs.
- Reranking improves the retrieval metrics reliably (`context_recall` +0.09,
  `context_precision` +0.06; p ≤ 0.003 in a sign test on per-question changes).
  The gain in `factual_correctness` (+0.035) is not significant (p ≈ 0.23).
- Read two metrics with care. `context_precision` is unreliable for
  `comparison` questions (about 0.3), where the reference is a bare yes/no or a
  short name. `faithfulness` can penalize correct answers that are inferred
  rather than quoted. The judge is the same model as the generator, which may
  inflate absolute scores; it likely affects both configurations alike.
- With reasoning before the answer, `factual_correctness` on the reranked run
  goes from 0.738 to 0.755, which is not significant (p ≈ 0.77). With `gpt-4o`
  as the generator it is 0.778 (p ≈ 0.20).

## Quick start

Python 3.11+. The `embedding-benchmark-harness` package is a dependency.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# or, to develop both projects side by side:
pip install -e ../embedding-benchmark-harness
```

An OpenAI key is needed for generation: set `OPENAI_API_KEY`, or pass a dotenv
file with `--env-file`.

```bash
# dense retrieval only, 200 questions
python -m rag_eval.run --n 200 --out runs/dense_200.jsonl

# with cross-encoder reranking of the top-30 candidates
python -m rag_eval.run --n 200 --rerank --out runs/dense_rerank_200.jsonl

# vary how many paragraphs the generator reads
python -m rag_eval.run --n 200 --context-k 10 --out runs/dense_200_ctx10.jsonl

# score a run file with the RAGAS metrics (about 2000 judge calls per 200 questions)
python -m rag_eval.ragas_eval runs/dense_rerank_200.jsonl

# compare retrievers without any LLM call, with paired significance
python -m rag_eval.retrieval_eval --retrievers dense dense+rerank --compare

# time each retriever, one question at a time
python -m rag_eval.latency
```

The first run downloads `bge-m3` (about 2 GB) and encodes the corpus; the
reranker is another download of about 2 GB. On a CPU the corpus takes about
9 minutes and the reranker about 20 minutes for 200 questions. Embeddings,
reranker scores and LLM answers are cached on disk, so repeating a run is fast
and free.

Options of `rag_eval.run`:

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 20 | Number of questions |
| `--out` | `runs/dense.jsonl` | Output run file |
| `--model` | `bge_m3` | Embedding model name from the `embedding-benchmark-harness` registry |
| `--chat-model` | `gpt-4o-mini` | OpenAI model that writes the answers |
| `--retrieve-k` | 10 | Ranking depth, used for `recall@k` |
| `--context-k` | 5 | Paragraphs passed to the generator |
| `--rerank` | off | Rerank dense candidates with the cross-encoder |
| `--candidates-k` | 30 | Candidates handed to the reranker |
| `--cot` | off | Let the model reason briefly and read the answer from its `Final answer:` line |
| `--env-file` | none | Dotenv file with `OPENAI_API_KEY` |

The script prints `recall@k`, `all_gold@k`, EM and F1 and writes a JSONL run
file.

`rag_eval.ragas_eval` reads a run file and writes `runs/ragas_<name>.jsonl`:
one object per question with its id and the four metric scores (`null` where a
metric failed). Options: `--n` (first N records), `--judge` (default
`gpt-4o-mini`), `--metrics` (a subset of the four), `--concurrency`
(default 8), `--out`, `--env-file`. Judge
replies are cached on disk, so repeating a run is free; a full 200-question run
took about 15 minutes.

`rag_eval.retrieval_eval` compares retrievers with no LLM call: `all_gold@k`
and `recall@k` for all questions and for each question type, and with
`--compare` a paired comparison of every retriever with the first one listed.
Choose from `dense`, `bm25`, `hybrid`, `dense+rerank`, `hybrid+rerank` and
`hybrid+rerank50`. Rerankers are built one at a time, because each holds a
cross-encoder of about 2 GB.

## Command line

`cli.py` is one entry point over the modules above:

```bash
# run a pipeline (paid: prints an estimate and asks first)
python cli.py run --n 200 --rerank --out runs/mine.jsonl

# score a run file: free metrics, and the RAGAS judge with --ragas (paid, asks)
python cli.py eval runs/mine.jsonl
python cli.py eval runs/mine.jsonl --ragas

# paired comparison of run files; the first one is the baseline
python cli.py compare runs/dense_200.jsonl runs/dense_rerank_200.jsonl
```

`run` takes the flags of `rag_eval.run`, and `eval` the flags of
`rag_eval.ragas_eval`, plus `--ragas`, `--ks` and `--yes`. `eval` and
`compare` work on a run file from any RAG pipeline that fits the format below:
`eval` gives EM and F1, and `recall@k` and `all_gold@k` when the records carry
`retrieved` and `gold_titles`.

```
comparison                    metric  mean diff  95% CI              better/worse  p (sign)
dense_rerank_200 - dense_200  em      +0.0450    [-0.0050, +0.0950]  19/10         0.136
dense_rerank_200 - dense_200  f1      +0.0592    [+0.0065, +0.1136]  28/14         0.044
```

Paid steps do not start by themselves. Before a model is called, the command
prints the number of requests, tokens and an approximate price, then asks
`[y/N]`. `--yes` skips the question; without a terminal and without `--yes`
nothing starts. The estimate is an upper bound for a fresh run (cached replies
are free). Its prices are approximate list prices that may be out of date, a
model without a known price gets no dollar figure, and for RAGAS only the
total of about 10.5 judge requests per question was measured.

Exit codes: 0 done, 1 a paid step was not confirmed, 2 invalid input (an
unreadable or invalid file is reported with its path and line).

## Run file format

One JSON object per question. A run file from any RAG pipeline works, as long
as it has the required fields:

| Field | Meaning | Required |
|---|---|---|
| `question_id`, `question` | The question and a non-empty id | yes |
| `contexts` | Paragraphs shown to the generator (`Title: …\nText: …`) | yes |
| `answer` | The model's answer (may be empty) | yes |
| `reference` | The gold answer | yes |
| `retrieved` | Ranked titles returned by retrieval | no |
| `gold_titles` | The supporting paragraph titles | no |

`retrieved` and `gold_titles` are only needed for the retrieval metrics. Other
fields in a line are ignored. Reading validates every line, and an error names
the file and the line (`runs/x.jsonl:17: …`). `question`, `contexts`, `answer`
and `reference` are the names RAGAS expects.

## Project layout

| Module | Role |
|---|---|
| [rag_eval/data.py](rag_eval/data.py) | Pinned, seeded HotpotQA subset: corpus and questions |
| [rag_eval/retrieve.py](rag_eval/retrieve.py) | `DenseRetriever`: cached paragraph embeddings, batched search |
| [rag_eval/rerank.py](rag_eval/rerank.py) | `RerankedRetriever`: cross-encoder reordering with a score cache |
| [rag_eval/generate.py](rag_eval/generate.py) | `ChatModel` protocol, OpenAI adapter, answer cache |
| [rag_eval/metrics.py](rag_eval/metrics.py) | `recall_at_k`, `all_gold_at_k`, EM, F1 |
| [rag_eval/records.py](rag_eval/records.py) | Run-file record and JSONL read/write |
| [rag_eval/run.py](rag_eval/run.py) | The run: retrieve, generate, summarize, save |
| [rag_eval/ragas_eval.py](rag_eval/ragas_eval.py) | RAGAS scoring of a run file with an LLM judge |
| [rag_eval/bm25.py](rag_eval/bm25.py) | `BM25Retriever`: `bm25s` with stemming and stopwords |
| [rag_eval/hybrid.py](rag_eval/hybrid.py) | `rrf_fuse` (reciprocal rank fusion) and `HybridRetriever` |
| [rag_eval/compare.py](rag_eval/compare.py) | Paired comparison of retrievers per question |
| [rag_eval/retrieval_eval.py](rag_eval/retrieval_eval.py) | Retrieval-only evaluation with tables and comparisons |
| [rag_eval/latency.py](rag_eval/latency.py) | Time per question for each retriever |
| [rag_eval/evaluate.py](rag_eval/evaluate.py) | Free metrics (EM, F1, recall, all-gold) for any run file |
| [rag_eval/cost.py](rag_eval/cost.py) | Approximate cost of paid steps and the confirmation before them |
| [rag_eval/cli.py](rag_eval/cli.py) | The `run`, `eval` and `compare` commands (`cli.py` launches it) |

## Design notes

- **Reproducible data.** The dataset revision is pinned and the subset is drawn
  with a fixed seed, so the same call always gives the same questions and
  corpus. Titles are paragraph ids.
- **Metrics without an LLM.** EM and F1 reuse the normalization of the official
  HotpotQA evaluation script, including its rule that `yes`/`no` answers get no
  partial credit.
- **Caches that cannot go stale.** The answer cache key hashes the model name
  (which includes its sampling settings) together with both prompts; the
  reranker cache keys on the question and passage text; embedding caches check
  the document ids. Changing a prompt never returns an old answer.
- **Paid steps ask first.** A model call that costs money is announced with an
  estimate and needs a yes; a script that forgets `--yes` stops instead of
  spending.
- **Shared statistics.** The paired bootstrap and the sign test are
  `significance.py` of `embedding-benchmark-harness`, so the numbers of both
  projects are directly comparable.
- **Swappable parts.** Encoder, chat model and scorer are small protocols, so
  tests run on fakes, without model downloads or API keys.
- **Why `gpt-4o-mini`.** The run needs `temperature=0`. Claude Haiku 5.5 rejects
  sampling parameters, so it cannot be pinned this way; the `ChatModel`
  protocol makes adding another provider a small class.
- **Yes/no questions** are handled with one instruction in the system prompt:
  reply exactly `yes` or `no`.

## Tests

```bash
pytest
```

The suite (267 tests) covers the metrics on toy data, the retrievers, rank
fusion, comparisons, generation, the run pipeline and RAGAS scoring with fake
models. Tests marked `integration` download the real HotpotQA split and need
network on the first run.

## Status

Done: data, retrieval (dense, BM25, hybrid), reranking, generation, metrics,
run files with validation, RAGAS scoring, paired comparisons, command line.
