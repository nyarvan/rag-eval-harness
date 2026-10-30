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

Retrieval quality (`all_gold@k`: both supporting paragraphs found in the top k,
the key metric for multi-hop questions):

| | @2 | @5 | @10 |
|---|---|---|---|
| dense | 0.460 | 0.745 | 0.875 |
| dense + rerank (top-30 candidates) | 0.665 | 0.885 | 0.945 |

By question type at `@5`: bridge 0.706 → 0.865, comparison 0.967 → 1.000.

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

Caveats: one sample of 200 questions, one generator, one seed. Differences of a
few points are not significant; for the reranking gain, 19 questions improved
and 10 got worse in a paired comparison (p ≈ 0.14).

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
| `--retrieve-k` | 10 | Ranking depth, used for `recall@k` |
| `--context-k` | 5 | Paragraphs passed to the generator |
| `--rerank` | off | Rerank dense candidates with the cross-encoder |
| `--candidates-k` | 30 | Candidates handed to the reranker |
| `--env-file` | none | Dotenv file with `OPENAI_API_KEY` |

The script prints `recall@k`, `all_gold@k`, EM and F1 and writes a JSONL run
file.

`rag_eval.ragas_eval` reads a run file and writes `runs/ragas_<name>.jsonl`:
one object per question with its id and the four metric scores (`null` where a
metric failed). Options: `--n` (first N records), `--judge` (default
`gpt-4o-mini`), `--concurrency` (default 8), `--out`, `--env-file`. Judge
replies are cached on disk, so repeating a run is free; a full 200-question run
took about 15 minutes.

## Run file format

One JSON object per question:

| Field | Meaning |
|---|---|
| `question_id`, `question` | The HotpotQA question |
| `contexts` | Paragraphs shown to the generator (`Title: …\nText: …`) |
| `answer` | The model's answer |
| `reference` | The gold answer |
| `retrieved` | Ranked titles returned by retrieval |
| `gold_titles` | The supporting paragraph titles |

`question`, `contexts`, `answer` and `reference` are the names RAGAS expects.

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

The suite (119 tests) covers the metrics on toy data, retrieval, reranking,
generation, the run pipeline and RAGAS scoring with fake models. Tests marked `integration`
download the real HotpotQA split and need network on the first run.

## Status

Done: data, retrieval, reranking, generation, metrics, run files, RAGAS scoring.
Planned: a single command-line entry point (`cli.py` is an empty placeholder).
