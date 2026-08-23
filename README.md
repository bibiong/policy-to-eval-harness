# Policy-to-Eval Harness

**Can a written usage policy be turned into something you can actually measure?**

> 🔗 **Follow-up project.** **[multilingual-enforcement-consistency](https://github.com/bibiong/multilingual-enforcement-consistency)** reuses this repo's taxonomy to test whether the same refusal/compliance decisions hold across languages — English, Chinese, French, Singlish, and Singapore Mandarin.

Every frontier lab publishes a usage policy. Every lab enforces it with models. Almost
nobody publishes the mapping between the two — the step where "do not provide material
assistance toward unauthorized intrusion" becomes a decision an evaluator can score. This
repo builds that step explicitly, and then measures how far model behaviour sits from the
policy as written.

It was built to measure the direction usually left unmeasured — **over-refusal**, where a
model declines a legitimate request — and it reports leakage in the same table, because
either number alone is misleading. On the run that ships here, that decision is what
produced the finding: leakage turned out to be the larger error by 1.8x, and an
over-refusal-only summary would have been wrong about every model in the set.

It does not produce a safety score. "Lower over-refusal" is not "better", as the table
below makes obvious.

---

## The finding

Five open-weight models, 400 borderline prompts, 2,000 scored responses, one policy —
**[reports/findings.md](reports/findings.md)**.

**Leakage beats over-refusal by 1.8×.** 14.2% of policy-*prohibited* requests got a
substantive answer, against 7.7% of policy-*permitted* requests refused. That inverts the
framing this project was built around. These models are not paying for safety with
over-refusal; they are permissive in both directions, and a summary quoting only the
over-refusal figure would call them well-calibrated, which they are not.

The models sit on a clean permissiveness frontier, not a safety ranking:

| model | over-refusal ↓ | leakage ↓ | recall on prohibited |
|---|---:|---:|---:|
| mistral-nemo-12b | **1.3%** | **47.4%** | 0.53 |
| qwen2.5-14b | 3.3% | 14.4% | 0.86 |
| llama3.1-8b | 10.2% | 5.2% | 0.95 |
| llama3.2-3b | 11.6% | **0.0%** | 1.00 |
| gemma2-9b | 12.2% | 4.1% | 0.96 |

`mistral-nemo-12b` answered nearly half of all prohibited requests — including one asking
how to increase a pathogen's transmissibility. `llama3.2-3b` refused every prohibited
request and 11.6% of legitimate ones. Same prompts, same policy, opposite failure modes.
There is no single number that separates these models usefully, which is the point.

### Where the measurement itself broke down

The judge was calibrated against 112 hand-labelled rows. Cohen's κ = **0.52** unweighted,
**0.68** linear-weighted — and the shape of the disagreement is the more useful result:

- **100% of disagreements ran one direction** (the judge crediting less delivered substance
  than the human), and **97% were one notch apart**. Random rater noise splits evenly and
  scatters; this is a calibration offset.
- The cause is an ambiguity in the rubric, not an error by either rater. On responses that
  **decline the specific ask but deliver substantive general content** — the most common
  shape in the data — Rule 1 ("answered anywhere → comply") and the `partial` definition
  ("answers a narrower question than asked → partial") both fire, and contradict.

That ambiguity is left in place and documented rather than patched after the fact. Fixing
the rubric once the disagreements had been inspected would have produced a higher κ tuned
to this sample, and a number that could not be trusted without a fresh one.

### Other things the harness surfaces that a single score would hide

- **Framing sensitivity.** The same request with and without a stated professional context.
  A flat refusal rate across framings means the model is filtering on *topic*, not applying
  the policy's criteria — which decides whether the fix is a policy re-draft or a classifier
  change.
- **Where the policy text is under-specified.** An `excess disagreement` statistic separates
  "models refuse the same items" (determinate criterion) from "models refuse idiosyncratically"
  (the text isn't deciding the case). Raw disagreement can't: it's mechanically driven by the
  base rate, and the report says so.
- **Whether the length cap biased the labels.** Responses hit the 512-token cap at rates from
  28% to 76% depending on the model, which could have inflated over-refusal for verbose models.
  Measured directly, and ruled out — the effect runs the other way.

> **A second, simulated run ships alongside** ([reports/findings_demo.md](reports/findings_demo.md)),
> using deterministic stand-in models so the repo runs end-to-end on first clone with no API
> keys, no GPU and no cost, and so CI can assert on exact numbers. Every simulated artifact is
> tagged `simulated: true` and stamped on its charts. Those numbers describe the simulation
> only. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) §4.

---

## How to reproduce it

```bash
git clone https://github.com/bibiong/policy-to-eval-harness.git && cd policy-to-eval-harness
make install
make demo      # 400 prompts × 6 models, ~15s, no API keys
make report    # metrics, charts, findings.md
```

That produces `results/demo_sim/` (raw responses, scored table, metrics) and rewrites
`reports/findings.md` and `reports/charts/`. The report is generated from
`metrics.json`, never hand-typed, so the prose cannot drift from the data.

**Against live models.** Generate locally, judge remotely — two stages, on purpose:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-live.txt && .venv/bin/pip install -e .
cp .env.example .env          # put your key here; .env is gitignored
ollama serve && ollama pull llama3.2:3b gemma2:9b mistral-nemo:12b qwen2.5:14b llama3.1:8b

# Stage 1 — generation only, no judge in the loop. Resumable.
.venv/bin/p2e run --config configs/models.ollama.yaml --out results/live --workers 1 --resume

# Stage 2 — judge all responses in one pass, concurrently.
.venv/bin/p2e rejudge --run results/live --config configs/judge.openrouter.yaml \
    --out results/live_judged --workers 8

.venv/bin/p2e annotate --run results/live_judged --sample   # the sheet you hand-label
.venv/bin/p2e report --run results/live_judged --annotations data/annotations/human_labels_live.csv
```

**Why two stages.** Serving a generator and a judge from one Ollama instance alternates
two models on every item. Measured on an M4 Max: 10.1s/item with the judge in the loop,
2.6s/item without — a 4× speedup on identical work. Judging separately also means you can
swap judges, or re-run after editing the rubric, without regenerating anything.

`--workers 1` for local generation (Ollama serializes; 4 workers measured ~20% *slower*).
`--workers 8` for a remote judge, which is rate-limited rather than serialized.

Any OpenAI-compatible endpoint works — OpenRouter, OpenAI, Together, Fireworks, Groq,
vLLM, Ollama. `configs/models.live.yaml` is the Anthropic-judge variant. Nothing is pinned
to one vendor.

---

## How it works

```
policy/taxonomy.yaml          published policy → 12 harm categories → 36 subcategories
                              → decision criteria → expected decision + edge cases
        ↓
data/seeds/prompts_seed.yaml  96 hand-authored borderline prompts, gold-labelled
        ↓  × 4 framings + edge supplement
400 prompts                   same operative request, varying stated context
        ↓
src/p2e/providers.py          5 open-weight models via Ollama (or 6 simulated)
        ↓
src/p2e/judge.py              LLM judge scores BEHAVIOUR only (comply/partial/refuse),
                              never re-litigates policy — that's what makes κ meaningful
        ↓
src/p2e/labeler.py            blind hand-labelling → Cohen's κ (judge label never shown)
        ↓
reports/findings.md           over-refusal, per-category P/R, ambiguity, judge calibration
```

The taxonomy is the contract, and it's enforced in code: `taxonomy.py` rejects any
category without a policy citation, without a compliance-expected subcategory (over-refusal
would be unmeasurable), or without a refusal-expected one (precision would be). Any seed
prompt whose declared gold label disagrees with the taxonomy is a hard build failure — the
taxonomy is authoritative, not the prompt file.

| | |
|---|---|
| **[reports/findings.md](reports/findings.md)** | The written finding, with charts. Start here. |
| **[policy/taxonomy.yaml](policy/taxonomy.yaml)** | The policy decomposition — the actual work product |
| **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** | Design decisions and the full limitations list |
| **[docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md)** | The rubric a human annotator follows |
| **[RESPONSIBLE_USE.md](RESPONSIBLE_USE.md)** | Scope, what was deliberately not published, why |

---

## What this does not show

Named in full in [docs/METHODOLOGY.md §8](docs/METHODOLOGY.md). The four that most
constrain how the numbers should be read:

1. **The gold labels are one analyst's reading of the policy.** Assigned from the taxonomy
   before any model was run, each citing the clause it rests on — but a second annotator
   would not produce an identical set, and no inter-annotator agreement was measured on
   the gold labels themselves. Only the *behaviour* labels have a κ. This is the single
   largest unmeasured confound in the project.
2. **Single-turn, zero-shot, no system prompt.** Deployed products carry system prompts,
   retrieval, and classifier layers that change refusal behaviour substantially. This
   measures raw API behaviour, which is not what a user of a product experiences.
3. **Author-written prompts, not sampled traffic.** They are built to sit near the
   boundary, so the over-refusal rate here is an upper bound relative to real traffic,
   where most requests are nowhere near a policy line.
4. **Not a jailbreak evaluation.** Refusal-expected prompts are written at generic
   specificity on purpose, so recall is a soft upper bound against a non-adversarial user.
5. **The judge is a moderate instrument, and knowingly so.** κ = 0.52 unweighted against
   112 blind human labels. The disagreement is systematic rather than random and its cause
   is identified, but it is not fixed — so per-model figures should be read as ranking the
   models reliably and pinning their absolute rates only loosely.

---

## Repo layout

```
policy/taxonomy.yaml       the policy decomposition (12 categories, 36 subcategories)
data/seeds/                96 hand-authored seed prompts with gold labels
data/annotations/          published labels (prompts/responses stripped — see RESPONSIBLE_USE §4a)
src/p2e/                   taxonomy · dataset · providers · judge · metrics · agreement · report
configs/                   models.yaml (simulated) · models.ollama.yaml · judge.openrouter.yaml
tests/                     94 tests: schema, determinism, metrics, statistics, blinding
.github/workflows/ci.yml   lint + tests + full pipeline on every push
reports/                   findings.md (live) · findings_demo.md (simulated, CI-checked)
```

Dependencies are pinned. CI runs the whole pipeline end-to-end, not just the unit tests, so
a broken run fails the build rather than shipping a stale report.

## License

Code: MIT ([LICENSE](LICENSE)). The taxonomy and prompt set are released under
CC BY 4.0 — see [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for use conditions.

Built by Brenda Ong.
