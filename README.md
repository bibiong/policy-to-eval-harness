# Policy-to-Eval Harness

**Can a written usage policy be turned into something you can actually measure?**

Every frontier lab publishes a usage policy. Every lab enforces it with models. Almost
nobody publishes the mapping between the two — the step where "do not provide material
assistance toward unauthorized intrusion" becomes a decision an evaluator can score. This
repo builds that step explicitly, and then measures how far model behaviour sits from the
policy as written.

It measures the direction that is usually left unmeasured: **over-refusal**. Not "which
model is safer" — refusing a legitimate request is a real failure with a real cost, it is
systematically under-reported because it is less alarming than the opposite error, and
unlike measuring harmful compliance it can be studied and published without generating a
single piece of harmful content.

---

## The finding

In the run shipped with this repo — **[reports/findings.md](reports/findings.md)** —
**26.8% of policy-permitted requests did not get a substantive answer** (95% CI
24.8–28.9%, n=1,818), against 6.9% leakage in the other direction. The spread across
models on identical prompts ran from 14.5% to 44.9%. Over-refusal concentrated in
`self_harm_adjacent` (36.0%) and `minors_safety` (34.7%) — the two categories where the
legitimate traffic is clinicians, safeguarding staff, and people trying to help someone
in crisis, and where a non-answer costs the most.

Three results the harness produces that a single "safety score" would hide:

- **Framing sensitivity.** The same request under a stated professional context versus no
  context. Models whose refusal rate is flat across framings are filtering on *topic*, not
  applying the policy's criteria — a distinction that determines whether the fix is a
  policy re-draft or a classifier change.
- **Where the policy text itself is under-specified.** An `excess disagreement` statistic
  isolates whether models refuse *the same* items (determinate criterion) or refuse
  idiosyncratically (the criterion isn't deciding the case). Raw disagreement can't do
  this — it's mechanically driven by the base rate, and the report says so.
- **Whether the judge can be trusted.** Cohen's κ = 0.73 against a hand-labelled
  stratified sample, with the confusion matrix shown, because an uncalibrated LLM judge
  makes every number above it unfalsifiable.

> ### ⚠️ The shipped run is simulated
>
> The six models in the default config are **deterministic stand-ins**, not real models,
> so the repo runs end-to-end on first clone with no API keys, no GPU, and no cost — and
> so CI can assert on exact numbers. Every simulated output is tagged `simulated: true`,
> every chart carries a "SIMULATED RUN" stamp, and the report opens with a banner.
>
> **The numbers above are a property of that simulation and are not measurements of any
> deployed model.** They demonstrate that the instrument works. Point the harness at live
> models with `configs/models.live.yaml` and every artifact regenerates from real data.
> See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) §4.

---

## How to reproduce it

```bash
git clone <this-repo> && cd policy-to-eval-harness
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
src/p2e/providers.py          6 models (open-weight + API, or simulated)
        ↓
src/p2e/judge.py              LLM judge scores BEHAVIOUR only (comply/partial/refuse),
                              never re-litigates policy — that's what makes κ meaningful
        ↓
data/annotations/             stratified sample, hand-labelled → Cohen's κ
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

---

## Repo layout

```
policy/taxonomy.yaml       the policy decomposition (12 categories, 36 subcategories)
data/seeds/                96 hand-authored seed prompts with gold labels
data/annotations/          the stratified sample used for judge calibration
src/p2e/                   taxonomy · dataset · providers · judge · metrics · agreement · report
configs/                   models.yaml (simulated) · models.ollama.yaml · judge.openrouter.yaml
tests/                     67 tests: schema, determinism, metric correctness, statistics
.github/workflows/ci.yml   lint + tests + full pipeline on every push
reports/                   findings.md and charts, both regenerated by `make report`
```

Dependencies are pinned. CI runs the whole pipeline end-to-end, not just the unit tests, so
a broken run fails the build rather than shipping a stale report.

## License

Code: MIT ([LICENSE](LICENSE)). The taxonomy and prompt set are released under
CC BY 4.0 — see [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for use conditions.

Built by Brenda Ong.
