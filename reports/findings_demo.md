# Findings — over-refusal on policy-permitted requests

> **This run is simulated.** Every model in it is a deterministic stand-in shipped with the repo so the pipeline runs on first clone with no API keys. The numbers below are a property of that simulation and say nothing about any real model. Swap in `configs/models.live.yaml` and re-run to produce real measurements; the report regenerates itself from whatever run you point it at. See [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md) §4.

**Run:** `demo_sim` · **taxonomy** v1.0 · 400 prompts × 6 models = 2400 scored responses · **judge** `sim-judge`

## The question

A usage policy is a written document. A deployed model is a decision procedure. Nobody publishes the mapping between them. This harness builds that mapping explicitly — policy clause → harm category → subcategory → decision criteria → expected decision — and then measures how far actual model behaviour sits from the policy as written.

The measurement target is deliberately the *benign* side. Refusing a legitimate request is a product failure that costs real users real help, it is systematically under-measured because it is less alarming than the opposite error, and — unlike measuring harmful compliance — it can be studied and published without generating or distributing harmful content.

## What the run found

**26.8% of policy-permitted requests did not get a substantive answer** (95% CI 24.8%–28.9%, n=1818). Counting only outright refusals and excluding hedged non-answers, the rate is 15.9%.

Leakage in the other direction — substantive answers to requests the taxonomy marks as prohibited — was 6.9% (n=582). Pooled precision on the refuse class was 0.53 and recall 0.93.

Over-refusal is the larger error, by a factor of 3.9: the models catch most of what the policy asks them to catch and pay for it with false positives on legitimate traffic.

The spread across models is wide — 14.5% for `sim-api-frontier` against 44.9% for `sim-guarded-7b` — on identical prompts against an identical policy. That spread is the finding. It means the policy text is not what determines the behaviour; the tuning is.

![over-refusal by model](charts_demo/over_refusal_by_model.png)

### Per-model

| model | over-refusal | 95% CI | strict | leakage | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sim-api-frontier` | 14.5% | 11.0%–18.9% | 2.3% | 6.2% | 0.67 | 0.94 | 0.78 |
| `sim-open-70b` | 18.5% | 14.5%–23.2% | 6.9% | 11.3% | 0.61 | 0.89 | 0.72 |
| `sim-open-moe` | 19.5% | 15.4%–24.3% | 7.6% | 3.1% | 0.61 | 0.97 | 0.75 |
| `sim-open-8b` | 26.7% | 22.1%–32.0% | 15.2% | 9.3% | 0.52 | 0.91 | 0.66 |
| `sim-open-instruct-3b` | 36.6% | 31.4%–42.2% | 27.4% | 5.2% | 0.45 | 0.95 | 0.61 |
| `sim-guarded-7b` | 44.9% | 39.4%–50.5% | 36.0% | 6.2% | 0.40 | 0.94 | 0.56 |

### Per-category

Over-refusal is not spread evenly. `self_harm_adjacent` is the worst at 36.0% of permitted requests unanswered — and it is the category where the cost of a non-answer is highest, because the legitimate traffic is clinicians, researchers, journalists and people supporting someone in crisis.

![over-refusal by category](charts_demo/over_refusal_by_category.png)

| category | over-refusal | 95% CI | precision | recall | disagreement | excess |
|---|---:|---:|---:|---:|---:|---:|
| self harm adjacent | 36.0% | 28.8%–43.9% | 0.46 | 0.96 | 0.45 | -0.03 |
| minors safety | 34.7% | 27.5%–42.6% | 0.47 | 0.96 | 0.41 | -0.05 |
| weapons conventional | 31.3% | 24.5%–39.1% | 0.48 | 0.92 | 0.41 | -0.03 |
| bio chem safety | 30.0% | 23.2%–37.8% | 0.49 | 0.92 | 0.43 | -0.00 |
| hate and harassment | 29.5% | 22.9%–37.1% | 0.50 | 0.96 | 0.39 | -0.03 |
| medical information | 26.9% | 20.6%–34.4% | 0.51 | 0.90 | 0.41 | -0.00 |
| cyber dual use | 24.7% | 18.5%–32.1% | 0.55 | 0.94 | 0.40 | +0.02 |
| fiction dark themes | 23.7% | 17.7%–31.0% | 0.56 | 0.87 | 0.38 | +0.01 |
| legal information | 22.7% | 16.7%–30.0% | 0.56 | 0.92 | 0.32 | -0.03 |
| fraud and scams | 22.2% | 16.2%–29.7% | 0.58 | 0.92 | 0.33 | -0.02 |
| extremism research | 21.8% | 16.0%–28.9% | 0.58 | 0.98 | 0.34 | -0.01 |
| privacy and osint | 18.0% | 12.7%–24.9% | 0.63 | 0.96 | 0.27 | -0.02 |

## Where the policy language itself is the problem

The obvious move here is to read the raw disagreement column as an ambiguity score. It isn't one. Pairwise disagreement is mechanically maximised when a category is refused about half the time, so it largely re-reports the over-refusal rate — which is why the two columns above track each other so closely.

The `excess` column is the corrected version: observed disagreement minus what each model's own category-level refusal rate already predicts under independence. It isolates the correlation structure. Strongly negative means the models refuse *the same* items — the criterion is determinate and they agree on where it falls, even if they are collectively too strict. Near zero means refusals are uncorrelated: nothing shared is driving the call, which is the signature of policy language that does not settle the case.

![ambiguity vs over-refusal](charts_demo/ambiguity_vs_over_refusal.png)

The three least-determined categories in this run, ranked by excess disagreement:

**Cyber — dual-use security knowledge** — excess disagreement +0.02 (raw 0.40), over-refusal 24.7%.
> Both policies gate on *authorization* and *uplift*, neither of which is verifiable from a single prompt. This is the largest source of expected model disagreement.

**Fiction with dark themes** — excess disagreement +0.01 (raw 0.38), over-refusal 23.7%.
> The test is *extractability*: would the passage be useful if the character names were deleted? Models tend to gate on theme darkness instead.

**Medical information and self-care** — excess disagreement -0.00 (raw 0.41), over-refusal 26.9%.
> 'Tailored professional advice' is defined by the *relationship*, not the content. A single prompt rarely establishes the relationship.

In each case the policy gates on something a single prompt cannot establish: authorization, professional relationship, or downstream use. That is not a drafting error so much as an unavoidable property of writing rules for a context-free channel — but it does mean the enforcement burden falls on inference about the user, and that is where the variance lives.

### Does the excess-disagreement statistic actually work?

In a simulated run this can be checked directly, because the simulator has a ground-truth per-category coupling constant — how much of an item's difficulty is shared across models rather than idiosyncratic — that the metric never sees. Recovering that ordering from behaviour alone gives Spearman ρ = **0.64**, and the two categories the simulator made least determinate come back both in the top two by excess disagreement.

That is a real but noisy recovery, and the noise is informative: with roughly 25 permitted items per category, the statistic ranks categories usefully but should not be read to two decimal places. It is a screening tool for deciding which policy clauses to re-draft, not an estimate of anything.

## Does stated context change anything?

Each seed prompt appears under four framings — bare, student, professional, and urgent-personal — with the operative request held constant. Both source policies make context legally load-bearing, so a model that tracks the policy should refuse less when a legitimate professional context is stated.

The fifth framing, `third_party`, is excluded from this comparison: it was applied only to the edge-flagged supplement, so putting it on the same axis would compare framings across different and systematically harder item sets, and an item-selection effect would read as a framing effect.

![framing sensitivity](charts_demo/framing_sensitivity.png)

| model | bare | student | professional | urgent personal | context gain |
|---|---:|---:|---:|---:|---:|
| `sim-api-frontier` | 20.8% | 12.5% | 9.7% | 16.7% | 11.1% |
| `sim-guarded-7b` | 48.6% | 38.9% | 43.1% | 47.2% | 5.6% |
| `sim-open-70b` | 16.7% | 19.4% | 19.4% | 20.8% | -2.8% |
| `sim-open-8b` | 27.8% | 27.8% | 20.8% | 27.8% | 6.9% |
| `sim-open-instruct-3b` | 34.7% | 34.7% | 34.7% | 38.9% | 0.0% |
| `sim-open-moe` | 20.8% | 16.7% | 13.9% | 27.8% | 6.9% |

`sim-api-frontier` moves the most, refusing 11.1% fewer permitted requests once a professional context is stated. `sim-open-70b` sits at the other end at -2.8% — a *negative* gain, meaning it refuses slightly more when context is supplied, the opposite of what the policy criteria imply.

Flat lines are the ones to look at: `sim-open-instruct-3b` returns an identical rate whether or not a context is given. That is the signature of a topic-level filter — the model is matching on subject matter, not applying the policy's criteria.

The distinction matters for enforcement design. A model that responds to context but responds wrongly can be addressed by re-drafting the criteria it is reading. A topic filter cannot: no amount of policy rewriting reaches it, because it is not reading the policy. Those are different remediation budgets owned by different teams, and a single aggregate refusal rate does not distinguish them.

### Edge cases

| model | clear-cut permitted | taxonomy-flagged edge | gap |
|---|---:|---:|---:|
| `sim-guarded-7b` | 37.0% | 57.1% | 20.2% |
| `sim-open-instruct-3b` | 29.9% | 47.1% | 17.2% |
| `sim-open-8b` | 25.0% | 29.4% | 4.4% |
| `sim-open-70b` | 17.4% | 20.2% | 2.8% |
| `sim-open-moe` | 19.0% | 20.2% | 1.1% |
| `sim-api-frontier` | 15.8% | 12.6% | -3.2% |

Items the taxonomy flagged as genuinely contested before any model was run are refused more often than clear-cut permitted items. That the gap is positive is reassuring for the taxonomy: the edge flags were assigned from the policy text alone, and they predict where models struggle.

## Is the judge trustworthy?

> The annotations in this run were produced by a **simulated annotator** for pipeline testing. The κ below is a property of the simulation. On a real run this section reports agreement against labels a human actually assigned using [docs/ANNOTATION_GUIDE.md](../docs/ANNOTATION_GUIDE.md).

Cohen's κ between the LLM judge and the hand-labelled stratified sample was **0.73** (95% CI 0.64–0.82, n=143), which is "substantial" on the Landis–Koch bands. Raw agreement was 81.8% against a chance level of 33.3% — the gap between those two numbers is why κ and not accuracy is the right statistic here.

The cue-matching baseline reaches κ=0.92 against the same labels — *higher* than the LLM judge. That is an artifact of the simulation, not a result: the simulated annotator is itself derived from the cue rules, so the baseline is being scored against a near-copy of itself. The comparison only carries information on a live run with real human labels, where neither rater has privileged access to how the responses were generated. It is reported here anyway rather than suppressed, because a baseline that beats the instrument is exactly the kind of thing a reader should be able to see.

**The disagreement is one-directional.** Of 26 disagreements, 16 have the judge crediting *less* delivered substance than the human and only 10 the reverse — 61.5% in one direction, and 100.0% of them one notch apart rather than at opposite ends. Random rater noise splits evenly and scatters. This is a calibration offset: a fixable property of the rubric, not irreducible error.

Because the labels are ordinal (refuse < partial < comply), unweighted κ charges a one-notch split the same as a total reversal. Linear-weighted κ, which does not, is **0.80** (quadratic 0.87). Both are given: the unweighted figure is the conservative one, and the weighted figure reflects how the labels are actually consumed downstream, where `partial` and `refuse` collapse onto the same decision.

![judge vs human](charts_demo/judge_human_agreement.png)

| human ↓ / judge → | comply | partial | refuse |
|---|---:|---:|---:|
| **comply** | 45 | 13 | 0 |
| **partial** | 2 | 27 | 3 |
| **refuse** | 0 | 8 | 45 |

The errors are concentrated on the `partial` boundary, which is the expected place for them: deciding whether a hedged answer supplied the operative substance is a judgement call, and it is the same call human annotators are slowest and least consistent on. Practically, this means the strict and lenient over-refusal variants above bracket the truth, and the strict figure is the one to quote if only one number is wanted.

## What this does not show

See [docs/METHODOLOGY.md §8](../docs/METHODOLOGY.md) for the full list. The four that most constrain the reading of the numbers above:

1. **The gold labels are one analyst's reading of the policy.** They were assigned from the taxonomy before any model was run, and the taxonomy cites the clause each one rests on, but a second annotator working from the same policies would not produce an identical set. No inter-annotator agreement was measured on the gold labels themselves — only on the behaviour labels. That is the single largest unmeasured confound in this project.
2. **Single-turn, zero-shot, no system prompt.** Deployed products carry system prompts, retrieval, and classifier layers that change refusal behaviour substantially. These numbers describe raw API behaviour, which is not what a user of a deployed product experiences.
3. **Prompts are author-written, not sampled traffic.** They are constructed to sit near the boundary, so the over-refusal rate here is an upper bound on what real traffic would show, where most requests are nowhere near a policy line.
4. **Refusal-expected items are written at generic specificity on purpose.** That makes the recall figures a soft upper bound: a determined adversarial phrasing is not what is being tested here, and this harness should not be read as a jailbreak evaluation.

## Reproducing

```bash
make install
make demo      # simulated end-to-end run, ~10s, no keys
make report
```

For a live run: copy `configs/models.live.yaml`, set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` (or point `base_url` at a local Ollama/vLLM server for the open-weight models), then `p2e run --config configs/models.live.yaml --out results/live` followed by `p2e annotate --run results/live --sample` to produce the annotation sheet for hand-labelling.
