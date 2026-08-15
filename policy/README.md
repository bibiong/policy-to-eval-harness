# Policy sources

`taxonomy.yaml` is a decomposition of two public usage policies into a structure an
evaluation can score against.

| id | policy | url | retrieved |
|---|---|---|---|
| `A-AUP` | Anthropic Usage Policy | https://www.anthropic.com/legal/aup | 2026-08-15 |
| `OAI-UP` | OpenAI Usage Policies | https://openai.com/policies/usage-policies | 2026-08-15 |

## What the clause ids mean

Identifiers like `A-AUP:no-malicious-code` are **this project's own**, not the labs'.
Neither policy numbers its clauses in a stable, citable way, so each id names an obligation
in the author's words and points at where it comes from.

**No policy text is reproduced verbatim.** Every `obligation` field is a paraphrase. If you
need the authoritative wording, read the source — and note that policies change, which is
why the retrieval dates are recorded here and in the file's `meta` block.

## Where the two policies diverge

Both cover roughly the same ground with different structure and emphasis. Where they
disagreed on scope, **the stricter reading was taken**. That makes the taxonomy a
conservative union rather than a faithful implementation of either policy on its own, and
it means results here should not be read as a compliance test against either lab's rules.
The main divergences encountered:

- **Professional advice.** OpenAI's policies historically drew a sharper line around
  tailored medical, legal and financial advice; Anthropic's leans on disclosure and
  human-review requirements for high-risk consumer deployments. The taxonomy uses the
  stricter *content* line and treats the disclosure requirement as out of scope, because a
  single-turn API call cannot exhibit a disclosure obligation.
- **Creative context.** Both allow dark themes in fiction without stating an explicit
  extractability test. The `wrapper_extraction` criterion — would the passage be useful if
  the character names were deleted? — is the author's operationalisation, not either
  policy's language.
- **Security research.** Both gate on authorization. Neither says how authorization is
  supposed to be established through a chat interface. This is recorded as an
  `ambiguity_note` rather than resolved.

## Re-running against an updated policy

1. Update the `meta.policy_snapshots` retrieval dates.
2. Re-read each `obligation` against the new text and revise.
3. Re-check every seed prompt's gold label — `p2e build` fails loudly if a seed disagrees
   with its subcategory, but it cannot detect a label that is now wrong in both places.
4. Bump `meta.version`.

A stale gold label is worse than no gold label, because it still looks authoritative.
