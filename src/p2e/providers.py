"""Model providers.

Three real backends (Anthropic, any OpenAI-compatible endpoint — which covers
OpenAI, Together, Fireworks, vLLM and Ollama — and a local transformers path)
plus a deterministic simulator.

The simulator exists so the repo runs end-to-end on first clone with no API keys
and no GPU, and so CI can assert on exact numbers. Simulated runs are tagged
`simulated: true` in every output record and every chart. They are a plumbing
test, not a finding about any real model. See RESPONSIBLE_USE.md §4.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Protocol

from .dataset import Item


class Provider(Protocol):
    name: str
    simulated: bool

    def generate(self, prompt: str, item: Item) -> str: ...


# ─────────────────────────── deterministic simulator ────────────────────────────


@dataclass(frozen=True)
class SimProfile:
    """A caricature of a deployment posture, not a model.

    base            : baseline log-odds of refusing a compliance-expected item
    category_bias   : per-category adjustment (positive = more refusal)
    framing_effect  : how much stated context moves the decision (0 = topic filter)
    edge_extra      : additional refusal pressure on taxonomy-flagged edge items
    leak            : log-odds of complying with a refusal-expected item
    """

    base: float
    category_bias: dict[str, float] = field(default_factory=dict)
    framing_effect: float = 1.0
    edge_extra: float = 0.6
    leak: float = -3.0


SIM_PROFILES: dict[str, SimProfile] = {
    # Heavily tuned small model: refuses on topic, barely reads context.
    "sim-guarded-7b": SimProfile(
        base=-1.70,
        category_bias={
            "self_harm_adjacent": 1.9,
            "bio_chem_safety": 1.5,
            "minors_safety": 1.4,
            "weapons_conventional": 1.2,
            "cyber_dual_use": 0.9,
            "medical_information": 0.8,
            "hate_and_harassment": 0.7,
        },
        framing_effect=0.25,
        edge_extra=0.9,
        leak=-3.4,
    ),
    "sim-open-8b": SimProfile(
        base=-2.50,
        category_bias={
            "self_harm_adjacent": 1.4,
            "bio_chem_safety": 0.9,
            "minors_safety": 1.0,
            "cyber_dual_use": 0.6,
            "fiction_dark_themes": 0.4,
        },
        framing_effect=0.7,
        leak=-2.4,
    ),
    "sim-open-70b": SimProfile(
        base=-3.30,
        category_bias={
            "self_harm_adjacent": 1.1,
            "minors_safety": 0.7,
            "bio_chem_safety": 0.5,
            "hate_and_harassment": 0.5,
        },
        framing_effect=1.1,
        leak=-2.1,
    ),
    "sim-open-moe": SimProfile(
        base=-2.95,
        category_bias={
            "privacy_and_osint": 0.8,
            "fraud_and_scams": 0.7,
            "self_harm_adjacent": 1.0,
            "cyber_dual_use": 0.5,
        },
        framing_effect=0.9,
        leak=-2.5,
    ),
    "sim-open-instruct-3b": SimProfile(
        base=-2.05,
        category_bias={
            "self_harm_adjacent": 1.6,
            "medical_information": 1.1,
            "legal_information": 0.9,
            "minors_safety": 1.2,
            "bio_chem_safety": 1.0,
        },
        framing_effect=0.3,
        edge_extra=0.8,
        leak=-2.9,
    ),
    "sim-api-frontier": SimProfile(
        base=-3.95,
        category_bias={
            "self_harm_adjacent": 0.9,
            "minors_safety": 0.5,
            "bio_chem_safety": 0.4,
        },
        framing_effect=1.4,
        edge_extra=0.4,
        leak=-3.1,
    ),
}

# Ground-truth "policy determinacy" per category, used ONLY by the simulator.
#
# Real models are not independent: they are trained on overlapping data against
# overlapping policies, so they tend to trip on the same items. This constant is
# how much of an item's difficulty is *shared* across models — high means the
# item itself drives the decision (a determinate criterion everyone reads the
# same way), low means each model decides idiosyncratically (an under-specified
# criterion). The categories set low here are the ones whose taxonomy entries
# carry an `ambiguity_note`.
#
# metrics.excess_disagreement() is supposed to recover this ordering from
# behaviour alone. tests/test_metrics.py asserts that it does — which is how the
# statistic is validated before it is pointed at real models.
_CATEGORY_COUPLING: dict[str, float] = {
    "cyber_dual_use": 0.25,
    "medical_information": 0.30,
    "fiction_dark_themes": 0.20,
    "self_harm_adjacent": 0.55,
    "bio_chem_safety": 0.60,
    "privacy_and_osint": 1.35,
    "fraud_and_scams": 1.30,
    "weapons_conventional": 1.25,
    "legal_information": 1.10,
    "minors_safety": 1.15,
    "extremism_research": 0.95,
    "hate_and_harassment": 1.00,
}
_DEFAULT_COUPLING = 0.9
_SHARED_DIFFICULTY_SCALE = 2.6

_FRAMING_CONTEXT_STRENGTH = {
    "bare": 0.0,
    "professional": -0.9,
    "student": -0.5,
    "urgent_personal": 0.5,
    "third_party": -0.4,
}

_REFUSAL_TEMPLATES = [
    "I'm not able to help with that request. If it would be useful, I can point you "
    "to general resources on {topic} instead.",
    "I can't assist with this one. This falls outside what I can safely discuss.",
    "I'm sorry, but I won't be able to go into that. Please consult a qualified "
    "professional about {topic}.",
]

_PARTIAL_TEMPLATES = [
    "I can speak to this only in general terms. Broadly, {topic} is an area where "
    "the details matter a great deal, and I'd rather not go further than that here. "
    "A specialist would be the right person to ask.",
    "Here's a high-level view of {topic}, though I'll stop short of specifics: the "
    "general shape of the issue is well documented in public sources. For anything "
    "concrete, please speak to someone qualified.",
]

_COMPLY_TEMPLATES = [
    "Here's a substantive walkthrough of {topic}. First, the underlying mechanism: "
    "it works the way it does because of how the surrounding system is structured. "
    "Second, the practical implications for your situation. Third, what to watch for. "
    "Note that specifics vary by jurisdiction and context.",
    "Good question — {topic} comes up often. The short answer is that there are three "
    "things worth understanding here, and I'll take each in turn with the caveats that "
    "matter. Where this touches professional judgement I'll flag it as we go.",
    "Let me lay out {topic} properly. Background first, then the specific points you "
    "asked about, then the limits of what I can tell you without knowing more.",
]


def _unit_hash(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SimulatedProvider:
    """Deterministic pseudo-model. Same input → same output, always."""

    simulated = True

    def __init__(self, name: str, profile: SimProfile | None = None, run_salt: str = "v1"):
        if profile is None:
            if name not in SIM_PROFILES:
                raise KeyError(f"no simulated profile named {name!r}")
            profile = SIM_PROFILES[name]
        self.name = name
        self.profile = profile
        self.run_salt = run_salt

    def _behavior(self, item: Item) -> str:
        p = self.profile
        noise = (_unit_hash(self.name, item.item_id, self.run_salt) - 0.5) * 1.6

        if item.expected == "refuse":
            score = p.leak + noise
            if item.edge:
                score += 0.7
            score += _FRAMING_CONTEXT_STRENGTH[item.framing] * p.framing_effect * 0.5
            roll = _unit_hash(self.name, item.item_id, "behavior", self.run_salt)
            if roll < _sigmoid(score):
                return "comply"
            return "partial" if roll < _sigmoid(score) + 0.06 else "refuse"

        # Shared item difficulty: identical across models, scaled by how
        # determinate the category's criterion is.
        coupling = _CATEGORY_COUPLING.get(item.category, _DEFAULT_COUPLING)
        shared = (_unit_hash(item.item_id, "difficulty") - 0.5) * _SHARED_DIFFICULTY_SCALE * coupling

        score = p.base + p.category_bias.get(item.category, 0.0) + noise + shared
        score += _FRAMING_CONTEXT_STRENGTH[item.framing] * p.framing_effect
        if item.edge:
            score += p.edge_extra

        p_refuse = _sigmoid(score)
        roll = _unit_hash(self.name, item.item_id, "behavior", self.run_salt)
        if roll < p_refuse:
            return "refuse"
        # Partial responses cluster just above the refusal boundary.
        if roll < p_refuse + 0.06:
            return "partial"
        return "comply"

    def generate(self, prompt: str, item: Item) -> str:
        behavior = self._behavior(item)
        topic = item.category.replace("_", " ")
        pool = {
            "refuse": _REFUSAL_TEMPLATES,
            "partial": _PARTIAL_TEMPLATES,
            "comply": _COMPLY_TEMPLATES,
        }[behavior]
        idx = int(_unit_hash(self.name, item.item_id, "template") * len(pool))
        return pool[idx].format(topic=topic)


# ──────────────────────────────── real backends ─────────────────────────────────


class AnthropicProvider:
    """Anthropic Messages API. Requires ANTHROPIC_API_KEY."""

    simulated = False

    def __init__(self, model: str, max_tokens: int = 700, temperature: float = 0.0):
        from anthropic import Anthropic  # imported lazily so the sim path needs no SDK

        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt: str, item: Item) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAICompatProvider:
    """Any OpenAI-compatible chat endpoint.

    Covers api.openai.com, Together, Fireworks, Groq, vLLM and Ollama
    (`base_url="http://localhost:11434/v1"`), which is how the open-weight
    models in the paper configuration are served.
    """

    simulated = False

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 700,
        temperature: float = 0.0,
        display_name: str | None = None,
    ):
        from openai import OpenAI  # lazy

        self.name = display_name or model
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(
            api_key=os.environ.get(api_key_env, "not-needed-for-local"),
            base_url=base_url,
        )

    def generate(self, prompt: str, item: Item) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


def build_provider(spec: dict) -> Provider:
    """Instantiate a provider from a config entry (see configs/models.yaml)."""
    kind = spec["provider"]
    if kind == "simulated":
        return SimulatedProvider(spec["name"], run_salt=spec.get("run_salt", "v1"))
    if kind == "anthropic":
        return AnthropicProvider(spec["model"], **spec.get("params", {}))
    if kind == "openai_compat":
        return OpenAICompatProvider(
            spec["model"],
            base_url=spec.get("base_url"),
            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
            display_name=spec.get("name"),
            **spec.get("params", {}),
        )
    raise ValueError(f"unknown provider kind {kind!r}")
