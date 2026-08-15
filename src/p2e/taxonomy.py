"""Load and validate the policy taxonomy.

The taxonomy is the contract between the written policy and the eval: every
prompt in the dataset must map to a (category, subcategory) that exists here,
and every subcategory carries the policy-derived expected decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_DECISIONS = {"comply", "refuse"}


@dataclass(frozen=True)
class Subcategory:
    id: str
    expected: str
    criteria: str


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    policy_refs: tuple[str, ...]
    obligation: str
    subcategories: tuple[Subcategory, ...]
    edge_cases: tuple[str, ...]
    ambiguity_note: str | None

    def subcategory(self, sub_id: str) -> Subcategory:
        for sub in self.subcategories:
            if sub.id == sub_id:
                return sub
        raise KeyError(f"{self.id} has no subcategory {sub_id!r}")


@dataclass(frozen=True)
class Taxonomy:
    version: str
    categories: tuple[Category, ...]

    def category(self, cat_id: str) -> Category:
        for cat in self.categories:
            if cat.id == cat_id:
                return cat
        raise KeyError(f"unknown category {cat_id!r}")

    @property
    def category_ids(self) -> list[str]:
        return [c.id for c in self.categories]

    def expected_decision(self, cat_id: str, sub_id: str) -> str:
        return self.category(cat_id).subcategory(sub_id).expected


def load_taxonomy(path: str | Path = "policy/taxonomy.yaml") -> Taxonomy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cats: list[Category] = []
    seen: set[str] = set()

    for entry in raw["categories"]:
        cat_id = entry["id"]
        if cat_id in seen:
            raise ValueError(f"duplicate category id {cat_id!r}")
        seen.add(cat_id)

        subs: list[Subcategory] = []
        for sub in entry["subcategories"]:
            if sub["expected"] not in VALID_DECISIONS:
                raise ValueError(
                    f"{cat_id}/{sub['id']}: expected must be one of {sorted(VALID_DECISIONS)}"
                )
            subs.append(Subcategory(sub["id"], sub["expected"], sub["criteria"]))

        if not any(s.expected == "comply" for s in subs):
            raise ValueError(f"{cat_id}: no compliance-expected subcategory — over-refusal unmeasurable")
        if not any(s.expected == "refuse" for s in subs):
            raise ValueError(f"{cat_id}: no refusal-expected subcategory — precision unmeasurable")
        if not entry.get("policy_refs"):
            raise ValueError(f"{cat_id}: every category must cite at least one policy clause")

        cats.append(
            Category(
                id=cat_id,
                name=entry["name"],
                policy_refs=tuple(entry["policy_refs"]),
                obligation=entry["obligation"].strip(),
                subcategories=tuple(subs),
                edge_cases=tuple(entry.get("edge_cases", ())),
                ambiguity_note=(entry.get("ambiguity_note") or "").strip() or None,
            )
        )

    return Taxonomy(version=str(raw["meta"]["version"]), categories=tuple(cats))
