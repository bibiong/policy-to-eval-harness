"""Schema, invariant, and determinism tests for the policy layer."""

from __future__ import annotations

import pytest
import yaml
from p2e.dataset import PRIMARY_FRAMINGS, TARGET_SIZE, build_dataset, read_dataset, write_dataset
from p2e.taxonomy import load_taxonomy

SEEDS = "data/seeds/prompts_seed.yaml"


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


@pytest.fixture(scope="module")
def items(taxonomy):
    return build_dataset(taxonomy, SEEDS)


def test_taxonomy_has_twelve_categories(taxonomy):
    assert len(taxonomy.categories) == 12


def test_every_category_cites_a_policy_clause(taxonomy):
    for cat in taxonomy.categories:
        assert cat.policy_refs, f"{cat.id} cites no policy clause"
        assert all(":" in ref for ref in cat.policy_refs), f"{cat.id} has a malformed clause id"


def test_every_category_can_measure_both_error_directions(taxonomy):
    """Without both, over-refusal or precision is undefined for that category."""
    for cat in taxonomy.categories:
        expectations = {s.expected for s in cat.subcategories}
        assert "comply" in expectations, f"{cat.id}: over-refusal unmeasurable"
        assert "refuse" in expectations, f"{cat.id}: precision unmeasurable"


def test_ambiguity_notes_exist_where_claimed(taxonomy):
    """The report leans on these; a missing one would render an empty quote block."""
    flagged = [c for c in taxonomy.categories if c.ambiguity_note]
    assert len(flagged) >= 4
    assert all(len(c.ambiguity_note) > 40 for c in flagged)


def test_dataset_is_exactly_400_items(items):
    assert len(items) == TARGET_SIZE


def test_dataset_is_benign_weighted(items):
    """The measurement target is over-refusal, so permitted items must dominate."""
    benign = sum(1 for i in items if i.expected == "comply")
    assert benign / len(items) > 0.7


def test_all_twelve_categories_are_represented(items, taxonomy):
    assert {i.category for i in items} == set(taxonomy.category_ids)


def test_item_ids_are_unique(items):
    assert len({i.item_id for i in items}) == len(items)


def test_every_seed_appears_under_every_primary_framing(items):
    by_seed: dict[str, set[str]] = {}
    for item in items:
        by_seed.setdefault(item.seed_id, set()).add(item.framing)
    for seed_id, framings in by_seed.items():
        assert set(PRIMARY_FRAMINGS) <= framings, f"{seed_id} missing a primary framing"


def test_framing_preserves_the_operative_request(items):
    """A framing prefix must not alter the seed text — otherwise the framing
    comparison is confounded with a content change."""
    bare = {i.seed_id: i.prompt for i in items if i.framing == "bare"}
    for item in items:
        if item.framing != "bare":
            assert item.prompt.endswith(bare[item.seed_id]), item.item_id


def test_framing_does_not_change_the_gold_label(items):
    by_seed: dict[str, set[str]] = {}
    for item in items:
        by_seed.setdefault(item.seed_id, set()).add(item.expected)
    assert all(len(v) == 1 for v in by_seed.values())


def test_build_is_deterministic(taxonomy):
    a = build_dataset(taxonomy, SEEDS)
    b = build_dataset(taxonomy, SEEDS)
    assert [i.item_id for i in a] == [i.item_id for i in b]
    assert [i.prompt for i in a] == [i.prompt for i in b]


def test_roundtrip_through_jsonl(items, tmp_path):
    path = write_dataset(items, tmp_path / "d.jsonl")
    assert read_dataset(path) == items


def test_seed_labels_must_match_the_taxonomy(taxonomy, tmp_path):
    """The taxonomy is authoritative; a prompt cannot quietly override it."""
    raw = yaml.safe_load(open(SEEDS, encoding="utf-8"))
    raw["seeds"][0]["expected"] = "refuse" if raw["seeds"][0]["expected"] == "comply" else "comply"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy is authoritative"):
        build_dataset(taxonomy, bad)


def test_seed_subcategories_all_exist(taxonomy):
    raw = yaml.safe_load(open(SEEDS, encoding="utf-8"))
    for seed in raw["seeds"]:
        taxonomy.category(seed["category"]).subcategory(seed["subcategory"])


def test_edge_flags_are_present_but_not_dominant(items):
    edge = sum(1 for i in items if i.edge)
    assert 0 < edge < len(items) * 0.5
