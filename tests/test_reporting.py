"""Tests for the run manifest and attack-success labelling."""

from pathlib import Path

from goldx.reporting import (
    _n_cases,
    _success_label,
    load_manifest,
    write_manifest,
)


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = {
        "source_images": 9,
        "attacks_per_image": 2,
        "attack_attempts": 18,
        "attack_successes": 15,
    }
    write_manifest(tmp_path, manifest)
    assert load_manifest(tmp_path) == manifest


def test_load_manifest_absent_is_none(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) is None


def test_n_cases_counts_distinct_case_target() -> None:
    records = [
        {"case": "a", "target": 1},
        {"case": "a", "target": 1},  # same heatmap row, same case/target
        {"case": "a", "target": 2},
        {"case": "b", "target": 1},
    ]
    assert _n_cases(records) == 3


def test_success_label_uses_attempts_denominator(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"attack_attempts": 18})
    records = [
        {"case": "a", "target": 1},
        {"case": "b", "target": 2},
    ]
    assert _success_label(tmp_path, records) == "2/18"


def test_success_label_none_without_manifest(tmp_path: Path) -> None:
    assert _success_label(tmp_path, [{"case": "a", "target": 1}]) is None
