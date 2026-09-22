import json
from copy import deepcopy

import pytest

from blackjack.experiment_data import atomic_json
from blackjack.research import checkpoint_identity, comparison_summary, freeze_checkpoint, freeze_run


def checkpoint(root):
    atomic_json(root / "training_report.json", {"test": {"accuracy": 0.9}, "test_deferred": False})
    atomic_json(root / "rl_agent_config.json", {"temperature": [1, 1, 1]})
    atomic_json(root / "encoder/config.json", {"model_type": "test"})
    atomic_json(root / "tokenizer/tokenizer.json", {"vocab": {"hit": 0}})
    (root / "model.safetensors").write_bytes(b"frozen-weights")
    return root


def test_snapshot_binds_tokenizer_config_and_weights_without_mutating_source(tmp_path):
    source = checkpoint(tmp_path / "source")
    identity = checkpoint_identity(source)
    frozen = freeze_checkpoint(source, tmp_path / "snapshot", identity)
    assert checkpoint_identity(frozen) == identity
    atomic_json(source / "tokenizer/tokenizer.json", {"vocab": {"stand": 0}})
    assert checkpoint_identity(source) != identity
    assert checkpoint_identity(frozen) == identity
    assert freeze_checkpoint(source, frozen, identity) == frozen
    with pytest.raises(ValueError, match="changed while freezing"):
        freeze_checkpoint(source, tmp_path / "invalid", identity)
    assert not (tmp_path / "invalid").exists()
    (frozen / "model.safetensors").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="Frozen checkpoint changed"):
        freeze_checkpoint(source, frozen, identity)


def test_sealed_or_incomplete_candidates_cannot_enter_return_comparison(tmp_path):
    source = checkpoint(tmp_path / "source")
    atomic_json(source / "training_report.json", {"test": None, "test_deferred": True})
    with pytest.raises(ValueError, match="sealed"):
        checkpoint_identity(source)
    (source / "encoder/config.json").unlink()
    with pytest.raises(ValueError, match="complete local"):
        checkpoint_identity(source)


def test_research_resume_preserves_original_deadline_and_input_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("blackjack.research.time.time", lambda: 1000)
    config = {"hours": 1, "seed": 42, "checkpoints": {"candidate": {"tokenizer": "v1"}}}
    frozen = freeze_run(tmp_path, config)
    assert frozen["deadline_unix"] == 4600
    monkeypatch.setattr("blackjack.research.time.time", lambda: 1200)
    assert freeze_run(tmp_path, config) == frozen
    changed = deepcopy(config)
    changed["checkpoints"]["candidate"]["tokenizer"] = "v2"
    with pytest.raises(ValueError, match="configuration changed"):
        freeze_run(tmp_path, changed)
    with pytest.raises(ValueError, match="configuration changed"):
        freeze_run(tmp_path, {**config, "seed": 43})
    monkeypatch.setattr("blackjack.research.time.time", lambda: 4600)
    with pytest.raises(TimeoutError, match="cannot be reset"):
        freeze_run(tmp_path, config)
    assert json.loads((tmp_path / "run.json").read_text()) == frozen


def test_research_does_not_adopt_legacy_outputs_without_a_frozen_run(tmp_path):
    atomic_json(tmp_path / "candidate/fresh/config.json", {"seed": "legacy"})
    with pytest.raises(ValueError, match="lacks"):
        freeze_run(tmp_path, {"hours": 1})


def test_four_comparison_summary_adjusts_uncertainty_in_original_wager_units(tmp_path):
    for mode in ("fresh", "continuous"):
        atomic_json(
            tmp_path / f"{mode}-comparison.json",
            {
                "paired_candidate_minus": {
                    "basic": {"mean_difference": 0.003, "ci95": [0.00104, 0.00496]},
                    "baseline": {"mean_difference": 0, "ci95": [-0.00196, 0.00196]},
                }
            },
        )
    result = comparison_summary(tmp_path)
    assert len(result["comparisons"]) == 4
    pair = result["comparisons"]["continuous:candidate-minus-basic"]
    assert pair["units_per_100_rounds"] == pytest.approx(0.3)
    assert pair["familywise_ci95"] == pytest.approx([0.05022945, 0.54977055], abs=1e-7)
    assert pair["familywise_ci95"][0] < pair["nominal_ci95"][0]
    assert pair["familywise_ci95"][1] > pair["nominal_ci95"][1]
