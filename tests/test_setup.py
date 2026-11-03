"""Stage 1 tests: configuration loading and deterministic seeding."""

from pathlib import Path

import numpy as np
import pytest

from pathrl.utils.config import PROJECT_ROOT, deep_merge, load_config, save_config
from pathrl.utils.hardware import hardware_info
from pathrl.utils.seeding import derive_seeds, make_rng, set_global_seed

DEBUG = PROJECT_ROOT / "configs" / "experiments" / "debug.yaml"


def test_default_config_has_required_sections():
    cfg = load_config()
    for section in ("env", "rewards", "paths"):
        assert section in cfg
    assert cfg["rewards"] == {"step": -1.0, "collision": -50.0, "goal": 100.0}


def test_experiment_file_overrides_only_listed_keys():
    cfg = load_config(DEBUG)
    assert cfg["num_seeds"] == 2
    assert cfg["env"]["slip_prob"] == 0.0  # untouched key inherited from default


def test_dotted_overrides_are_typed():
    cfg = load_config(overrides=["env.grid_size=50", "env.slip_prob=0.1"])
    assert cfg["env"]["grid_size"] == 50 and isinstance(cfg["env"]["grid_size"], int)
    assert cfg["env"]["slip_prob"] == pytest.approx(0.1)


def test_unknown_override_key_fails_loudly():
    with pytest.raises(KeyError):
        load_config(overrides=["env.grid_szie=50"])


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_save_config_round_trip(tmp_path: Path):
    cfg = load_config()
    out = tmp_path / "resolved.yaml"
    save_config(cfg, out)
    assert load_config(default=out) == cfg


def test_derived_seeds_are_deterministic_and_distinct():
    a = derive_seeds(123, 30)
    assert a == derive_seeds(123, 30)
    assert len(set(a)) == 30
    assert a != derive_seeds(124, 30)


def test_rng_streams_reproduce():
    assert np.array_equal(make_rng(7).random(5), make_rng(7).random(5))


def test_torch_seeding_reproduces_network_init():
    torch = pytest.importorskip("torch")
    set_global_seed(42)
    w1 = torch.nn.Linear(4, 4).weight.detach().clone()
    set_global_seed(42)
    w2 = torch.nn.Linear(4, 4).weight.detach().clone()
    assert torch.equal(w1, w2)


def test_hardware_info_reports_basics():
    info = hardware_info()
    assert info["python"].startswith("3.11")
    assert info["cpu_count"] >= 1
