"""Stage 11 tests: the dashboard script runs headlessly and its buttons produce real results."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from pathrl.utils.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "src" / "pathrl" / "dashboard" / "app.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_loads():
    at = _app()
    assert at.title[0].value == "DynamicPathfinding-RL"
    assert len(at.tabs) == 4


def test_simulator_and_comparison_buttons_run():
    at = _app()
    at.button[0].click().run()   # Run episode
    assert not at.exception, at.exception
    assert any(m.label == "Outcome" for m in at.metric)
    at.button[1].click().run()   # Compare planners
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1


def test_live_q_learning_run(tmp_path, monkeypatch):
    log = tmp_path / "runs.csv"
    monkeypatch.setenv("PATHRL_DASHBOARD_LOG", str(log))  # never write test runs into outputs/results
    at = _app()
    at.number_input[1].set_value(300).run()   # small episode budget
    at.button[2].click().run()                # Run experiment (Q-Learning default)
    assert not at.exception, at.exception
    assert any(m.label == "Converged" for m in at.metric)
    assert log.exists() and len(log.read_text().splitlines()) == 2  # header + this run
