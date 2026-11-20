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
    assert any("Plan the path." in m.value for m in at.markdown)
    assert len(at.tabs) == 5


def test_simulator_and_comparison_buttons_run():
    at = _app()
    at.button(key="run_episode").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Outcome" for m in at.metric)
    at.button(key="compare").click().run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1


def test_live_q_learning_run(tmp_path, monkeypatch):
    log = tmp_path / "runs.csv"
    monkeypatch.setenv("PATHRL_DASHBOARD_LOG", str(log))  # never write test runs into outputs/results
    at = _app()
    at.number_input(key="budget").set_value(300).run()   # small episode budget
    at.button(key="train").click().run()       # Train (Q-Learning default)
    assert not at.exception, at.exception
    assert any(m.label == "Converged" for m in at.metric)
    assert log.exists() and len(log.read_text().splitlines()) == 2  # header + this run
