"""Stage 7 tests: Q-network shapes, replay buffer, DQN vs Double DQN targets, checkpoints, training."""

import numpy as np
import pytest
import torch

from pathrl.algorithms.reinforcement_learning import DQN, DQNConfig, DoubleDQN
from pathrl.algorithms.reinforcement_learning.network import QNetwork
from pathrl.algorithms.reinforcement_learning.replay_buffer import ReplayBuffer
from pathrl.environment import EnvConfig, GridMap, GridWorldEnv, generate_map
from pathrl.evaluation.rollout import evaluate_policy, run_policy_episode
from pathrl.evaluation.scenarios import ScenarioSampler, make_map_set


@pytest.mark.parametrize("side", [7, 10, 20, 30, 50, 100])
def test_network_outputs_one_q_value_per_action(side):
    net = QNetwork(side, n_actions=4)
    assert net(torch.zeros(5, 3, side, side)).shape == (5, 4)
    assert QNetwork(side, n_actions=8)(torch.zeros(2, 3, side, side)).shape == (2, 8)


def test_replay_buffer_overwrites_oldest_and_samples_shapes():
    buf = ReplayBuffer(capacity=5, obs_shape=(3, 4, 4), seed=0)
    for i in range(8):
        obs = np.full((3, 4, 4), i % 2, dtype=np.float32)
        buf.add(obs, action=i % 4, reward=float(i), next_obs=obs, terminated=(i == 7))
    assert len(buf) == 5 and buf.pos == 3
    assert sorted(buf.rewards.tolist()) == [3.0, 4.0, 5.0, 6.0, 7.0]  # 0, 1, 2 were overwritten
    obs, a, r, nxt, d = buf.sample(16)
    assert obs.shape == (16, 3, 4, 4) and obs.dtype == np.float32 and a.shape == (16,)
    assert set(r.tolist()) <= {3.0, 4.0, 5.0, 6.0, 7.0}


def _agents():
    cfg = DQNConfig(gamma=0.9, device="cpu")
    return DQN((3, 5, 5), 4, cfg, seed=0), DoubleDQN((3, 5, 5), 4, cfg, seed=0)


def test_dqn_and_double_dqn_targets_differ_exactly_as_defined():
    dqn, ddqn = _agents()
    next_obs = torch.rand(8, 3, 5, 5)
    rewards, terminated = torch.full((8,), -1.0), torch.zeros(8)
    terminated[0] = 1.0
    with torch.no_grad():
        online_q, target_q = ddqn.online(next_obs), ddqn.target(next_obs)
        for p in ddqn.target.parameters():  # make target differ from online
            p.add_(0.1 * torch.randn_like(p))
        target_q = ddqn.target(next_obs)
        dqn.target.load_state_dict(ddqn.target.state_dict())
        dqn.online.load_state_dict(ddqn.online.state_dict())
    y_dqn = dqn.compute_targets(rewards, next_obs, terminated)
    y_ddqn = ddqn.compute_targets(rewards, next_obs, terminated)
    expected_dqn = rewards + 0.9 * (1 - terminated) * target_q.max(1).values
    expected_ddqn = rewards + 0.9 * (1 - terminated) * target_q.gather(1, online_q.argmax(1, keepdim=True)).squeeze(1)
    assert torch.allclose(y_dqn, expected_dqn) and torch.allclose(y_ddqn, expected_ddqn)
    assert y_dqn[0] == -1.0 and y_ddqn[0] == -1.0  # terminal: no bootstrap
    assert torch.all(y_ddqn <= y_dqn + 1e-6)  # evaluating any action <= evaluating the max


def test_target_network_is_a_separate_copy():
    dqn, _ = _agents()
    buf = ReplayBuffer(100, (3, 5, 5), seed=0)
    for _ in range(100):
        buf.add(np.random.rand(3, 5, 5) > 0.5, 1, -1.0, np.random.rand(3, 5, 5) > 0.5, False)
    before = [p.clone() for p in dqn.target.parameters()]
    for _ in range(5):
        dqn.learn_step(buf)
    assert all(torch.equal(a, b) for a, b in zip(before, dqn.target.parameters()))  # unchanged
    dqn.sync_target()
    assert all(torch.equal(a, b) for a, b in zip(dqn.online.parameters(), dqn.target.parameters()))


def test_act_does_not_change_parameters():
    dqn, _ = _agents()
    before = [p.clone() for p in dqn.online.parameters()]
    for _ in range(10):
        dqn.act(np.random.rand(3, 5, 5).astype(np.float32))
    assert all(torch.equal(a, b) for a, b in zip(before, dqn.online.parameters()))


def test_checkpoint_round_trip(tmp_path):
    for agent in _agents():
        obs = np.random.rand(3, 5, 5).astype(np.float32)
        agent.save(tmp_path / f"{agent.name}.pt")
        loaded = DQN.load(tmp_path / f"{agent.name}.pt")
        assert type(loaded) is type(agent)
        assert np.allclose(loaded.q_values(obs), agent.q_values(obs))
        assert loaded.config.gamma == agent.config.gamma


def test_same_seed_same_initial_network():
    cpu = DQNConfig(device="cpu")
    a, b = DQN((3, 6, 6), config=cpu, seed=3), DQN((3, 6, 6), config=cpu, seed=3)
    assert all(torch.equal(x, y) for x, y in zip(a.online.parameters(), b.online.parameters()))


def test_map_sets_are_disjoint_between_train_and_test():
    cfg = EnvConfig(grid_size=10, placement="random")
    train = make_map_set(cfg, 200, seed=1)
    test = make_map_set(cfg, 200, seed=2)
    key = lambda m: (m.walls.tobytes(), m.start, m.goal)
    assert not ({key(m) for m in train} & {key(m) for m in test})


def test_sampler_converts_map_dynamics_to_env():
    walker_map = generate_map(EnvConfig(grid_size=8, dynamics="random_walk", change_interval=1),
                              np.random.default_rng(0))
    static_env = GridWorldEnv(EnvConfig(grid_size=8), map_seed=0)
    env, options = ScenarioSampler([static_env], [walker_map]).sample(np.random.default_rng(0))
    env.reset(seed=0, options=options)
    assert env.grid_map.dynamics == "static" and env.obstacles.count == 0


@pytest.mark.slow
@pytest.mark.parametrize("algo", [DQN, DoubleDQN])
def test_dqn_learns_tiny_corridor(algo):
    """End-to-end: both variants solve a 5x5 map with a single gap within a small budget."""
    m = GridMap.from_ascii("""
    S . # . .
    . . # . .
    . . . . .
    . . # . .
    . . # . G
    """)
    env = GridWorldEnv(EnvConfig(grid_size=5, max_steps_factor=2), grid_map=m)
    cfg = DQNConfig(total_steps=6000, learning_starts=500, target_update_interval=500,
                    train_frequency=1, epsilon_decay_fraction=0.6, learning_rate=1e-3, batch_size=32,
                    device="cpu")
    agent = algo((3, 5, 5), 4, cfg, seed=0)
    log = agent.train(ScenarioSampler([env]), seed=0, eval_interval=20,
                      evaluator=lambda p: evaluate_policy(p, env, 3, seed=1))
    result = run_policy_episode(env, agent, seed=5)
    assert result["success"] and result["steps"] == 8
    assert log.train_time_s > 0 and log.gradient_steps > 0 and len(log.evaluations) > 0
