"""
test_policy_integration.py — SIMULATED robot + REAL policy API.

The SO-101 hardware is mocked (no serial port, no camera), but the policy
HTTP calls go to the REAL pi05_base REST endpoint instead of a mock server.

Skipped by default — enable by setting POLICY_URL:

    POLICY_URL=http://127.0.0.1:8000 \
    POLICY_TASK="pick up the red cube" \
        uv run pytest tests/test_policy_integration.py -v -s
"""

import math
import os
import time

import numpy as np
import pytest
import requests

# --- import the script under test (now main.py) ---
try:
    from robot_hackathon import main as m
except ImportError:
    import importlib.util
    from pathlib import Path

    _path = Path(__file__).resolve().parents[1] / "src" / "robot_hackathon" / "main.py"
    _spec = importlib.util.spec_from_file_location("main", _path)
    m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(m)

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

POLICY_URL = os.environ.get("POLICY_URL")  # e.g. http://127.0.0.1:8000
POLICY_TASK = os.environ.get("POLICY_TASK", "pick up the red cube")
REQUEST_TIMEOUT = 15.0  # real inference (and first-call model warmup) can be slow

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
#  Simulated hardware (same fakes as test_read_so101.py — consider moving     #
#  them into tests/conftest.py to share between both files)                   #
# --------------------------------------------------------------------------- #


class MockSO101Follower:
    def __init__(self, config):
        self.config = config
        self.is_connected = False
        self.connected_port = None
        self.sent_actions = []
        self._step = 0
        self._rng = np.random.default_rng(seed=0)

    def connect(self):
        self.connected_port = self.config.port
        self.is_connected = True

    def get_observation(self):
        assert self.is_connected
        self._step += 1
        obs = {}
        for i, joint in enumerate(JOINTS):
            obs[f"{joint}.pos"] = math.sin(0.1 * self._step + i) * 45.0
        for name, cam in self.config.cameras.items():
            obs[name] = self._rng.integers(
                0, 256, (cam.height, cam.width, 3), dtype=np.uint8
            )
        return obs

    def send_action(self, action):
        assert self.is_connected
        self.sent_actions.append(dict(action))

    def disconnect(self):
        self.is_connected = False


class MockSO101Leader:
    def __init__(self, config):
        self.config = config
        self.is_connected = False

    def connect(self):
        self.is_connected = True

    def get_action(self):
        return {f"{j}.pos": 0.0 for j in JOINTS}

    def disconnect(self):
        self.is_connected = False


# --------------------------------------------------------------------------- #
#  Fixtures / helpers                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_policy():
    """Skip unless POLICY_URL is set AND the server answers."""
    if not POLICY_URL:
        pytest.skip("set POLICY_URL=http://host:port to run this integration test")
    try:
        requests.get(POLICY_URL, timeout=3)  # any HTTP response = server is up
    except requests.RequestException as e:
        pytest.skip(f"policy server at {POLICY_URL} unreachable: {e}")
    return POLICY_URL


@pytest.fixture
def mock_hw(monkeypatch):
    created = {}

    def follower_factory(config):
        created["robot"] = MockSO101Follower(config)
        return created["robot"]

    def leader_factory(config):
        created["leader"] = MockSO101Leader(config)
        return created["leader"]

    monkeypatch.setattr(m, "SOFollower", follower_factory)
    monkeypatch.setattr(m, "SOLeader", leader_factory)
    return created


def run_main(monkeypatch, argv, iterations=3):
    monkeypatch.setattr("sys.argv", ["main.py", *argv])
    real_snapshot = m.snapshot
    calls = {"n": 0}

    def wrapped(robot):
        calls["n"] += 1
        if calls["n"] > iterations:
            raise KeyboardInterrupt
        return real_snapshot(robot)

    monkeypatch.setattr(m, "snapshot", wrapped)
    m.main()


def assert_valid_action(action: dict):
    assert action, "empty action"
    assert all(math.isfinite(v) for v in action.values()), (
        f"non-finite values: {action}"
    )
    assert all(abs(v) < 1e4 for v in action.values()), f"garbage values: {action}"


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #


def test_real_policy_returns_valid_action(real_policy):
    """Direct client call: simulated observation -> real server -> valid action."""
    cfg = m.SOFollowerRobotConfig(
        port=m.FOLLOWER_PORT, id="integration-test", cameras=m.CAMERAS
    )
    robot = MockSO101Follower(cfg)
    robot.connect()
    data = m.snapshot(robot)  # simulated motors + camera frames

    client = m.PolicyClient(
        real_policy, model="pi05_base", task=POLICY_TASK, timeout=REQUEST_TIMEOUT
    )
    t0 = time.perf_counter()
    action = client.get_action(data)
    dt = time.perf_counter() - t0
    print(f"\npolicy latency: {dt * 1000:.0f} ms   action: {action}")

    assert action is not None, (
        "server returned no usable action (check payload schema/logs)"
    )
    assert_valid_action(action)


def test_main_loop_against_real_policy(mock_hw, monkeypatch, real_policy):
    """Full pipeline: main.py loop with simulated arm driven by the real API."""
    # give the real server more headroom than the built-in 1s timeout
    real_client_cls = m.PolicyClient
    monkeypatch.setattr(
        m,
        "PolicyClient",
        lambda url, model, task: real_client_cls(
            url, model, task, timeout=REQUEST_TIMEOUT
        ),
    )

    run_main(
        monkeypatch,
        argv=["--policy-url", real_policy, "--policy-task", POLICY_TASK],
        iterations=3,
    )

    robot = mock_hw["robot"]
    assert robot.connected_port == m.FOLLOWER_PORT
    assert robot.is_connected is False  # clean disconnect
    assert len(robot.sent_actions) >= 1, "no policy actions were applied"
    for action in robot.sent_actions:
        assert_valid_action(action)
    print(f"\n{len(robot.sent_actions)}/3 policy actions applied to the simulated arm")
