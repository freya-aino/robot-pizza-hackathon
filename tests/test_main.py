"""
test_read_so101.py — test read_so101.py with the SO-101 hardware mocked out.

No real robot / leader arm / camera / serial port is touched:
  - SOFollower and SOLeader are replaced by fakes at the module boundary
    (they assert the configured ports FOLLOWER_PORT / LEADER_PORT are used)
  - the pi05_base policy REST API is replaced by a real local HTTP server
    on an ephemeral port

Run:
    uv add --dev pytest        # once
    uv run pytest tests/test_read_so101.py -v
"""

import base64
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

# --- import the script under test (works with or without an installed package)
try:
    from robot_hackathon import main as m
except ImportError:
    import importlib.util
    from pathlib import Path

    _path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "robot_hackathon"
        / "read_so101.py"
    )
    _spec = importlib.util.spec_from_file_location("read_so101", _path)
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


# --------------------------------------------------------------------------- #
#  Hardware fakes (same API as lerobot 0.6.x SOFollower / SOLeader)           #
# --------------------------------------------------------------------------- #


class MockSO101Follower:
    """Pretends to be the follower arm on FOLLOWER_PORT."""

    def __init__(self, config):
        self.config = config
        self.is_connected = False
        self.connected_port = None
        self.sent_actions = []  # recorded for assertions
        self._step = 0
        self._rng = np.random.default_rng(seed=0)

    def connect(self):
        self.connected_port = self.config.port
        self.is_connected = True

    def get_observation(self):
        assert self.is_connected, "get_observation() before connect()"
        self._step += 1
        obs = {}
        for i, joint in enumerate(JOINTS):  # smooth fake motor motion
            obs[f"{joint}.pos"] = math.sin(0.1 * self._step + i) * 45.0
        for name, cam in self.config.cameras.items():  # fake camera frames
            obs[name] = self._rng.integers(
                0, 256, (cam.height, cam.width, 3), dtype=np.uint8
            )
        return obs

    def send_action(self, action):
        assert self.is_connected, "send_action() before connect()"
        self.sent_actions.append(dict(action))

    def disconnect(self):
        self.is_connected = False


class MockSO101Leader:
    """Pretends to be the leader arm on LEADER_PORT; returns increasing actions."""

    def __init__(self, config):
        self.config = config
        self.is_connected = False
        self.connected_port = None
        self._n = 0

    def connect(self):
        self.connected_port = self.config.port
        self.is_connected = True

    def get_action(self):
        assert self.is_connected, "get_action() before connect()"
        self._n += 1
        return {f"{joint}.pos": float(self._n) for joint in JOINTS}

    def disconnect(self):
        self.is_connected = False


# --------------------------------------------------------------------------- #
#  Fixtures / helpers                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_hw(monkeypatch):
    """Patch SOFollower/SOLeader inside read_so101 so no serial port is opened."""
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
    """Run m.main() for `iterations` loop passes, then stop via KeyboardInterrupt."""
    monkeypatch.setattr("sys.argv", ["read_so101.py", *argv])
    real_snapshot = m.snapshot
    calls = {"n": 0}

    def wrapped(robot):
        calls["n"] += 1
        if calls["n"] > iterations:
            raise KeyboardInterrupt  # main() catches this and shuts down cleanly
        return real_snapshot(robot)

    monkeypatch.setattr(m, "snapshot", wrapped)
    m.main()
    return calls["n"]


class MockPolicyServer(HTTPServer):
    """HTTP server that records every request payload it receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received: list[dict] = []


class PolicyHandler(BaseHTTPRequestHandler):
    """Mock pi05_base REST server: validates the request, returns action=1.0 everywhere."""

    server: MockPolicyServer  # set by the server on each request; tells type checkers the type

    def do_POST(self):
        assert self.path == "/act", f"unexpected path {self.path}"
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.received.append(payload)
        body = json.dumps({"action": {f"{j}.pos": 1.0 for j in JOINTS}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep test output clean


@pytest.fixture
def policy_server():
    server = MockPolicyServer(("127.0.0.1", 0), PolicyHandler)  # port 0 = ephemeral
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", server.received
    server.shutdown()
    thread.join(timeout=2)


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #


def test_snapshot_returns_full_dict():
    cfg = m.SOFollowerRobotConfig(port=m.FOLLOWER_PORT, id="test", cameras=m.CAMERAS)
    robot = MockSO101Follower(cfg)
    robot.connect()

    data = m.snapshot(robot)

    assert set(data) == {"timestamp", "motors", "cameras"}
    assert isinstance(data["timestamp"], float)
    assert set(data["motors"]) == set(JOINTS)
    assert all(isinstance(v, float) for v in data["motors"].values())
    frame = data["cameras"]["top"]
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8


def test_read_only_mode(mock_hw, monkeypatch):
    run_main(monkeypatch, argv=[], iterations=3)

    robot = mock_hw["robot"]
    assert robot.connected_port == m.FOLLOWER_PORT  # correct port was used
    assert robot.is_connected is False  # disconnect() ran in finally
    assert robot.sent_actions == []  # read-only never commands the arm
    assert "leader" not in mock_hw  # leader never connected


def test_teleop_mode(mock_hw, monkeypatch):
    run_main(monkeypatch, argv=["--teleop"], iterations=3)

    robot, leader = mock_hw["robot"], mock_hw["leader"]
    assert robot.connected_port == m.FOLLOWER_PORT
    assert leader.connected_port == m.LEADER_PORT
    assert len(robot.sent_actions) == 3  # one action per loop iteration
    for action in robot.sent_actions:
        assert set(action) == {f"{j}.pos" for j in JOINTS}
        assert all(isinstance(v, float) for v in action.values())


def test_policy_mode(mock_hw, monkeypatch, policy_server):
    url, received = policy_server
    run_main(
        monkeypatch,
        argv=["--policy-url", url, "--policy-task", "pick the cube"],
        iterations=3,
    )

    robot = mock_hw["robot"]
    assert len(robot.sent_actions) == 3
    assert all(a["shoulder_pan.pos"] == 1.0 for a in robot.sent_actions)
    assert "leader" not in mock_hw  # policy mode doesn't touch the leader

    assert len(received) == 3
    payload = received[0]
    assert payload["model"] == "pi05_base"
    assert payload["task"] == "pick the cube"
    assert set(payload["motors"]) == set(JOINTS)
    assert set(payload["images"]) == {"top"}
    jpeg = base64.b64decode(payload["images"]["top"])
    assert jpeg[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_policy_client_parses_list_action(monkeypatch):
    """Server may also return a plain 6-float list instead of a dict."""
    client = m.PolicyClient("http://unused", model="pi05_base", task="x")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"action": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}

    monkeypatch.setattr(client._session, "post", lambda *a, **k: FakeResp())
    action = client.get_action({"motors": {}, "cameras": {}})
    assert action == {f"{j}.pos": float(v) for j, v in zip(JOINTS, range(1, 7))}


def test_policy_client_returns_none_when_server_down():
    """Dead server -> None -> main loop holds position instead of crashing."""
    client = m.PolicyClient(
        "http://127.0.0.1:9", model="pi05_base", task="x", timeout=0.2
    )
    data = {
        "motors": {j: 0.0 for j in JOINTS},
        "cameras": {"top": np.zeros((4, 4, 3), dtype=np.uint8)},
    }
    assert client.get_action(data) is None
