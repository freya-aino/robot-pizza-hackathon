"""
read_so101.py — read ALL data (camera frames + motor positions) from a
LeRobot SO-101 follower arm as a plain Python dict.

lerobot 0.6.x API (SOFollower / SOLeader).

Modes
-----
1) Read-only (nothing else may be connected to the robot):
       uv run src/robot_hackathon/read_so101.py

2) Teleop with the leader arm, logging in the same process:
       uv run src/robot_hackathon/read_so101.py --teleop

3) Remote policy (pi05_base behind a REST API) — DISABLED by default,
   enabled by passing --policy-url:
       uv run src/robot_hackathon/read_so101.py \
           --policy-url http://127.0.0.1:8000 \
           --policy-task "pick up the red cube"

Data dict (every loop iteration):
{
    "timestamp": 1732145.123,
    "motors":  {"shoulder_pan": float, "shoulder_lift": float, "elbow_flex": float,
                "wrist_flex": float, "wrist_roll": float, "gripper": float},
    "cameras": {"top": np.ndarray(H, W, 3) uint8 RGB},
    "action":  {"shoulder_pan.pos": float, ...}   # only in teleop/policy modes
}
"""

import argparse
import base64
import io
import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

# ----------------------------- configuration --------------------------------
FOLLOWER_PORT = "/dev/ttyACM0"  # follower arm  (Windows: "COM3")
LEADER_PORT = "/dev/ttyACM1"  # leader arm
FOLLOWER_ID = "my_follower"
LEADER_ID = "my_leader"

CAMERAS = {
    # dict key = key in the returned data dict
    "top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
}

FPS = 30

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# -----------------------------------------------------------------------------


def connect_follower() -> SOFollower:
    cfg = SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, cameras=CAMERAS)
    robot = SOFollower(cfg)
    try:
        robot.connect()  # loads cached calibration or runs calibration once
    except (OSError, RuntimeError) as e:
        raise SystemExit(
            f"\n[ERROR] Could not open {FOLLOWER_PORT}.\n"
            "Is `lerobot teleoperate` / `lerobot record` already running?\n"
            "A serial port can only be owned by ONE process at a time.\n"
            f"Original error: {e}"
        ) from e
    return robot


def connect_leader() -> SOLeader:
    leader = SOLeader(SOLeaderTeleopConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect()
    return leader


def snapshot(robot: SOFollower) -> dict:
    """Read everything once and return it as a plain dict."""
    obs = robot.get_observation()  # {'shoulder_pan.pos': ..., 'top': np.array, ...}
    data = {"timestamp": time.time(), "motors": {}, "cameras": {}}
    for key, value in obs.items():
        if key.endswith(".pos"):
            data["motors"][key.removesuffix(".pos")] = float(value)
        else:
            data["cameras"][key] = value  # np.ndarray (H, W, 3) uint8 RGB
    return data


class PolicyClient:
    """
    Thin REST client for an inference server hosting pi05_base.

    Server contract (adapt `endpoint` / keys here if your server differs):
        POST {base_url}/act
        request : {
            "model":  "pi05_base",
            "task":   "<language instruction>",
            "motors": {"shoulder_pan": float, ...},
            "images": {"top": "<base64-encoded JPEG, RGB>", ...}
        }
        response: {"action": {"shoulder_pan.pos": float, ...}}
                  or {"action": [float, ...]} in JOINT_ORDER
    """

    def __init__(self, base_url: str, model: str, task: str, timeout: float = 1.0):
        import requests  # lazy import: policy mode is off by default

        self._requests = requests
        self._session = requests.Session()
        self.endpoint = base_url.rstrip("/") + "/act"
        self.model = model
        self.task = task
        self.timeout = timeout

    @staticmethod
    def _encode_image(rgb) -> str:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def get_action(self, data: dict) -> dict | None:
        """Send observation, receive action. Returns None on failure (arm holds position)."""
        payload = {
            "model": self.model,
            "task": self.task,
            "motors": data["motors"],
            "images": {
                name: self._encode_image(f) for name, f in data["cameras"].items()
            },
        }
        try:
            r = self._session.post(self.endpoint, json=payload, timeout=self.timeout)
            r.raise_for_status()
        except self._requests.RequestException as e:
            print(f"\n[policy] request failed: {e} — holding position")
            return None
        action = r.json()["action"]
        if isinstance(action, dict):
            return {k: float(v) for k, v in action.items()}
        return {f"{j}.pos": float(v) for j, v in zip(JOINT_ORDER, action)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--teleop",
        action="store_true",
        help="drive the follower with the leader arm while logging",
    )
    p.add_argument(
        "--policy-url",
        default=None,
        help="base URL of a REST policy server (pi05_base). Disabled when omitted.",
    )
    p.add_argument("--policy-model", default="pi05_base")
    p.add_argument(
        "--policy-task",
        default="",
        help="language instruction sent to the policy (VLA models need one)",
    )
    p.add_argument("--fps", type=float, default=FPS)
    args = p.parse_args()

    robot = connect_follower()

    policy, leader = None, None
    if args.policy_url:
        policy = PolicyClient(args.policy_url, args.policy_model, args.policy_task)
        print(
            f"Policy mode: POST {policy.endpoint} (model={policy.model}, task={policy.task!r})"
        )
        if args.teleop:
            print(
                "NOTE: --teleop ignored while --policy-url is set (policy has control)."
            )
    elif args.teleop:
        leader = connect_leader()

    period = 1.0 / args.fps
    print("Reading data... Ctrl+C to stop.")
    try:
        while True:
            t0 = time.perf_counter()

            data = snapshot(robot)

            action = None
            if policy is not None:
                action = policy.get_action(data)
                if action is not None:
                    robot.send_action(action)
            elif leader is not None:
                action = {k: float(v) for k, v in leader.get_action().items()}
                robot.send_action(action)
            if action is not None:
                data["action"] = action

            # ---- do whatever you want with `data` here ----
            motors = "  ".join(f"{k}={v:7.2f}" for k, v in data["motors"].items())
            cams = "  ".join(f"{k}:{v.shape}" for k, v in data["cameras"].items())
            mode = "policy" if policy else ("teleop" if leader else "read-only")
            print(
                f"\r[{mode}] motors | {motors}   cams | {cams}   ", end="", flush=True
            )

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        if leader is not None:
            leader.disconnect()
        robot.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
