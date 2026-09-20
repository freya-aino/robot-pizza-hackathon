"""read_so101.py — read ALL data (camera frames + motor positions) from a
LeRobot SO-101 follower arm as a plain Python dict.

lerobot 0.6.x API (SOFollower / SOLeader).

Modes
    Read-only:
        uv run src/robot_hackathon/read_so101.py

    Teleop with the leader arm:
        uv run src/robot_hackathon/read_so101.py --teleop

    Remote policy:
        uv run src/robot_hackathon/read_so101.py \
            --policy-url http://127.0.0.1:8000 \
            --policy-task "pick up the red cube"

Calibration is specified as a file, e.g.
    --follower-calib-file ~/.cache/huggingface/lerobot/my_follower/calibration.json

Data dict (every loop iteration):
{
    "timestamp": 1732145.123,
    "motors":  {"shoulder_pan": float, ...},
    "cameras": {"top": np.ndarray(H, W, 3) uint8 RGB},
    "action":  {"shoulder_pan.pos": float, ...}   # teleop/policy only
}
"""

import argparse
import base64
import io
import time
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

# ----------------------------- configuration --------------------------------
FOLLOWER_PORT = "/dev/ttyACM0"  # follower arm  (Windows: "COM3")
LEADER_PORT = "/dev/ttyACM1"  # leader arm
FOLLOWER_ID = "my_awesome_follower_arm"
LEADER_ID = "my_awesome_leader_arm"

# Default calibration cache when the user does not override it.
DEFAULT_CALIB_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
FOLLOWER_CALIB_FILE = DEFAULT_CALIB_ROOT / FOLLOWER_ID / "calibration.json"
LEADER_CALIB_FILE = DEFAULT_CALIB_ROOT / LEADER_ID / "calibration.json"

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
# ----------------------------------------------------------------------------


def _build_config(cfg_cls, base_kwargs, calib_file: Path | None):
    """
    Instantiate a lerobot config dataclass, mapping a user-supplied calibration
    *file* to the correct config field. If the class only knows about a
    calibration directory, we use the file's parent directory.
    """
    if calib_file is not None:
        calib_file = Path(calib_file)

    fields = getattr(cfg_cls, "__dataclass_fields__", {})

    if "calibration_file" in fields:
        # Preferred: the config accepts a single calibration file.
        base_kwargs["calibration_file"] = calib_file

    elif "calibration_dir" in fields:
        # Fall back: the config only accepts a directory.
        if calib_file is None:
            base_kwargs["calibration_dir"] = None
        elif calib_file.suffix == "" or (calib_file.exists() and calib_file.is_dir()):
            # Looks like a directory path; use it as-is.
            base_kwargs["calibration_dir"] = calib_file
        else:
            # Looks like a file path; hand the directory to LeRobot.
            base_kwargs["calibration_dir"] = calib_file.parent
            if "calibration_filename" in fields:
                base_kwargs["calibration_filename"] = calib_file.name
            else:
                print(
                    f"[WARN] {cfg_cls.__name__} ignores file names; using "
                    f"calibration_dir={calib_file.parent}. If the cache file "
                    f"is not the default name, calibration may be regenerated."
                )
    else:
        if calib_file is not None:
            print(
                f"[WARN] {cfg_cls.__name__} has no calibration field; "
                f"ignoring {calib_file}."
            )

    return cfg_cls(**base_kwargs)


def connect_follower(calib_file: Path | None = None) -> SOFollower:
    cfg = _build_config(
        SOFollowerRobotConfig,
        {"port": FOLLOWER_PORT, "id": FOLLOWER_ID, "cameras": CAMERAS},
        calib_file,
    )
    robot = SOFollower(cfg)
    try:
        robot.connect()  # loads cached calibration or runs it once
    except (OSError, RuntimeError) as e:
        raise SystemExit(
            f"\n[ERROR] Could not open {FOLLOWER_PORT}.\n"
            "Is lerobot teleoperate / lerobot record already running?\n"
            "A serial port can only be owned by ONE process at a time.\n"
            f"Original error: {e}"
        ) from e
    return robot


def connect_leader(calib_file: Path | None = None) -> SOLeader:
    cfg = _build_config(
        SOLeaderTeleopConfig,
        {"port": LEADER_PORT, "id": LEADER_ID},
        calib_file,
    )
    leader = SOLeader(cfg)
    leader.connect()
    return leader


def snapshot(robot: SOFollower) -> dict:
    """Read everything once and return it as a plain dict."""
    obs = robot.get_observation()
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
                name: self._encode_image(frame)
                for name, frame in data["cameras"].items()
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

    # ---- explicit calibration files ----
    p.add_argument(
        "--follower-calib-file",
        type=Path,
        default=FOLLOWER_CALIB_FILE,
        help="path to the follower calibration cache file",
    )
    p.add_argument(
        "--leader-calib-file",
        type=Path,
        default=LEADER_CALIB_FILE,
        help="path to the leader calibration cache file",
    )

    args = p.parse_args()

    print(f"Follower calibration file: {args.follower_calib_file}")
    print(f"Leader calibration file:   {args.leader_calib_file}")

    robot = connect_follower(args.follower_calib_file)

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
        leader = connect_leader(args.leader_calib_file)

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
