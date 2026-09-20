"""main.py — read ALL data (camera frames + motor positions) from a
LeRobot SO-101 follower arm as a plain Python dict, then send it to an
OpenPI-compatible inference endpoint.

Modes
    Read-only:
        uv run src/robot_hackathon/main.py

    Teleop with the leader arm:
        uv run src/robot_hackathon/main.py --teleop

    Remote OpenPI inference:
        uv run src/robot_hackathon/main.py \
            --inference-url http://127.0.0.1:8000 \
            --policy-task "pick up the red cube"

    Log observations to another endpoint:
        uv run src/robot_hackathon/main.py \
            --data-url http://127.0.0.1:9000/data

Calibration is specified as a file, e.g.
    --follower-calib-file ~/.cache/huggingface/lerobot/my_follower/calibration.json

Data dict (every loop iteration):
{
    "timestamp": 1732145.123,
    "motors":  {"shoulder_pan": float, ...},
    "cameras": {"top": np.ndarray(H, W, 3) uint8 RGB},
    "action":  {"shoulder_pan.pos": float, ...}   # teleop/inference only
}
"""

import argparse
import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

# LeRobot imports are optional at module-load time.  This lets the script
# print --help and run in lightweight containers / smoke tests without the
# heavy robotics stack, while still connecting to real hardware when available.
try:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    _LEROBOT_IMPORT_ERROR = None
except ImportError as _lerobot_err:  # pragma: no cover
    OpenCVCameraConfig = SOFollower = SOFollowerRobotConfig = None
    SOLeader = SOLeaderTeleopConfig = None
    _LEROBOT_IMPORT_ERROR = _lerobot_err

# ----------------------------- configuration --------------------------------
FOLLOWER_PORT = "/dev/ttyACM0"  # follower arm  (Windows: "COM3")
LEADER_PORT = "/dev/ttyACM1"  # leader arm
FOLLOWER_ID = "my_awesome_follower_arm"
LEADER_ID = "my_awesome_leader_arm"

# Default calibration cache when the user does not override it.
DEFAULT_CALIB_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
FOLLOWER_CALIB_FILE = DEFAULT_CALIB_ROOT / FOLLOWER_ID / "calibration.json"
LEADER_CALIB_FILE = DEFAULT_CALIB_ROOT / LEADER_ID / "calibration.json"


@dataclass
class CameraCfg:
    index_or_path: int | str
    width: int
    height: int
    fps: int


CAMERAS = {
    # dict key = key in the returned data dict
    "top": CameraCfg(index_or_path=0, width=640, height=480, fps=30),
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
    if SOFollower is None:
        raise RuntimeError(
            "lerobot is required to connect to the real follower arm."
        ) from _LEROBOT_IMPORT_ERROR

    lerobot_cameras = {
        name: OpenCVCameraConfig(
            index_or_path=c.index_or_path,
            width=c.width,
            height=c.height,
            fps=c.fps,
        )
        for name, c in CAMERAS.items()
    }

    cfg = _build_config(
        SOFollowerRobotConfig,
        {"port": FOLLOWER_PORT, "id": FOLLOWER_ID, "cameras": lerobot_cameras},
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
    if SOLeader is None:
        raise RuntimeError(
            "lerobot is required to connect to the real leader arm."
        ) from _LEROBOT_IMPORT_ERROR

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


# --------------------------------------------------------------------------- #
#  OpenPI-compatible data formatting & network clients                        #
# --------------------------------------------------------------------------- #


def encode_image(rgb) -> str:
    """Encode an RGB numpy array as a base64 JPEG string."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def format_for_openpi(data: dict, task: str = "") -> dict:
    """
    Convert the internal `data` dict into the JSON schema expected by an
    OpenPI inference server:

        {
            "observation": {
                "state":  [6 floats in JOINT_ORDER],
                "images": {"top": "<base64 JPEG>", ...}
            },
            "prompt": "<language instruction>"
        }
    """
    return {
        "observation": {
            "state": [float(data["motors"][j]) for j in JOINT_ORDER],
            "images": {
                name: encode_image(frame) for name, frame in data["cameras"].items()
            },
        },
        "prompt": task,
    }


class PolicyClient:
    """
    Thin REST client for an OpenPI-compatible inference server.

    Server contract:
        POST {base_url}/act
        request : format_for_openpi(data, task)
        response: {"action": <list or dict>}
    """

    def __init__(self, base_url: str, model: str, task: str, timeout: float = 1.0):
        import requests  # lazy import: inference mode is off by default

        self._requests = requests
        self._session = requests.Session()
        self.endpoint = base_url.rstrip("/") + "/act"
        self.model = model  # kept for CLI compatibility; not sent to OpenPI
        self.task = task
        self.timeout = timeout

    def get_action(self, data: dict) -> dict | None:
        """Send observation, receive action. Returns None on failure (arm holds position)."""
        payload = format_for_openpi(data, self.task)
        try:
            r = self._session.post(self.endpoint, json=payload, timeout=self.timeout)
            r.raise_for_status()
        except self._requests.RequestException as e:
            print(f"\n[policy] {self.endpoint} request failed: {e} — holding position")
            return None

        resp = r.json()
        if not isinstance(resp, dict) or "action" not in resp:
            print("\n[policy] response did not contain 'action' — holding position")
            return None

        return self._parse_action(resp["action"])

    def _parse_action(self, action) -> dict:
        """Accept a dict of joint actions or a list/array of floats."""
        if isinstance(action, dict):
            return {k: float(v) for k, v in action.items()}

        # OpenPI may return an action chunk: [[step0], [step1], ...]
        if isinstance(action, list) and len(action) and isinstance(action[0], list):
            action = action[0]

        return {f"{j}.pos": float(v) for j, v in zip(JOINT_ORDER, action)}


class DataClient:
    """
    Fire-and-forget client that POSTs every processed observation to a
    remote endpoint (logger, dataset collector, etc.).
    """

    def __init__(self, base_url: str, task: str = "", timeout: float = 1.0):
        import requests

        self._requests = requests
        self._session = requests.Session()
        self.endpoint = base_url.rstrip("/") + "/data"
        self.task = task
        self.timeout = timeout

    def send(self, data: dict) -> dict | None:
        payload = format_for_openpi(data, self.task)
        try:
            r = self._session.post(self.endpoint, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except self._requests.RequestException as e:
            print(f"\n[data] {self.endpoint} request failed: {e}")
            return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--teleop",
        action="store_true",
        help="drive the follower with the leader arm while logging",
    )

    # --- URL for sending data to an OpenPI-compatible inference server ---
    p.add_argument(
        "--inference-url",
        "--policy-url",  # legacy alias
        dest="inference_url",
        default=None,
        help="base URL of an OpenPI-compatible inference server, e.g. http://127.0.0.1:8000",
    )
    p.add_argument(
        "--policy-model",
        default="pi05_base",
        help="model name (kept for CLI compatibility; not used by OpenPI)",
    )
    p.add_argument(
        "--policy-task",
        default="",
        help="language instruction sent to the inference endpoint (VLA models need one)",
    )

    # --- optional second URL to log observations ---
    p.add_argument(
        "--data-url",
        default=None,
        help="base URL to which every processed observation is POSTed (/data)",
    )
    p.add_argument(
        "--data-task",
        default="",
        help="language prompt included in --data-url payloads",
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

    policy, leader, data_client = None, None, None

    if args.inference_url:
        policy = PolicyClient(args.inference_url, args.policy_model, args.policy_task)
        print(f"Inference mode: POST {policy.endpoint} (task={policy.task!r})")
        if args.teleop:
            print(
                "NOTE: --teleop ignored while --inference-url is set (policy has control)."
            )
    elif args.teleop:
        leader = connect_leader(args.leader_calib_file)

    if args.data_url:
        data_client = DataClient(args.data_url, args.data_task or args.policy_task)
        print(f"Data logging: POST {data_client.endpoint}")

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

            if data_client is not None:
                data_client.send(data)

            if action is not None:
                data["action"] = action

            motors = "  ".join(f"{k}={v:7.2f}" for k, v in data["motors"].items())
            cams = "  ".join(f"{k}:{v.shape}" for k, v in data["cameras"].items())
            mode = "inference" if policy else ("teleop" if leader else "read-only")
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
