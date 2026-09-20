# robot-hackathon

Read camera frames and motor positions from a LeRobot SO-101 follower arm and
send processed observations to an OpenPI-compatible inference endpoint
(`pi05_base` / `pi05_aloha`). The project can be run locally with `uv`
(and optionally `devenv`) or paired with the included Docker image for the
OpenPI policy server.

## Repository layout


src/robot_hackathon/main.py   # robot reader + OpenPI client

Dockerfile                    # OpenPI pi05_base serving image

tests/                        # unit, integration & Docker smoke tests


## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Docker (only if you run the OpenPI server container)
- A LeRobot SO-101 follower arm plugged in (for real-robot modes)
- Optional: [devenv](https://devenv.sh/) for a reproducible Nix-based shell

## Install dependencies

```bash
uv sync

If you use devenv:

devenv shell   # drops you into a shell with all dependencies
# or
devenv up      # if you have processes defined in devenv.nix


Usage

1. Docker — OpenPI inference server

The Dockerfile builds an OpenPI serving image but does not download the

checkpoint. At runtime you mount your own pi05_base checkpoint directory

and the Paligemma tokenizer.

# Build
docker build -t openpi-server .

# Run (mount a local checkpoint + tokenizer)
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/pi05_base:/mnt/model/checkpoint:ro \
  -v /path/to/paligemma_tokenizer.model:/mnt/model/tokenizer/paligemma_tokenizer.model:ro \
  openpi-server


The server exposes an OpenPI /act endpoint on port 8000.


    Tip: If you do not mount a tokenizer, the container will try to download

    gs://big_vision/paligemma_tokenizer.model on first inference.


2. Local — robot-hackathon CLI via uv

Expose the CLI in pyproject.toml:

[project.scripts
]
robot-hackathon = "robot_hackathon.main:main"


Then run it through uv:

# Show help
uv run robot-hackathon --help

# Read-only logging
uv run robot-hackathon

# Teleoperate with the leader arm
uv run robot-hackathon --teleop

# Drive the follower from the OpenPI Docker server
uv run robot-hackathon \
  --inference-url http://localhost:8000 \
  --policy-task "pick up the red cube"

# Also POST every processed observation to a second endpoint
uv run robot-hackathon \
  --inference-url http://localhost:8000 \
  --policy-task "pick up the red cube" \
  --data-url http://localhost:9000/data

# Use a specific calibration cache
uv run robot-hackathon \
  --follower-calib-file ~/.cache/huggingface/lerobot/my_follower/calibration.json



    --policy-url is kept as a legacy alias for --inference-url.


3. Optional devenv workflow

If you have a devenv.nix / devenv.yaml, the same uv commands work inside

the devenv shell:

devenv shell
uv run robot-hackathon --inference-url http://localhost:8000 --policy-task "stack the blocks"


CLI reference

Flag	Description
--teleop	Drive the follower with the leader arm
--inference-url URL	OpenPI-compatible inference endpoint
--policy-task TEXT	Language instruction sent to the model
--data-url URL	Optional logging/collector URL (/data is appended)
--data-task TEXT	Language prompt included in --data-url payloads
--follower-calib-file PATH	Follower calibration cache
--leader-calib-file PATH	Leader calibration cache
--fps FLOAT	Control loop rate (default: 30)

Calibration

Default calibration is read from:

~/.cache/huggingface/lerobot/<arm_id>/calibration.json

Override with --follower-calib-file / --leader-calib-file.

Camera names & OpenPI configs

main.py sends one camera named top. If your OpenPI policy config expects a

different image key (e.g. cam_high for pi05_aloha), rename the camera in

src/robot_hackathon/main.py:

CAMERAS = {
    "cam_high": CameraCfg(index_or_path=0, width=640, height=480, fps=30),
}


Running tests

# Fast unit tests (mocked hardware)
uv run pytest tests/test_main.py -v

# Build & run a temporary Dockerfile that ends with `python main.py --help`
uv run pytest tests/test_docker.py -v

# Integration test against a real OpenPI server
INFERENCE_URL=http://localhost:8000 \
POLICY_TASK="pick up the red cube" \
  uv run pytest tests/test_policy_integration.py -v -s


Troubleshooting


    Serial port busy: Only one process may own /dev/ttyACM0. Close

    lerobot teleoperate / lerobot record first.

    Policy server unreachable: Verify the Docker container is running and

    port 8000 is mapped.

    Wrong image/state keys: Check the OpenPI policy config

    (pi05_aloha, etc.) for the expected observation keys.


