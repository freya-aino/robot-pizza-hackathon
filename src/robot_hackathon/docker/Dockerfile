# syntax=docker/dockerfile:1
# Build from this directory (no other local repository files required):
# docker build -t openpi-pi05-base .
# Run with the checkpoint and tokenizer already included:
# docker run --rm --gpus all -p 8000:8000 openpi-pi05-base

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Keep the system dependencies from the official serving Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git git-lfs linux-headers-generic build-essential clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Override with a commit SHA for a reproducible build.
ARG OPENPI_REF=main
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git . \
    && GIT_LFS_SKIP_SMUDGE=1 git checkout "${OPENPI_REF}" \
    && GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    PYTHONUNBUFFERED=1 \
    OPENPI_DATA_HOME=/openpi_assets

RUN uv venv --python 3.11.9 "$UV_PROJECT_ENVIRONMENT"
# Install the official lockfile dependencies AND the project itself, since
# this image runs independently of any local repository mount.
RUN --mount=type=cache,target=/root/.cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev

# Apply the same Transformers replacements as the official Dockerfile.
RUN /.venv/bin/python -c "import pathlib, shutil, transformers; shutil.copytree('src/openpi/models_pytorch/transformers_replace', pathlib.Path(transformers.__file__).parent, dirs_exist_ok=True)"

RUN mkdir -p /openpi_assets
# Store downloads in an image layer, not a BuildKit cache mount. The full
# checkpoint includes normalization assets. Download the tokenizer as well
# so inference does not need to fetch it on first startup. No GPU is needed.
RUN /.venv/bin/python -c "from pathlib import Path; from openpi.shared import download; checkpoint = download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'); download.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}); Path('/opt/pi05_base').symlink_to(checkpoint, target_is_directory=True)"
EXPOSE 8000

# Use the checkpoint packaged during the build. Do not mount an empty
# directory over /openpi_assets, which would hide the packaged downloads.
# pi05_aloha supplies the ALOHA observation/action transforms for base weights.
CMD ["/.venv/bin/python", "scripts/serve_policy.py", "--port", "8000", "policy:checkpoint", "--policy.config", "pi05_aloha", "--policy.dir", "/opt/pi05_base"]
