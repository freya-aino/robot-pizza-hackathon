# syntax=docker/dockerfile:1
# OpenPI pi05_base / pi05_aloha serving image.
#
# The model checkpoint and tokenizer are loaded from mounted directories at
# runtime instead of being baked into the image.
#
# Build:
#   docker build -t openpi-pi05-base .
#
# Run (mount your local checkpoint and tokenizer):
#   docker run --rm --gpus all -p 8000:8000 \
#       -v /local/pi05_base:/mnt/model/checkpoint:ro \
#       -v /local/paligemma_tokenizer.model:/mnt/model/tokenizer/paligemma_tokenizer.model:ro \
#       openpi-pi05-base
#
# From main.py:
#   --inference-url http://host:8000 --policy-task "pick up the red cube"

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git git-lfs linux-headers-generic build-essential clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Override with a commit SHA for a reproducible build.
ARG OPENPI_REF=main
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git . \
    && git checkout "${OPENPI_REF}" \
    && git submodule update --init --recursive

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    PYTHONUNBUFFERED=1 \
    OPENPI_DATA_HOME=/openpi_assets

RUN uv venv --python 3.11.9 "$UV_PROJECT_ENVIRONMENT"
RUN --mount=type=cache,target=/root/.cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev

# Apply the same Transformers replacements as the official Dockerfile.
RUN /.venv/bin/python -c "import pathlib, shutil, transformers; shutil.copytree('src/openpi/models_pytorch/transformers_replace', pathlib.Path(transformers.__file__).parent, dirs_exist_ok=True)"

# Runtime-mount locations.
ENV OPENPI_CHECKPOINT_DIR=/mnt/model/checkpoint
ENV OPENPI_TOKENIZER_PATH=/mnt/model/tokenizer/paligemma_tokenizer.model

EXPOSE 8000

# Symlink the mounted tokenizer into the OpenPI download cache so the server
# skips the GCS download when a tokenizer was mounted.  If it was not mounted,
# the server will fall back to downloading it.
CMD ["/bin/sh", "-c", "mkdir -p /openpi_assets/gs/big_vision && ln -sf ${OPENPI_TOKENIZER_PATH} /openpi_assets/gs/big_vision/paligemma_tokenizer.model && exec /.venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_aloha --policy.dir ${OPENPI_CHECKPOINT_DIR}"]