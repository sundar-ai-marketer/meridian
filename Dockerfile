# NOTICE: This file is new in this fork and does not exist in the original
# google/meridian source.
#
# Container for someone who would rather not touch their local Python install.
# The Python minor version and uv installer are fixed, while the base image and
# apt packages remain platform-dependent. Two stages:
# `builder` has compilers in case a pinned
# dependency needs to build from source, `runtime` copies out only the venv
# and the source tree (editable install needs the source at runtime, not
# just at build time) so those compilers do not end up in the shipped image.
#
# Image size depends on the target platform and the dependency wheels selected
# by uv. Measure the image for the platform you intend to deploy.
#
# Verify after changes with:
#
#   docker build -t meridian .
#   docker run --rm meridian                       # runs the quickstart demo
#   docker run --rm meridian python scripts/run_tests.py
#   docker run --rm -it meridian bash               # interactive shell

FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6 AS builder

# Keep the resolver itself stable across rebuilds. The lockfile still records
# the package artifacts and hashes; this version only controls how that lock
# is interpreted and installed.
ARG UV_VERSION=0.11.14

# build-essential: some pinned scientific-python dependency versions may not
# ship a wheel for every platform this gets built on, and fall back to
# compiling from source. Not present in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Copy the full source before installing: this mirrors scripts/setup.sh,
# which installs editable (`pip install -e`). An editable install is a
# pointer back at this source tree, not a real copy -- so the tree has to
# exist at runtime too, not just here at build time.
COPY . /app

# Setuptools' proto build recursively discovers every `*.proto` below its
# source root. Remove host build artifacts before the isolated build so a
# previous local editable install cannot be mistaken for source input.
RUN rm -rf /app/build /app/proto/build /app/dist /app/proto/dist

# Install from the universal lock and select the supported CPU extras. The
# lock contains metadata for the optional CUDA extra, but it is intentionally
# not selected here because it has no Apple Silicon wheels.
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync \
    --python /opt/venv/bin/python \
    --frozen \
    --no-default-groups \
    --extra dev \
    --extra colab \
    --extra schema \
    --extra mlflow \
    --extra geox \
    --extra scenarioplanner

# An editable install never runs setup.py's `build`, so the `compile_scss`
# step that generates the report stylesheet never fires and every rendered
# report comes out unstyled (see scripts/compile_report_css.py for the full
# explanation). Compile it explicitly, same as scripts/setup.sh does.
RUN python scripts/compile_report_css.py
RUN python -m pip check

FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6 AS runtime

RUN useradd --create-home --uid 1000 meridian

COPY --from=builder /opt/venv /opt/venv
COPY --chown=meridian:meridian --from=builder /app /app

ENV PATH="/opt/venv/bin:${PATH}" \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER meridian

# Default: the same one-shot demo `make demo` runs, on the bundled sample
# data. Override the command for anything else, e.g.:
#   docker run --rm meridian python scripts/run_tests.py
#   docker run --rm -it meridian bash
CMD ["python", "examples/quickstart.py"]
