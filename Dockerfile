# NOTICE: This file is new in this fork and does not exist in the original
# google/meridian source.
#
# Container for someone who would rather not touch their local Python install.
# The Python minor version is fixed, while the base image, apt packages and
# dependency resolution remain platform- and time-dependent. Two stages:
# `builder` has compilers in case a pinned
# dependency needs to build from source, `runtime` copies out only the venv
# and the source tree (editable install needs the source at runtime, not
# just at build time) so those compilers do not end up in the shipped image.
#
# Image size depends on the target platform and the dependency wheels selected
# by pip. Measure the image for the platform you intend to deploy.
#
# Verify after changes with:
#
#   docker build -t meridian .
#   docker run --rm meridian                       # runs the quickstart demo
#   docker run --rm meridian python -m pytest meridian scenarioplanner -q -n auto --dist=worksteal
#   docker run --rm -it meridian bash               # interactive shell

FROM python:3.11-slim AS builder

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

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Copy the full source before installing: this mirrors scripts/setup.sh,
# which installs editable (`pip install -e`). An editable install is a
# pointer back at this source tree, not a real copy -- so the tree has to
# exist at runtime too, not just here at build time.
COPY . /app

# Use the same extras as scripts/setup.sh and the README, with the local schema
# package installed first so this image cannot silently use a different build.
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e proto --config-settings editable_mode=strict \
    && pip install -e ".[dev,colab,schema,mlflow,geox,scenarioplanner]"

# An editable install never runs setup.py's `build`, so the `compile_scss`
# step that generates the report stylesheet never fires and every rendered
# report comes out unstyled (see scripts/compile_report_css.py for the full
# explanation). Compile it explicitly, same as scripts/setup.sh does.
RUN python scripts/compile_report_css.py

FROM python:3.11-slim AS runtime

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
#   docker run --rm meridian python -m pytest meridian scenarioplanner -q -n auto --dist=worksteal
#   docker run --rm -it meridian bash
CMD ["python", "examples/quickstart.py"]
