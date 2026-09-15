# NOTICE: This file is new in this fork and does not exist in the original
# google/meridian source.
#
# Reproducible container for someone who would rather not touch their local
# Python install. Two stages: `builder` has compilers in case a pinned
# dependency needs to build from source, `runtime` copies out only the venv
# and the source tree (editable install needs the source at runtime, not
# just at build time) so those compilers do not end up in the shipped image.
#
# SIZE WARNING, for real: this project depends on TensorFlow and JAX. Do not
# expect a small image -- TensorFlow alone typically adds multiple GB, and
# JAX/jaxlib, arviz, pandas, scipy and matplotlib stack on top of that. A
# realistic expectation is several GB. That has not been measured for this
# exact image (see below), so treat it as an order-of-magnitude expectation,
# not a verified number.
#
# NOT BUILT: nobody has run `docker build` on this file. It has only been
# checked statically (structure and command review; see the report that
# shipped with this file). Build and smoke-test it yourself before relying
# on it in any pipeline:
#
#   docker build -t meridian .
#   docker run --rm meridian                       # runs the quickstart demo
#   docker run --rm meridian python -m pytest meridian -q -n auto
#   docker run --rm -it meridian bash               # interactive shell

FROM python:3.11-slim AS builder

# build-essential: some pinned scientific-python dependency versions may not
# ship a wheel for every platform this gets built on, and fall back to
# compiling from source. Not present in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
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

# Same extras scripts/setup.sh and the README both use.
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e ".[dev,colab,schema,mlflow,geox,scenarioplanner]"

# An editable install never runs setup.py's `build`, so the `compile_scss`
# step that generates the report stylesheet never fires and every rendered
# report comes out unstyled (see scripts/compile_report_css.py for the full
# explanation). Compile it explicitly, same as scripts/setup.sh does.
RUN python scripts/compile_report_css.py

FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 meridian

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENV PATH="/opt/venv/bin:${PATH}" \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER meridian

# Default: the same one-shot demo `make demo` runs, on the bundled sample
# data. Override the command for anything else, e.g.:
#   docker run --rm meridian python -m pytest meridian -q -n auto
#   docker run --rm -it meridian bash
CMD ["python", "examples/quickstart.py"]
