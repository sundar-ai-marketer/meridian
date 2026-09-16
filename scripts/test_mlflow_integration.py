#!/usr/bin/env python3
# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Verify real Meridian autologging against a temporary local MLflow service.

The server binds only to loopback, uses a temporary SQLite database and artifact
directory, and stops after the check. This does not test a cloud account or its
authentication. The small MCMC fit checks integration, not statistical quality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from test_end_to_end import run


def verify(root: Path) -> dict:
  with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
  uri = f'http://127.0.0.1:{port}'
  with (root / 'server.log').open('w') as log:
    server = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'mlflow',
            'server',
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--backend-store-uri',
            f'sqlite:///{root}/tracking.db',
            '--artifacts-destination',
            str(root / 'artifacts'),
            '--serve-artifacts',
            '--workers',
            '1',
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
      deadline = time.monotonic() + 60
      while time.monotonic() < deadline:
        if server.poll() is not None:
          raise RuntimeError('The local MLflow server exited before readiness.')
        try:
          with urllib.request.urlopen(uri + '/health', timeout=1) as response:
            if response.status == 200:
              break
        except (OSError, urllib.error.URLError):
          time.sleep(0.25)
      else:
        raise RuntimeError('The local MLflow server did not become ready.')

      import mlflow
      from meridian.mlflow import autolog

      mlflow.set_tracking_uri(uri)
      mlflow.set_experiment('meridian-local-http-smoke')
      autolog.autolog(log_metrics=False)
      try:
        with mlflow.start_run() as current:
          evidence = run(root / 'e2e')
          mlflow.log_metric(
              'serialization_roi_max_delta',
              evidence['serialization_roi_max_delta'],
          )
          mlflow.log_artifact(str(root / 'e2e/results.json'))
          run_id = current.info.run_id
      finally:
        autolog.autolog(disable=True)

      client = mlflow.tracking.MlflowClient()
      saved = client.get_run(run_id)
      if saved.info.status != 'FINISHED':
        raise AssertionError('The tracked model run did not finish.')
      if saved.data.params.get('sample_posterior.n_adapt') != '20':
        raise AssertionError(
            'Meridian autologging did not record the sampler options.'
        )
      if saved.data.metrics.get('serialization_roi_max_delta') != 0:
        raise AssertionError(
            'MLflow did not preserve the logged integration metric.'
        )
      if not saved.info.artifact_uri.startswith('mlflow-artifacts:'):
        raise AssertionError('Artifacts must use the HTTP artifact service.')
      downloaded = Path(
          client.download_artifacts(run_id, 'results.json', str(root))
      )
      if json.loads(downloaded.read_text()) != evidence:
        raise AssertionError(
            'Artifact upload/download changed the result JSON.'
        )
      return {
          'status': 'PASS',
          'scope': (
              'local HTTP tracking and artifacts; no external authentication'
          ),
          'parameter_count': len(saved.data.params),
          'artifact_round_trip': 'exact',
          'integration': evidence,
      }
    except Exception:
      log.flush()
      print((root / 'server.log').read_text(), file=sys.stderr)
      raise
    finally:
      # The MLflow CLI starts a server subprocess. Stop its whole process
      # group so the temporary listener cannot outlive this check.
      def stop_server(sig):
        try:
          if os.name == 'posix':
            os.killpg(server.pid, sig)
          elif sig == signal.SIGTERM:
            server.terminate()
          else:
            server.kill()
        except ProcessLookupError:
          pass

      stop_server(signal.SIGTERM)
      try:
        server.wait(timeout=15)
      except subprocess.TimeoutExpired:
        stop_server(signal.SIGKILL)
        server.wait()


if __name__ == '__main__':
  with tempfile.TemporaryDirectory(prefix='meridian-mlflow-http-') as directory:
    print(json.dumps(verify(Path(directory)), indent=2))
