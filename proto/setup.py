# Copyright 2026 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTICE: This file was modified from the original google/meridian source.
# See the NOTICE file at the repository root for details.

"""The setup.py file for MMM Unified Schema."""

import logging
import pathlib
import subprocess
import sys
import tempfile

import setuptools
from setuptools.command import build


def _toml_load(path):
  if sys.version_info[:2] >= (3, 11):
    import tomllib  # pylint: disable=g-import-not-at-top

    return tomllib.load(path)
  else:
    # for python<3.11
    import tomli  # pylint: disable=g-import-not-at-top

    return tomli.load(path)


class ProtoBuild(setuptools.Command):
  """Custom command to build proto files."""

  def initialize_options(self):
    with open("pyproject.toml", "rb") as f:
      cfg = _toml_load(f).get("tool", {}).get("unified_schema_builder")
      self._root = pathlib.Path(*cfg.get("proto_root").split("/"))
      self._deps = {}
      for url_with_tag in cfg.get("github_includes"):
        tag = None
        url = url_with_tag
        if "@" in url_with_tag:
          url, tag = url_with_tag.split("@")
        folder = url.split("/")[-1].split(".")[0]
        self._deps[folder] = (url, tag)
      self._srcs = list(self._root.rglob("*.proto"))

  def finalize_options(self):
    pass

  def _check_protoc_version(self):
    command = [sys.executable, "-m", "grpc_tools.protoc", "--version"]
    try:
      out = subprocess.run(
          command,
          check=True,
          text=True,
          capture_output=True,
      ).stdout
    except subprocess.CalledProcessError as e:
      logging.error(
          "Unable to determine protoc version; command %r failed:\n%s",
          command,
          (e.stderr or "").strip(),
      )
      raise
    out = out.strip() if out else ""
    if out.startswith("libprotoc"):
      return int(out.split()[1].split(".")[0])
    return 0

  def _run_cmds(self, commands):
    for c in commands:
      command = [str(arg) for arg in c]
      logging.info("Running command %r", command)
      try:
        subprocess.run(command, capture_output=True, text=True, check=True)
      except subprocess.CalledProcessError as e:
        logging.error(
            "Unified Schema compilation failed for command %r:\n%s",
            command,
            (e.stderr or "").strip(),
        )
        raise
    return 0

  def _compile_proto_in_place(self, includes):
    i = [f"-I{include_path}" for include_path in includes]
    srcs_folders = [src for src in self._srcs]
    commands = [
        [sys.executable, "-m", "grpc_tools.protoc"]
        + i
        + ["--python_out=.", str(src)]
        for src in srcs_folders
    ]
    return self._run_cmds(commands)

  def _pull_deps(self, root):
    cmds = []
    for folder, (url, tag) in self._deps.items():
      target_path = root / folder
      target_path.mkdir(parents=True, exist_ok=True)
      if tag:
        cmds.append(
            [
                "git",
                "clone",
                "--quiet",
                "--depth=1",
                "--branch",
                tag,
                url,
                str(target_path),
            ]
        )
      else:
        cmds.append(
            [
                "git",
                "clone",
                "--quiet",
                "--depth=1",
                url,
                str(target_path),
            ]
        )
    return self._run_cmds(cmds)

  def run(self):
    protoc_major_version = self._check_protoc_version()
    if protoc_major_version < 27:
      raise RuntimeError(
          "Unified Schema compilation requires protoc major version 27 or "
          f"newer; found {protoc_major_version}."
      )

    with tempfile.TemporaryDirectory() as t:
      temp_root = pathlib.Path(t)
      self._pull_deps(temp_root)
      includes = [self._root] + [temp_root / path for path in self._deps.keys()]
      self._compile_proto_in_place(includes)


class CustomBuild(build.build):
  sub_commands = [
      ("compile_unified_schema_proto", None)
  ] + build.build.sub_commands


if __name__ == "__main__":
  setuptools.setup(
      cmdclass={
          "build": CustomBuild,
          "compile_unified_schema_proto": ProtoBuild,
      }
  )
