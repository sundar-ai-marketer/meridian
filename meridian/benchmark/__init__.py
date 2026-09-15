# Copyright 2026 Meridian fork contributors.
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
#
# NOTICE: This file is new in this fork and does not exist in the
# original google/meridian source.

"""Performance benchmarking for Meridian."""

from meridian.benchmark.benchmark import BenchmarkConfig
from meridian.benchmark.benchmark import BenchmarkResult
from meridian.benchmark.benchmark import environment_info
from meridian.benchmark.benchmark import run_benchmark


__all__ = [
    'BenchmarkConfig',
    'BenchmarkResult',
    'environment_info',
    'run_benchmark',
]
