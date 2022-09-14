# ------------------------------------------------------------------------
# Copyright (c) 2022-present, SeetaCloud, Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Unittest utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import argparse

# The global argument parser
parser = argparse.ArgumentParser(add_help=False)


def run_tests(argv=None):
    """Run tests under the current ``__main__``."""
    if argv is None:
        _, remaining = parser.parse_known_args()
        argv = [sys.argv[0]] + remaining
    unittest.main(argv=argv)
