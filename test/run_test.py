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
"""Command line to run tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import subprocess

import argparse

TESTS_AND_SOURCES = [
    ('codewithgpu/test_data', 'codewithgpu.data'),
    ('codewithgpu/test_inference', 'codewithgpu.inference'),
]

TESTS = [t[0] for t in TESTS_AND_SOURCES]
SOURCES = [t[1] for t in TESTS_AND_SOURCES]


def parse_args():
    parser = argparse.ArgumentParser(
        description='run the unittests',
        epilog='where TESTS is any of: {}'.format(', '.join(TESTS)))
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='print verbose information')
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='print error information only')
    parser.add_argument(
        '-c',
        '--coverage',
        action='store_true',
        help='run coverage for unittests')
    parser.add_argument(
        '-x',
        '--exclude',
        nargs='+',
        choices=TESTS,
        metavar='TESTS',
        default=[],
        help='select a set of tests to exclude')
    parser.add_argument(
        '--ignore-distributed-blocklist',
        action='store_true',
        help='always run block-listed distributed tests')
    return parser.parse_args()


def get_base_command(args):
    """Return the base running command."""
    if args.coverage:
        executable = ['coverage', 'run', '--parallel-mode']
    else:
        executable = [sys.executable]
    return executable


def get_selected_tests(args, tests, sources):
    """Return the selected tests."""
    for exclude_test in args.exclude:
        tests_copy = tests[:]
        for i, test in enumerate(tests_copy):
            if test.startswith(exclude_test):
                tests.pop(i)
                sources.pop(i)
    return tests, sources


def main():
    """The main procedure."""
    args = parse_args()
    base_command = get_base_command(args)
    tests, sources = get_selected_tests(args, TESTS, SOURCES)
    for i, test in enumerate(tests):
        command = base_command[:]
        if args.coverage:
            if sources[i]:
                command.extend(['--source ', sources[i]])
        command.append(test + '.py')
        if args.verbose:
            command.append('--verbose')
        elif args.quiet:
            command.append('--quiet')
        subprocess.call(' '.join(command), shell=True)
    if args.coverage:
        subprocess.call(['coverage', 'combine'])
        subprocess.call(['coverage', 'html'])


if __name__ == '__main__':
    main()
