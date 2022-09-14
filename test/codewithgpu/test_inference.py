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
"""Test inference module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import queue
import unittest

import codewithgpu
from codewithgpu.utils.unittest_util import run_tests


class TestCommand(unittest.TestCase):
    """Test command.."""

    def test_inference_command(self):
        input_queue = queue.Queue(10)
        output_queue = queue.Queue(10)
        command = codewithgpu.InferenceCommand(
            input_queue, output_queue, batch_size=2, batch_timeout=0.01)
        input_queue.put((0, 'data1'))
        input_queue.put((-1, None))
        command.run()

    def test_serving_command(self):
        command = codewithgpu.ServingCommand()
        command.run()


if __name__ == '__main__':
    run_tests()
