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
"""Record dataset example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import multiprocessing

import codewithgpu


if __name__ == '__main__':
    # Firstly, write data to the records.
    features = {'a': ['float'], 'b': ['int'], 'c': ['bytes'], 'd': 'string'}
    data1 = {'a': [1., 2., 3.], 'b': [4, 5, 6], 'c': [b'7', b'8', b'9'], 'd': '1'}
    data2 = {'a': [2., 3., 4.], 'b': [5, 6, 7], 'c': [b'8', b'9', b'10'], 'd': '2'}
    path_to_records = os.path.join(tempfile.gettempdir(), 'my_records')
    if os.path.exists(path_to_records):
        shutil.rmtree(path_to_records)
    os.makedirs(path_to_records)
    with codewithgpu.RecordWriter(path_to_records, features) as writer:
        writer.write(data1)
        writer.write(data2)

    # Next, create a prefetching queue.
    batch_size = 64
    output_queue = multiprocessing.Queue(batch_size)

    # Finally, create and start a dataset reader.
    dataset_reader = codewithgpu.DatasetReader(path_to_records, output_queue)
    dataset_reader.start()

    # Enjoy the training loop.
    for i in range(10):
        print(output_queue.get())
