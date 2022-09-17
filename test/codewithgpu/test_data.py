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
"""Test data module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import queue
import shutil
import tempfile
import unittest

import codewithgpu
from codewithgpu.utils.unittest_util import run_tests
import numpy


class TestRecord(unittest.TestCase):
    """Test record components."""

    def test_writer_and_reader(self):
        path = tempfile.gettempdir() + '/test_record'
        features = {'a': ['float'],
                    'b': {'bb': ['int']},
                    'c': [['bytes']],
                    'd': 'string',
                    'e': [{'ee': 'int'}]}
        data = {'a': [1., 2., 3.],
                'b': {'bb': [4, 5, 6]},
                'c': [[b'7', b'8', b'9']],
                'd': 'data',
                'e': [{'ee': 1}, {'ee': 2}]}
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        with codewithgpu.RecordWriter(path, features, max_examples=2) as writer:
            for i in range(5):
                unique_data = copy.deepcopy(data)
                unique_data['d'] += str(i)
                writer.write(unique_data)
        try:
            writer.write(data)
        except RuntimeError:
            pass
        dataset = codewithgpu.RecordDataset(path)
        self.assertEqual(dataset._features, writer._features)
        self.assertEqual(dataset.size, 5)
        self.assertEqual(len(dataset), 5)
        dataset.seek(0)
        data['d'] = 'data0'
        self.assertEqual(data, dataset.read())
        dataset.reset()
        for data in dataset:
            pass
        data['d'] = 'data0'
        self.assertEqual(data, dataset[0])
        data['d'] = 'data3'
        self.assertEqual(data, dataset[3])
        self.assertEqual(dataset.tell(), 4)
        dataset.close()
        output_queue = queue.Queue(10)
        for shuffle, initial_fill in [(False, 1), (True, 1), (True, 1024)]:
            reader = codewithgpu.DatasetReader(
                path, output_queue,
                dataset_getter=codewithgpu.RecordDataset,
                shuffle=shuffle, initial_fill=initial_fill)
            reader._init_dataset()
            for _ in range(2):
                reader.push_example()
            reader._dataset.close()
        self.assertEqual(data['a'], output_queue.get()['a'])


class TestTFRecord(unittest.TestCase):
    """Test tfrecord components."""

    def assertEqual(self, first, second):
        if isinstance(first, numpy.ndarray):
            self.assertEqual(first.shape, second.shape)
            self.assertEqual(first.dtype, second.dtype)
            self.assertEqual(first.tolist(), second.tolist())
        elif isinstance(first, dict):
            for k in first.keys():
                self.assertEqual(first[k], second[k])
        elif isinstance(first, (tuple, list)):
            for i in range(len(first)):
                self.assertEqual(first[i], second[i])
        else:
            super(TestTFRecord, self).assertEqual(first, second)

    def test_writer_and_reader(self):
        path = tempfile.gettempdir() + '/test_tfrecord'
        features = {'a': ['float'],
                    'b': ['int', (3,)],
                    'c': ['bytes', ()],
                    'd': 'string'}
        data = {'a': [1., 2., 3.],
                'b': numpy.array([4, 5, 6]),
                'c': b'7',
                'd': 'data'}
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        with codewithgpu.TFRecordWriter(path, features, max_examples=2) as writer:
            for i in range(5):
                unique_data = copy.deepcopy(data)
                unique_data['d'] += str(i)
                writer.write(unique_data)
        try:
            writer.write(data)
        except RuntimeError:
            pass
        dataset = codewithgpu.TFRecordDataset(path)
        self.assertEqual(dataset._features, writer._features)
        self.assertEqual(dataset.size, 5)
        self.assertEqual(len(dataset), 5)
        dataset.seek(0)
        data['d'] = 'data0'
        self.assertEqual(data, dataset.read())
        dataset.reset()
        for data in dataset:
            pass
        data['d'] = 'data0'
        self.assertEqual(data, dataset[0])
        data['d'] = 'data3'
        self.assertEqual(data, dataset[3])
        self.assertEqual(dataset.tell(), 4)
        dataset.close()
        output_queue = queue.Queue(10)
        for shuffle, initial_fill in [(False, 1), (True, 1), (True, 1024)]:
            reader = codewithgpu.DatasetReader(
                path, output_queue,
                dataset_getter=codewithgpu.TFRecordDataset,
                shuffle=shuffle, initial_fill=initial_fill)
            reader._init_dataset()
            for _ in range(2):
                reader.push_example()
            reader._dataset.close()
        self.assertEqual(data['a'], output_queue.get()['a'])


if __name__ == '__main__':
    run_tests()
