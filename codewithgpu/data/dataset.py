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
"""Record dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import struct

try:
    from codewithgpu.data import record_pb2
    from codewithgpu.data import tf_record_pb2
except (ImportError, TypeError):
    from codewithgpu.utils import deprecation
    record_pb2 = deprecation.NotInstalled('protobuf<4.0.0')
    tf_record_pb2 = deprecation.NotInstalled('protobuf<4.0.0')
from codewithgpu.data.record import RecordDecoder
from codewithgpu.data.tf_record import TFRecordDecoder


class RecordDataset(object):
    """Dataset to load data from the record files."""

    def __init__(self, path):
        """Create a ``RecordDataset``.

        Parameters
        ----------
        path : str
            The path containing record files.

        """
        self._data_files = []
        self._indices = []
        self._size = 0
        self._create_indices(path)
        with open(os.path.join(path, 'METADATA')) as f:
            meta_data = json.load(f)
            self._features = meta_data['features']
            if self._size != meta_data['entries']:
                raise ValueError('Mismatched number of indices and entries. {} vs. {}'
                                 .format(self._size, meta_data['entries']))
        self._cursor = 0
        self._shard_id = None
        self._shard_loader = None

    @property
    def size(self):
        """Return the total number of examples.

        Returns
        -------
        int
            The number of examples.

        """
        return self._size

    def read(self):
        """Read and return the next example.

        Returns
        -------
        Dict
            The data example.

        """
        if self._cursor >= self._size:
            raise StopIteration
        pos, size, shard_id = self._indices[self._cursor]
        if self._shard_id != shard_id:
            self._shard_id = shard_id
            if self._shard_loader is not None:
                self._shard_loader.close()
            self._shard_loader = open(self._data_files[shard_id], 'rb')
        if self._shard_loader.tell() != pos:
            self._shard_loader.seek(pos)
        self._cursor += 1
        message = record_pb2.FeatureMap()
        message.ParseFromString(self._shard_loader.read(size))
        return RecordDecoder.decode(message, self._features)

    def close(self):
        """Close the dataset."""
        self.reset()

    def seek(self, offset):
        """Move cursor to the given offset.

        Parameters
        ----------
        offset : int
            The value for new cursor.

        """
        self._cursor = offset

    def reset(self):
        """Reset the dataset."""
        self._cursor = 0
        self._shard_id = None
        if self._shard_loader is not None:
            self._shard_loader.close()

    def tell(self):
        """Return the cursor.

        Returns
        -------
        int
            The cursor.

        """
        return self._cursor

    def _create_indices(self, path):
        """Create the dataset indices."""
        index_files = filter(lambda x: x.endswith('.index'), os.listdir(path))
        index_files = [os.path.join(path, x) for x in index_files]
        index_files.sort()
        for i, index_file in enumerate(index_files):
            data_file = index_file.replace('.index', '.data')
            if not os.path.exists(data_file):
                raise FileNotFoundError('Excepted data file: %s' % data_file)
            self._data_files.append(data_file)
            with open(index_file, 'r') as f:
                lines = f.readlines()
                self._size += len(lines)
                for line in lines:
                    pos, size = line.split()
                    self._indices.append((int(pos), int(size), i))

    def __getitem__(self, index):
        """Return example at the given index.

        Parameters
        ----------
        index : int
            The index of desired example.

        Returns
        -------
        Dict
            The data example.

        """
        self.seek(int(index))
        return self.read()

    def __iter__(self):
        """Return the iterator.

        Returns
        -------
        codewithgpu.RecordDataset
            The iterator.

        """
        return self

    def __len__(self):
        """Return dataset size.

        Returns
        -------
        int
            The number of examples in the dataset.

        """
        return self._size

    def __next__(self):
        """Read and return the next example.

        Returns
        -------
        Dict
            The data example.

        """
        return self.read()


class TFRecordDataset(RecordDataset):
    """Dataset to load data from the tfrecord files."""

    def read(self):
        """Read and return the next example.

        Returns
        -------
        Dict
            The data example.

        """
        if self._cursor >= self._size:
            raise StopIteration
        pos, size, shard_id = self._indices[self._cursor]
        if self._shard_id != shard_id:
            self._shard_id = shard_id
            if self._shard_loader is not None:
                self._shard_loader.close()
            self._shard_loader = open(self._data_files[shard_id], 'rb')
        if self._shard_loader.tell() != pos:
            self._shard_loader.seek(pos)
        self._cursor += 1
        data = self._shard_loader.read(size)
        length = struct.unpack('q', data[:8])[0]
        data = data[12:12 + length]  # Omit length and crc32 of length.
        message = tf_record_pb2.Example()
        message.ParseFromString(data)
        return TFRecordDecoder.decode(message.features, self._features)
