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
"""Read and write record files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

try:
    from codewithgpu.data import record_pb2
except (ImportError, TypeError):
    from codewithgpu.utils import deprecation
    record_pb2 = deprecation.NotInstalled('protobuf<4.0.0')


class FeatureType(object):
    """Record feature type."""

    BYTES = 'BYTES'
    STRING = 'STRING'
    FLOAT = FLOAT32 = FLOAT64 = 'FLOAT32'
    INT = INT32 = INT64 = 'INT64'


class RecordEncoder(object):
    """Encode data to protobuf messages."""

    @classmethod
    def encode(cls, data, feature_type):
        """Encode the data."""
        message = record_pb2.FeatureMap()
        cls.encode_map(data, message, feature_type)
        return message

    @classmethod
    def encode_feature(cls, data, feature, feature_type):
        """Encode the feature."""
        if feature_type == FeatureType.BYTES:
            feature.s = data
        elif feature_type == FeatureType.FLOAT32:
            feature.f = data
        elif feature_type == FeatureType.INT64:
            feature.i = data
        elif feature_type == FeatureType.STRING:
            feature.s = data.encode()
        else:
            raise TypeError('Unsupported feature type: ' + feature_type)

    @classmethod
    def encode_list(cls, data, message, feature_type):
        """Encode the list container."""
        container = message.container
        for v in data:
            feature = container.add()
            if isinstance(v, (list, tuple)):
                cls.encode_list(v, feature.feature_list, feature_type[0])
            elif isinstance(v, dict):
                cls.encode_map(v, feature.feature_map, feature_type[0])
            else:
                cls.encode_feature(v, feature, feature_type[0])

    @classmethod
    def encode_map(cls, data, message, feature_type):
        """Encode the map container."""
        container = message.container
        for k, v in data.items():
            feature = record_pb2.Feature()
            if isinstance(v, (list, tuple)):
                cls.encode_list(v, feature.feature_list, feature_type[k])
                container[k].CopyFrom(feature)
            elif isinstance(v, dict):
                cls.encode_map(v, feature.feature_map, feature_type[k])
                container[k].CopyFrom(feature)
            else:
                cls.encode_feature(v, feature, feature_type[k])
                container[k].CopyFrom(feature)


class RecordDecoder(object):
    """Decode data from protobuf messages."""

    @classmethod
    def decode(cls, message, feature_type):
        """Decode the data."""
        return cls.decode_map(message, feature_type)

    @classmethod
    def decode_feature(cls, feature, feature_type):
        """Decode the feature."""
        if feature_type == FeatureType.BYTES:
            return feature.s
        elif feature_type == FeatureType.FLOAT32:
            return feature.f
        elif feature_type == FeatureType.INT64:
            return feature.i
        elif feature_type == FeatureType.STRING:
            return feature.s.decode()
        else:
            raise Exception('Unsupported feature type: ' + feature_type)

    @classmethod
    def decode_list(cls, message, feature_type):
        """Decode the list container."""
        feature_type, container = feature_type[0], message.container
        if isinstance(feature_type, list):
            return [cls.decode_list(feature.feature_list, feature_type)
                    for feature in container]
        elif isinstance(feature_type, dict):
            return [cls.decode_map(feature.feature_map, feature_type)
                    for feature in container]
        else:
            return [cls.decode_feature(feature, feature_type)
                    for feature in container]

    @classmethod
    def decode_map(cls, message, feature_type):
        """Decode the map container."""
        data, container = {}, message.container
        for k, v in feature_type.items():
            if isinstance(v, list):
                data[k] = cls.decode_list(container[k].feature_list, v)
            elif isinstance(v, dict):
                data[k] = cls.decode_map(container[k].feature_map, v)
            else:
                data[k] = cls.decode_feature(container[k], v)
        return data


class RecordWriter(object):
    """Write data to the record file."""

    VERSION = 1

    def __init__(
        self,
        path,
        features,
        max_examples=2**63 - 1,
        zfill_width=5,
    ):
        """Create a ``RecordWriter``.

        Parameters
        ----------
        path : str
            The path to write the record files.
        features : Dict
            The feature descriptors.
        max_examples : int, optional
            The max examples of a single record file.
        zfill_width : int, optional, default=5
            The width of zfill for naming record files.

        """
        self._path = path
        self._features = self._get_features(features)
        self._entries = 0
        self._shard_id = -1
        self._examples = 0
        self._max_examples = max_examples
        self._data_template = path + '/{0:0%d}.data' % zfill_width
        self._index_template = path + '/{0:0%d}.index' % zfill_width
        self._data_writer = None
        self._index_writer = None
        self._writing = True

    def write(self, data):
        """Write data to the record file.

        Parameters
        ----------
        data : Dict
            Data matching the feature descriptors.

        """
        if self._writing:
            self._maybe_new_shard()
            message = RecordEncoder.encode(data, self._features)
            current = self._data_writer.tell()
            self._data_writer.write(message.SerializeToString())
            self._index_writer.write(
                str(current) + ' ' +
                str(self._data_writer.tell() - current) + '\n')
            self._entries += 1
            self._examples += 1
        else:
            raise RuntimeError('Writer has been closed.')

    def close(self):
        """Close the writer."""
        if self._writing:
            if self._data_writer is not None:
                self._write_meta_data()
                self._data_writer.close()
                self._index_writer.close()
            self._writing = False

    @classmethod
    def _get_features(cls, descriptor):
        """Return feature type from the descriptor."""
        if isinstance(descriptor, dict):
            for k, v in descriptor.items():
                descriptor[k] = cls._get_features(v)
            return descriptor
        elif isinstance(descriptor, list):
            return [cls._get_features(v) for v in descriptor]
        else:
            return getattr(FeatureType, descriptor.upper())

    def _maybe_new_shard(self):
        """Create the shard file handles."""
        if self._examples >= self._max_examples or self._data_writer is None:
            self._examples = 0
            self._shard_id += 1
            data_file = self._data_template.format(self._shard_id)
            index_file = self._index_template.format(self._shard_id)
            for file in (data_file, index_file):
                if os.path.exists(file):
                    raise ValueError('File %s existed.' % file)
            if self._data_writer is not None:
                self._data_writer.close()
                self._index_writer.close()
            self._data_writer = open(data_file, 'wb')
            self._index_writer = open(index_file, 'w')

    def _write_meta_data(self):
        """Write meta data."""
        meta_data = {'entries': self._entries,
                     'features': self._features,
                     'version': self.VERSION}
        with open(os.path.join(self._path, 'METADATA'), 'w') as f:
            json.dump(meta_data, f, indent=2)

    def __enter__(self):
        """Enter a **with** block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit a **with** block and close the file."""
        self.close()
