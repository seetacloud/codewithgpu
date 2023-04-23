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
"""Read and write tfrecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import struct
import zlib

try:
    import numpy as np
except ImportError:
    from codewithgpu.utils import deprecation
    np = deprecation.NotInstalled('numpy')

try:
    from codewithgpu.data import tf_record_pb2
except (ImportError, TypeError):
    from codewithgpu.utils import deprecation
    tf_record_pb2 = deprecation.NotInstalled('protobuf<4.0.0')
from codewithgpu.data.record import RecordWriter


class FeatureType(object):
    """Record feature type."""

    BYTES = 'bytes'
    STRING = 'string'
    FLOAT = FLOAT32 = FLOAT64 = 'float32'
    INT = INT32 = INT64 = 'int64'

    @staticmethod
    def get_default_value(dtype):
        """Return the default value of given data type."""
        if dtype == 'string' or dtype == 'bytes':
            return ''
        return 0.0 if dtype == 'float32' else 0


class TFRecordEncoder(object):
    """Encode data to protobuf messages."""

    @classmethod
    def encode(cls, data, feature_type):
        """Encode the data."""
        message = tf_record_pb2.Example()
        cls.encode_map(data, message.features, feature_type)
        return message

    @classmethod
    def encode_length_and_crc32(cls, message):
        """Encode data with length and crc32."""
        def compute_crc32(value):
            crc = zlib.crc32(bytes(value))
            crc = crc & 0xffffffff if crc < 0 else crc
            crc = numpy.array(crc, 'uint32')
            crc = (crc >> 15) | (crc << 17).astype('uint32')
            return int((crc + 0xa282ead8).astype('uint32'))
        ret = bytes()
        data = message.SerializeToString()
        length = len(data)
        ret += struct.pack('q', length)
        ret += struct.pack('I', compute_crc32(length))
        ret += data
        ret += struct.pack('I', compute_crc32(data))
        return ret

    @classmethod
    def encode_feature(cls, data, feature, feature_type):
        """Encode the feature."""
        dtype = feature_type[1]
        if dtype == FeatureType.BYTES:
            feature.bytes_list.value.extend(data)
        elif dtype == FeatureType.FLOAT32:
            feature.float_list.value.extend(data)
        elif dtype == FeatureType.INT64:
            feature.int64_list.value.extend(data)
        elif dtype == FeatureType.STRING:
            feature.bytes_list.value.extend([v.encode() for v in data])
        else:
            raise TypeError('Unsupported data type: ' + dtype)

    @classmethod
    def encode_map(cls, data, message, feature_type):
        """Encode the map container."""
        container = message.feature
        for k, v in data.items():
            if hasattr(v, 'tolist'):
                v = v.tolist()
            if not isinstance(v, (tuple, list)):
                v = [v]
            cls.encode_feature(v, container[k], feature_type[k])


class TFRecordDecoder(object):
    """Decode data from protobuf messages."""

    @classmethod
    def decode(cls, message, feature_type):
        """Decode the data."""
        return cls.decode_map(message, feature_type)

    @classmethod
    def decode_feature(cls, feature, feature_type):
        """Decode the feature."""
        shape, dtype = feature_type[:2]
        if dtype == FeatureType.BYTES:
            data = list(feature.bytes_list.value)
        elif dtype == FeatureType.FLOAT32:
            data = list(feature.float_list.value)
        elif dtype == FeatureType.INT64:
            data = list(feature.int64_list.value)
        elif dtype == FeatureType.STRING:
            data = [v.decode() for v in feature.bytes_list.value]
        else:
            raise Exception('Unsupported data type: ' + dtype)
        if shape is not None:
            if len(shape) == 0:
                return data[0]
            return numpy.array(data, dtype).reshape(shape)
        return data

    @classmethod
    def decode_map(cls, message, feature_type):
        """Decode the map container."""
        data, container = {}, message.feature
        for k, v in feature_type.items():
            data[k] = cls.decode_feature(container[k], v)
        return data


class TFRecordWriter(RecordWriter):
    """Write data to the tfrecord file."""

    VERSION = 1

    def __init__(
        self,
        path,
        features,
        max_examples=2**63 - 1,
        zfill_width=5,
    ):
        """Create a ``TRRecordWriter``.

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
        super(TFRecordWriter, self).__init__(
            path, features, max_examples, zfill_width)

    def write(self, data):
        """Write data to the record file.

        Parameters
        ----------
        data : Dict
            Data matching the feature descriptors.

        """
        if self._writing:
            self._maybe_new_shard()
            message = TFRecordEncoder.encode(data, self._features)
            current = self._data_writer.tell()
            self._data_writer.write(TFRecordEncoder.encode_length_and_crc32(message))
            self._index_writer.write(
                str(current) + ' ' +
                str(self._data_writer.tell() - current) + '\n')
            self._entries += 1
            self._examples += 1
        else:
            raise RuntimeError('Writer has been closed.')

    @classmethod
    def _get_features(cls, descriptor):
        """Return feature type from the descriptor."""
        if isinstance(descriptor, dict):
            for k, v in descriptor.items():
                descriptor[k] = cls._get_features(v)
            return descriptor
        elif isinstance(descriptor, (tuple, list)):
            dtype = getattr(FeatureType, descriptor[0].upper())
            shape = list(descriptor[1]) if len(descriptor) > 1 else None
            default_value = FeatureType.get_default_value(dtype)
            default_value = descriptor[2] if len(descriptor) > 2 else default_value
            return [shape, dtype, default_value]
        else:
            dtype, shape = getattr(FeatureType, descriptor.upper()), []
            default_value = FeatureType.get_default_value(dtype)
            return [shape, dtype, default_value]
