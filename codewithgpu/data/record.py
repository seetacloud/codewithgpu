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

import os
import shutil
import uuid

from codewithgpu.data import dataset_pb2
from codewithgpu.utils import proto_util


class FeatureType(object):
    """Record feature type."""

    BYTES = 'ByteArray'
    BYTEARRAY = 'ByteArray'
    FLOAT = 'Float'
    FLOAT32 = 'Float'
    FLOAT64 = 'Float'
    INT = 'Int'
    INT32 = 'Int'
    INT64 = 'Int'
    STRING = 'String'
    FILE = 'File'


class RecordEncoder(object):
    """Encode data to protobuf messages."""

    ROOT = None
    SIZE_THRESHOLD = 50 * (1 << 20)  # 50MB
    FILE_ROOT = "files"

    @classmethod
    def encode(cls, value, feature_type):
        """Encode the data."""
        message = dataset_pb2.FeatureMap()
        cls.encode_map(value, message, feature_type)
        return message

    @classmethod
    def encode_meta(cls, feature_type, id, version):
        """Encode the meta data."""
        message = dataset_pb2.Meta()
        message.version = version
        message.id = id
        message.record_type.CopyFrom(cls.encode_feature_type(feature_type))
        return message

    @classmethod
    def encode_feature_type(cls, feature_type):
        """Encode the feature type."""
        message = dataset_pb2.RecordType()
        if isinstance(feature_type, dict):
            message.basicType = dataset_pb2.RecordType.BasicType.map
            properties = message.properties
            for k, v in feature_type.items():
                message.keys.append(k)
                properties[k].CopyFrom(cls.encode_feature_type(v))
        elif isinstance(feature_type, list):
            message.basicType = dataset_pb2.RecordType.BasicType.list
            message.item.CopyFrom(cls.encode_feature_type(feature_type[0]))
        elif feature_type == FeatureType.BYTEARRAY:
            message.basicType = dataset_pb2.RecordType.BasicType.bytes
        elif feature_type == FeatureType.FLOAT:
            message.basicType = dataset_pb2.RecordType.BasicType.float64
        elif feature_type == FeatureType.INT:
            message.basicType = dataset_pb2.RecordType.BasicType.int64
        elif feature_type == FeatureType.STRING:
            message.basicType = dataset_pb2.RecordType.BasicType.string
        elif feature_type == FeatureType.STRING:
            message.basicType = dataset_pb2.RecordType.BasicType.file
        else:
            raise TypeError('Unsupported feature type: ' + feature_type)
        return message

    @classmethod
    def encode_feature(cls, value, message, feature_type):
        """Encode the feature value."""
        if feature_type == FeatureType.INT:
            message.int64_field = value
        elif feature_type == FeatureType.BYTEARRAY:
            message.bytes_field = value
        elif feature_type == FeatureType.STRING:
            message.bytes_field = value.encode()
        elif feature_type == FeatureType.FLOAT:
            message.float64_field = value
        elif feature_type == FeatureType.FILE:
            file = dataset_pb2.FileContent()
            size = os.path.getsize(value)
            file.size = size
            if size > cls.SIZE_THRESHOLD:
                with open(value, 'rb') as fi:
                    file.type = dataset_pb2.FileContent.FileType.large
                    file.bytes_field = fi.read(512)
                rel_path = os.path.join(cls.FILE_ROOT, uuid.uuid1().hex)
                target_path = os.path.join(cls.ROOT, rel_path)
                file.file_path = rel_path
                shutil.copyfile(value, target_path)
            else:
                with open(value, 'rb') as f:
                    file.type = dataset_pb2.FileContent.FileType.small
                    file.bytes_field = f.read()
                file.file_path = ''
            message.file_field.CopyFrom(file)
        else:
            raise TypeError('Unsupported feature type: ' + feature_type)

    @classmethod
    def encode_map(cls, value, message, feature_type):
        """Encode the map container."""
        container = message.map
        for k, v in value.items():
            assert isinstance(k, str)
            if isinstance(v, (list, tuple)):
                feature = dataset_pb2.Feature()
                cls.encode_list(v, feature.feature_list_field, feature_type[k])
                container[k].CopyFrom(feature)
            elif isinstance(v, dict):
                feature = dataset_pb2.Feature()
                cls.encode_map(v, feature.feature_map_field, feature_type[k])
                container[k].CopyFrom(feature)
            else:
                feature = dataset_pb2.Feature()
                cls.encode_feature(v, feature, feature_type[k])
                container[k].CopyFrom(feature)

    @classmethod
    def encode_list(cls, value, message, feature_type):
        """Encode the list container."""
        container = message.list
        for i, v in enumerate(value):
            feature = container.add()
            if isinstance(v, (list, tuple)):
                cls.encode_list(v, feature.feature_list_field, feature_type[0])
            elif isinstance(v, dict):
                cls.encode_map(v, feature.feature_map_field, feature_type[0])
            else:
                cls.encode_feature(v, feature, feature_type[0])


class RecordDecoder(object):
    """Decode data from protobuf messages."""

    @classmethod
    def decode(cls, message, feature_type):
        """Decode the data."""
        return cls.decode_map(message, feature_type)

    @classmethod
    def decode_feature_type(cls, message):
        """Decode the feature type."""
        if message.basicType == dataset_pb2.RecordType.BasicType.string:
            return FeatureType.STRING
        elif message.basicType == dataset_pb2.RecordType.BasicType.int64:
            return FeatureType.INT
        elif message.basicType == dataset_pb2.RecordType.BasicType.bytes:
            return FeatureType.BYTEARRAY
        elif message.basicType == dataset_pb2.RecordType.BasicType.float64:
            return FeatureType.FLOAT
        elif message.basicType == dataset_pb2.RecordType.BasicType.file:
            return FeatureType.FILE
        elif message.basicType == dataset_pb2.RecordType.BasicType.list:
            return [cls.decode_feature_type(message.item)]
        elif message.basicType == dataset_pb2.RecordType.BasicType.map:
            feature_type = {}
            for k, v in message.properties.items():
                feature_type[k] = cls.decode_feature_type(v)
            return feature_type
        else:
            raise TypeError('Unsupported feature type: ' + message.basicType)

    @classmethod
    def decode_feature(cls, message, feature_type):
        """Decode the feature value."""
        if feature_type == FeatureType.INT:
            return message.int64_field
        elif feature_type == FeatureType.BYTEARRAY:
            return message.bytes_field
        elif feature_type == FeatureType.FLOAT:
            return message.float64_field
        elif feature_type == FeatureType.STRING:
            try:
                return message.bytes_field.decode()
            except UnicodeDecodeError:
                return message.bytes_field.decode('GBK')
        elif feature_type == FeatureType.FILE:
            return {'type': message.file_field.type,
                    'content': message.file_field.bytes_field,
                    'file_path': message.file_field.file_path,
                    'size': message.file_field.size}
        else:
            raise Exception('Unsupported feature type: ' + feature_type)

    @classmethod
    def decode_map(cls, message, feature_type):
        """Decode the map container."""
        data, container = {}, message.map
        for k, v in feature_type.items():
            if isinstance(v, list):
                data[k] = cls.decode_list(container[k].feature_list_field, v)
            elif isinstance(v, dict):
                data[k] = cls.decode_map(container[k].feature_map_field, v)
            else:
                data[k] = cls.decode_feature(container[k], v)
        return data

    @classmethod
    def decode_list(cls, message, feature_type):
        """Decode the list container."""
        data, container, feature_type = [], message.list, feature_type[0]
        if isinstance(feature_type, list):
            for i in range(len(container)):
                data.append(cls.decode_list(container[i].feature_list_field, feature_type))
        elif isinstance(feature_type, dict):
            for i in range(len(container)):
                data.append(cls.decode_map(container[i].feature_map_field, feature_type))
        else:
            for i in range(len(container)):
                data.append(cls.decode_feature(container[i], feature_type))
        return data


class RecordWriter(object):
    """Write data to the records file."""

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
            The path to write the record file.
        features : str
            The feature descriptors.
        max_examples : int, optional
            The max examples of a single record file.
        zfill_width : int, optional, default=5
            The width of zfill for naming record files.

        """
        self._shard_id = -1
        self._examples = 0
        self._max_examples = max_examples
        self._data_template = path + '/{0:0%d}.data' % zfill_width
        self._index_template = path + '/{0:0%d}.index' % zfill_width
        self._feature_type = self._get_feature_type(features)
        self._data_writer = None
        self._index_writer = None
        self._writing = True
        self._maybe_new_shard()
        self._write_meta_data(path)

    def write(self, data):
        """Write data to the records file.

        Parameters
        ----------
        data : Dict
            Data matching the feature descriptors.

        """
        if self._writing:
            message = RecordEncoder.encode(data, self._feature_type)
            current = self._data_writer.tell()
            self._data_writer.write(message.SerializeToString())
            self._index_writer.write(
                str(current) + ' ' +
                str(self._data_writer.tell() - current) + '\n')
            self._examples += 1
            self._maybe_new_shard()
        else:
            raise RuntimeError('Writer has been closed.')

    def close(self):
        """Close the writer."""
        if self._writing:
            self._data_writer.close()
            self._index_writer.close()
            self._writing = False

    @classmethod
    def _get_feature_type(cls, descriptor):
        """Return the feature type from the descriptor."""
        if isinstance(descriptor, dict):
            for k, v in descriptor.items():
                descriptor[k] = cls._get_feature_type(v)
            return descriptor
        elif isinstance(descriptor, list):
            return [cls._get_feature_type(v) for v in descriptor]
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

    def _write_meta_data(self, path):
        """Write the meta data."""
        message = RecordEncoder.encode_meta(
            self._feature_type, '', self.VERSION)
        with open(os.path.join(path, 'METADATA'), 'w') as f:
            f.write(proto_util.message_to_text(message))

    def __enter__(self):
        """Enter a **with** block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit a **with** block and close the file."""
        self.close()
