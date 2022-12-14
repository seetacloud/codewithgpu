# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tf_record.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tf_record.proto',
  package='codewithgpu.tensorflow',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n\x0ftf_record.proto\x12\x16\x63odewithgpu.tensorflow\"\x1a\n\tBytesList\x12\r\n\x05value\x18\x01 \x03(\x0c\"\x1e\n\tFloatList\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"\x1e\n\tInt64List\x12\x11\n\x05value\x18\x01 \x03(\x03\x42\x02\x10\x01\"\xbc\x01\n\x07\x46\x65\x61ture\x12\x37\n\nbytes_list\x18\x01 \x01(\x0b\x32!.codewithgpu.tensorflow.BytesListH\x00\x12\x37\n\nfloat_list\x18\x02 \x01(\x0b\x32!.codewithgpu.tensorflow.FloatListH\x00\x12\x37\n\nint64_list\x18\x03 \x01(\x0b\x32!.codewithgpu.tensorflow.Int64ListH\x00\x42\x06\n\x04kind\"\x9b\x01\n\x08\x46\x65\x61tures\x12>\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32-.codewithgpu.tensorflow.Features.FeatureEntry\x1aO\n\x0c\x46\x65\x61tureEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12.\n\x05value\x18\x02 \x01(\x0b\x32\x1f.codewithgpu.tensorflow.Feature:\x02\x38\x01\"=\n\x07\x45xample\x12\x32\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32 .codewithgpu.tensorflow.FeaturesB\x03\xf8\x01\x01\x62\x06proto3')
)




_BYTESLIST = _descriptor.Descriptor(
  name='BytesList',
  full_name='codewithgpu.tensorflow.BytesList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='codewithgpu.tensorflow.BytesList.value', index=0,
      number=1, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=43,
  serialized_end=69,
)


_FLOATLIST = _descriptor.Descriptor(
  name='FloatList',
  full_name='codewithgpu.tensorflow.FloatList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='codewithgpu.tensorflow.FloatList.value', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=71,
  serialized_end=101,
)


_INT64LIST = _descriptor.Descriptor(
  name='Int64List',
  full_name='codewithgpu.tensorflow.Int64List',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='codewithgpu.tensorflow.Int64List.value', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=103,
  serialized_end=133,
)


_FEATURE = _descriptor.Descriptor(
  name='Feature',
  full_name='codewithgpu.tensorflow.Feature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bytes_list', full_name='codewithgpu.tensorflow.Feature.bytes_list', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='float_list', full_name='codewithgpu.tensorflow.Feature.float_list', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int64_list', full_name='codewithgpu.tensorflow.Feature.int64_list', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='kind', full_name='codewithgpu.tensorflow.Feature.kind',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=136,
  serialized_end=324,
)


_FEATURES_FEATUREENTRY = _descriptor.Descriptor(
  name='FeatureEntry',
  full_name='codewithgpu.tensorflow.Features.FeatureEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='codewithgpu.tensorflow.Features.FeatureEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='codewithgpu.tensorflow.Features.FeatureEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=403,
  serialized_end=482,
)

_FEATURES = _descriptor.Descriptor(
  name='Features',
  full_name='codewithgpu.tensorflow.Features',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature', full_name='codewithgpu.tensorflow.Features.feature', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FEATURES_FEATUREENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=327,
  serialized_end=482,
)


_EXAMPLE = _descriptor.Descriptor(
  name='Example',
  full_name='codewithgpu.tensorflow.Example',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='features', full_name='codewithgpu.tensorflow.Example.features', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=484,
  serialized_end=545,
)

_FEATURE.fields_by_name['bytes_list'].message_type = _BYTESLIST
_FEATURE.fields_by_name['float_list'].message_type = _FLOATLIST
_FEATURE.fields_by_name['int64_list'].message_type = _INT64LIST
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['bytes_list'])
_FEATURE.fields_by_name['bytes_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['float_list'])
_FEATURE.fields_by_name['float_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['int64_list'])
_FEATURE.fields_by_name['int64_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURES_FEATUREENTRY.fields_by_name['value'].message_type = _FEATURE
_FEATURES_FEATUREENTRY.containing_type = _FEATURES
_FEATURES.fields_by_name['feature'].message_type = _FEATURES_FEATUREENTRY
_EXAMPLE.fields_by_name['features'].message_type = _FEATURES
DESCRIPTOR.message_types_by_name['BytesList'] = _BYTESLIST
DESCRIPTOR.message_types_by_name['FloatList'] = _FLOATLIST
DESCRIPTOR.message_types_by_name['Int64List'] = _INT64LIST
DESCRIPTOR.message_types_by_name['Feature'] = _FEATURE
DESCRIPTOR.message_types_by_name['Features'] = _FEATURES
DESCRIPTOR.message_types_by_name['Example'] = _EXAMPLE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BytesList = _reflection.GeneratedProtocolMessageType('BytesList', (_message.Message,), {
  'DESCRIPTOR' : _BYTESLIST,
  '__module__' : 'tf_record_pb2'
  # @@protoc_insertion_point(class_scope:codewithgpu.tensorflow.BytesList)
  })
_sym_db.RegisterMessage(BytesList)

FloatList = _reflection.GeneratedProtocolMessageType('FloatList', (_message.Message,), {
  'DESCRIPTOR' : _FLOATLIST,
  '__module__' : 'tf_record_pb2'
  # @@protoc_insertion_point(class_scope:codewithgpu.tensorflow.FloatList)
  })
_sym_db.RegisterMessage(FloatList)

Int64List = _reflection.GeneratedProtocolMessageType('Int64List', (_message.Message,), {
  'DESCRIPTOR' : _INT64LIST,
  '__module__' : 'tf_record_pb2'
  # @@protoc_insertion_point(class_scope:codewithgpu.tensorflow.Int64List)
  })
_sym_db.RegisterMessage(Int64List)

Feature = _reflection.GeneratedProtocolMessageType('Feature', (_message.Message,), {
  'DESCRIPTOR' : _FEATURE,
  '__module__' : 'tf_record_pb2'
  # @@protoc_insertion_point(class_scope:codewithgpu.tensorflow.Feature)
  })
_sym_db.RegisterMessage(Feature)

Features = _reflection.GeneratedProtocolMessageType('Features', (_message.Message,), {

  'FeatureEntry' : _reflection.GeneratedProtocolMessageType('FeatureEntry', (_message.Message,), {
    'DESCRIPTOR' : _FEATURES_FEATUREENTRY,
    '__module__' : 'tf_record_pb2'
    # @@protoc_insertion_point(class_scope:codewithgpu.tensorflow.Features.FeatureEntry)
    })
  ,
  'DESCRIPTOR' : _FEATURES,
  '__module__' : 'tf_record_pb2'
  # @@protoc_insertion_point(class_scope:codewithgpu.tensorflow.Features)
  })
_sym_db.RegisterMessage(Features)
_sym_db.RegisterMessage(Features.FeatureEntry)

Example = _reflection.GeneratedProtocolMessageType('Example', (_message.Message,), {
  'DESCRIPTOR' : _EXAMPLE,
  '__module__' : 'tf_record_pb2'
  # @@protoc_insertion_point(class_scope:codewithgpu.tensorflow.Example)
  })
_sym_db.RegisterMessage(Example)


DESCRIPTOR._options = None
_FLOATLIST.fields_by_name['value']._options = None
_INT64LIST.fields_by_name['value']._options = None
_FEATURES_FEATUREENTRY._options = None
# @@protoc_insertion_point(module_scope)
