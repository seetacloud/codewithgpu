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
"""CodeWithGPU Python Client."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Classes
from codewithgpu.data.dataset import RecordDataset
from codewithgpu.data.dataset import TFRecordDataset
from codewithgpu.data.reader import DatasetReader
from codewithgpu.data.record import RecordWriter
from codewithgpu.data.tf_record import TFRecordWriter
from codewithgpu.inference.command import InferenceCommand
from codewithgpu.inference.command import ServingCommand
from codewithgpu.inference.module import InferenceModule
from codewithgpu.model.download import download

# Version
from codewithgpu.version import version as __version__

# Attributes
__all__ = [_s for _s in dir() if not _s.startswith('_')]
