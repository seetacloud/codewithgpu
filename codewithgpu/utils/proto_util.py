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
"""Protocol buffer utilities."""

import google.protobuf.text_format


def message_to_text(message):
    """Convert message to the text."""
    return google.protobuf.text_format.MessageToString(message)


def parse_from_text(message, text):
    """Parse message from the text."""
    google.protobuf.text_format.Parse(text, message)
    return message
