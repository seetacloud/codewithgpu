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
"""Inference module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class InferenceModule(object):
    """Inference module."""

    def __init__(self, model):
        """Create a ``InferenceModule``.

        Parameters
        ----------
        model : object
            The built inference model.

        """
        self.model = model

    def get_results(self, inputs):
        """Return the inference results.

        Parameters
        ----------
        inputs : Sequence
            A batch of input examples.

        Returns
        -------
        Sequence
            The result of each example in the batch.

        """

    def get_time_diffs(self):
        """Return the time differences.

        Returns
        -------
        Dict[str, number]
            The time differences.

        """
