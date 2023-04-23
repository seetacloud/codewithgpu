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
"""Dataset reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

try:
    import numpy as np
except ImportError:
    from codewithgpu.utils import deprecation
    np = deprecation.NotInstalled('numpy')

from codewithgpu.data.dataset import RecordDataset


class DatasetReader(multiprocessing.Process):
    """Read examples from a dataset.

    An external queue is required to prefetch examples:

    ```python
    batch_size = 128
    output_queue = multiprocessing.Queue(batch_size)
    reader = codewithgpu.DatasetReader('/path/to/dataset', output_queue)
    ```

    Shuffle is supported to randomly sampling into a sequence buffer:

    ```python
    shuffle_reader = codewithgpu.DatasetReader(
        '/path/to/dataset', output_queue,
        # It is recommended to set a buffer size larger than
        # the batch size to make batches of single node more diverse.
        # Default value 1024 is sufficient for most case.
        shuffle=True, initial_fill=1024,
    )
    ```

    Partitions are available over distributed nodes:

    ```python
    distributed_reader = codewithgpu.DataReader(
        '/path/to/dataset', output_queue,
        partition_id=rank, num_partitions=world_size,
    )
    ```

    """

    class BufferBound(object):
        """Record the boundary of current buffer."""

        def __init__(self, start, end):
            self.start, self.end = start, end

        @property
        def is_depleted(self):
            return self.start == self.end

    def __init__(
        self,
        path,
        output_queue,
        dataset_getter=None,
        partition_id=0,
        num_partitions=1,
        stick_to_partition=True,
        shuffle=False,
        initial_fill=1024,
        seed=1337,
        **kwargs
    ):
        """Create a ``DatasetReader``.

        Parameters
        ----------
        path : str
            The dataset path.
        output_queue : multiprocessing.Queue
            The queue to push output examples.
        dataset_getter : callable, optional
            The callable to create dataset.
        partition_id : int, optional, default=0
            The index of partition to read.
        num_partitions : int, optional, default=1
            The total number of partitions over dataset.
        stick_to_partition : bool, optional, default=True
            Fix the partition id after each epoch or not.
        shuffle : bool, optional, default=False
            Whether to shuffle the data.
        initial_fill : int, optional, default=1024
            The length of sampling sequence for shuffle.
        seed : int, optional, default=1337
            The random seed to use instead.

        """
        super(DatasetReader, self).__init__(daemon=True)
        self._path = path
        self._output_queue = output_queue
        self._dataset_getter = dataset_getter or RecordDataset
        self._partition_id = partition_id
        self._num_partitions = num_partitions
        self._shuffle = shuffle
        self._initial_fill = initial_fill
        self._seed = seed
        self._stick_to_partition = stick_to_partition
        self._first, self._current, self._last = 0, 0, 0
        self._partition_size = 0
        self._dataset_size = 0
        self._buffer_seq = []
        self._buffer_bounds = []
        self._kwargs = kwargs

    def before_first(self):
        """Move the cursor before begin."""
        self._current = self._first
        self._dataset.seek(self._first)

    def next_example(self):
        """Return the next example."""
        self._current += 1
        return self._dataset.read()

    def reset(self):
        """Reset the dataset."""
        # Redirect to the adjacent part if available.
        if not self._stick_to_partition:
            self._partition_id = (self._partition_id + 1) % self._num_partitions
        self._first = self._partition_id * self._partition_size
        self._last = min(self._first + self._partition_size, self._dataset_size)
        self.before_first()
        # Use new boundary to avoid sampling duplicates
        # when buffer size is greater than dataset size.
        counter = self._buffer_bounds[-1].end
        self._buffer_bounds.append(self.BufferBound(counter, counter))

    def push_example(self):
        """Push an example into the output queue."""
        # Pop the depleted buffer if necessary.
        if self._buffer_bounds[0].is_depleted:
            self._buffer_bounds.pop(0)
        pop_bound = self._buffer_bounds[0]
        push_bound = self._buffer_bounds[-1]
        pop_offset = 0
        if self._shuffle:
            # Sample a random offset.
            pop_range = pop_bound.end - pop_bound.start
            pop_offset = np.random.randint(0, pop_range)
        # Pop an example from the buffer.
        i = pop_bound.start % len(self._buffer_seq)
        j = (pop_bound.start + pop_offset) % len(self._buffer_seq)
        self._output_queue.put(self._buffer_seq[j])
        self._buffer_seq[j] = self._buffer_seq[i]
        # Push an example into the buffer.
        k = push_bound.end % len(self._buffer_seq)
        self._buffer_seq[k] = self.next_example()
        # Increase the buffer boundary.
        push_bound.end += 1
        pop_bound.start += 1
        # Reset the cursor if necessary.
        if self._current >= self._last:
            self.reset()

    def run(self):
        """Start the process."""
        self._init_dataset()
        # Persist a loop to push examples.
        while True:
            self.push_example()

    def _init_dataset(self):
        """Initialize the dataset."""
        np.random.seed(self._seed)
        # Instantiate the dataset here to avoid a fork of process.
        self._dataset = self._dataset_getter(path=self._path)
        # Compute the partitions.
        self._dataset_size = self._dataset.size
        self._partition_size = (self._dataset_size +
                                self._num_partitions - 1) // self._num_partitions
        # Fill the initial buffer to support random sampling.
        self._buffer_bounds.append(self.BufferBound(0, 0))
        self.reset()
        for _ in range(self._initial_fill):
            self._buffer_bounds[-1].end += 1
            self._buffer_seq.append(self.next_example())
            if self._current >= self._last:
                self.reset()
