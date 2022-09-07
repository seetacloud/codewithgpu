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
"""Inference command."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import time
import multiprocessing

try:
    import cv2
    import flask
    import numpy as np
except ImportError:
    cv2 = flask = np = None

from codewithgpu.inference.module import InferenceModule


class ServingCommand(object):
    """Command to run serving."""

    def __init__(self, app_library='flask'):
        """Create a ``ServingCommand``.

        Parameters
        ----------
        app_library : str, optional, default='flask'
            The application library.

        """
        self.app_library = app_library
        self.example_id = multiprocessing.Value('i', 0)

    def get_image(self):
        """Return an image.

        Returns
        -------
        Tuple[int, numpy.array]
            The global index and image array.

        """
        return getattr(self, 'get_image_' + self.app_library)()

    def get_image_flask(self):
        """Return an image from the flask app.

        Returns
        -------
        Tuple[int, numpy.array]
            The global index and image array.

        """
        img, img_base64, img_bytes = None, '', b''
        try:
            req = flask.request.get_json(force=True)
            img_base64 = req['image']
        except KeyError:
            err_msg = 'Not found "image" in data.'
            flask.abort(flask.Response(err_msg, 400))
        try:
            img_base64 = img_base64.split(",")[-1]
            img_bytes = base64.b64decode(img_base64)
        except Exception as e:
            err_msg = 'Decode image bytes failed. Detail: ' + str(e)
            flask.abort(flask.Response(err_msg, 400))
        try:
            img = np.frombuffer(img_bytes, 'uint8')
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        except Exception as e:
            err_msg = 'Decode image bytes. Detail: ' + str(e)
            flask.abort(flask.Response(err_msg, 400))
        if img is None:
            err_msg = 'Bad image type.'
            flask.abort(flask.Response(err_msg, 415))
        with self.example_id.get_lock():
            self.example_id.value += 1
            example_id = self.example_id.value
        return example_id, img

    def run(self):
        """Main loop to make the serving outputs."""


class InferenceCommand(object):
    """Command to run inference."""

    def __init__(
        self,
        input_queue,
        output_queue,
        batch_size=1,
        batch_timeout=None,
    ):
        """Create a ``InferenceCommand``.

        Parameters
        ----------
        input_queue : multiprocessing.Queue
            The queue to pull input examples.
        output_queue : multiprocessing.Queue
            The queue to push output results.
        batch_size : int, optional, default=1
            The inference batch size.
        batch_timeout : number, optional
            The wait time for a complete batch.

        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

    def build_env(self):
        """Build the environment."""

    def build_model(self):
        """Build and return the model.

        Returns
        -------
        object
            The built inference model.

        """
        return self

    def build_module(self, model):
        """Build and return the inference module.

        Parameters
        ----------
        model : object
            The built inference model.

        """
        return InferenceModule(model)

    def send_results(self, module, indices, examples):
        """Send the batch results.

        Parameters
        ----------
        module : codewithgpu.InferenceModule
            The inference module.
        indices : Sequence[int]
            The global index of each example.
        examples : Sequence
            A batch of input examples.

        """
        results = module.get_results(examples)
        for i, outputs in enumerate(results):
            self.output_queue.put((indices[i], outputs))

    def run(self):
        """Main loop to make the inference outputs."""
        self.build_env()
        model = self.build_model()
        module = self.build_module(model)
        must_stop = False
        while not must_stop:
            indices, examples = [], []
            deadline, timeout = None, None
            for i in range(self.batch_size):
                if self.batch_timeout and i == 1:
                    deadline = time.monotonic() + self.batch_timeout
                if self.batch_timeout and i >= 1:
                    timeout = deadline - time.monotonic()
                try:
                    index, example = self.input_queue.get(timeout=timeout)
                    if index < 0:
                        must_stop = True
                        break
                    indices.append(index)
                    examples.append(example)
                except Exception:
                    pass
            if len(examples) == 0:
                continue
            self.send_results(module, indices, examples)
