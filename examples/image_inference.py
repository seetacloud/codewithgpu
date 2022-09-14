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
"""Image inference example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import base64
import multiprocessing
import time
import logging

import codewithgpu
import numpy as np
import torch


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='simple image application')
    parser.add_argument(
        '--batch_size',
        type=float,
        default=1,
        help='max number of examples in a batch')
    parser.add_argument(
        '--batch_timeout',
        type=float,
        default=1,
        help='timeout to wait for a batch')
    parser.add_argument(
        '--queue_size',
        type=int,
        default=512,
        help='size of the memory queue')
    parser.add_argument(
        '--app',
        default='gradio',
        help='application framework')
    parser.add_argument(
        '--processes',
        type=int,
        default=1,
        help='number of flask processes')
    parser.add_argument(
        '--port',
        type=int,
        default=5050,
        help='listening port')
    return parser.parse_args()


class InferenceModule(codewithgpu.InferenceModule):
    """Inference module."""

    def __init__(self, model):
        super(InferenceModule, self).__init__(model)

    @torch.no_grad()
    def get_results(self, imgs):
        """Return the inference results."""
        max_shape = np.max(np.stack([img.shape for img in imgs]), 0)
        output_shape = [len(imgs)] + list(max_shape)
        im_batch = np.full(output_shape, 127, imgs[0].dtype)
        for i, img in enumerate(imgs):
            copy_slices = (slice(0, d) for d in img.shape)
            im_batch[(i,) + tuple(copy_slices)] = img
        x = torch.from_numpy(im_batch.transpose(0, 3, 1, 2))
        x = self.model(x.float()).permute(0, 2, 3, 1).byte()
        return [img[0] for img in np.split(x.numpy().copy(), x.size(0))]


class InferenceCommand(codewithgpu.InferenceCommand):
    """Command to run inference."""

    def __init__(self, input_queue, output_queue, kwargs):
        super(InferenceCommand, self).__init__(input_queue, output_queue)
        self.kwargs = kwargs

    def build_env(self):
        """Build the environment."""
        self.batch_size = self.kwargs.get('batch_size', 1)
        self.batch_timeout = self.kwargs.get('batch_timeout', None)

    def build_model(self):
        """Build and return the model."""
        return torch.nn.Upsample(scale_factor=0.5, mode='bilinear')

    def build_module(self, model):
        """Build and return the inference module."""
        return InferenceModule(model)

    def send_results(self, module, indices, imgs):
        """Send the batch results."""
        results = module.get_results(imgs)
        for i, out_img in enumerate(results):
            self.output_queue.put((indices[i], out_img))


class ServingCommand(codewithgpu.ServingCommand):
    """Command to run serving."""

    def __init__(self, output_queue):
        super(ServingCommand, self).__init__(app_library='flask')
        self.output_queue = output_queue
        self.output_dict = multiprocessing.Manager().dict()

    def run(self):
        """Main loop to make the serving outputs."""
        while True:
            img_id, out_img = self.output_queue.get()
            self.output_dict[img_id] = out_img


def build_flask_app(queues, command):
    """Build the flask application."""
    import flask
    app = flask.Flask('codewithgpu.simple_image_inference')

    @app.route("/upload", methods=['POST'])
    def upload():
        img_id, img = command.get_image()
        queues[img_id % len(queues)].put((img_id, img))
        return flask.jsonify({'image_id': img_id})

    @app.route("/get", methods=['POST'])
    def get():
        def try_get(retry_time=0.005):
            try:
                req = flask.request.get_json(force=True)
                img_id = req['image_id']
            except KeyError:
                err_msg, img_id = 'Not found "image_id" in data.', ''
                flask.abort(flask.Response(err_msg))
            while img_id not in command.output_dict:
                time.sleep(retry_time)
            return img_id, command.output_dict.pop(img_id)
        img_id, out_img = try_get(retry_time=0.005)
        out_img_bytes = base64.b64encode(out_img)
        logging.info('ImageId = %d' % (img_id))
        return flask.jsonify({'image': out_img_bytes.decode()})
    return app


def build_gradio_app(queues, command):
    """Build the gradio application."""
    def upload_and_get(img):
        if img is None or img.size == 0:
            return None
        with command.example_id.get_lock():
            command.example_id.value += 1
            img_id = command.example_id.value
        queues[img_id % len(queues)].put((img_id, img))
        while img_id not in command.output_dict:
            time.sleep(0.005)
        out_img = command.output_dict.pop(img_id)
        logging.info('ImageId = %d,' % img_id)
        return out_img
    import gradio
    app = gradio.Interface(
        upload_and_get,
        gradio.Image(show_label=False),
        gradio.Image(show_label=False))
    return app


if __name__ == '__main__':
    args = parse_args()
    logging.info('Called with args:\n' + str(args))

    # Build actors.
    queues = [multiprocessing.Queue(args.queue_size) for _ in range(2)]
    commands = [InferenceCommand(
        queues[i], queues[-1], kwargs={
            'batch_size': args.batch_size,
            'batch_timeout': args.batch_timeout,
            'verbose': i == 0,
        }) for i in range(1)]
    commands += [ServingCommand(queues[-1])]
    actors = [multiprocessing.Process(target=command.run) for command in commands]
    for actor in actors:
        actor.start()

    # Build app.
    if args.app == 'flask':
        app = build_flask_app(queues[:-1], commands[-1])
        app.run(host='0.0.0.0', port=args.port,
                threaded=args.processes == 1, processes=args.processes)
    elif args.app == 'gradio':
        app = build_gradio_app(queues[:-1], commands[-1])
        app.queue(concurrency_count=args.processes)
        app.launch(server_name='0.0.0.0', server_port=args.port)
    else:
        raise ValueError('Unsupported application framework: ' + args.app)
