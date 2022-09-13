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
"""Bench models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import copy
import json
import logging
import sys
import subprocess
import time


BENCHMARKS = [
    # Model Training.
    ('models/resnet/bench_*.py', 'resnet50.train'),
    ('models/vit/bench_*.py', 'vit_base_patch16_224.train'),
    ('models/mobilenet_v3/bench_*.py', 'mobilenet_v3_large.train'),
    # Model Inference.
    ('models/resnet/bench_*.py', 'resnet50.eval'),
    ('models/vit/bench_*.py', 'vit_base_patch16_224.eval'),
    ('models/mobilenet_v3/bench_*.py', 'mobilenet_v3_large.eval'),
]


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Run the benchmarks.')
    parser.add_argument('--precision', default='float16', help='compute precision')
    parser.add_argument('--device', default=0, help='compute device')
    parser.add_argument('--backend', default='torch', help='compute backend')
    parser.add_argument('--metric', nargs='+', default=['throughout'],
                        help='performance metrics')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='print error information only')
    parser.add_argument('-f', '--filename', default='',
                        help='Save results to the specified file')
    return parser.parse_args()


def get_base_command(args):
    """Return the base command."""
    cmd = [sys.executable, '{}', '--model', '{}']
    cmd += ['--train'] if args.train else []
    cmd += ['--precision', args.precision]
    cmd += ['--device', str(args.device)]
    return cmd


def get_device_name(backend, device_index):
    """Return the device name."""
    if backend == 'torch':
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(device_index)
        elif torch.mps.is_available():
            import dragon
            return dragon.mps.get_device_name(device_index)
        else:
            return 'CPU'
    elif 'vm' in backend:
        import dragon
        if dragon.cuda.is_available():
            return dragon.cuda.get_device_name(device_index)
        elif dragon.mps.is_available():
            return dragon.mps.get_device_name(device_index)
        else:
            return 'CPU'
    return 'Unknown'


def get_backend_name(backend):
    """Return the backend name."""
    if backend == 'torch':
        version = subprocess.check_output(
            '%s -c "import torch;print(torch.__version__)"'
            % sys.executable, shell=True).decode('ascii').strip()
        return 'torch-%s' % version
    elif backend == 'tf':
        version = subprocess.check_output(
            '%s -c "import tensorflow;print(tensorflow.__version__)"'
            % sys.executable, shell=True).decode('ascii').strip()
    elif 'vm' in backend:
        version = subprocess.check_output(
            '%s -c "import dragon;print(dragon.__version__)"'
            % sys.executable, shell=True).decode('ascii').strip()
        return 'seeta-dragon-%s' % version
    return backend


def get_model_args(args, model):
    """Return the model-specific args."""
    args = copy.deepcopy(args)
    model = model.split('.')
    args.model = model.pop(0)
    presets = {'train': ('train', True),
               'eval': ('train', False),
               'float16': ('precision', 'float16'),
               'float32': ('precision', 'float32')}
    for k, v in presets.items():
        if k in model:
            setattr(args, v[0], v[1])
    return args


def get_results(output, keys):
    """Extract results from the output string."""
    results = collections.defaultdict(list)
    for line in output.splitlines():
        if not line.startswith('{'):
            continue
        if not line.endswith('}'):
            continue
        metrics = eval(line)
        for k in keys:
            if k in metrics:
                results[k].append(metrics[k])
    for k in results.keys():
        results[k].pop(0)  # Warmup.
        results[k] = sum(results[k]) / len(results[k])
    return results


def main():
    """Main procedure."""
    args = parse_args()
    logging.getLogger().setLevel('ERROR' if args.quiet else 'INFO')
    log_handler = logging.StreamHandler(sys.stderr)
    log_handler.terminator = ''
    log_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(log_handler)
    all_results = []
    for count, (script, model) in enumerate(BENCHMARKS):
        model_args = get_model_args(args, model)
        base_command = get_base_command(model_args)
        logging.info('[%d/%d] bench %s ... '
                     % (count + 1, len(BENCHMARKS), model))
        script = script.replace('*', args.backend)
        command = (' '.join(base_command)).format(script, model_args.model)
        output = subprocess.check_output(command, shell=True)
        output = output.decode('ascii').strip()
        results = collections.OrderedDict()
        results['device'] = get_device_name(args.backend, args.device)
        results['backend'] = get_backend_name(args.backend)
        results['model'] = model
        results.update(get_results(output, args.metric))
        all_results.append(results)
        logging.info('ok\n')
    if not args.filename:
        args.filename = '../{}.json'.format(time.strftime(
            '%Y%m%d_%H%M%S', time.localtime(time.time())))
    with open(args.filename, 'w') as f:
        json.dump(all_results, f)


if __name__ == '__main__':
    main()
