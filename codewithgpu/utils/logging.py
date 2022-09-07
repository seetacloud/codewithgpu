# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------
"""Logging utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import logging as _logging
import os
import sys as _sys
import threading


_logger = None
_logger_lock = threading.Lock()


def get_logger():
    global _logger
    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger
    _logger_lock.acquire()
    try:
        if _logger:
            return _logger
        logger = _logging.getLogger('codewithgpu')
        logger.setLevel('INFO')
        logger.propagate = False
        logger._is_root = True
        if True:
            # Determine whether we are in an interactive environment.
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if _sys.ps1:
                    _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = _sys.flags.interactive
            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                logger.setLevel('INFO')
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr
            # Add the output handler.
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(_logging.Formatter('%(levelname)s %(message)s'))
            logger.addHandler(_handler)
        _logger = logger
        return _logger
    finally:
        _logger_lock.release()


def _detailed_msg(msg):
    file, lineno = inspect.stack()[:3][2][1:3]
    return "{}:{}] {}".format(os.path.split(file)[-1], lineno, msg)


def log(level, msg, *args, **kwargs):
    get_logger().log(level, _detailed_msg(msg), *args, **kwargs)


def debug(msg, *args, **kwargs):
    get_logger().debug(_detailed_msg(msg), *args, **kwargs)


def error(msg, *args, **kwargs):
    get_logger().error(_detailed_msg(msg), *args, **kwargs)
    assert 0


def fatal(msg, *args, **kwargs):
    get_logger().fatal(_detailed_msg(msg), *args, **kwargs)
    assert 0


def info(msg, *args, **kwargs):
    get_logger().info(_detailed_msg(msg), *args, **kwargs)


def warning(msg, *args, **kwargs):
    get_logger().warning(_detailed_msg(msg), *args, **kwargs)


def get_verbosity():
    """Return how much logging output will be produced."""
    return get_logger().getEffectiveLevel()


def set_verbosity(v):
    """Set the threshold for what messages will be logged."""
    get_logger().setLevel(v)


def set_formatter(fmt=None, datefmt=None):
    """Set the formatter."""
    handler = _logging.StreamHandler(_sys.stderr)
    handler.setFormatter(_logging.Formatter(fmt, datefmt))
    logger = get_logger()
    logger.removeHandler(logger.handlers[0])
    logger.addHandler(handler)
