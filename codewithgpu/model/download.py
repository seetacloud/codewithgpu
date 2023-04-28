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

from codewithgpu.utils.cg_cli import get_os_config
import os


def download(model_name, target_directory=None):
    if target_directory is None or target_directory == "":
        target_directory = os.getcwd()
    else:
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
    os_config = get_os_config()
    os_config.download(model_name, target_directory)


def cli_download(args):
    download(args.model, args.target_directory)

