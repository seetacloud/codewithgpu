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
import argparse


def download(model_name, target_directory=None):
    if len(model_name.split("/")) < 2:
        print("ERROR: model name `{}` format invalid. should be <model_name>/<file_name>".format(model_name))
        return
    if target_directory is None or target_directory == "":
        target_directory = os.getcwd()
    else:
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
    os_config = get_os_config()
    os_config.download(model_name, target_directory)


def download_cli():
    parser = argparse.ArgumentParser(description="CodeWithGPU.com CLI tools.")
    subparsers = parser.add_subparsers(title="down", metavar="<down>")
    parser_down = subparsers.add_parser("down", help='download model use command: `cg down <model_name>/<file_name> -t <target directory>`')
    parser_down.add_argument('model', type=str, help='model name. should be <model_name>/<file_name> format')
    parser_down.add_argument('-t', '--target_directory', type=str, default=None,
                             help='set download directory. default: current directory')
    args = parser.parse_args()
    if "model" not in args:
        print("ERROR: model parameter not found. use `cg down --help` for help")
        print("1. If download model. please use command: `cg down <model_name>/<file_name> -t <target directory>`")
        return
    download(args.model, target_directory=args.target_directory)


if __name__ == '__main__':
    # pass
    download_cli()
    # download("test/cg-linux", "/tmp")
