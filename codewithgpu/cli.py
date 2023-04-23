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

import argparse
from codewithgpu.model import download
from codewithgpu.utils import cg_cli


def main_cli():
    parser = argparse.ArgumentParser(description="CodeWithGPU.com CLI tools.")
    subparsers = parser.add_subparsers(title="down", metavar="<down>")
    # download
    parser_down = subparsers.add_parser("down",
                                        help='download model use command: `cg down <model_name>/<file_name> -t <target directory>`')
    parser_down.add_argument('model', type=str, help='model name. should be <model_name>/<file_name> format')
    parser_down.add_argument('-t', '--target_directory', type=str, default=None,
                             help='set download directory. default: current directory')
    parser_down.set_defaults(func=download.cli_download)
    # upgrade
    parser_upgrade = subparsers.add_parser("upgrade",
                                           help='upgrade cli tools use command: `cg upgrade`')
    parser_upgrade.set_defaults(func=cg_cli.upgrade_cli)
    args = parser.parse_args()
    if "func" not in args:
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main_cli()
