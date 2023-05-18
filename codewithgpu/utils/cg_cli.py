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

import os
import sys
import stat
import requests


class OSConfig:
    def __init__(self):
        pass

    def get_cli_path(self):
        return ""

    def try_download_cg_cli(self):
        path = self.get_cli_path()
        if os.path.exists(path):
            return
        print("Init CodeWithGPU CLI Tools. Please wait...")
        cli_dir = os.path.dirname(path)
        if not os.path.exists(cli_dir):
            os.makedirs(cli_dir)
        try:
            response = requests.get(url=self.URL)
        except Exception as e:
            print("Init failed. error reason: ", e)
            exit()
        try:
            with open(path, "wb") as fo:
                fo.write(response.content)
        except Exception as e:
            print("Init failed. error reason: ", e)
            os.remove(path)
            exit()
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXO)
        print("Init success!")

    def download(self, model_name, target_directory):
        self.try_download_cg_cli()
        target_path_name = model_name
        if "win" in sys.platform:
            target_path_name = model_name.replace("/", "\\")
        print(">>> download to ", os.path.join(target_directory, target_path_name))
        os.system("{} down {} -t {}".format(self.get_cli_path(), model_name, target_directory))

    def upload(self, local_file, token):
        self.try_download_cg_cli()
        os.system("{} upload {} --token {}".format(self.get_cli_path(), local_file, token))

    def upgrade(self):
        print("Upgrade CodeWithGPU CLI Tools. Please wait...")
        path = self.get_cli_path()
        if os.path.exists(path):
            os.remove(path)
        cli_dir = os.path.dirname(path)
        if not os.path.exists(cli_dir):
            os.makedirs(cli_dir)
        try:
            response = requests.get(url=self.URL)
        except Exception as e:
            print("Upgrade failed. error reason: ", e)
            exit()
        try:
            with open(path, "wb") as fo:
                fo.write(response.content)
        except Exception as e:
            print("Upgrade failed. error reason: ", e)
            os.remove(path)
            exit()
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXO)
        print("Upgrade success!")


class Windows(OSConfig):
    URL = "https://autodl-public.ks3-cn-beijing.ksyuncs.com/tool/cg-win.exe"

    def __init__(self):
        super(Windows, self).__init__()

    def get_cli_path(self):
        return os.path.dirname(__file__) + "\\cg-win.exe"


class Linux(OSConfig):
    URL = "https://autodl-public.ks3-cn-beijing.ksyuncs.com/tool/cg-linux"

    def __init__(self):
        super(Linux, self).__init__()

    def get_cli_path(self):
        return os.path.dirname(__file__) + "/cg-linux"


class MacOS(OSConfig):
    URL = "https://autodl-public.ks3-cn-beijing.ksyuncs.com/tool/cg-mac"

    def __init__(self):
        super(MacOS, self).__init__()

    def get_cli_path(self):
        return os.path.dirname(__file__) + "/cg-mac"


def get_os_config():
    if "linux" in sys.platform:
        return Linux()
    if "win" in sys.platform:
        return Windows()
    if "darwin" in sys.platform:
        return MacOS()
    raise Exception("CodeWithGPU CLI Tools not support os `{}`.".format(sys.platform))


def upgrade_cli(args):
    config = get_os_config()
    config.upgrade()
