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
"""Python setup script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_py
import setuptools.command.install


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=None)
    args, unknown = parser.parse_known_args()
    args.git_version = None
    args.long_description = ''
    sys.argv = [sys.argv[0]] + unknown
    if args.version is None and os.path.exists('version.txt'):
        with open('version.txt', 'r') as f:
            args.version = f.read().strip()
    if os.path.exists('.git'):
        try:
            git_version = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd='./')
            args.git_version = git_version.decode('ascii').strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if os.path.exists('README.md'):
        with open(os.path.join('README.md'), encoding='utf-8') as f:
            args.long_description = f.read()
    return args


def clean_builds():
    for path in ['build', 'codewithgpu.egg-info']:
        if os.path.exists(path):
            shutil.rmtree(path)
    if os.path.exists('codewithgpu/version.py'):
        os.remove('codewithgpu/version.py')


def find_packages(top):
    """Return the python sources installed to package."""
    packages = []
    for root, _, _ in os.walk(top):
        if os.path.exists(os.path.join(root, '__init__.py')):
            packages.append(root)
    return packages


def find_package_data(top):
    """Return the external data installed to package."""
    protos = ['data/record.proto', 'data/tf_record.proto']
    return protos


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Enhanced 'build_py' command."""

    def build_packages(self):
        with open('codewithgpu/version.py', 'w') as f:
            f.write("from __future__ import absolute_import\n"
                    "from __future__ import division\n"
                    "from __future__ import print_function\n\n"
                    "version = '{}'\n"
                    "git_version = '{}'\n".format(args.version, args.git_version))
        protoc = self.get_finalized_command('install').protoc
        if protoc is not None:
            cmd = '{} -I codewithgpu/data --python_out codewithgpu/data '
            cmd += 'codewithgpu/data/record.proto '
            cmd += 'codewithgpu/data/tf_record.proto'
            subprocess.call(cmd.format(protoc), shell=True)
        self.packages = find_packages('codewithgpu')
        super(BuildPyCommand, self).build_packages()

    def build_package_data(self):
        self.package_data = {'codewithgpu': find_package_data('codewithgpu')}
        super(BuildPyCommand, self).build_package_data()


class InstallCommand(setuptools.command.install.install):
    """Enhanced 'install' command."""

    user_options = setuptools.command.install.install.user_options
    user_options += [('protoc=', None, "path to the protobuf compiler")]

    def initialize_options(self):
        self.protoc = None
        super(InstallCommand, self).initialize_options()
        self.old_and_unmanageable = True


args = parse_args()
setuptools.setup(
    name='codewithgpu-test',
    version=args.version,
    description='CodeWithGPU Python Client',
    long_description=args.long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/seetacloud/codewithgpu',
    author='SeetaCloud',
    license='Apache License',
    packages=find_packages('codewithgpu'),
    cmdclass={'build_py': BuildPyCommand, 'install': InstallCommand},
    install_requires=['numpy',
                      'protobuf>=3.9.1,<=3.20.1',
                      'opencv-python',
                      'flask',
                      'gradio'],
    entry_points={
          "console_scripts": [
              "cg = codewithgpu.model.download:download_cli",
          ],
      },
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: Apache Software License',
                 'Programming Language :: C++',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3 :: Only',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
clean_builds()
