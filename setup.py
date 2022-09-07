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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import subprocess

import setuptools
import setuptools.command.install

version = git_version = None
if os.path.exists('version.txt'):
    with open('version.txt', 'r') as f:
        version = f.read().strip()
if os.path.exists('.git'):
    try:
        git_version = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd='./')
        git_version = git_version.decode('ascii').strip()
    except (OSError, subprocess.CalledProcessError):
        pass


def clean_builds():
    for path in ['build', 'codewithgpu.egg-info']:
        if os.path.exists(path):
            shutil.rmtree(path)


def find_packages(top):
    """Return the python sources installed to package."""
    packages = []
    for root, _, _ in os.walk(top):
        if os.path.exists(os.path.join(root, '__init__.py')):
            packages.append(root)
    return packages


class InstallCommand(setuptools.command.install.install):
    """Enhanced 'install' command."""

    def initialize_options(self):
        super(InstallCommand, self).initialize_options()
        self.old_and_unmanageable = True


setuptools.setup(
    name='codewithgpu',
    version=version,
    description='CodeWithGPU Python Client',
    url='https://github.com/seetacloud/codewithgpu',
    author='SeetaCloud',
    license='Apache License',
    packages=find_packages('codewithgpu'),
    package_dir={'codewithgpu': 'codewithgpu'},
    cmdclass={'install': InstallCommand},
    install_requires=[
        'flask',
        'gradio',
        'opencv-python',
        'numpy',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
clean_builds()
