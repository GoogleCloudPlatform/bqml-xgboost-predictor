# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install BQML XGBoost Predictor."""

import setuptools

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setuptools.setup(
    name='bqml_xgboost_predictor',
    version='0.0.1',
    description='BQML XGBoost Predictor',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='https://github.com/GoogleCloudPlatform/bqml-xgboost-predictor',
    license='Apache 2.0',
    packages=['bqml_xgboost_predictor'],
    install_requires=['absl-py', 'xgboost'],
    keywords='xgboost bqml',
)
