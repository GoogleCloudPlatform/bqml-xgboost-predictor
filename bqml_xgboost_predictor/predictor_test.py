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

"""Tests for third_party.py.bqml_xgboost_predictor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import flags
from absl.testing import absltest
from bqml_xgboost_predictor import predictor

# pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 9):  # `importlib.resources.files` was added in 3.9
  import importlib.resources as importlib_resources
else:
  import importlib_resources

FLAGS = flags.FLAGS


class PredictorTest(absltest.TestCase):

  def test_boosted_tree_regressor(self):

    model_path = str(
        importlib_resources.files('bqml_xgboost_predictor').joinpath(
            'testdata/boosted_tree_regressor_model'))
    test_predictor = predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'b',
        'f3': 3,
        'f2': ['a']
    }, {
        'f1': 'f',
        'f2': ['c', 'a', 'a', 'f'],
        'f3': 0
    }])['predicted_label']
    self.assertAlmostEqual(1.9788086414337158, predict_output[0])
    self.assertAlmostEqual(1.9364699125289917, predict_output[1])

  def test_boosted_tree_classifier(self):
    model_path = str(
        importlib_resources.files('bqml_xgboost_predictor').joinpath(
            'testdata/boosted_tree_classifier_model'))
    test_predictor = predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'b',
        'f3': 3,
        'f2': ['a']
    }, {
        'f1': 'f',
        'f2': ['c', 'a', 'a', 'f'],
        'f3': 0
    }])
    self.assertEqual('2', predict_output[0]['predicted_label'])
    self.assertEqual('2', predict_output[1]['predicted_label'])
    self.assertEqual(['3', '2', '1'], predict_output[0]['label_values'])
    self.assertSequenceAlmostEqual(
        [0.23010218143463135, 0.5752021670341492, 0.1946956366300583],
        predict_output[0]['label_probs'])
    self.assertSequenceAlmostEqual(
        [0.19618307054042816, 0.47606906294822693, 0.3277478516101837],
        predict_output[1]['label_probs'])

  def test_target_encode(self):
    model_path = str(
        importlib_resources.files('bqml_xgboost_predictor').joinpath(
            'testdata/target_encode_model'))
    test_predictor = predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'b',
        'f3': 'c',
        'f2': ['a']
    }, {
        'f1': 'f',
        'f2': ['c', 'a', 'a', 'f'],
        'f3': 'a'
    }])
    self.assertEqual('2', predict_output[0]['predicted_label'])
    self.assertEqual('2', predict_output[1]['predicted_label'])
    self.assertEqual(['3', '2', '1'], predict_output[0]['label_values'])
    self.assertSequenceAlmostEqual(
        [0.19618307054042816, 0.47606906294822693, 0.3277478516101837],
        predict_output[0]['label_probs'])
    self.assertSequenceAlmostEqual(
        [0.19618307054042816, 0.47606906294822693, 0.3277478516101837],
        predict_output[1]['label_probs'])


if __name__ == '__main__':
  absltest.main()
