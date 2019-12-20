# Copyright 2019 The PlaNet Authors. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def reward_int(state, graph, params):
 
  features = []
  for mdl in range(graph.config.num_models):
      features.append(graph.cell[mdl].mean_features_from_state(state[mdl]))
  features = tf.convert_to_tensor(features)
  mean, variance = tf.nn.moments(features, axes=[0])
  reward, _ = tf.nn.moments(variance, axes=[2])

  return tf.reduce_sum(reward, 1)

def reward(state, graph, params):

    features = graph.cell[0].features_from_state(state[0])
    reward = graph.heads.reward(features).mean()

    return tf.reduce_sum(reward,1)
