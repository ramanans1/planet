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


def reward(state, num_model, graph, params):
  # print('GRAPH-------------')
  # for k,v in graph.items():
  #     print('KEY',k)
  #     #print('VAL',v)
  #TODO: Compute variance, Right now operating similarly as what Planet does
  features = []
  for mdl in range(num_model):
      features.append(graph.cell[mdl].features_from_state(state[mdl]))
  features = tf.convert_to_tensor(features)
  mean, variance = tf.nn.moments(features, axes=[0])
  reward, _ = tf.nn.moments(variance, axes=[2])
  #print(features)
  #print(variance)
  #assert 1==2

  #features = graph.cell[0].features_from_state(state[0])
  #print(features)
  #print(graph.heads[0].image(features).mean())
  #reward = graph.heads[0].reward(features).mean()
  print(reward)
  #assert 1==2

  return tf.reduce_sum(reward, 1)
