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

from planet import tools


def cross_entropy_method(
    cell, objective_fn, state, obs_shape, action_shape, horizon, graph,
    amount=1000, topk=100, iterations=10, min_action=-1, max_action=1):
  num_model = 2
  obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
  original_batch = tools.shape(tools.nested.flatten(state)[0])[0]
  #original_batch = tools.shape(tools.nested.flatten(state[0])[0])[0]
  initial_state = tools.nested.map(lambda tensor: tf.tile(
      tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)
  # for mdl in range(num_model):
  #     initial_state.append(tools.nested.map(lambda tensor: tf.tile(
  #         tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state[mdl]))
  #extended_batch = tools.shape(tools.nested.flatten(initial_state[0])[0])[0]
  extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
  use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
  obs = tf.zeros((extended_batch, horizon) + obs_shape)

  def iteration(mean_and_stddev, _):
    all_model_states = []
    mean, stddev = mean_and_stddev
    # Sample actioperformn proposals from belief.
    normal = tf.random_normal((original_batch, amount, horizon) + action_shape)
    action = normal * stddev[:, None] + mean[:, None]
    action = tf.clip_by_value(action, min_action, max_action)
    # Evaluate proposal actions.
    action = tf.reshape(
        action, (extended_batch, horizon) + action_shape)
    for mdl in range(num_model):
        (_, state), _ = tf.nn.dynamic_rnn(
            cell[mdl], (0 * obs, action, use_obs), initial_state=initial_state)
        all_model_states.append(state)

    return_ = objective_fn(all_model_states, num_model)
    return_ = tf.reshape(return_, (original_batch, amount))
    # Re-fit belief to the best ones.
    _, indices = tf.nn.top_k(return_, topk, sorted=False)
    indices += tf.range(original_batch)[:, None] * amount
    best_actions = tf.gather(action, indices)
    mean, variance = tf.nn.moments(best_actions, 1)
    stddev = tf.sqrt(variance + 1e-6)
    return mean, stddev

  mean = tf.zeros((original_batch, horizon) + action_shape)
  stddev = tf.ones((original_batch, horizon) + action_shape)
  if iterations < 1:
    return mean
  mean, stddev = tf.scan(
      iteration, tf.range(iterations), (mean, stddev), back_prop=False)
  mean, stddev = mean[-1], stddev[-1]  # Select belief at last iterations.
  return mean
