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

from tensorflow_probability import distributions as tfd
import tensorflow as tf

from planet.tools import nested

##### Initialize the MPC agent to one of the models (lets say 0), and with every begin episode, choose a different model

class MPCAgent(object):

  def __init__(self, batch_env, step, is_training, should_log, config):
    self._batch_env = batch_env
    self._step = step  # Trainer step, not environment step.
    self._is_training = is_training
    self._should_log = should_log
    self._config = config
    self._num_models = config.num_models
    #self._cell = config.cell
    #self._modelsampler = tfd.Uniform(low=0.0,high=2.0)
    #self._model = tf.dtypes.cast(self._modelsampler.sample(),tf.int32)
    self._cell = config.cell ### Initialize with the 0th model
    #state = self._cell[0].zero_state(len(batch_env), tf.float32) #Using a type of cell to init the state
    #print(state)
    var_like = lambda x: tf.get_local_variable(
        x.name.split(':')[0].replace('/', '_') + '_var',
        shape=x.shape,
        initializer=lambda *_, **__: tf.zeros_like(x), use_resource=True)
    self._state = []
    for mdl in range(self._num_models):
        self._state.append(nested.map(var_like, self._cell[mdl].zero_state(len(batch_env), tf.float32)))
        # print('asdfasdf',self._state)

    #self._state = nested.map(var_like, state)
    self._prev_action = tf.get_local_variable(
        'prev_action_var', shape=self._batch_env.action.shape,
        initializer=lambda *_, **__: tf.zeros_like(self._batch_env.action),
        use_resource=True)

  def begin_episode(self, agent_indices):
    reset_state = []
    for mdl in range(self._num_models):
        state = nested.map(
            lambda tensor: tf.gather(tensor, agent_indices),
            self._state[mdl])
        reset_state.append(nested.map(
            lambda var, val: tf.scatter_update(var, agent_indices, 0 * val),
            self._state[mdl], state, flatten=True))

    reset_prev_action = self._prev_action.assign(
        tf.zeros_like(self._prev_action))
    controldep = reset_state[0]
    for mdl in range(1,self._num_models):
        controldep += reset_state[mdl]

    with tf.control_dependencies(controldep + (reset_prev_action,)):
        return tf.constant('')

  def perform(self, agent_indices, observ):
    observ = self._config.preprocess_fn(observ)
    embedded = self._config.encoder({'image': observ[:, None]})[:, 0]

    prev_action = self._prev_action + 0
    with tf.control_dependencies([prev_action]):
      use_obs = tf.ones(tf.shape(agent_indices), tf.bool)[:, None]
      state = []
      for mdl in range(self._num_models):
          tmpstate = nested.map(lambda tensor: tf.gather(tensor, agent_indices),self._state[mdl])
          _, tmpstate = self._cell[mdl]((embedded, prev_action, use_obs), tmpstate)
          state.append(tmpstate)
    action = self._config.planner(
        self._config.cell, self._config.objective, state,
        embedded.shape[1:].as_list(),
        prev_action.shape[1:].as_list())
    action = action[:, 0]
    if self._config.exploration:
      scale = self._config.exploration.scale
      if self._config.exploration.schedule:
        scale *= self._config.exploration.schedule(self._step)
      action = tfd.Normal(action, scale).sample()
    action = tf.clip_by_value(action, -1, 1)
    remember_action = self._prev_action.assign(action)
    remember_state = nested.map(
        lambda var, val: tf.scatter_update(var, agent_indices, val),
        self._state[0], state[0], flatten=True)
    for mdl in range(1,self._num_models):
        remember_state += nested.map(
            lambda var, val: tf.scatter_update(var, agent_indices, val),
            self._state[mdl], state[mdl], flatten=True)
    with tf.control_dependencies(remember_state + (remember_action,)):
      return tf.identity(action), tf.constant('')

  def experience(self, agent_indices, *experience):
    return tf.constant('')

  def end_episode(self, agent_indices):
    return tf.constant('')
