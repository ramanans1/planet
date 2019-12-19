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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from planet import tools


def one_step_model(state, prev_action):

    data_shape=[1024]
    num_layers=1
    activation=tf.nn.relu
    units=1024
    state = tf.stop_gradient(state)
    inputs = tf.concat([state, prev_action], -1)
    for _ in range(num_layers):
        hidden = tf.layers.dense(inputs, units, activation )
        inputs = tf.concat([hidden, prev_action], -1)

    mean = tf.layers.dense(inputs, int(np.prod(data_shape)), None)
    mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)

    return mean
