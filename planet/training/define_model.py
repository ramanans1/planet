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

import functools

import tensorflow as tf

from planet import tools
from planet.training import define_summaries
from planet.training import utility


def define_model(data, trainer, config):
  tf.logging.info('Build TensorFlow compute graph.')
  dependencies = []
  cleanups = []
  step = trainer.step
  global_step = trainer.global_step
  phase = trainer.phase

#Disagreement additions

  cell = []
  for mdl in range(config.num_models):
      with tf.variable_scope('model_no'+str(mdl)):
          cell.append(config.cell())
          kwargs = dict(create_scope_now_=True)

  encoder = tf.make_template('encoder', config.encoder, **kwargs)
  #heads = tools.AttrDict(_unlocked=True)
  heads = tools.AttrDict(_unlocked=True)
  #dummy_features = cell.features_from_state(cell.zero_state(1, tf.float32))
  dummy_features = cell[0].features_from_state(cell[0].zero_state(1, tf.float32))

  for key, head in config.heads.items():
    print('KEYHEAD', key)
    name = 'head_{}'.format(key)
    kwargs = dict(create_scope_now_=True)
    if key in data:
      kwargs['data_shape'] = data[key].shape[2:].as_list()
    elif key == 'action_target':
      kwargs['data_shape'] = data['action'].shape[2:].as_list()
    #heads[key] = tf.make_template(name, head, **kwargs)
    heads[key] = tf.make_template(name, head, **kwargs)
    heads[key](dummy_features)  # Initialize weights.

  embedded = encoder(data)
  with tf.control_dependencies(dependencies):
    embedded = tf.identity(embedded)

  graph = tools.AttrDict(locals())
  posterior = []
  prior = []

  bagging_size = int(0.8*config.batch_shape[0])
  sample_with_replacement = tf.random.uniform([config.num_models, bagging_size], minval=0, maxval=config.batch_shape[0],
                                                dtype= tf.int32)

  for mdl in range(config.num_models):
    with tf.variable_scope('model_no'+str(mdl)):
      bootstrap_action_data = tf.gather(data['action'], sample_with_replacement[mdl,:], axis=0)
      bootstrap_embedded = tf.gather(embedded, sample_with_replacement[mdl,:], axis=0)
      tmp_prior, tmp_posterior = tools.unroll.closed_loop(
          cell[mdl], bootstrap_embedded, bootstrap_action_data, config.debug)
      prior.append(tmp_prior)
      posterior.append(tmp_posterior)

  graph = tools.AttrDict(locals())
  objectives = utility.compute_objectives(
      posterior, prior, data, graph, config)

  summaries, grad_norms = utility.apply_optimizers(
      objectives, trainer, config)

  graph = tools.AttrDict(locals())
  # Active data collection.
  with tf.variable_scope('collection'):
    with tf.control_dependencies(summaries):  # Make sure to train first.
      for name, params in config.train_collects.items():
        schedule = tools.schedule.binary(
            step, config.batch_shape[0],
            params.steps_after, params.steps_every, params.steps_until)
        summary, _ = tf.cond(
            tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
            functools.partial(
                utility.simulate_episodes, config, params, graph, cleanups,
                expensive_summaries=False, gif_summary=False, name=name),
            lambda: (tf.constant(''), tf.constant(0.0)),
            name='should_collect_' + name)
        summaries.append(summary)
  print('AFTER ACTIVE DATA COLLECT')
  # Compute summaries.
  graph = tools.AttrDict(locals())
  # for k,v in graph.items():
  #     print('KEEY',k)
  #assert 1==2
  #TODO: Determine if summary from one model is enough
  summary, score = tf.cond(
      trainer.log,
      lambda: define_summaries.define_summaries(graph, config, cleanups),
      lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
      name='summaries')
  summaries = tf.summary.merge([summaries, summary])
  #TODO: Determine if objective and grad norm printed from only one model is enough
  # Objectives
  dependencies.append(utility.print_metrics(
      {ob.name: ob.value for ob in objectives},
      step, config.print_metrics_every, 'objectives'))
  dependencies.append(utility.print_metrics(
      grad_norms, step, config.print_metrics_every, 'grad_norms'))
  with tf.control_dependencies(dependencies):
    score = tf.identity(score)
  print('Code runs?')
  #assert 1==2
  return score, summaries, cleanups
