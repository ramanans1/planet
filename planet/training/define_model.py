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
  num_model = 2

#Disagreement additions

  cell = []
  encoder = []
  heads = []
  dummy_features = []
  embedded = []
  objectives = []
  grad_norms = []
  summaries = []
  prior = []
  posterior = []
  # Instantiate network blocks.
  for mdl in range(num_model):
      #cell = config.cell()
      with tf.variable_scope('model_no'+str(mdl)):
          cell.append(config.cell())
          kwargs = dict(create_scope_now_=True)
          #encoder = tf.make_template('encoder', config.encoder, **kwargs)
          encoder.append(tf.make_template('encoder', config.encoder, **kwargs))
          #heads = tools.AttrDict(_unlocked=True)
          heads.append(tools.AttrDict(_unlocked=True))
          #dummy_features = cell.features_from_state(cell.zero_state(1, tf.float32))
          dummy_features.append(cell[mdl].features_from_state(cell[mdl].zero_state(1, tf.float32)))

          for key, head in config.heads.items():
            print('KEYHEAD', key)
            name = 'head_{}'.format(key)
            kwargs = dict(create_scope_now_=True)
            if key in data:
              kwargs['data_shape'] = data[key].shape[2:].as_list()
            elif key == 'action_target':
              kwargs['data_shape'] = data['action'].shape[2:].as_list()
            #heads[key] = tf.make_template(name, head, **kwargs)
            heads[mdl][key] = tf.make_template(name, head, **kwargs)
            heads[mdl][key](dummy_features[mdl])  # Initialize weights.

      # Apply and optimize model.
          #embedded = encoder(data)
          embedded.append(encoder[mdl](data))
          with tf.control_dependencies(dependencies):
            #embedded = tf.identity(embedded)
            embedded[mdl] = tf.identity(embedded[mdl])

          graph = tools.AttrDict(locals())
          # for k,v in local_graph.items():
          #     print('KEEY',k)
          # prior, posterior = tools.unroll.closed_loop(
          #     cell, embedded, data['action'], config.debug)
          tmp_prior, tmp_posterior = tools.unroll.closed_loop(
              cell[mdl], embedded[mdl], data['action'], config.debug)
          prior.append(tmp_prior)
          posterior.append(tmp_posterior)
          # objectives = utility.compute_objectives(
          #     posterior, prior, data, graph, config)
          objectives.append(utility.compute_objectives(
              posterior[mdl], prior[mdl], data, graph, config, mdl))

          tmp_summaries, tmp_grad_norms = utility.apply_optimizers(
              objectives[mdl], trainer, config)
          summaries.append(tmp_summaries)
          grad_norms.append(tmp_grad_norms)

  #summaries = tf.summary.merge([summ for summ in summaries])
  summaries = summaries[0] #TODO: Figure out why all summaries cannot be computed
  graph = tools.AttrDict(locals())
  for k,v in graph.items():
      print('DISAGREE KEY', k)
  #assert 1==2
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
      {ob.name: ob.value for ob in objectives[0]},
      step, config.print_metrics_every, 'objectives'))
  dependencies.append(utility.print_metrics(
      grad_norms[0], step, config.print_metrics_every, 'grad_norms'))
  with tf.control_dependencies(dependencies):
    score = tf.identity(score)
  print('Code runs?')
  assert 1==2
  return score, summaries, cleanups
