# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to compute V-trace off-policy actor critic targets.
For details and theory see:
"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.
See https://arxiv.org/abs/1802.01561 for the full paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

from agent.KLDiv import KL_from_logits

DRTraceFromLogitsReturns = namedtuple(
    'DRTraceFromLogitsReturns',
    ['vs', 'qs', 'advantages', 'log_rhos',
     'behaviour_action_log_probs', 'target_action_log_probs'])

DRTraceReturns = namedtuple('DRTraceReturns', ['vs', 'qs', 'advantages'])


def log_probs_from_logits_and_actions(policy_logits, actions):
    policy_logits = tf.convert_to_tensor(policy_logits, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    return -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=policy_logits, labels=actions)


def from_logits(
        behaviour_policy_logits, target_policy_logits, actions,
        discounts, rewards, v_values, q_values, target_values, bootstrap_value,
        clip_rho_threshold=[0.95, 1.05], clip_c_threshold=[0.95, 1.05], laser_threshold=0.01,
        name='drtrace_from_logits'):
    behaviour_policy_logits = tf.convert_to_tensor(
        behaviour_policy_logits, dtype=tf.float32)
    target_policy_logits = tf.convert_to_tensor(
        target_policy_logits, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    # Make sure tensor ranks are as expected.
    # The rest will be checked by from_action_log_probs.
    behaviour_policy_logits.shape.assert_has_rank(3)
    target_policy_logits.shape.assert_has_rank(3)
    actions.shape.assert_has_rank(2)

    with tf.name_scope(name, values=[
        behaviour_policy_logits, target_policy_logits, actions,
        discounts, rewards, v_values, q_values, target_values, bootstrap_value]):
        target_action_log_probs = log_probs_from_logits_and_actions(
            target_policy_logits, actions)
        behaviour_action_log_probs = log_probs_from_logits_and_actions(
            behaviour_policy_logits, actions)
        log_rhos = target_action_log_probs - behaviour_action_log_probs
        if laser_threshold is not None:
            lambdas = KL_from_logits(target_policy_logits, behaviour_policy_logits)
            lambdas = tf.cast(lambdas < laser_threshold, tf.float32)
        else:
            lambdas = tf.ones_like(actions, tf.float32)
        vtrace_returns = from_importance_weights(
            log_rhos=log_rhos,
            lambdas=lambdas,
            discounts=discounts,
            rewards=rewards,
            v_values=v_values,
            q_values=q_values,
            target_values=target_values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=clip_rho_threshold,
            clip_c_threshold=clip_c_threshold)

        return DRTraceFromLogitsReturns(
            log_rhos=log_rhos,
            behaviour_action_log_probs=behaviour_action_log_probs,
            target_action_log_probs=target_action_log_probs,
            **vtrace_returns._asdict()
        )


def from_importance_weights(
        log_rhos, lambdas, discounts, rewards, v_values, q_values, target_values, bootstrap_value,
        clip_rho_threshold=[0.95, 1.05], clip_c_threshold=[0.95, 1.05],
        name='vtrace_from_importance_weights'):
    log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
    lambdas = tf.convert_to_tensor(lambdas, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    v_values = tf.convert_to_tensor(v_values, dtype=tf.float32)
    q_values = tf.convert_to_tensor(q_values, dtype=tf.float32)
    target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)

    # Make sure tensor ranks are consistent.
    rho_rank = log_rhos.shape.ndims
    v_values.shape.assert_has_rank(rho_rank)
    q_values.shape.assert_has_rank(rho_rank)
    target_values.shape.assert_has_rank(rho_rank)
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)
    discounts.shape.assert_has_rank(rho_rank)
    rewards.shape.assert_has_rank(rho_rank)

    ###########################################
    log_rhos = tf.transpose(log_rhos, perm=[1, 0])
    lambdas = tf.transpose(lambdas, perm=[1, 0])
    discounts = tf.transpose(discounts, perm=[1, 0])
    rewards = tf.transpose(rewards, perm=[1, 0])
    v_values = tf.transpose(v_values, perm=[1, 0])
    q_values = tf.transpose(q_values, perm=[1, 0])
    target_values = tf.transpose(target_values, perm=[1, 0])
    ###########################################

    with tf.name_scope(name, values=[
        log_rhos, discounts, rewards, v_values, q_values, target_values, bootstrap_value]):
        rhos = tf.exp(log_rhos)

        if clip_c_threshold is not None:
            clipped_c_rhos = tf.clip_by_value(
                rhos, clip_c_threshold[0], clip_c_threshold[1], name="cs")
        else:
            clipped_c_rhos = rhos
        cs = lambdas * clipped_c_rhos

        if clip_rho_threshold is not None:
            clipped_rho_rhos = tf.clip_by_value(
                rhos, clip_rho_threshold[0], clip_rho_threshold[1], name="rhos")
        else:
            clipped_rho_rhos = rhos
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = tf.concat(
            [target_values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = clipped_rho_rhos * (rewards + discounts * values_t_plus_1 - q_values)

        sequences = (discounts, cs, deltas)

        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.
        def scanfunc(acc, sequence_item):
            discount_t, c_t, delta_t = sequence_item
            return delta_t + discount_t * c_t * acc

        initial_values = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = tf.scan(
            fn=scanfunc,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            back_prop=False,
            reverse=True,  # Computation starts from the back.
            name='scan')

        # Add V(x_s) to get v_s.
        vs = tf.add(vs_minus_v_xs, v_values, name='vs')

        # Advantage for policy gradient.
        vs_t_plus_1 = tf.concat([
            vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        qs = rewards + discounts * vs_t_plus_1
        advantages = qs - v_values

        ###########################################
        vs = tf.transpose(vs, perm=[1, 0])
        qs = tf.transpose(qs, perm=[1, 0])
        advantages = tf.transpose(advantages, perm=[1, 0])
        ###########################################

        # Make sure no gradients backpropagated through the returned values.
        return DRTraceReturns(vs=tf.stop_gradient(vs),
                             qs=tf.stop_gradient(qs),
                             advantages=tf.stop_gradient(advantages))


def gae(lambdas, discounts, rewards, v_values, q_values, target_values, bootstrap_value,
        name='generalized_advantage_estimator'):
    lambdas = tf.convert_to_tensor(lambdas, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    v_values = tf.convert_to_tensor(v_values, dtype=tf.float32)
    q_values = tf.convert_to_tensor(q_values, dtype=tf.float32)
    target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)

    # Make sure tensor ranks are consistent.
    rho_rank = lambdas.shape.ndims
    v_values.shape.assert_has_rank(rho_rank)
    q_values.shape.assert_has_rank(rho_rank)
    target_values.shape.assert_has_rank(rho_rank)
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)
    discounts.shape.assert_has_rank(rho_rank)
    rewards.shape.assert_has_rank(rho_rank)

    ###########################################
    lambdas = tf.transpose(lambdas, perm=[1, 0])
    discounts = tf.transpose(discounts, perm=[1, 0])
    rewards = tf.transpose(rewards, perm=[1, 0])
    v_values = tf.transpose(v_values, perm=[1, 0])
    q_values = tf.transpose(q_values, perm=[1, 0])
    target_values = tf.transpose(target_values, perm=[1, 0])
    ###########################################

    with tf.name_scope(name, values=[
        discounts, rewards, v_values, q_values, target_values, bootstrap_value]):

        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = tf.concat(
            [target_values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = (rewards + discounts * values_t_plus_1 - q_values)

        sequences = (discounts, lambdas, deltas)

        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.
        def scanfunc(acc, sequence_item):
            discount_t, c_t, delta_t = sequence_item
            return delta_t + discount_t * c_t * acc

        initial_values = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = tf.scan(
            fn=scanfunc,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            back_prop=False,
            reverse=True,  # Computation starts from the back.
            name='scan')

        # Add V(x_s) to get v_s.
        vs = tf.add(vs_minus_v_xs, v_values, name='vs')

        # Advantage for policy gradient.
        vs_t_plus_1 = tf.concat([
            vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        qs = rewards + discounts * vs_t_plus_1
        advantages = qs - v_values

        ###########################################
        vs = tf.transpose(vs, perm=[1, 0])
        qs = tf.transpose(qs, perm=[1, 0])
        advantages = tf.transpose(advantages, perm=[1, 0])
        ###########################################

        # Make sure no gradients backpropagated through the returned values.
        return DRTraceReturns(vs=tf.stop_gradient(vs),
                             qs=tf.stop_gradient(qs),
                             advantages=tf.stop_gradient(advantages))