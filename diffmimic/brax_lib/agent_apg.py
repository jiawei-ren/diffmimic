# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analytic policy gradient training."""

import functools
import time
from typing import Callable, Optional, Tuple

from absl import logging
from brax import envs
from brax.envs import wrappers
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.apg import networks as apg_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.io import model
import flax
import jax
import jax.numpy as jnp
import optax

from diffmimic.brax_lib import acting
from diffmimic.utils.io import serialize_qp

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  normalizer_params: running_statistics.RunningStatisticsState
  policy_params: Params


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def train(environment: envs.Env,
          episode_length: int,
          action_repeat: int = 1,
          num_envs: int = 1,
          max_devices_per_host: Optional[int] = None,
          num_eval_envs: int = 128,
          learning_rate: float = 1e-4,
          seed: int = 0,
          truncation_length: Optional[int] = None,
          max_gradient_norm: float = 1e9,
          num_evals: int = 1,
          normalize_observations: bool = False,
          deterministic_eval: bool = False,
          network_factory: types.NetworkFactory[
              apg_networks.APGNetworks] = apg_networks.make_apg_networks,
          progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
          eval_environment: Optional[envs.Env] = None,
          eval_episode_length: Optional[int] = None,
          save_dir: Optional[str] = None,
          use_linear_scheduler: Optional[bool] = False,
          ):
  """Direct trajectory optimization training."""
  best_pose_error = 1e8

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)
  device_count = local_devices_to_use * process_count

  if truncation_length is not None:
    assert truncation_length > 0

  num_evals_after_init = max(num_evals - 1, 1)

  assert num_envs % device_count == 0
  env = environment
  env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  env = wrappers.VmapWrapper(env)
  env = wrappers.AutoResetWrapper(env)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  apg_network = network_factory(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=normalize)
  make_policy = apg_networks.make_inference_fn(apg_network)
  if use_linear_scheduler:
      lr_scheduler = optax.linear_schedule(init_value=learning_rate, end_value=1e-5, transition_steps=num_evals_after_init)
      optimizer = optax.adam(learning_rate=lr_scheduler)
  else:
      optimizer = optax.adam(learning_rate=learning_rate)

  def env_step(carry: Tuple[envs.State, PRNGKey], step_index: int,
               policy: types.Policy):
    env_state, key = carry
    key, key_sample = jax.random.split(key)
    actions = policy(env_state.obs, key_sample)[0]
    nstate = env.step(env_state, actions)
    if truncation_length is not None:
      nstate = jax.lax.cond(
          jnp.mod(step_index + 1, truncation_length) == 0.,
          jax.lax.stop_gradient, lambda x: x, nstate)

    return (nstate, key), (nstate.reward, env_state.obs, nstate.metrics)

  def loss(policy_params, normalizer_params, key):
    key_reset, key_scan = jax.random.split(key)
    env_state = env.reset(
        jax.random.split(key_reset, num_envs // process_count))
    f = functools.partial(
        env_step, policy=make_policy((normalizer_params, policy_params)))
    (rewards,
     obs, metrics) = jax.lax.scan(f, (env_state, key_scan),
                         (jnp.array(range(episode_length // action_repeat))))[1]
    return -jnp.mean(rewards), (rewards, obs, metrics)

  loss_grad = jax.grad(loss, has_aux=True)

  def clip_by_global_norm(updates):
    g_norm = optax.global_norm(updates)
    trigger = g_norm < max_gradient_norm
    return jax.tree_util.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
        updates)

  def training_epoch(training_state: TrainingState, key: PRNGKey):
    key, key_grad = jax.random.split(key)
    grad_raw, (rewards, obs, metrics) = loss_grad(training_state.policy_params,
                          training_state.normalizer_params, key_grad)
    grad = clip_by_global_norm(grad_raw)
    grad = jax.lax.pmean(grad, axis_name='i')
    grad_raw = jax.lax.pmean(grad_raw, axis_name='i')
    params_update, optimizer_state = optimizer.update(
        grad, training_state.optimizer_state)
    policy_params = optax.apply_updates(training_state.policy_params,
                                        params_update)

    normalizer_params = running_statistics.update(
        training_state.normalizer_params, obs, pmap_axis_name=_PMAP_AXIS_NAME)
    metrics = {
        'grad_norm': optax.global_norm(grad_raw),
        'params_norm': optax.global_norm(policy_params),
        'loss': -1 * rewards,
        **metrics
    }
    return TrainingState(
        optimizer_state=optimizer_state,
        normalizer_params=normalizer_params,
        policy_params=policy_params), metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  training_walltime = 0

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(training_state: TrainingState,
                                 key: PRNGKey) -> Tuple[TrainingState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    (training_state, metrics) = training_epoch(training_state, key)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (episode_length * num_envs) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, metrics

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, eval_key = jax.random.split(local_key)

  # The network key should be global, so that networks are initialized the same
  # way for different processes.
  policy_params = apg_network.policy_network.init(global_key)
  del global_key

  training_state = TrainingState(
      optimizer_state=optimizer.init(policy_params),
      policy_params=policy_params,
      normalizer_params=running_statistics.init_state(
          specs.Array((env.observation_size,), jnp.float32)))
  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

  eval_episode_length = episode_length if not eval_episode_length else eval_episode_length
  if not eval_environment:
      eval_env = env
  else:
      eval_env = eval_environment
      eval_env = wrappers.EpisodeWrapper(eval_env, eval_episode_length, action_repeat)
      eval_env = wrappers.VmapWrapper(eval_env)
      eval_env = wrappers.AutoResetWrapper(eval_env)
  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=eval_episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  # Run initial eval
  if process_id == 0 and num_evals > 1:
    metrics, _ = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.policy_params)),
        training_metrics={})
    best_pose_error = min(metrics['eval/episode_pose_error'], best_pose_error)
    metrics['eval/best_pose_error'] = best_pose_error
    logging.info(metrics)
    progress_fn(0, metrics)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    # optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state,
     training_metrics) = training_epoch_with_timing(training_state, epoch_keys)

    if process_id == 0:
      # Run evals.
      metrics, qp_list = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params)),
          training_metrics)
      best_pose_error = min(metrics['eval/episode_pose_error'], best_pose_error)
      metrics['eval/best_pose_error'] = best_pose_error
      logging.info(metrics)
      progress_fn(it + 1, metrics)
      if save_dir is not None:
          params = _unpmap(
              (training_state.normalizer_params, training_state.policy_params))
          eval_traj = serialize_qp(qp_list)
          if best_pose_error == metrics['eval/episode_pose_error']:
            model.save_params(save_dir + '/params_best.pkl', params)
            with open(save_dir+'/eval_traj_best.npy', 'wb') as f:
                jnp.save(f, eval_traj)
          if (it+1) % 10 == 0:
            with open(save_dir+f'/eval_traj_{it+1}.npy', 'wb') as f:
                jnp.save(f, eval_traj)



  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params))
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
