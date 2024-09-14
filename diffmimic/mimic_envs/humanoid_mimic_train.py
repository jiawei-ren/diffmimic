import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env
from .humanoid_mimic import HumanoidMimic
from .losses import *
import jax


class HumanoidMimicTrain(HumanoidMimic):
    """Trains a humanoid to mimic reference motion."""

    def __init__(self, total_length, rollout_length, early_termination, demo_replay_mode, err_threshold, replay_rate,
                 **kwargs):
        super().__init__(**kwargs)
        self.total_length = total_length
        self.rollout_length = rollout_length
        self.early_termination = early_termination
        self.demo_replay_mode = demo_replay_mode
        self.err_threshold = err_threshold
        self.replay_rate = replay_rate

    def reset(self, rng: jp.ndarray) -> env.State:
        reward, done, zero = jp.zeros(3)
        step_index = jp.randint(rng, high=self.total_length-self.rollout_length+1)   # random state initialization (RSI)
        qp = self._get_ref_state(step_index)
        metrics = {'step_index': step_index, 'pose_error': zero, 'fall': zero}
        obs = self._get_obs(qp, step_index=step_index)
        state = env.State(qp, obs, reward, done, metrics)
        if self.demo_replay_mode != 'none':
            state.metrics.update(replay=jp.zeros(1)[0])
        if self.demo_replay_mode == 'random':
            replay_key, rng = jp.random_split(rng)
            state.metrics.update(replay_key=rng)
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super(HumanoidMimicTrain, self).step(state, action)
        if self.early_termination:
            state = state.replace(done=state.metrics['fall'])
        if self.demo_replay_mode != 'none':
            state = self._demo_replay(state)
        return state

    def _demo_replay(self, state) -> env.State:
        qp = state.qp
        ref_qp = self._get_ref_state(state.metrics['step_index'])
        if self.demo_replay_mode == 'threshold':
            error = loss_l2_pos(qp, ref_qp)
            replay = jp.where(error > self.err_threshold, jp.float32(1), jp.float32(0))
        elif self.demo_replay_mode == 'random':
            replay_key, key = jax.random.split(state.metrics['replay_key'])
            state.metrics.update(replay_key=replay_key)
            replay = jp.where(jax.random.bernoulli(key, p=self.replay_rate), jp.float32(1), jp.float32(0))
        else:
            raise NotImplementedError
        qp = jp.tree_map(lambda x: x*(1 - replay), qp) + jp.tree_map(lambda x: x*replay, ref_qp)
        obs = self._get_obs(qp, state.metrics['step_index'])
        state.metrics.update(replay=replay)
        return state.replace(qp=qp, obs=obs)
