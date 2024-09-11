import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env
from .system_configs import get_system_cfg
from diffmimic.utils.io import deserialize_qp
from .losses import *
from diffmimic.utils.rotation6d import quaternion_to_rotation_6d


class HumanoidMimic(env.Env):
    """Trains a humanoid to mimic reference motion."""

    def __init__(self, system_config, reference_traj, obs_type='timestamp', cyc_len=None, reward_scaling=1.,
                 rot_weight=1., vel_weight=0., ang_weight=0.):
        super().__init__(config=get_system_cfg(system_config))
        self.reference_qp = deserialize_qp(reference_traj)
        self.reference_len = reference_traj.shape[0]
        self.reward_scaling = reward_scaling
        self.obs_type = obs_type
        self.cycle_len = cyc_len if cyc_len is not None else self.reference_len
        self.rot_weight = rot_weight
        self.vel_weight = vel_weight
        self.ang_weight = ang_weight

    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        reward, done, zero = jp.zeros(3)
        qp = self._get_ref_state(zero)
        metrics = {'step_index': zero, 'pose_error': zero, 'fall': zero}
        obs = self._get_obs(qp, step_index=zero)
        state = env.State(qp, obs, reward, done, metrics)
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        step_index = state.metrics['step_index'] + 1
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, step_index)
        ref_qp = self._get_ref_state(step_idx=step_index)
        reward = -1 * (mse_pos(qp, ref_qp) +
                       self.rot_weight * mse_rot(qp, ref_qp) +
                       self.vel_weight * mse_vel(qp, ref_qp) +
                       self.ang_weight * mse_ang(qp, ref_qp)
                       ) * self.reward_scaling
        fall = jp.where(qp.pos[0, 2] < 0.2, jp.float32(1), jp.float32(0))
        fall = jp.where(qp.pos[0, 2] > 1.7, jp.float32(1), fall)
        state.metrics.update(
            step_index=step_index,
            pose_error=loss_l2_relpos(qp, ref_qp),
            fall=fall,
        )
        state = state.replace(qp=qp, obs=obs, reward=reward)
        return state

    def _get_obs(self, qp: brax.QP, step_index: jp.ndarray) -> jp.ndarray:
        """Observe humanoid body position, velocities, and angles."""
        pos, rot, vel, ang = qp.pos[:-1], qp.rot[:-1], qp.vel[:-1], qp.ang[:-1]  # Remove floor
        rot_6d = quaternion_to_rotation_6d(rot)
        rel_pos = (pos - pos[0])[1:]

        if self.obs_type == 'timestamp':
            phi = (step_index % self.cycle_len) / self.cycle_len
            obs = jp.concatenate([rel_pos.reshape(-1), rot_6d.reshape(-1), vel.reshape(-1), ang.reshape(-1),
                                  phi[None]], axis=-1)
        elif self.obs_type == 'target_state':
            target_qp = self._get_ref_state(step_idx=step_index + 1)
            target_pos, target_rot, target_vel, target_ang = target_qp.pos[:-1], target_qp.rot[:-1], target_qp.vel[:-1], target_qp.ang[:-1]
            target_rot_6d = quaternion_to_rotation_6d(target_rot)
            obs = jp.concatenate([pos.reshape(-1), rot.reshape(-1), vel.reshape(-1), ang.reshape(-1),
                                  target_pos.reshape(-1), target_rot_6d.reshape(-1), target_vel.reshape(-1), target_ang.reshape(-1)
                                  ], axis=-1)
        else:
            raise NotImplementedError
        return obs

    def _get_ref_state(self, step_idx) -> brax.QP:
        mask = jp.where(step_idx == jp.arange(0, self.reference_len), jp.float32(1), jp.float32(0))
        ref_state = jp.tree_map(lambda x: (mask @ x.transpose(1, 0, 2)), self.reference_qp)
        return ref_state
