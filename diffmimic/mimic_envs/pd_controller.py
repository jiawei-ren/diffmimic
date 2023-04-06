from typing import Tuple

from brax import jumpy as jp
from brax.physics.base import P, QP
from brax.physics.actuators import Angle


def compute_pd_control(target_angles, current_angles, current_velocities, Kp, Kd):
    error = target_angles - current_angles
    error_derivative = -current_velocities
    control_output = Kp * error + Kd * error_derivative
    return control_output


def apply_reduced(self, act: jp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
  axis, angle = self.joint.axis_angle(qp_p, qp_c)
  vel = tuple([jp.dot(qp_c.ang - qp_p.ang, ax) for ax in axis])
  axis, angle, vel = jp.array(axis), jp.array(angle), jp.array(vel)

  target_angles = act * jp.pi * 1.2

  torque = compute_pd_control(target_angles, angle, vel, Kp=self.strength, Kd=self.strength/10)
  torque = jp.sum(jp.vmap(jp.multiply)(axis, torque), axis=0)

  dang_p = -self.joint.body_p.inertia * torque
  dang_c = self.joint.body_c.inertia * torque

  return dang_p, dang_c

Angle.apply_reduced = apply_reduced