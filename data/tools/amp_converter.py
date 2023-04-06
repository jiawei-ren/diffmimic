import os
import glob
import numpy as np
from diffmimic.mimic_envs.system_configs import get_system_cfg
from brax import math
from brax.physics import bodies
from brax.physics.base import QP, vec_to_arr
from data.tools.rotation_utils.conversions import *
from data.tools.rotation_utils.quaternion import *

from diffmimic.mimic_envs import register_mimic_env
import json
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

register_mimic_env()

CFG_AMP = get_system_cfg('amp')


def convert_to_qp(ase_poses, ase_vel, ase_ang, pelvis_trans):
    qp_list = []
    for i in range(ase_poses.shape[0]):
        qp = QP.zero(shape=(len(CFG_AMP.bodies),))
        body = bodies.Body(CFG_AMP)
        # set any default qps from the config
        joint_idxs = []
        j_idx = 0
        for j in CFG_AMP.joints:
            beg = joint_idxs[-1][1][1] if joint_idxs else 0
            dof = len(j.angle_limit)
            joint_idxs.append((j, (beg, beg + dof), j_idx))
            j_idx += 1
        lineage = {j.child: j.parent for j in CFG_AMP.joints}
        depth = {}
        for child, parent in lineage.items():
            depth[child] = 1
            while parent in lineage:
                parent = lineage[parent]
                depth[child] += 1
        joint_idxs = sorted(joint_idxs, key=lambda x: depth.get(x[0].parent, 0))
        joint = [j for j, _, _ in joint_idxs]
        joint_order = [i for _, _, i in joint_idxs]

        # update qp in depth order
        joint_body = jp.array([
            (body.index[j.parent], body.index[j.child]) for j in joint
        ])
        joint_off = jp.array([(vec_to_arr(j.parent_offset),
                               vec_to_arr(j.child_offset)) for j in joint])

        local_rot = ase_poses[i][1:]
        world_vel = ase_vel[i][1:]
        world_ang = ase_ang[i][1:]

        def init(qp):
            pos = jp.index_update(qp.pos, 0, pelvis_trans[i])
            rot = ase_poses[i][0] / jp.norm(ase_poses[i][0])  # important
            rot = jp.index_update(qp.rot, 0, rot)
            vel = jp.index_update(qp.vel, 0, ase_vel[i][0])
            ang = jp.index_update(qp.ang, 0, ase_ang[i][0])
            qp = qp.replace(pos=pos, rot=rot, vel=vel, ang=ang)
            return qp

        qp = init(qp)
        amp_rot = local_rot[joint_order]
        world_vel = world_vel[joint_order]
        world_ang = world_ang[joint_order]

        num_joint_dof = sum(len(j.angle_limit) for j in CFG_AMP.joints)
        num_joints = len(CFG_AMP.joints)
        takes = []
        for j, (beg, end), _ in joint_idxs:
            arr = list(range(beg, end))
            arr.extend([num_joint_dof] * (3 - len(arr)))
            takes.extend(arr)
        takes = jp.array(takes, dtype=int)

        def to_dof(a):
            b = np.zeros([num_joint_dof])
            for idx, (j, (beg, end), _) in enumerate(joint_idxs):
                b[beg:end] = a[idx, :end - beg]
            return b

        def to_3dof(a):
            a = jp.concatenate([a, jp.array([0.0])])
            a = jp.take(a, takes)
            a = jp.reshape(a, (num_joints, 3))
            return a

        # build local rot and ang per joint
        joint_rot = jp.array(
            [math.euler_to_quat(vec_to_arr(j.rotation)) for j in joint])
        joint_ref = jp.array(
            [math.euler_to_quat(vec_to_arr(j.reference_rotation)) for j in joint])

        def local_rot_ang(_, x):
            angles, vels, rot, ref = x
            axes = jp.vmap(math.rotate, [True, False])(jp.eye(3), rot)
            ang = jp.dot(axes.T, vels).T
            rot = ref
            for axis, angle in zip(axes, angles):
                # these are euler intrinsic rotations, so the axes are rotated too:
                axis = math.rotate(axis, rot)
                next_rot = math.quat_rot_axis(axis, angle)
                rot = math.quat_mul(next_rot, rot)
            return (), (rot, ang)

        def local_rot_ang_inv(_, x):
            angles, vels, rot, ref = x
            axes = jp.vmap(math.rotate, [True, False])(jp.eye(3), math.quat_inv(rot))
            ang = jp.dot(axes.T, vels).T
            rot = ref
            for axis, angle in zip(axes, angles):
                # these are euler intrinsic rotations, so the axes are rotated too:
                axis = math.rotate(axis, rot)
                next_rot = math.quat_rot_axis(axis, angle)
                rot = math.quat_mul(next_rot, rot)
            return (), (rot, ang)

        amp_rot = quaternion_to_euler(amp_rot)
        xs = (amp_rot, world_ang, joint_rot, joint_ref)
        _, (amp_rot, _) = jp.scan(local_rot_ang_inv, (), xs, len(joint))
        amp_rot = quaternion_to_euler(amp_rot)
        amp_rot = to_3dof(to_dof(amp_rot))

        xs = (amp_rot, world_ang, joint_rot, joint_ref)
        _, (amp_rot, _) = jp.scan(local_rot_ang, (), xs, len(joint))

        def set_qp(carry, x):
            qp, = carry
            (body_p, body_c), (off_p, off_c), local_rot, world_ang, world_vel = x
            local_rot = local_rot / jp.norm(local_rot)  # important
            world_rot = math.quat_mul(qp.rot[body_p], local_rot)
            world_rot = world_rot / jp.norm(world_rot)  # important
            local_pos = off_p - math.rotate(off_c, local_rot)
            world_pos = qp.pos[body_p] + math.rotate(local_pos, qp.rot[body_p])
            world_vel = qp.vel[body_p] + math.rotate(local_pos, math.euler_to_quat(qp.ang[body_p]))
            pos = jp.index_update(qp.pos, body_c, world_pos)
            rot = jp.index_update(qp.rot, body_c, world_rot)
            vel = jp.index_update(qp.vel, body_c, world_vel)
            ang = jp.index_update(qp.ang, body_c, world_ang)
            qp = qp.replace(pos=pos, rot=rot, vel=vel, ang=ang)
            return (qp,), ()

        xs = (joint_body, joint_off, amp_rot, world_ang, world_vel)
        (qp,), () = jp.scan(set_qp, (qp,), xs, len(joint))

        # any trees that have no body qp overrides in the config are moved above
        # the xy plane.  this convenience operation may be removed in the future.
        fixed = {j.child for j in joint}
        root_idx = {
            b.name: [i]
            for i, b in enumerate(CFG_AMP.bodies)
            if b.name not in fixed
        }
        for j in joint:
            parent = j.parent
            while parent in lineage:
                parent = lineage[parent]
            if parent in root_idx:
                root_idx[parent].append(body.index[j.child])

        for children in root_idx.values():
            zs = jp.array([
                bodies.min_z(jp.take(qp, c), CFG_AMP.bodies[c]) for c in children
            ])
            min_z = min(jp.amin(zs), 0)
            children = jp.array(children)
            pos = jp.take(qp.pos, children) - min_z * jp.array([0., 0., 1.])
            pos = jp.index_update(qp.pos, children, pos)
            qp = qp.replace(pos=pos)
        qp_list.append(qp)
    return qp_list


def convert_to_states(qp_list, target_dt):
    demo_traj = []
    frame_count = 0
    start_pos = np.zeros(3)
    total_frames = int(20 / target_dt) + 1
    while True:
        for i in range(1, len(qp_list)):
            qp = qp_list[i]
            pos = qp.pos.copy()
            pos[:-1, :2] += start_pos[:2]
            demo_traj.append(
                np.concatenate([pos.reshape(-1), qp.rot.reshape(-1), qp.vel.reshape(-1), qp.ang.reshape(-1)], axis=-1))
            frame_count += 1
            if frame_count == total_frames:
                demo_traj = np.stack(demo_traj, axis=0)
                return demo_traj
        start_pos[:2] += qp_list[i].pos[0, :2]


def convert(x):
    x = np.array(x)
    if x.shape[0] == 3:
        x = x[[0, 2, 1]]
        x[1] *= -1
        return x
    if x.shape[0] == 1:
        x = euler_to_quaternion(np.array([0, -1 * x[0], 0]))
        return x
    else:
        x = x[[0, 1, 3, 2]]
        x[2] *= -1
        return x


def interpolate(y, dt, target_dt, gt=None):
    x = np.arange(y.shape[0]) * dt
    x_target = np.arange(int(y.shape[0] * dt / target_dt)) * target_dt
    cs = CubicSpline(x, y)
    vel = cs.derivative()(x_target)
    vel_smooth = gaussian_filter1d(vel, sigma=2 * dt / target_dt, axis=0)
    if gt is not None:
        plt.plot(x, gt)
        plt.plot(x_target, vel, 'x')
        plt.plot(x_target, vel_smooth, '--')
        plt.show()
    return cs(x_target), vel_smooth


def _compute_angular_velocity(r, time_delta: float):
    # assume the second last dimension is the time axis
    diff_quat_data = quat_identity_like(r)
    diff_quat_data[:-1, :] = quat_mul_norm(
        r[1:, :], quat_inverse(r[:-1, :])
    )
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    angular_velocity = diff_axis * diff_angle[..., None] / time_delta
    return angular_velocity


def get_ang_vel(rot, dt, target_dt, gt=None):
    rot = rot[..., [1, 2, 3, 0]]
    x = np.arange(rot.shape[0]) * dt
    x_target = np.arange(int(rot.shape[0] * dt / target_dt)) * target_dt
    rotations = Rotation.from_quat(rot)
    spline = RotationSpline(x, rotations)
    ang = _compute_angular_velocity(spline(x_target, 0).as_quat(), target_dt)  # [x,y,z,w]
    ang_smoothed = gaussian_filter1d(ang, sigma=2 * dt / target_dt, axis=0, mode="nearest")

    if gt is not None:
        plt.plot(x, gt, 'x')
        plt.plot(x_target, ang_smoothed, '-')
        plt.show()

    return ang_smoothed


def get_rot(rot, dt, target_dt):
    rot = rot[..., [1, 2, 3, 0]]
    nframe = rot.shape[0]
    x = np.arange(nframe) * dt
    x_target = np.arange(int(nframe * dt / target_dt)) * target_dt

    rotations = Rotation.from_quat(rot)
    spline = RotationSpline(x, rotations)

    return spline(x_target, 0).as_quat()[:, [3, 0, 1, 2]]


if __name__ == '__main__':
    for fps in ['30']:
        for fname in glob.glob('/PATH/TO/MOTION/*.json'):
            action = os.path.basename(fname).split('.')[0]
            with open(fname, 'r') as f:
                ase_motion = json.load(f)
            ase_poses = []
            pelvis_trans = []
            slice = [0, 1, 3, 4, 4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]
            for i in range(1, len(slice)):
                slice[i] += slice[i - 1]
            dt = ase_motion['Frames'][0][0]
            for frame in ase_motion['Frames']:
                _, root_pos, root_rot, chest_rot, neck_rot, right_hip_rot, right_knee_rot, right_ankle_rot, right_shoulder_rot, right_elbow_rot, left_hip_rot, left_knee_rot, left_ankle_rot, left_shoulder_rot, left_elbow_rot = \
                    [convert(frame[beg:end]) for beg, end in zip(slice[:-1], slice[1:])]
                ase_poses.append(np.stack(
                    [root_rot, chest_rot, neck_rot, right_shoulder_rot, right_elbow_rot, left_shoulder_rot,
                     left_elbow_rot, right_hip_rot, right_knee_rot, right_ankle_rot, left_hip_rot, left_knee_rot,
                     left_ankle_rot]))
                pelvis_trans.append(root_pos)

            ase_poses = np.array(ase_poses)
            pelvis_trans = np.array(pelvis_trans)
            ase_ang = np.zeros_like(ase_poses)[..., :-1]
            ase_vel = np.zeros_like(ase_poses)[..., :-1]
            target_dt = {
                'orig': dt,
                '16': 0.0625,
                '30': 0.0333
            }[fps]

            _qp_list = convert_to_qp(ase_poses, ase_vel, ase_ang, pelvis_trans)
            abs_poses = np.stack([qp.rot for qp in _qp_list], axis=0)
            ase_poses_interp = np.stack([get_rot(ase_poses[:, i, :], dt, target_dt) for i in range(ase_poses.shape[1])],
                                        axis=1)
            ase_ang_interp = np.stack(
                [get_ang_vel(abs_poses[:, i, :], dt, target_dt) for i in range(ase_poses.shape[1])], axis=1)

            pelvis_trans_interp, pelvis_trans_vel_interp = interpolate(pelvis_trans, dt, target_dt)
            ase_vel_interp = np.zeros_like(ase_ang_interp)
            ase_vel_interp[:, 0] = pelvis_trans_vel_interp

            qp_list = convert_to_qp(ase_poses_interp, ase_vel_interp, ase_ang_interp, pelvis_trans_interp)
            print(action, len(qp_list))
            demo_traj = convert_to_states(qp_list, target_dt)
            with open('../demo_humanoid/{}_fps_{}_20s.npy'.format(action, fps), 'wb') as f:
                np.save(f, demo_traj)