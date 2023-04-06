import numpy as np
from diffmimic.mimic_envs.system_configs import get_system_cfg
from brax import jumpy as jp
from brax import math
from brax.physics import bodies
from brax.physics.base import QP, vec_to_arr
from data.tools.rotation_utils.conversions import *
import glob, os

from diffmimic.mimic_envs import register_mimic_env
register_mimic_env()

CFG_SWORD = get_system_cfg('sword')

def convert_to_qp(ase_poses, ase_vel, ase_ang, pelvis_trans):
    ase_poses = ase_poses[..., [3, 0, 1, 2]]   # important, [x, y, z, w] -> [w, x, y, z]
    qp_list = []
    for i in range(ase_poses.shape[0]):
        qp = QP.zero(shape=(len(CFG_SWORD.bodies),))
        body = bodies.Body(CFG_SWORD)
        # set any default qps from the config
        joint_idxs = []
        j_idx = 0
        for j in CFG_SWORD.joints:
          beg = joint_idxs[-1][1][1] if joint_idxs else 0
          dof = len(j.angle_limit)
          joint_idxs.append((j, (beg, beg + dof), j_idx))
          j_idx += 1
        lineage = {j.child: j.parent for j in CFG_SWORD.joints}
        depth = {}
        for child, parent in lineage.items():
          depth[child] = 1
          while parent in lineage:
            parent = lineage[parent]
            depth[child] += 1
        joint_idxs = sorted(joint_idxs, key=lambda x: depth.get(x[0].parent, 0))
        joint = [j for j, _, _ in joint_idxs]
        joint_order =[i for _, _, i in joint_idxs]

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
            qp = qp.replace(pos=pos, rot =rot, vel=vel, ang=ang)
            return qp

        qp = init(qp)
        amp_rot = local_rot[joint_order]
        world_vel = world_vel[joint_order]
        world_ang = world_ang[joint_order]

        num_joint_dof = sum(len(j.angle_limit) for j in CFG_SWORD.joints)
        num_joints = len(CFG_SWORD.joints)
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
        qp_list.append(qp)
    return qp_list

def convert_to_states(qp_list):
    demo_traj = []
    for i in range(len(qp_list)):
        qp = qp_list[i]
        demo_traj.append(
            np.concatenate([qp.pos.reshape(-1), qp.rot.reshape(-1), qp.vel.reshape(-1), qp.ang.reshape(-1)], axis=-1))
    demo_traj = np.stack(demo_traj, axis=0)
    return demo_traj

if __name__ == '__main__':
    for fname in glob.glob('/PATH/TO/MOTION/*.npy'):
        action = os.path.basename(fname).split('.')[0]
        ase_motion = np.load(fname, allow_pickle=True).item()
        n_frame = ase_motion['rotation']['arr'].shape[0]
        print(action, n_frame, ase_motion['fps'])
        body_idx = [k for k,v in enumerate(ase_motion['skeleton_tree']['node_names']) if v not in ['sword', 'shield', 'left_hand']]
        ase_poses = ase_motion['rotation']['arr'][:, body_idx, :]
        ase_vel = ase_motion['global_velocity']['arr'][:, body_idx, :]
        ase_ang = ase_motion['global_angular_velocity']['arr'][:, body_idx, :]
        pelvis_trans = ase_motion['root_translation']['arr']
        qp_list = convert_to_qp(ase_poses, ase_vel, ase_ang, pelvis_trans)
        demo_traj = convert_to_states(qp_list)
        with open('../demo_swordshield/{}.npy'.format(action), 'wb') as f:
            np.save(f, demo_traj)