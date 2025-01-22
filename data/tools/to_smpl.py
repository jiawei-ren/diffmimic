import os
import numpy as np
from diffmimic.mimic_envs.system_configs import *
from brax.physics import bodies
from brax.physics.base import QP, vec_to_arr
from data.tools.rotation_utils.conversions import *

from quaternion import qinv_np, qmul_np

CFG_AMP = process_system_cfg(get_system_cfg('smpl'))

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

joint_body = np.array([
    (body.index[j.parent], body.index[j.child]) for j in joint
])
joint_off = np.array([(vec_to_arr(j.parent_offset),
                        vec_to_arr(j.child_offset)) for j in joint])


body_p = [int(joint_body[i][0]) for i,j in enumerate(joint_order)]
body_c = [int(joint_body[i][1]) for i,j in enumerate(joint_order)]
i_idx = [i for i,j in enumerate(joint_order)]
j_idx = [j+1 for i,j in enumerate(joint_order)]


def invert_process(qp_rot, qp_pos):
    smpl_rot = np.zeros_like(qp_rot)
    smpl_pos = np.zeros_like(qp_pos)
    for t in range(qp_rot.shape[-3]):
        # print(smpl_pos.shape)
        smpl_pos[..., t, 0, :] = qp_pos[..., t, 0, :]
        smpl_rot[..., t, 0, :] = qp_rot[..., t, 0, :]

        world_rot = qp_rot[..., t, body_c, :]
        parent_rot = qp_rot[..., t, body_p, :]
        local_rot = qmul_np(qinv_np(parent_rot), world_rot)

        smpl_rot[..., t, j_idx, :] = local_rot[..., i_idx, :]

    return smpl_rot, smpl_pos


def convert_to_smpl(demo_traj):
    demo_traj = np.array(demo_traj)
    batch_dims = demo_traj.shape[:-1]
    qp_rot = demo_traj[..., 57:57+18*4].reshape(batch_dims + (18, 4))
    qp_pos = demo_traj[..., :57].reshape(batch_dims + (19, 3))
    qp_pos = qp_pos[..., :1, :]
    smpl_rot, smpl_pos = invert_process(qp_rot, qp_pos)
    smpl_rot = quaternion_to_axis_angle(torch.from_numpy(smpl_rot))
    demo_traj_smpl = np.concatenate([smpl_pos, smpl_rot], axis=-2)
    return demo_traj_smpl

if __name__ == '__main__':
    in_file = '../demo_amass/75_09_stageii.npy'
    action = os.path.basename(in_file).split('.')[0]
    demo_traj = np.load(in_file)
    res = convert_to_smpl(demo_traj)
    with open('../demo_smpl/{}.npy'.format(action), 'wb') as f:
        np.save(f, res)
