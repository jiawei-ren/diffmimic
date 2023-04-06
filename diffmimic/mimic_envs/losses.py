from diffmimic.utils.rotation6d import quaternion_to_rotation_6d


def loss_l2_relpos(qp, ref_qp):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    relpos, ref_relpos = (pos - pos[0])[1:], (ref_pos - ref_pos[0])[1:]
    relpos_loss = (((relpos - ref_relpos) ** 2).sum(-1) ** 0.5).mean()
    relpos_loss = relpos_loss
    return relpos_loss


def loss_l2_pos(qp, ref_qp):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    pos_loss = (((pos - ref_pos) ** 2).sum(-1)**0.5).mean()
    return pos_loss


def mse_pos(qp, ref_qp):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    pos_loss = ((pos - ref_pos) ** 2).sum(-1).mean()
    return pos_loss


def mse_rot(qp, ref_qp):
    rot, ref_rot = quaternion_to_rotation_6d(qp.rot[:-1]), quaternion_to_rotation_6d(ref_qp.rot[:-1])
    rot_loss = ((rot - ref_rot) ** 2).sum(-1).mean()
    return rot_loss


def mse_vel(qp, ref_qp):
    vel, ref_vel = qp.vel[:-1], ref_qp.vel[:-1]
    vel_loss = ((vel - ref_vel) ** 2).sum(-1).mean()
    return vel_loss


def mse_ang(qp, ref_qp, reduce='mean'):
    ang, ref_ang = qp.ang[:-1], ref_qp.ang[:-1]
    ang_loss = ((ang - ref_ang) ** 2).sum(-1).mean()
    return ang_loss
