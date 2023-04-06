from brax import jumpy as jp


def quat_imaginary(x):
    """
    imaginary components of the quaternion
    """
    return x[..., :3]

def quat_rotate(rot, vec):
    """
    Rotate a 3D vector with the 3D rotation
    """
    other_q = jp.concatenate([vec, jp.zeros_like(vec[..., :1])], axis=-1)
    return quat_imaginary(quat_mul(quat_mul(rot, other_q), quat_conjugate(rot)))

def quat_mul(a, b):
    """
    quaternion multiplication
    """
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return jp.stack([x, y, z, w], axis=-1)

def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0)
    q = (1 - 2 * z) * q
    return q

def quat_abs(x):
    """
    quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    x = jp.safe_norm(x, axis=-1)
    return x

def quat_unit(x):
    """
    normalized quaternion with norm of 1
    """
    norm = quat_abs(x)[..., None]
    return x / norm

def quat_conjugate(x):
    """
    quaternion with its imaginary part negated
    """
    return jp.concatenate([-x[..., :3], x[..., 3:]], axis=-1)

def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q

def quat_inverse(x):
    """
    The inverse of the rotation
    """
    return quat_conjugate(x)

def quat_mul_norm(x, y):
    """
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_normalize(quat_mul(x, y))

def quat_angle_axis(x):
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    s = 2 * (x[..., 3] ** 2) - 1
    angle = jp.arccos(jp.clip(s, a_min=jp.ones_like(s)*-1, a_max=jp.ones_like(s))) # just to be safe
    axis = x[..., :3]
    norm = jp.safe_norm(axis, axis=-1)[..., None]
    axis /= jp.clip(norm, a_min=jp.ones_like(norm)*1e-9, a_max=jp.ones_like(norm)*1e9)
    return angle, axis


def quat_identity(shape):
    """
    Construct 3D identity rotation given shape
    """
    w = jp.ones(shape + [1])
    xyz = jp.zeros(shape + [3])
    q = jp.concatenate([xyz, w], axis=-1)
    return quat_normalize(q)


def quat_identity_like(x):
    """
    Construct identity 3D rotation with the same shape
    """
    return quat_identity(list(x.shape[:-1]))



def quat_diff_theta(q0, q1):
    return quat_angle_axis(quat_mul_norm(q0, quat_inverse(q1)))[0]