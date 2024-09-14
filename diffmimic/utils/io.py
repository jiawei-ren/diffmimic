import brax.v1 as brax
import jax.numpy as jnp
from brax.v1 import QP


def deserialize_qp(nparray) -> brax.QP:
    """
    Get QP from a trajectory numpy array
    """
    num_bodies = nparray.shape[-1] // 13    # pos (,3) rot (,4) vel (,3) ang (,3)
    batch_dims = nparray.shape[:-1]
    slices = [num_bodies * x for x in [0, 3, 7, 10, 13]]
    pos = jnp.reshape(nparray[..., slices[0]:slices[1]], batch_dims + (num_bodies, 3))
    rot = jnp.reshape(nparray[..., slices[1]:slices[2]], batch_dims + (num_bodies, 4))
    vel = jnp.reshape(nparray[..., slices[2]:slices[3]], batch_dims + (num_bodies, 3))
    ang = jnp.reshape(nparray[..., slices[3]:slices[4]], batch_dims + (num_bodies, 3))
    return QP(pos=pos, rot=rot, vel=vel, ang=ang)


def serialize_qp(qp) -> jnp.array:
    """
    Serialize QP to a trajectory numpy array
    """
    pos = qp.pos
    rot = qp.rot
    vel = qp.vel
    ang = qp.ang
    batch_dim = pos.shape[:-2]
    nparray = []
    nparray.append(pos.reshape(batch_dim + (-1,)))
    nparray.append(rot.reshape(batch_dim + (-1,)))
    nparray.append(vel.reshape(batch_dim + (-1,)))
    nparray.append(ang.reshape(batch_dim + (-1,)))
    return jnp.concatenate(nparray, axis=-1)
