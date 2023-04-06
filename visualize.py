import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
from diffmimic.utils.io import deserialize_qp, serialize_qp
import streamlit as st
from diffmimic.mimic_envs import register_mimic_env
register_mimic_env()

st.title('DiffMimic - Visualization Tool')

uploaded_file = st.file_uploader("Choose a file")
rollout_traj = None
action_traj = None

if uploaded_file is not None:
    rollout_traj = np.load(uploaded_file)

if rollout_traj is not None:
    if len(rollout_traj.shape) == 3:
        seed = st.slider('Random seed', 0, rollout_traj.shape[1]-1, 0)
        rollout_traj = rollout_traj[:, seed]
    if rollout_traj.shape[-1] == 182:
        system_cfg = 'humanoid'
    elif rollout_traj.shape[-1] == 247:
        system_cfg = 'smpl'
    else:
        system_cfg = 'swordshield'

    init_qp = deserialize_qp(rollout_traj[0])
    rollout_qp = [deserialize_qp(rollout_traj[i]) for i in range(rollout_traj.shape[0])]
    rollout_traj = serialize_qp(deserialize_qp(rollout_traj))

    env = envs.get_environment(
        env_name="humanoid_mimic",
        system_config=system_cfg,
        reference_traj=rollout_traj,
    )
    components.html(html.render(env.sys, rollout_qp), height=500)
