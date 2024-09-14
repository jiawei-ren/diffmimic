import functools
import numpy as np
import jax.numpy as jnp
from absl import flags, app
import yaml
import brax.v1 as brax
from brax.v1 import envs
from brax.v1.io import metrics
from brax.training.agents.apg import networks as apg_networks
from diffmimic.utils import AttrDict
from diffmimic.mimic_envs import register_mimic_env
import diffmimic.brax_lib.agent_diffmimic as dmm

register_mimic_env()

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'configs/AMP/backflip.yaml', help='Experiment configuration.')


def main(argv):
    with open(FLAGS.config, 'r') as f:
        args = AttrDict(yaml.safe_load(f))

    logdir = "logs/exp"
    for k, v in args.items():
        if k == 'ref':
            logdir += f"_{v.split('/')[-1].split('.')[0]}"
        else:
            logdir += f"_{v}"

    demo_traj = jnp.array(np.load(args.ref))
    demo_len = demo_traj.shape[0]
    args.ep_len = min(args.ep_len, demo_len)
    args.cycle_len = min(args.get('cycle_len', demo_len), demo_len)
    args.ep_len_eval = min(args.get('ep_len_eval', demo_len), demo_len)

    train_env = envs.get_environment(
        env_name="humanoid_mimic_train",
        system_config=args.system_config,
        reference_traj=demo_traj,
        obs_type=args.get('obs_type', 'timestamp'),
        cyc_len=args.cycle_len,
        total_length=args.ep_len_eval,
        rollout_length=args.ep_len,
        early_termination=args.get('early_termination', False),
        demo_replay_mode=args.demo_replay_mode,
        err_threshold=args.threshold,
        replay_rate=args.get('replay_rate', 0.05),
        reward_scaling=args.get('reward_scaling', 1.),
        rot_weight=args.rot_weight,
        vel_weight=args.vel_weight,
        ang_weight=args.ang_weight
    )

    eval_env = envs.get_environment(
        env_name="humanoid_mimic",
        system_config=args.system_config,
        reference_traj=demo_traj,
        obs_type=args.get('obs_type', 'timestamp'),
        cyc_len=args.cycle_len,
        rot_weight=args.rot_weight,
        vel_weight=args.vel_weight,
        ang_weight=args.ang_weight
    )

    with metrics.Writer(logdir) as writer:
        make_inference_fn, params, _ = dmm.train(
            seed=args.seed,
            environment=train_env,
            eval_environment=eval_env,
            episode_length=args.ep_len-1,
            eval_episode_length=args.ep_len_eval-1,
            num_envs=args.num_envs,
            num_eval_envs=args.num_eval_envs,
            learning_rate=args.lr,
            num_evals=args.max_it+1,
            max_gradient_norm=args.max_grad_norm,
            network_factory=functools.partial(apg_networks.make_apg_networks, hidden_layer_sizes=(512, 256)),
            normalize_observations=args.normalize_observations,
            save_dir=logdir,
            progress_fn=writer.write_scalars,
            use_linear_scheduler=args.use_lr_scheduler,
            truncation_length=args.get('truncation_length', None),
        )


if __name__ == '__main__':
    app.run(main)
