import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity_rma.config.quadruped.unitree_go2.pos.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
from isaaclab_rl.rsl_rl.new_modules.actor_critic_o1 import ActorCritic_o1

env_cfg = Go2FlatEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env = ManagerBasedRLEnv(cfg=env_cfg)

model = ActorCritic_o1(
    num_actor_obs=236, 
    num_critic_obs=236,
    num_actions=12,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
)

load_state = torch.load('logs/rsl_rl/go2_velocity_rma_flat/2025-05-13_14-39-38_teacher/model_30500.pt', weights_only=True)['model_state_dict']
model.load_state_dict(load_state)
model.eval()
pi = model.act_inference

# simulate physics
obs, _ = env.reset()
count = 0
while simulation_app.is_running():
    with torch.inference_mode():
        # action = torch.zeros_like(env.action_manager.action)
        action = pi(obs['policy'].to('cpu'))
        obs, rew, terminated, truncated, info = env.step(action)
        # print(rew)
        count += 1

env.close()
simulation_app.close()
