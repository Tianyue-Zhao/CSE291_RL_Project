import gym
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt
from mani_skill2.utils.wrappers import RecordEpisode
from sac import SAC
from replay_memory import ReplayMemory
from pathlib import Path

# Evaluation script for manipulation environments

# Environment Control
env_id = "LiftCube-v1"
obs_mode = "state"
control_mode = "pd_ee_delta_pos"
reward_mode = "dense"
max_env_steps = 200

# File parameters
video_location = 'eval_videos/'
vis_location = 'eval_vis/'
load_from = 'snapshots/model_100000.data'

# Run parameters
episodes = 10

# Create paths
directory = Path.cwd()
video_dir = directory / video_location
vis_dir = directory / vis_location
video_dir.mkdir(exist_ok=True)
vis_dir.mkdir(exist_ok=True)

# Make the environment
env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode)
env = RecordEpisode(
    env,
    str(video_dir),
    render_mode = 'cameras',
    info_on_video = True
)

# Load the agent
lr = 0.0003
alpha = 0.2
eval_alpha = 0.01
args = {
    'cuda': True,
    'automatic_entropy_tuning': True,
    'alpha': alpha,
    'lr': lr,
    'gamma': 0.9,
    'tau': 0.008,
    'policy_type': 'Gaussian',
    'target_update_interval': 1, # Don't understand why this is 1 by default
    'hidden_size': 256 # Chose a relative small hidden size for small environment
}
# The SAC algorithm file expects the arguments as attributes
class Arg_Attribute:
    def __init__(self, args):
        for key, value in args.items():
            setattr(self, key, value)
args = Arg_Attribute(args)
agent = SAC(env.observation_space.shape[0], env.action_space, args)
assert(not load_from == "")
agent.load_checkpoint(load_from)

# Initialize records
for i in range(episodes):
    obs = env.reset()
    episode_reward = []
    episode_q1 = []
    episode_q2 = []
    for j in range(max_env_steps):
        action = agent.select_action(obs, evaluate=True)
        q1, q2 = agent.query_q(obs, action, eval_alpha)
        episode_q1.append(q1)
        episode_q2.append(q2)
        obs, reward, done, info = env.step(action)
        episode_reward.append(reward)
        if done:
            break
    # Graph the actual and predicted rewards
    reward_acc = [0] * len(episode_reward)
    reward_cur = 0
    for k in range(len(episode_reward)):
        reward_cur *= args.gamma
        reward_cur += episode_reward[-k-1]
        reward_acc[-k-1] = reward_cur
    fig, ax = plt.subplots()
    ax.plot(reward_acc, label="Actual")
    ax.plot(episode_q1, label="Q1")
    ax.plot(episode_q2, label="Q2")
    plt.savefig(str(vis_dir / "episode_{}.png".format(i)))
    print("Episode: {}, Reward: {}".format(i, sum(episode_reward)))