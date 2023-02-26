import gym
import numpy as np
import os
import mani_skill2.envs
import matplotlib.pyplot as plt
from mani_skill2.utils.wrappers import RecordEpisode
from sac import SAC
from replay_memory import ReplayMemory
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Training script for easy manipulation environments
# with state observation and dense rewards
# Uses SAC from existing repository on Github

# Environment Control
# env_id = "LiftCube-v1"
env_id = "StackCube-v1"
obs_mode = "state"
control_mode = "pd_ee_delta_pos"
reward_mode = "dense"
max_env_steps = 200

# File parameters
video_location = 'videos/'
snapshot_location = 'snapshots/'
metric_location = 'metrics/'

# Training parameters
steps_to_train = 500000
initial_exploration_steps = 10000 # Steps to randomly sample actions
lr = 0.0003
alpha = 0.2
batch_size = 256
seed = 21283489
replay_size = 2000000
snapshot_every = 20000
load_from = ""

# Create paths
directory = Path.cwd()
video_dir = directory / video_location
snapshot_dir = directory / snapshot_location
metric_dir = directory / metric_location
video_dir.mkdir(exist_ok=True)
snapshot_dir.mkdir(exist_ok=True)
metric_dir.mkdir(exist_ok=True)
sw = SummaryWriter(metric_dir)

# Make the environment
env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode)
env = RecordEpisode(
    env,
    str(video_dir),
    render_mode = 'cameras',
    info_on_video = True
)
obs = env.reset()
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
memory = ReplayMemory(replay_size, seed)
total_steps = 0

if(not load_from == ''):
    agent.load_checkpoint(str(snapshot_dir / load_from))
    print("Agent loaded checkpoint {}".format(load_from))
    total_steps = int(load_from.split('_')[-1].split('.')[0])

# Fill replay buffer with random actions
num_episodes = 0
episode_steps = 0
for i in range(initial_exploration_steps):
    if(load_from == ''):
        action = env.action_space.sample()
    else:
        action = agent.select_action(obs)
    next_obs, reward, done, _ = env.step(action)
    episode_steps += 1
    mask = 1 if episode_steps == max_env_steps else float(not done)
    memory.push(obs, action, reward, next_obs, mask)
    obs = next_obs
    if(done):
        num_episodes += 1
        episode_steps = 0
        obs = env.reset()
        print("Collected initial episode {}".format(num_episodes))

# Run the main training loop
episode_steps = 0
episode_reward = 0
# num_episodes = 0
obs = env.reset()
for i in range(initial_exploration_steps, steps_to_train):
    total_steps += 1
    action = agent.select_action(obs)
    metrics = agent.update_parameters(memory, batch_size, total_steps) 
    for (key, value) in metrics.items():
        sw.add_scalar(key, value, total_steps)

    next_state, reward, done, _ = env.step(action) # Step
    sw.add_scalar("Reward", reward, total_steps)
    episode_steps += 1
    episode_reward += reward
    mask = 1 if episode_steps == max_env_steps else float(not done)
    memory.push(obs, action, reward, next_state, mask) # Append transition to memory
    obs = next_state
    if(done):
        episode_steps = 0
        obs = env.reset()
        num_episodes += 1
        print("Episode {} finished with reward {}".format(num_episodes, episode_reward))
        sw.add_scalar("Episode Reward", episode_reward, i)
        episode_reward = 0
    
    if(total_steps % snapshot_every == 0):
        agent.save_checkpoint(str(snapshot_dir / "model_{}.data".format(total_steps)))
        print("Saved model at step {}".format(total_steps))
agent.save_checkpoint(str(snapshot_dir / "model_final.data"))