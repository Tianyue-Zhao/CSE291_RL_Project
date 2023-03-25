import gym
import numpy as np
import mani_skill2.envs
import torch
import pickle
import datetime
import json
import random
import faulthandler
from mani_skill2.utils.wrappers import RecordEpisode
from dm_env import specs
from dm_env import StepType
from replay_buffer import ReplayBufferStorage, make_replay_loader
from coit_chair import Agent
from utils import eval_mode
from dmc import ExtendedTimeStep
from pathlib import Path
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import sys

def log_metrics(metrics, steps):
    for (key, value) in metrics.items():
        sw.add_scalar(key, value, steps)
    with open(str(metric_dir) + '/metric_' + str(steps) + '.dict', 'wb') as output_file:
        pickle.dump(metrics, output_file)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# take a input from the command line
seed = sys.argv[1]
print("Training seed: {}".format(seed))
seed = int(seed)
set_seed_everywhere(seed)
task = "pushchair-v1"
# task = "pushchair-v2"

env_id = "PushChair-v1"
# env_id = "PushChair-v2"
obs_mode = "rgbd"
control_mode = "base_pd_joint_vel_arm_pd_joint_vel"
reward_mode = "dense"
max_env_steps = 200

num_train_frames = 500000
replay_buffer_frames = 2000000
snapshot_every = 20000
num_expl_steps = 10000
load_from = ""

training_location = 'training/'
video_location = 'videos/'
replay_location = 'replays/'
snapshot_location = 'snapshots/'
metric_location = 'metrics/'
single_run_location = task + '/' + str(seed) + '/'

# Get a string of the current time
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

faulthandler.enable()

directory = Path.cwd()
directory = directory / training_location
directory = directory / single_run_location
# directory = directory / current_time
directory.mkdir(parents=True, exist_ok=True)
video_dir = directory / video_location
replay_dir = directory / replay_location
snapshot_dir = directory / snapshot_location
metric_dir = directory / metric_location
video_dir.mkdir(exist_ok=True)
snapshot_dir.mkdir(exist_ok=True)
metric_dir.mkdir(exist_ok=True)
sw = SummaryWriter(metric_dir)
reward_record = directory / 'reward_history.pickle'

# Save a json file about the training information
training_info = {
    'env_id': env_id,
    'obs_mode': obs_mode,
    'control_mode': control_mode,
    'reward_mode': reward_mode,
    'max_env_steps': max_env_steps,
    'num_train_frames': num_train_frames,
    'replay_buffer_frames': replay_buffer_frames,
    'snapshot_every': snapshot_every,
    'num_expl_steps': num_expl_steps,
    'load_from': load_from,
    'training_seed': seed
}
info_file = str(directory / 'training_info.json')
with open(info_file, 'w') as outfile:
    json.dump(training_info, outfile)

env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode)
env = RecordEpisode(
    env,
    str(video_dir),
    render_mode = 'cameras',
    info_on_video = True
)
obs = env.reset()['image']['base_camera']['rgb'].transpose((2, 0, 1))
obs_stack = deque(maxlen=3)
obs_stack.append(obs)
obs_stack.append(obs)
obs_stack.append(obs)

# Agent configuration variables
stddev_schedule = 'linear(1.0,0.1,100000)'
learning_rate = 1e-4
augmented_learning_rating = 2e-6
obs_shape = (9,128,128)
stack_frames = 3 # Stack the 3 most recent frames
assert(obs_shape[0] % stack_frames == 0)
replay_shape = (obs_shape[0] // stack_frames,
                obs_shape[1], obs_shape[2])
action_shape = (20,)
feature_dim = 50
hidden_dim = 1024 #256 # Chose a relative small hidden size for small environment
critic_target_tau = 0.01
update_every_steps = 2
stddev_clip = 0.3
encoder_target_tau= 0.0001
alpha= 0.0
lam= 0.0
use_tb = True
metrics = None

data_specs = (specs.Array(replay_shape, np.uint8, 'observation'),
              specs.Array(action_shape, np.float32, 'action'),
              specs.Array((1,), np.float32, 'reward'),
              specs.Array((1,), np.float32, 'discount'))
replay_storage = ReplayBufferStorage(data_specs, replay_dir)
replay_loader = make_replay_loader(
    replay_dir, replay_buffer_frames, 256, 0, False,
     3, 0.99, stack_frames
)

agent = Agent(obs_shape, action_shape, 'cuda', learning_rate,
              augmented_learning_rating, feature_dim, hidden_dim, 
              critic_target_tau, num_expl_steps, update_every_steps, 
              stddev_schedule, stddev_clip, use_tb, 
              encoder_target_tau, alpha, lam)
prev_steps = 0  # An indication of how many steps the agent has already been trained for

if(not load_from == ''):
    agent.load(str(snapshot_dir / load_from))
    prev_steps = int(load_from.split('_')[-1].split('.')[0])

length_history = []
ep_length = 0
reward_history = []
ep_reward = 0
ep_num = 0
csv_location = str(directory / 'reward_and_length.csv')
record_csv = open(csv_location, 'w')
record_csv.write('Episode num, Episode reward\n')

# total_steps = 0
# if(not load_from == ''):
#     agent.load_checkpoint(str(snapshot_dir / load_from))
#     print("Agent loaded checkpoint {}".format(load_from))
#     total_steps = int(load_from.split('_')[-1].split('.')[0])

print("Episode 0")
for i in range(num_train_frames):
    # total_steps += 1
    if(ep_length == 200):  # An episode is done
        print("Reward " + str(ep_reward))
        sw.add_scalar('reward', ep_reward, ep_num)
        reward_history.append(ep_reward)
        length_history.append(ep_length)
        record_csv.write(str(ep_num) + ',' + str(ep_reward) + '\n')
        ep_length = 0
        ep_reward = 0
        obs = env.reset()['image']['base_camera']['rgb'].transpose((2, 0, 1))
        obs_stack.append(obs)
        obs_stack.append(obs)
        obs_stack.append(obs)

        step = ExtendedTimeStep(
            step_type = StepType.FIRST,
            reward = 0,
            discount = 1,
            observation = obs,
            action = np.zeros(action_shape, dtype=np.float32)
        )
        ep_num += 1
        replay_storage.add(step)

        # Write the reward history to a file
        with open(reward_record, 'wb') as f:
            pickle.dump(reward_history, f)

        print("Episode " + str(ep_num))

    with torch.no_grad(), eval_mode(agent):
        action = agent.act(np.concatenate(
            list(obs_stack), axis=0), i, eval_mode=False)
    if(i > num_expl_steps):
        metrics = agent.update(iter(replay_loader), i)
    if(metrics is not None and sw is not None):
        log_metrics(metrics, i)
        metrics = None
    obs, reward, done, info = env.step(action)
    obs = obs['image']['base_camera']['rgb'].transpose((2, 0, 1))
    obs_stack.append(obs)
    ep_reward += reward
    ep_length += 1
    if(ep_length < 200):
        step_status = StepType.MID
    else:
        step_status = StepType.LAST
    step = ExtendedTimeStep(
        step_type = step_status,
        reward = reward,
        discount = 1,    # Only pass the current frame instead of the stacked frames
        observation = obs,
        action = action
    )
    replay_storage.add(step)
    if(i % snapshot_every == 0):
        agent.save(str(snapshot_dir / ('snapshot_' + str(i + prev_steps) + '.data')))
agent.save(str(snapshot_dir / ('snapshot_' + str(i) + '.data')))
record_csv.close()