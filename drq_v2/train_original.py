#!/usr/bin/python3
import torch
import os
import numpy as np
import cv2
import pickle
import faulthandler
from dm_env import specs
from dm_env import StepType
from replay_buffer import ReplayBufferStorage, make_replay_loader
from pybullet_env import Manipulation_Env
from utils import eval_mode
from drqv2 import DrQV2Agent
from pathlib import Path
from dmc import ExtendedTimeStep
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

faulthandler.enable()

# Configuration variables
num_train_frames = 1600000 # Taken from the medium difficulty rating
num_train_frames = 40000

eval_run = False # If this run is an evaluation run
eval_episodes = 5
record_every = 1 # Record a video every record_every episodes
save_every = 200 # Save an agent snapshot every save_every episodes
replay_buffer_frames = 1000000

load_from = "" # Option to load a saved checkpoint

# Agent configuration variables
stddev_schedule = 'linear(1.0,0.1,100000)'
learning_rate = 1e-4
obs_shape = (9,84,84)
stack_frames = 3 # Stack the 3 most recent frames
assert(obs_shape[0] % stack_frames == 0)
replay_shape = (obs_shape[0] // stack_frames,
                obs_shape[1], obs_shape[2])
action_shape = (10,)
feature_dim = 50
hidden_dim = 1024
critic_target_tau = 0.01
num_expl_steps = 4000
update_every_steps = 2
stddev_clip = 0.3
use_tb = True
metrics = None
sw = None


# Create recording directories
directory = Path.cwd()
replay_dir = directory / "replays"
record_dir = directory / "record"
save_dir = directory / "snapshots"
metric_dir = directory / "metrics"
record_dir.mkdir(exist_ok=True)
save_dir.mkdir(exist_ok=True)
metric_dir.mkdir(exist_ok=True)

# Create the tensorboard logger
sw = SummaryWriter(log_dir = str(metric_dir))

def log_line(line):
    print(line)
    log_file.write(line + '\n')

def evaluate():
    train_env.enable_global_view()
    train_env.global_view = True
    eval_dir = directory / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    for i in range(eval_episodes):
        obs = train_env.reset()
        record = cv2.VideoWriter(str(eval_dir) + '/' + str(i) + '.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'), 30, (train_env.global_pixel * 2, train_env.global_pixel))
        record.write(obs[:3,:,:].transpose(1,2,0))
        done = False
        ep_length = 0
        ep_reward = 0

        pixel_length_ratio = 350
        line_width = 3
        print("Evaluate episode " + str(i))

        action = np.asarray([0] * 9 + [-1])

        while(not done):
            with torch.no_grad(), eval_mode(agent):
                prev_action = action
                action = agent.act(obs, ep_length, eval_mode = True)
            obs, reward, done, info = train_env.step(action)
            region_image = obs[:3,:,:].transpose(1,2,0)
            region_image = cv2.resize(region_image, (train_env.global_pixel, train_env.global_pixel))
            ep_reward += reward
            ep_length += 1
            global_image = train_env.global_image.astype(np.uint8)
            # Now draw the rectangle to describe the view region selected by the agent
            # Somehow opencv rectangle doesn't work so I'm drawing it manually
            cur_fov = train_env.camera_current[2]
            cur_x = train_env.camera_current[0]
            cur_y = train_env.camera_current[1]
            view_pixels = train_env.global_pixel / 2 * 1.4 / cur_fov
            upper_left = (cur_x * pixel_length_ratio - view_pixels + train_env.global_pixel / 2, -cur_y * pixel_length_ratio + view_pixels + train_env.global_pixel / 2)
            lower_right = (cur_x * pixel_length_ratio + view_pixels + train_env.global_pixel / 2, -cur_y * pixel_length_ratio - view_pixels + train_env.global_pixel / 2)
            upper_left = (int(upper_left[0]), int(upper_left[1]))
            lower_right = (int(lower_right[0]), int(lower_right[1]))
            global_image = cv2.rectangle(global_image, upper_left, lower_right, (255,0,0), 4)
            record.write(np.concatenate([region_image, global_image], axis=1))
        print("Reward " + str(ep_reward) + "  Length " + str(ep_length))
        record.release()

def log_metrics(metrics, steps):
    for (key, value) in metrics.items():
        sw.add_scalar(key, value, steps)
    with open(str(metric_dir) + '/metric_' + str(steps) + '.dict', 'wb') as output_file:
        pickle.dump(metrics, output_file)

log_file = open('training_log', 'w')

# Initialize environment
train_env = Manipulation_Env()

obs = train_env.reset()
done = False

length_history = []
ep_length = 0
reward_history = []
ep_reward = 0
ep_num = 0

data_specs = (specs.Array(replay_shape, np.uint8, 'observation'),
              specs.Array(action_shape, np.float32, 'action'),
              specs.Array((1,), np.float32, 'reward'),
              specs.Array((1,), np.float32, 'discount'))
replay_storage = ReplayBufferStorage(data_specs, replay_dir)
replay_loader = make_replay_loader(
    replay_dir, replay_buffer_frames, 256, 0, False,
     3, 0.99, stack_frames
)

agent = DrQV2Agent(obs_shape, action_shape, 'cuda', learning_rate,
    feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
    update_every_steps, stddev_schedule, stddev_clip, use_tb)

if(not load_from == ""):
    log_line("Loading saved checkpoint from " + str(load_from))
    agent = torch.load(load_from)

if(eval_run):
    assert(not load_from == "")  # Assert that a save snapshot is specified
    evaluate()
    quit()

log_line("Episode 0")

for i in range(num_train_frames):
    if(done):  # An episode is done
        log_line("Reward " + str(ep_reward))
        reward_history.append(ep_reward)
        length_history.append(ep_length)
        ep_length = 0
        ep_reward = 0
        obs = train_env.reset()

        if((ep_num % record_every == 0) and (ep_num > 0)):
            record.release()

        step = ExtendedTimeStep(
            step_type = StepType.FIRST,
            reward = 0,
            discount = 1,
            observation = obs[obs_shape[0] - replay_shape[0]:],
            action = np.zeros(action_shape, dtype=np.float32)
        )
        ep_num += 1

        replay_storage.add(step)

        log_line("Episode " + str(ep_num))
        if(ep_num % record_every == 0):
            record = cv2.VideoWriter(str(record_dir) + '/' + str(ep_num) + '.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84))
            record.write(obs[:3,:,:].transpose(1,2,0))

        if(ep_num % save_every == 0):
            agent_file = save_dir / (str(ep_num)+'_save')
            with agent_file.open('wb') as f:
                torch.save(agent, agent_file)

    with torch.no_grad(), eval_mode(agent):
        action = agent.act(obs, i, eval_mode=False)
    if(i > num_expl_steps):
        metrics = agent.update(iter(replay_loader), i)
    if(metrics is not None and sw is not None):
        log_metrics(metrics, i)
        metrics = None
    obs, reward, done, info = train_env.step(action)
    ep_reward += reward

    if(not done):
        step_status = StepType.MID
    else:
        step_status = StepType.LAST
    step = ExtendedTimeStep(
        step_type = step_status,
        reward = reward,
        discount = 1,    # Only pass the current frame instead of the stacked frames
        observation = obs[obs_shape[0] - replay_shape[0]:],
        action = action
    )
    replay_storage.add(step)
    ep_length += 1
    if((ep_num % record_every == 0) and (ep_num > 0)):
        record.write(obs[:3,:,:].transpose(1,2,0))

# Pickle reward history for possible analysis
with open(str(metric_dir) + '/' + 'reward_history_object', 'wb') as reward_file:
    pickle.dump(reward_history, reward_file, protocol = pickle.DEFAULT_PROTOCOL)

# Plot reward vs episode graph
fig, ax = plt.subplots(1,1)
ax.set_ylim([min(reward_history) - 2, 10])
ax.scatter([i+1 for i in range(len(reward_history))], reward_history, s=1)
plt.title("Total reward each episode")
plt.savefig(str(metric_dir) + '/' + 'reward_history.png')

log_file.close()