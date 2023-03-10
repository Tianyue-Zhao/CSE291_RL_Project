import gym
import numpy as np
import mani_skill2.envs
import torch
import pickle
from mani_skill2.utils.wrappers import RecordEpisode
from dm_env import specs
from dm_env import StepType
from replay_buffer import ReplayBufferStorage, make_replay_loader
from drqv2 import DrQV2Agent
from utils import eval_mode
from dmc import ExtendedTimeStep
from pathlib import Path
from collections import deque
from torch.utils.tensorboard import SummaryWriter

def log_metrics(metrics, steps):
    for (key, value) in metrics.items():
        sw.add_scalar(key, value, steps)
    with open(str(metric_dir) + '/metric_' + str(steps) + '.dict', 'wb') as output_file:
        pickle.dump(metrics, output_file)

env_id = "LiftCube-v1"
obs_mode = "rgbd"
control_mode = "pd_ee_delta_pos"
reward_mode = "dense"
max_env_steps = 200

num_train_frames = 300000
replay_buffer_frames = 500000
snapshot_every = 20000
load_from = ""

video_location = 'videos/'
replay_location = 'replays/'
snapshot_location = 'snapshots/'
metric_location = 'metrics/'


directory = Path.cwd()
video_dir = directory / video_location
replay_dir = directory / replay_location
snapshot_dir = directory / snapshot_location
metric_dir = directory / metric_location
video_dir.mkdir(exist_ok=True)
snapshot_dir.mkdir(exist_ok=True)
metric_dir.mkdir(exist_ok=True)
sw = SummaryWriter(metric_dir)

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
stddev_schedule = 'linear(1.0,0.1,10000)'
learning_rate = 1e-4
obs_shape = (9,128,128)
stack_frames = 3 # Stack the 3 most recent frames
assert(obs_shape[0] % stack_frames == 0)
replay_shape = (obs_shape[0] // stack_frames,
                obs_shape[1], obs_shape[2])
action_shape = (4,)
feature_dim = 50
hidden_dim = 1024
critic_target_tau = 0.01
num_expl_steps = 10000
update_every_steps = 2
stddev_clip = 0.3
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

agent = DrQV2Agent(obs_shape, action_shape, 'cuda', learning_rate,
    feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
    update_every_steps, stddev_schedule, stddev_clip, use_tb)

length_history = []
ep_length = 0
reward_history = []
ep_reward = 0
ep_num = 0

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

        print("Episode " + str(ep_num))

    with torch.no_grad(), eval_mode(agent):
        action = agent.act(np.concatenate(list(obs_stack), axis = 0), i, eval_mode=False)
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

#     if(total_steps % snapshot_every == 0):
#         agent.save_checkpoint(str(snapshot_dir / "model_{}.data".format(total_steps)))
#         print("Saved model at step {}".format(total_steps))
# agent.save_checkpoint(str(snapshot_dir / "model_final.data"))