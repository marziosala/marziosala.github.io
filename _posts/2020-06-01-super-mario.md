---
layout: splash
permalink: /super-mario/
title: "Solving Super Mario SNES using DQN"
header:
  overlay_image: /assets/images/super-mario/super-mario-splash.jpeg
excerpt: "Solving Super Mario SNES using DQN."
---

```python
from collections import deque, namedtuple
from pathlib import Path
import torch
import torch.nn as nn
import random
import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from tqdm.notebook import tnrange as trange
import pickle 
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
import gym
import numpy as np
import collections 
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
```


```python
class MaxAndSkipEnv(gym.Wrapper):
    
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym_super_mario_bros.make(env_name)
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return JoypadSpace(env, SIMPLE_MOVEMENT)
```


```python
class DQNSolver(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
```


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```


```python
env = make_env('SuperMarioBros-1-1-v0')
observation_space = env.observation_space.shape
action_space = env.action_space.n
```


```python
obs = env.reset()
for _ in range(100):
    obs, _, done, _ = env.step(random.randint(0, 1))
    if done: break
```


```python
fig, axes = plt.subplots(ncols=4, figsize=(12, 4))
for i in range(4):
    axes[i].imshow(obs[i], cmap=plt.get_cmap('gray'))
    axes[i].axis('off')
```


    
![png](/assets/images/super-mario/super-mario-1.png)
    



```python
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'done', 'reward'))
```


```python
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```


```python
class DQNAgent:

    def __init__(self, seed, input_dim, num_actions, learning_rate,
                 capacity, batch_size, gamma, sync_every, burnin, learn_every,
                 epsilon_start, epsilon_min, epsilon_decay,
                 save_every, logger):
        torch.manual_seed(seed)
        env.seed(seed + 1)
        env.action_space.seed(seed + 2)
        np.random.seed(seed + 3)
        random.seed(seed + 4)
        
        self.curr_step = 0
        self.num_actions = num_actions
        self.online_net = DQNSolver(input_dim, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

        self.target_net = DQNSolver(input_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        # Q_target parameters are frozen.
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.l1 = nn.SmoothL1Loss().to(device)
        self.gamma = gamma
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.sync_every = sync_every
        self.burnin = burnin
        self.learn_every = learn_every
        self.save_every = save_every
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.logger = logger
        
        self.tmp_dir = Path('./tmp/')

    def get_action(self, state, explore=True):
        # epsilon-greedy part, we select a random action
        if torch.rand(1).item() <= self.epsilon and explore:
            action = torch.randint(0, self.num_actions, (1,)).item()
        else:
            Q_row = self.online_net(state.to(device))
            action = torch.argmax(Q_row).unsqueeze(0).unsqueeze(0).item()
        if explore:
            self.curr_step += 1
        return action

    def remember(self, state, action, next_state, done, reward):
        # keep the memory on the CPU
        self.memory.push(state.to(device), action.to(device), next_state.to(device), done.to(device), reward.to(device))

    def learn(self):
        if self.curr_step < self.burnin:
            return np.nan
        
        if self.curr_step % self.learn_every != 0:
            return np.nan

        # if we don't have enough experience yet, we don't optimize and simply return
        if len(self.memory) < self.batch_size:
            return np.nan
        
        # sample for memory to create a batch of transitions
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        state = torch.cat(batch.state)
        next_state = torch.cat(batch.next_state)
        action = torch.cat(batch.action)
        done = torch.cat(batch.done)
        reward = torch.cat(batch.reward)

        with torch.no_grad():
            best_action = self.online_net(next_state).max(1)[1].view(-1, 1)
            next_state_value = self.target_net(next_state).gather(1, best_action).view(-1, 1)
        target = reward + self.gamma * torch.mul(next_state_value, 1 - done)
        #target = reward + torch.mul(
        #    (self.gamma * self.target_net(next_state).max(1).values.unsqueeze(1)), 1 - done)

        current = self.online_net(state).gather(1, action.unsqueeze(-1)).float()

        self.optimizer.zero_grad()
        loss = self.l1(current, target)
        loss.backward()
        #### FIXME TEST THIS
        #for param in self.online_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if self.curr_step % self.sync_every == 0:
            self.sync()

        if self.curr_step % self.save_every == 0:
            self.save(self.curr_step)

        logger.info(f'optimizing at step {self.curr_step}, loss: {loss.item():.2f}')

        return loss.item()
    
    def sync(self):
        logger.info(f"Synchronizing at step {self.curr_step}")
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, label):
        if episode is None:
            filenane = "net.pt"
            torch.save(self.online_net.state_dict(), filenane)
        else:
            filenane = self.tmp_dir / f"online-net-{label}.pt"
            torch.save(self.online_net.state_dict(), filenane)
            filenane = self.tmp_dir / f"target-net-{label}.pt"
            torch.save(self.target_net.state_dict(), filenane)
    
    def load(self):
        self.online_net.load_state_dict(torch.load("net.pt"))
        self.target_net.load_state_dict(self.online_net.state_dict())
```


```python
class MetricLogger:
    
    def __init__(self, msg_filename='mario.log', data_filename='mario.csv'):
        self.msg_filename = msg_filename
        with open(self.msg_filename, 'w') as f:
            f.write('Simulation starts\n')
        self.data_filename = data_filename
        with open(self.data_filename, 'w') as f:
            f.write('Episode,Length,Reward,Loss,X-Pos,Time\n')
    
    def append(self, episode, length, reward, loss, info):
        with open(self.data_filename, 'a') as f:
            f.write(f"{episode},{length},{reward},{loss},{info['x_pos']},{info['time']}\n")
    
    def info(self, message):
        with open(self.msg_filename, 'a') as f:
            f.write(message + '\n')
```


```python
input_dim = env.observation_space.shape
num_actions = env.action_space.n
print(f"Environment has a input dim {input_dim} and {num_actions} actions")
```

    Environment has a input dim (4, 84, 84) and 7 actions
    


```python
logger = MetricLogger()
```


```python
agent = DQNAgent(seed=42, input_dim=input_dim, num_actions=num_actions,
                 learning_rate=0.00025, capacity=20_000, batch_size=32, gamma=0.9,
                 sync_every=5_000, burnin=1, learn_every=1, save_every=10_000,
                 epsilon_start=1, epsilon_min=0.02, epsilon_decay=0.9999,
                 logger=logger)
test_every = 1_000
```


```python
episodes = 10000
episode_losses, episode_rewards, episode_lens = [], [], []
```


```python
for episode in (pbar := trange(1, episodes + 1)):
    state, done, episode_len, episode_reward, episode_loss = env.reset(), False, 0, 0.0, 0.0
    state = torch.tensor([state], dtype=torch.float)
    while not done:
        episode_len += 1
        try:
            env.render()
        except:
            pass
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        
        action = torch.tensor([action], dtype=torch.long)
        next_state = torch.tensor([next_state], dtype=torch.float)
        reward = torch.tensor([reward], dtype=torch.float).unsqueeze(0)
        done = torch.tensor([float(done)], dtype=torch.float).unsqueeze(0)
        agent.remember(state, action, next_state, done, reward)

        state = next_state
        episode_loss += agent.learn()
    
    logger.append(episode, episode_len, episode_reward, episode_loss, info)
    episode_losses.append(episode_loss / episode_len)
    episode_rewards.append(episode_reward)
    episode_lens.append(episode_len)
    avg_lens = np.array(episode_lens[-50:]).mean()
    avg_rewards = np.array(episode_rewards[-50:]).mean()
    pbar.set_description(f"{max(episode_lens)}/{avg_lens:.2f}/{avg_rewards:.2f}/{len(agent.memory)}/{agent.epsilon:.2f} ~ {agent.curr_step}")
```


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=10000.0), HTML(value='')))


    
    


```python
agent.save(None)
```


```python
agent.load()
agent.epsilon = 0.05
```


```python
env = make_env('SuperMarioBros-1-3-v0')
state = env.reset()
state = torch.tensor([state], dtype=torch.float)
frames = [env.render(mode='rgb_array').copy()]
done = False
total_reward = 0.0
while not done:
    action = agent.get_action(state)
    next_state, reward, done, info = env.step(action)
    next_state = torch.tensor([next_state], dtype=torch.float)
    total_reward += reward
    #print(info, reward)
    frames.append(env.render(mode='rgb_array').copy())
    state = next_state
    time.sleep(0.025)
    env.render()
print(f"Total reward: {total_reward}")
env.close()
```


```python
# np array with shape (frames, height, width, channels)
video = np.array(frames[:]) 

fig = plt.figure(figsize=(4, 4))
im = plt.imshow(video[0,:,:,:])
plt.axis('off')
plt.close() # this is required to not display the generated image

def init():
    im.set_data(video[0,:,:,:])

def animate(i):
    im.set_data(video[i,:,:,:])
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                               interval=100)
```


```python
anim.save('super-mario-video.gif', dpi=80, writer='imagemagick')
```

<img src='/assets/images/super-mario/super-mario-video.gif'/>


```python
filenane = "net.pt"
torch.save(agent.online_net.state_dict(), filenane)
```
