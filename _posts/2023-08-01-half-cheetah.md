---
layout: splash
permalink: /half-cheetah/
title: "A2C for the Half Cheetah Environment"
header:
  overlay_image: /assets/images/half-cheetah/half-cheetah-splash.png
excerpt: "Solving the Half Cheetah environment with the Advantage-Actor-Critic method."
---

```python
import time
from pathlib import Path
from datetime import datetime
import gymnasium as gym
import json
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
```


```python
from pyvirtualdisplay import Display
display = Display(visible=0, size=(800, 600))
display.start()
```




    <pyvirtualdisplay.display.Display at 0x7f8f6002e080>




```python
class Args:
    pass

args = Args()
args.env_id = "HalfCheetah-v4"
args.total_timesteps = 10_000_000
args.num_envs = 16
args.num_steps = 5
args.learning_rate = 5e-4
args.actor_layers = [64, 64]
args.critic_layers  = [64, 64]
args.gamma = 0.99
args.gae = 1.0
args.value_coef = 0.5
args.entropy_coef = 0.01
args.clip_grad_norm = 0.5
args.seed = 0

args.batch_size = int(args.num_envs * args.num_steps)
args.num_updates = int(args.total_timesteps // args.batch_size)
```


```python
def make_env(env_id, capture_video=False, run_dir="."):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos",
                episode_trigger=lambda x: x,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda state: np.clip(state, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk
```


```python
def compute_advantages(rewards, flags, values, last_value, args):
    advantages = torch.zeros((args.num_steps, args.num_envs))
    adv = torch.zeros(args.num_envs)

    for i in reversed(range(args.num_steps)):
        returns = rewards[i] + args.gamma * flags[i] * last_value
        delta = returns - values[i]

        adv = delta + args.gamma * args.gae * flags[i] * adv
        advantages[i] = adv

        last_value = values[i]

    return advantages
```


```python
class RolloutBuffer:
    def __init__(self, num_steps, num_envs, observation_shape, action_shape):
        self.states = np.zeros((num_steps, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.flags = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)

        self.step = 0
        self.num_steps = num_steps

    def push(self, state, action, reward, flag, value):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.flags[self.step] = flag
        self.values[self.step] = value

        self.step = (self.step + 1) % self.num_steps

    def get(self):
        return (
            torch.from_numpy(self.states),
            torch.from_numpy(self.actions),
            torch.from_numpy(self.rewards),
            torch.from_numpy(self.flags),
            torch.from_numpy(self.values),
        )
```


```python
class ActorCriticNet(nn.Module):
    def __init__(self, observation_shape, action_dim, actor_layers, critic_layers):
        super().__init__()

        self.actor_net = self._build_net(observation_shape, actor_layers)
        self.critic_net = self._build_net(observation_shape, critic_layers)

        self.actor_net.append(self._build_linear(actor_layers[-1], action_dim, std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic_net.append(self._build_linear(critic_layers[-1], 1, std=1.0))

    def _build_linear(self, in_size, out_size, apply_init=True, std=np.sqrt(2), bias_const=0.0):
        layer = nn.Linear(in_size, out_size)

        if apply_init:
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)

        return layer

    def _build_net(self, observation_shape, hidden_layers):
        layers = nn.Sequential()
        in_size = np.prod(observation_shape)

        for out_size in hidden_layers:
            layers.append(self._build_linear(in_size, out_size))
            layers.append(nn.Tanh())
            in_size = out_size

        return layers

    def forward(self, state):
        mean = self.actor_net(state)
        std = self.actor_logstd.expand_as(mean).exp()
        distribution = Normal(mean, std)

        action = distribution.sample()

        value = self.critic_net(state).squeeze(-1)

        return action, value

    def evaluate(self, states, actions):
        mean = self.actor_net(states)
        std = self.actor_logstd.expand_as(mean).exp()
        distribution = Normal(mean, std)

        log_probs = distribution.log_prob(actions).sum(-1)
        entropy = distribution.entropy().sum(-1)

        values = self.critic_net(states).squeeze(-1)

        return log_probs, values, entropy

    def critic(self, state):
        return self.critic_net(state).squeeze(-1)
```


```python
def train(args, run_name, run_dir):
    # Create tensorboard writer and save hyperparameters
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create vectorized environment(s)
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # Metadata about the environment
    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    action_dim = np.prod(action_shape)

    # Set seed for reproducibility
    if args.seed:
        torch.manual_seed(args.seed)
        state, _ = envs.reset(seed=args.seed)
    else:
        state, _ = envs.reset()

    # Create policy network and optimizer
    policy = ActorCriticNet(observation_shape, action_dim, args.actor_layers, args.critic_layers)
    optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-5)

    # Create buffers
    rollout_buffer = RolloutBuffer(args.num_steps, args.num_envs, observation_shape, action_shape)

    # Remove unnecessary variables
    del action_dim

    global_step = 0
    log_episodic_returns, log_episodic_lengths = [], []
    start_time = time.process_time()

    # Main loop
    for iter in tqdm(range(args.num_updates)):
        for _ in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            with torch.no_grad():
                # Get action
                action, value = policy(torch.from_numpy(state).float())

            # Perform action
            action = action.cpu().numpy()
            next_state, reward, terminated, truncated, infos = envs.step(action)

            # Store transition
            flag = 1.0 - np.logical_or(terminated, truncated)
            value = value.cpu().numpy()
            rollout_buffer.push(state, action, reward, flag, value)

            state = next_state

            if "final_info" not in infos:
                continue

            # Log episodic return and length
            for info in infos["final_info"]:
                if info is None:
                    continue

                log_episodic_returns.append(info["episode"]["r"])
                log_episodic_lengths.append(info["episode"]["l"])
                writer.add_scalar("rollout/episodic_return", np.mean(log_episodic_returns[-5:]), global_step)
                writer.add_scalar("rollout/episodic_length", np.mean(log_episodic_lengths[-5:]), global_step)

        # Get transition batch
        states, actions, rewards, flags, values = rollout_buffer.get()

        with torch.no_grad():
            last_value = policy.critic(torch.from_numpy(next_state).float())

        # Calculate advantages and TD target
        advantages = compute_advantages(rewards, flags, values, last_value, args)
        td_target = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch
        states = states.reshape(-1, *observation_shape)
        actions = actions.reshape(-1, *action_shape)
        td_target = td_target.reshape(-1)
        advantages = advantages.reshape(-1)

        # Compute losses
        log_probs, td_predict, entropy = policy.evaluate(states, actions)

        actor_loss = (-log_probs * advantages).mean()
        critic_loss = mse_loss(td_target, td_predict)
        entropy_loss = entropy.mean()

        loss = actor_loss + critic_loss * args.value_coef - entropy_loss * args.entropy_coef

        # Update policy network
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
        optimizer.step()

        # Log training metrics
        writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)
        writer.add_scalar("train/loss", loss, global_step)
        writer.add_scalar("train/actor_loss", actor_loss, global_step)
        writer.add_scalar("train/critic_loss", critic_loss, global_step)

        if iter % 1_000 == 0:
            torch.save(policy.state_dict(), f"{run_dir}/policy.pt")

    # Save final policy
    torch.save(policy.state_dict(), f"{run_dir}/policy.pt")
    print(f"Saved policy to {run_dir}/policy.pt")

    # Close the environment
    envs.close()
    writer.close()

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    mean_train_return = np.mean(log_episodic_returns[-indexes:])
    writer.add_scalar("rollout/mean_train_return", mean_train_return, global_step)

    return mean_train_return
```


```python
def eval_and_render(args, run_dir):
    # Create environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True, run_dir=run_dir)])

    # Metadata about the environment
    observation_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.shape
    action_dim = np.prod(action_shape)

    # Load policy
    policy = ActorCriticNet(observation_shape, action_dim, args.actor_layers, args.critic_layers)
    filename = f"{run_dir}/policy.pt"
    print(f"reading {filename}...")
    policy.load_state_dict(torch.load(filename))
    policy.eval()

    count_episodes = 0
    list_rewards = []

    state, _ = env.reset()

    # Run episodes
    while count_episodes < 30:
        with torch.no_grad():
            action, _ = policy(torch.from_numpy(state).float())

        action = action.cpu().numpy()
        state, _, _, _, infos = env.step(action)

        if "final_info" in infos:
            info = infos["final_info"][0]
            returns = info["episode"]["r"][0]
            count_episodes += 1
            list_rewards.append(returns)
            print(f"-> Episode {count_episodes}: {returns} returns")

    env.close()

    return np.mean(list_rewards)
```


```python
# Create run directory
run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
run_name = "A2C_PyTorch"
run_dir = Path(f"runs/a2c-{run_time}")
if not run_dir.exists():
    run_dir.mkdir()
with open(run_dir / "args.json", "w") as fp:
    json.dump(args.__dict__, fp)
```


```python
print(f"Commencing training of {run_name} on {args.env_id} for {args.total_timesteps} timesteps.")
print(f"Results will be saved to: {run_dir}")
mean_train_return = train(args=args, run_name=run_name, run_dir=run_dir)
print(f"Training - Mean returns achieved: {mean_train_return}.")
```


```python
print(f"Evaluating and capturing videos on {args.env_id}.")
mean_eval_return = eval_and_render(args=args, run_dir=run_dir)
print(f"Evaluation - Mean returns achieved: {mean_eval_return}.")
```


```python
from IPython.display import Video

Video('./half-cheetah-video.mp4')
```




<video src="/assets/videos/half-cheetah-video.mp4" controls  >
      Your browser does not support the <code>video</code> element.
    </video>


