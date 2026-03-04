import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import flappy_bird_gymnasium

# --------------------
# Simple DQN
# --------------------
class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    eval_env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

    obs, _ = env.reset()
    obs_dim = len(obs)
    act_dim = env.action_space.n

    policy = DQN(obs_dim, act_dim)
    target = DQN(obs_dim, act_dim)
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    memory = []

    gamma = 0.99
    batch_size = 32
    epsilon = 1.0
    eps_decay = 0.995
    eps_min = 0.05

    steps = 0

    for episode in range(1, 501):
        obs, _ = env.reset()
        state = torch.tensor(obs, dtype=torch.float32)
        total_reward = 0

        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randrange(act_dim)
            else:
                with torch.no_grad():
                    action = policy(state).argmax().item()

            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = None if done else torch.tensor(obs2, dtype=torch.float32)

            memory.append((state, action, reward, next_state, done))
            if len(memory) > 10_000:
                memory.pop(0)

            state = next_state if next_state is not None else state
            total_reward += reward
            steps += 1

            # Train
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards).unsqueeze(1)
                dones = torch.tensor(dones).unsqueeze(1)

                q_vals = policy(states).gather(1, actions)

                with torch.no_grad():
                    next_q = torch.zeros(batch_size, 1)
                    idx = [i for i, ns in enumerate(next_states) if ns is not None]
                    if idx:
                        next_states_t = torch.stack([next_states[i] for i in idx])
                        next_q[idx] = target(next_states_t).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * next_q * (~dones)

                loss = nn.MSELoss()(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % 1000 == 0:
                target.load_state_dict(policy.state_dict())

        epsilon = max(eps_min, epsilon * eps_decay)
        print(f"ep={episode} reward={total_reward:.2f} eps={epsilon:.3f}")

        # WATCH IT PLAY
        if episode % 50 == 0:
            obs, _ = eval_env.reset()
            s = torch.tensor(obs, dtype=torch.float32)
            done = False
            while not done:
                with torch.no_grad():
                    a = policy(s).argmax().item()
                obs, _, term, trunc, _ = eval_env.step(a)
                done = term or trunc
                if not done:
                    s = torch.tensor(obs, dtype=torch.float32)

    torch.save(policy.state_dict(), "dqn_flappy.pt")
    print("Model saved to dqn_flappy.pt")
    env.close()
    eval_env.close()


def play(model_path="dqn_flappy.pt"):
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    obs, _ = env.reset()
    obs_dim = len(obs)
    act_dim = env.action_space.n

    policy = DQN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()

    while True:
        obs, _ = env.reset()
        state = torch.tensor(obs, dtype=torch.float32)
        done = False
        score = 0
        while not done:
            with torch.no_grad():
                action = policy(state).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            if not done:
                state = torch.tensor(obs, dtype=torch.float32)
        print(f"score={score:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true", help="Load saved model and watch it play")
    parser.add_argument("--model", default="dqn_flappy.pt", help="Path to saved model")
    args = parser.parse_args()

    if args.play:
        play(args.model)
    else:
        main()
