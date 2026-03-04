# Flappy Bird DQN

Deep Q-Network that learns to play Flappy Bird from scratch. Trains for 500 episodes with epsilon-greedy exploration, replay buffer, and a target network. Every 50 episodes it opens a window so you can watch the agent play.

A pre-trained model (`dqn_flappy.pt`) is included so you can skip training and just watch it play.

## How the DQN works

- 3-layer neural net (128 hidden units) maps game observations to action values (flap or don't)
- Epsilon-greedy exploration starts at 100% random and decays to 5%
- Replay buffer stores the last 10k transitions, samples batches of 32
- Target network syncs every 1000 steps to stabilize training
- Discount factor of 0.99
